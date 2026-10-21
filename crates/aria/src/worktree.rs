//! Worktrees for subagents that write: each gets its own checkout at `<git root>/.aria-worktrees/<job id>`, made from
//! the person's uncommitted changes, so it sees the folder as they left it and two of them never step on each other or
//! on the person. When it is done, its changes come back as a plain `git apply` to the person's working tree: nothing is
//! committed or staged for them, and a patch that does not apply leaves the worktree in place for the person to look
//! at. Untracked files of the person are not copied into the worktree (`git stash create` leaves them out).

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

/// Folder, under the git root, that holds every worktree; also the line that keeps it out of `git status`.
const DIR: &str = ".aria-worktrees";
const EXCLUDE_LINE: &str = ".aria-worktrees/";

/// Runs git in `dir`. `git stash create` makes a commit object, which needs an identity the person may not have set,
/// so one is given here; it never ends up in the person's history.
fn git(dir: &Path, args: &[&str], stdin: Option<&[u8]>) -> io::Result<Output> {
    let mut command = Command::new("git");
    command
        .arg("-C")
        .arg(dir)
        .args(["-c", "user.name=Aria", "-c", "user.email=aria@localhost"])
        .args(args)
        .stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // The tests must not depend on the machine's git config; the person's repo config still applies.
    #[cfg(test)]
    command.env("GIT_CONFIG_GLOBAL", "/dev/null").env("GIT_CONFIG_NOSYSTEM", "1");
    let mut child = command.spawn()?;
    // Taken so it closes before the wait, or git would wait for more input.
    let written = match (stdin, child.stdin.take()) {
        (Some(input), Some(mut pipe)) => pipe.write_all(input),
        _ => Ok(()),
    };
    // Waited for even when the write failed, so no zombie is left behind.
    let out = child.wait_with_output()?;
    match written {
        // A git that stopped reading has failed; its status and stderr say why.
        Err(e) if e.kind() != io::ErrorKind::BrokenPipe => Err(e),
        _ => Ok(out),
    }
}

/// Like [`git`], but a failure is an error carrying git's own words, and the output is the trimmed stdout.
fn git_ok(dir: &Path, args: &[&str]) -> io::Result<String> {
    let out = git(dir, args, None)?;
    if !out.status.success() {
        return Err(io::Error::other(format!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The top of the git checkout `folder` is in, or `None` when it is not in one (a subagent there cannot get a
/// worktree).
pub fn git_root(folder: &Path) -> Option<PathBuf> {
    git_ok(folder, &["rev-parse", "--show-toplevel"]).ok().filter(|root| !root.is_empty()).map(PathBuf::from)
}

/// A subagent's checkout and the commit it started from, which its changes are measured against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Worktree {
    pub path: PathBuf,
    pub base: String,
}

/// Makes the worktree for job `id` under `root`. It starts from a stash commit of the person's uncommitted changes
/// (made without touching their folder or stash list), or from `HEAD` when there are none.
pub fn create(root: &Path, id: &str) -> io::Result<Worktree> {
    debug_assert!(!id.is_empty() && !id.contains('/') && id != "." && id != "..", "a job id names one folder: {id:?}");
    let stash = git_ok(root, &["stash", "create"])?;
    let base = if stash.is_empty() { git_ok(root, &["rev-parse", "HEAD"])? } else { stash };
    // Before the worktree exists, so the person's `git status` never shows it.
    exclude(root)?;
    let path = root.join(DIR).join(id);
    let path_str = path.to_string_lossy();
    git_ok(root, &["worktree", "add", "--detach", &path_str, &base])?;
    Ok(Worktree { path, base })
}

/// Adds `.aria-worktrees/` to the repository's own exclude file (shared by every worktree, never committed), once.
fn exclude(root: &Path) -> io::Result<()> {
    let common = PathBuf::from(git_ok(root, &["rev-parse", "--git-common-dir"])?);
    let common = if common.is_relative() { root.join(common) } else { common };
    let info = common.join("info");
    fs::create_dir_all(&info)?;
    let file = info.join("exclude");
    let current = match fs::read_to_string(&file) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e),
    };
    if current.lines().any(|line| line == EXCLUDE_LINE) {
        return Ok(());
    }
    let mut text = current;
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(EXCLUDE_LINE);
    text.push('\n');
    fs::write(&file, text)
}

/// What became of a subagent's changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Back {
    /// They are in the person's working tree, unstaged, and the worktree is gone.
    Applied,
    /// There were none; the worktree is gone.
    Empty,
    /// They clash with what the person changed since; the worktree is kept. Carries git's explanation.
    Conflict(String),
}

/// Brings the worktree's changes (edits, new and deleted files, binaries) back to the person's folder as a patch
/// against the commit it started from.
pub fn bring_back(root: &Path, wt: &Worktree) -> io::Result<Back> {
    // Staged in the worktree only, so new files are part of the diff; the person's index is never touched.
    git_ok(&wt.path, &["add", "-A"])?;
    // Plumbing, not `git diff`: the person's diff settings (`diff.noprefix`, an external diff, textconv) would change
    // the patch, and a patch without its `a/` `b/` prefixes applies with every path one folder short.
    let diff = git(&wt.path, &["diff-index", "-p", "--binary", "--cached", &wt.base], None)?;
    if !diff.status.success() {
        return Err(io::Error::other(format!("git diff: {}", String::from_utf8_lossy(&diff.stderr).trim())));
    }
    if diff.stdout.is_empty() {
        remove(root, &wt.path)?;
        return Ok(Back::Empty);
    }
    // `--whitespace=nowarn`: the person's `apply.whitespace` must neither refuse the patch nor rewrite it.
    let applied = git(root, &["apply", "--whitespace=nowarn"], Some(&diff.stdout))?;
    if !applied.status.success() {
        return Ok(Back::Conflict(String::from_utf8_lossy(&applied.stderr).trim().to_string()));
    }
    // The changes are in the person's folder now: a worktree that will not go is only a leftover, never an error,
    // or a retry would apply them twice.
    if let Err(error) = remove(root, &wt.path) {
        eprintln!("aria: worktree {} left behind: {error}", wt.path.display());
    }
    Ok(Back::Applied)
}

/// Removes a worktree, whatever is still in it: only called once its changes are back or the person dropped them.
pub fn remove(root: &Path, wt_path: &Path) -> io::Result<()> {
    let path = wt_path.to_string_lossy();
    git_ok(root, &["worktree", "remove", "--force", &path]).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process::Command;

    /// Runs git in `dir` for a test, blind to the person's own git config.
    fn git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(["-c", "user.email=a@b", "-c", "user.name=a"])
            .args(args)
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap()
    }

    /// A repo with `a.txt` committed as "1\n".
    fn repo() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        git(&root, &["init", "-q"]);
        fs::write(root.join("a.txt"), "1\n").unwrap();
        git(&root, &["add", "a.txt"]);
        git(&root, &["commit", "-q", "-m", "one"]);
        (dir, root)
    }

    fn read(path: &Path) -> String {
        fs::read_to_string(path).unwrap()
    }

    #[test]
    fn the_worktree_starts_from_the_persons_uncommitted_changes() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        fs::write(root.join("a.txt"), "2\n").unwrap();
        let before = git(&root, &["status", "--porcelain"]);
        let wt = create(&root, "j1").unwrap();
        assert_eq!(wt.path, root.join(".aria-worktrees/j1"));
        assert_eq!(read(&wt.path.join("a.txt")), "2\n");
        assert_eq!(read(&root.join("a.txt")), "2\n");
        assert_eq!(git(&root, &["status", "--porcelain"]), before);
    }

    #[test]
    fn a_clean_repo_starts_from_head() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        let head = git(&root, &["rev-parse", "HEAD"]).trim().to_string();
        let wt = create(&root, "j1").unwrap();
        assert_eq!(wt.base, head);
        assert_eq!(read(&wt.path.join("a.txt")), "1\n");
    }

    #[test]
    fn the_exclude_line_is_written_once() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        create(&root, "j1").unwrap();
        create(&root, "j2").unwrap();
        let exclude = read(&root.join(".git/info/exclude"));
        assert_eq!(exclude.lines().filter(|l| *l == ".aria-worktrees/").count(), 1);
        assert!(!git(&root, &["status", "--porcelain"]).contains(".aria-worktrees"));
    }

    #[test]
    fn changes_come_back_unstaged() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        let wt = create(&root, "j1").unwrap();
        fs::write(wt.path.join("a.txt"), "3\n").unwrap();
        fs::write(wt.path.join("new.txt"), "new\n").unwrap();
        assert_eq!(bring_back(&root, &wt).unwrap(), Back::Applied);
        assert_eq!(read(&root.join("a.txt")), "3\n");
        assert_eq!(read(&root.join("new.txt")), "new\n");
        assert_eq!(git(&root, &["diff", "--cached"]), "");
        assert!(!wt.path.exists());
    }

    #[test]
    fn the_persons_diff_config_does_not_shape_the_patch() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        git(&root, &["config", "diff.noprefix", "true"]);
        let wt = create(&root, "j1").unwrap();
        fs::create_dir(wt.path.join("src")).unwrap();
        fs::write(wt.path.join("src/new.txt"), "new\n").unwrap();
        assert_eq!(bring_back(&root, &wt).unwrap(), Back::Applied);
        assert_eq!(read(&root.join("src/new.txt")), "new\n");
        assert!(!root.join("new.txt").exists());
    }

    #[test]
    fn the_persons_whitespace_rule_does_not_refuse_the_patch() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        git(&root, &["config", "apply.whitespace", "error"]);
        let wt = create(&root, "j1").unwrap();
        fs::write(wt.path.join("a.txt"), "1  \n").unwrap();
        assert_eq!(bring_back(&root, &wt).unwrap(), Back::Applied);
        assert_eq!(read(&root.join("a.txt")), "1  \n");
    }

    #[test]
    fn a_clash_keeps_the_worktree() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        let wt = create(&root, "j1").unwrap();
        fs::write(wt.path.join("a.txt"), "3\n").unwrap();
        fs::write(root.join("a.txt"), "4\n").unwrap();
        assert!(matches!(bring_back(&root, &wt).unwrap(), Back::Conflict(_)));
        assert!(wt.path.exists());
        assert_eq!(read(&root.join("a.txt")), "4\n");
    }

    #[test]
    fn nothing_changed_is_empty() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        let wt = create(&root, "j1").unwrap();
        assert_eq!(bring_back(&root, &wt).unwrap(), Back::Empty);
        assert!(!wt.path.exists());
    }

    #[test]
    #[should_panic(expected = "a job id names one folder")]
    fn an_id_cannot_leave_the_worktrees_folder() {
        // Checked before any git runs, so no repo (and no `SERIAL`, which a panic would poison) is needed.
        let _ = create(Path::new("/nonexistent"), "../escape");
    }

    #[test]
    fn a_plain_folder_has_no_git_root() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(git_root(dir.path()), None);
    }
}
