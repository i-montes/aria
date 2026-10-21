//! Worktrees for subagents that write: each gets its own checkout at `<projects dir>/.aria-worktrees/<repo>-<job id>`,
//! made from the person's uncommitted changes, so it sees the folder as they left it and two of them never step on each
//! other or on the person. They live outside the person's repository, so its test runners, watchers and compilers never
//! see them, and nothing in the repository is edited besides git's own worktree bookkeeping. When it is done, its
//! changes come back as a plain `git apply` to the person's working tree: nothing is committed or staged for them, and a
//! patch that does not apply leaves the worktree in place for the person to look at. Untracked files of the person are
//! not copied into the worktree (`git stash create` leaves them out).

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Mutex, OnceLock};

/// Folder, in the projects folder, that holds every worktree.
pub const DIR: &str = ".aria-worktrees";

/// Settings that would make git run a program of the repository's choosing. The repository's config and hooks can be
/// written by a boxed command, and the dock runs git outside the box, so every git it runs turns them off: hooks, the
/// file-system monitor, signing and ssh here, filter drivers by [`neutral`] (which needs their names first).
const NEUTRAL: [&str; 10] = [
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.untrackedCache=false",
    "-c",
    "commit.gpgSign=false",
    "-c",
    "core.sshCommand=false",
];

/// Runs git in `dir` with [`NEUTRAL`] and then `extra` (from [`neutral`]). `git stash create` makes a commit object,
/// which needs an identity the person may not have set, so one is given here; it never ends up in the person's history.
fn git(dir: &Path, extra: &[String], args: &[&str], stdin: Option<&[u8]>) -> io::Result<Output> {
    let mut command = Command::new("git");
    command
        .arg("-C")
        .arg(dir)
        .args(NEUTRAL)
        .args(extra)
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
fn git_ok(dir: &Path, extra: &[String], args: &[&str]) -> io::Result<String> {
    let out = git(dir, extra, args, None)?;
    if !out.status.success() {
        return Err(io::Error::other(format!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The names of the filter drivers git sees in `dir`, from `filter.<name>.<setting>` keys.
fn filter_names(dir: &Path) -> io::Result<Vec<String>> {
    let out = git(dir, &[], &["config", "--null", "--get-regexp", r"^filter\."], None)?;
    // 1: no such key, which is the common case.
    if !out.status.success() && out.status.code() != Some(1) {
        return Err(io::Error::other(format!("git config: {}", String::from_utf8_lossy(&out.stderr).trim())));
    }
    let mut names: Vec<String> = Vec::new();
    for entry in out.stdout.split(|b| *b == 0) {
        // `--null`: the key, a newline, the value.
        let key = entry.split(|b| *b == b'\n').next().unwrap_or_default();
        let key = String::from_utf8_lossy(key);
        let Some(rest) = key.strip_prefix("filter.") else { continue };
        let Some((name, _)) = rest.rsplit_once('.') else { continue };
        if !names.iter().any(|n| n == name) {
            names.push(name.to_string());
        }
    }
    Ok(names)
}

/// The `-c` settings that turn off every filter driver git sees in any of `dirs` (an empty command is no filter), so
/// neither a clean nor a smudge program of the repository's runs and files go through as they are.
fn neutral(dirs: &[&Path]) -> io::Result<Vec<String>> {
    let mut extra = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for dir in dirs {
        for name in filter_names(dir)? {
            if seen.contains(&name) {
                continue;
            }
            for setting in ["clean=", "smudge=", "process=", "required=false"] {
                extra.push("-c".to_string());
                extra.push(format!("filter.{name}.{setting}"));
            }
            seen.push(name);
        }
    }
    Ok(extra)
}

/// The top of the git checkout `folder` is in, or `None` when it is not in one (a subagent there cannot get a
/// worktree).
pub fn git_root(folder: &Path) -> Option<PathBuf> {
    git_ok(folder, &[], &["rev-parse", "--show-toplevel"]).ok().filter(|root| !root.is_empty()).map(PathBuf::from)
}

/// A subagent's checkout and the commit it started from, which its changes are measured against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Worktree {
    pub path: PathBuf,
    pub base: String,
}

/// Makes the worktree for job `id` of the repository at `root`, as `<repo dir name>-<id>` in `worktrees`. It starts
/// from a stash commit of the person's uncommitted changes (made without touching their folder or stash list), or from
/// `HEAD` when there are none.
pub fn create(root: &Path, worktrees: &Path, id: &str) -> io::Result<Worktree> {
    debug_assert!(!id.is_empty() && !id.contains('/') && id != "." && id != "..", "a job id names one folder: {id:?}");
    let off = neutral(&[root])?;
    let stash = git_ok(root, &off, &["stash", "create"])?;
    let base = if stash.is_empty() { git_ok(root, &off, &["rev-parse", "HEAD"])? } else { stash };
    let repo = root.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_else(|| "repo".into());
    fs::create_dir_all(worktrees)?;
    let path = worktrees.join(format!("{repo}-{id}"));
    let path_str = path.to_string_lossy();
    git_ok(root, &off, &["worktree", "add", "--detach", &path_str, &base])?;
    Ok(Worktree { path, base })
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

/// One lock per repository: two subagents that finish at once never apply their patches into it together.
fn root_lock(root: &Path) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
    let mut locks = LOCKS.get_or_init(Mutex::default).lock().unwrap_or_else(|e| e.into_inner());
    locks.entry(root.to_path_buf()).or_default().clone()
}

/// Brings the worktree's changes (edits, new and deleted files, binaries) back to the person's folder as a patch
/// against the commit it started from. One at a time per repository.
pub fn bring_back(root: &Path, wt: &Worktree) -> io::Result<Back> {
    let lock = root_lock(root);
    let _one_at_a_time = lock.lock().unwrap_or_else(|e| e.into_inner());
    // Read again now: the subagent may have added a driver while it worked.
    let off = neutral(&[root, &wt.path])?;
    // Staged in the worktree only, so new files are part of the diff; the person's index is never touched.
    git_ok(&wt.path, &off, &["add", "-A"])?;
    // Plumbing, not `git diff`: the person's diff settings (`diff.noprefix`, an external diff, textconv) would change
    // the patch, and a patch without its `a/` `b/` prefixes applies with every path one folder short.
    let diff = git(&wt.path, &off, &["diff-index", "-p", "--binary", "--cached", &wt.base], None)?;
    if !diff.status.success() {
        return Err(io::Error::other(format!("git diff: {}", String::from_utf8_lossy(&diff.stderr).trim())));
    }
    if diff.stdout.is_empty() {
        remove_with(root, &wt.path, &off)?;
        return Ok(Back::Empty);
    }
    // `--whitespace=nowarn`: the person's `apply.whitespace` must neither refuse the patch nor rewrite it.
    let applied = git(root, &off, &["apply", "--whitespace=nowarn"], Some(&diff.stdout))?;
    if !applied.status.success() {
        return Ok(Back::Conflict(String::from_utf8_lossy(&applied.stderr).trim().to_string()));
    }
    // The changes are in the person's folder now: a worktree that will not go is only a leftover, never an error,
    // or a retry would apply them twice.
    if let Err(error) = remove_with(root, &wt.path, &off) {
        eprintln!("aria: worktree {} left behind: {error}", wt.path.display());
    }
    Ok(Back::Applied)
}

/// Removes a worktree, whatever is still in it: only called once its changes are back or the person dropped them.
pub fn remove(root: &Path, wt_path: &Path) -> io::Result<()> {
    remove_with(root, wt_path, &neutral(&[root])?)
}

fn remove_with(root: &Path, wt_path: &Path, off: &[String]) -> io::Result<()> {
    let path = wt_path.to_string_lossy();
    git_ok(root, off, &["worktree", "remove", "--force", &path]).map(|_| ())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::fs;
    use std::process::Command;

    /// Runs git in `dir` for a test, blind to the person's own git config.
    pub(crate) fn git(dir: &Path, args: &[&str]) -> String {
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
    pub(crate) fn repo() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        git(&root, &["init", "-q"]);
        fs::write(root.join("a.txt"), "1\n").unwrap();
        git(&root, &["add", "a.txt"]);
        git(&root, &["commit", "-q", "-m", "one"]);
        (dir, root)
    }

    /// Where the worktrees go: `.aria-worktrees` in a projects folder of its own.
    pub(crate) fn worktrees() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let spot = dir.path().canonicalize().unwrap().join(DIR);
        (dir, spot)
    }

    fn read(path: &Path) -> String {
        fs::read_to_string(path).unwrap()
    }

    #[test]
    fn the_worktree_starts_from_the_persons_uncommitted_changes() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        let (_spot, spot) = worktrees();
        fs::write(root.join("a.txt"), "2\n").unwrap();
        let before = git(&root, &["status", "--porcelain"]);
        let wt = create(&root, &spot, "j1").unwrap();
        assert_eq!(read(&wt.path.join("a.txt")), "2\n");
        assert_eq!(read(&root.join("a.txt")), "2\n");
        assert_eq!(git(&root, &["status", "--porcelain"]), before);
    }

    #[test]
    fn a_clean_repo_starts_from_head() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        let (_spot, spot) = worktrees();
        let head = git(&root, &["rev-parse", "HEAD"]).trim().to_string();
        let wt = create(&root, &spot, "j1").unwrap();
        assert_eq!(wt.base, head);
        assert_eq!(read(&wt.path.join("a.txt")), "1\n");
    }

    #[test]
    fn the_worktree_is_outside_the_repo_and_the_repo_is_not_edited() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        let (_spot, spot) = worktrees();
        let exclude = fs::read_to_string(root.join(".git/info/exclude")).ok();
        let one = create(&root, &spot, "j1").unwrap();
        let two = create(&root, &spot, "j2").unwrap();
        let name = root.file_name().unwrap().to_string_lossy().into_owned();
        assert_eq!(one.path, spot.join(format!("{name}-j1")));
        assert_eq!(two.path, spot.join(format!("{name}-j2")));
        assert!(!one.path.starts_with(&root));
        assert_eq!(fs::read_to_string(root.join(".git/info/exclude")).ok(), exclude, "no exclude line");
        assert_eq!(git(&root, &["status", "--porcelain"]), "");
        remove(&root, &one.path).unwrap();
        assert!(!one.path.exists());
        assert!(!git(&root, &["worktree", "list"]).contains(&format!("{name}-j1")));
    }

    #[test]
    fn changes_come_back_unstaged() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        let (_spot, spot) = worktrees();
        let wt = create(&root, &spot, "j1").unwrap();
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
        let (_spot, spot) = worktrees();
        git(&root, &["config", "diff.noprefix", "true"]);
        let wt = create(&root, &spot, "j1").unwrap();
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
        let (_spot, spot) = worktrees();
        git(&root, &["config", "apply.whitespace", "error"]);
        let wt = create(&root, &spot, "j1").unwrap();
        fs::write(wt.path.join("a.txt"), "1  \n").unwrap();
        assert_eq!(bring_back(&root, &wt).unwrap(), Back::Applied);
        assert_eq!(read(&root.join("a.txt")), "1  \n");
    }

    #[test]
    fn a_clash_keeps_the_worktree() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        let (_spot, spot) = worktrees();
        let wt = create(&root, &spot, "j1").unwrap();
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
        let (_spot, spot) = worktrees();
        let wt = create(&root, &spot, "j1").unwrap();
        assert_eq!(bring_back(&root, &wt).unwrap(), Back::Empty);
        assert!(!wt.path.exists());
    }

    #[test]
    fn the_repos_own_programs_never_run() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        let (_spot, spot) = worktrees();
        let marks = tempfile::tempdir().unwrap();
        let mark = |name: &str| marks.path().join(name).display().to_string();
        let hook = root.join(".git/hooks/post-checkout");
        fs::write(&hook, format!("#!/bin/sh\ntouch {}\n", mark("hook"))).unwrap();
        fs::set_permissions(&hook, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
        // A filter that would leave a mark and change what it reads.
        let clean = format!("touch {}; tr a-z A-Z", mark("clean"));
        let smudge = format!("touch {}; tr a-z A-Z", mark("smudge"));
        fs::write(root.join(".gitattributes"), "* filter=evil\n").unwrap();
        git(&root, &["add", ".gitattributes"]);
        git(&root, &["commit", "-q", "-m", "attributes"]);
        git(&root, &["config", "filter.evil.clean", &clean]);
        git(&root, &["config", "filter.evil.smudge", &smudge]);
        git(&root, &["config", "filter.evil.required", "true"]);
        // Last: the test's own git would run it.
        let monitor = marks.path().join("monitor.sh");
        fs::write(&monitor, format!("#!/bin/sh\ntouch {}\n", mark("fsmonitor"))).unwrap();
        fs::set_permissions(&monitor, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
        git(&root, &["config", "core.fsmonitor", monitor.to_str().unwrap()]);
        fs::write(root.join("a.txt"), "two\n").unwrap();
        let wt = create(&root, &spot, "j1").unwrap();
        assert_eq!(read(&wt.path.join("a.txt")), "two\n", "checked out as it is");
        fs::write(wt.path.join("a.txt"), "three\n").unwrap();
        fs::write(wt.path.join("new.txt"), "new\n").unwrap();
        assert_eq!(bring_back(&root, &wt).unwrap(), Back::Applied);
        assert_eq!(read(&root.join("a.txt")), "three\n", "no filter between the worktree and the folder");
        assert_eq!(read(&root.join("new.txt")), "new\n");
        let left: Vec<_> = fs::read_dir(marks.path()).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(left, vec![std::ffi::OsString::from("monitor.sh")], "no program of the repo ran");
    }

    #[test]
    fn a_driver_the_subagent_adds_does_not_run_either() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        let (_spot, spot) = worktrees();
        let marks = tempfile::tempdir().unwrap();
        let wt = create(&root, &spot, "j1").unwrap();
        // Written from inside the box while the subagent works: the common config and an attributes file.
        let clean = format!("touch {}; tr a-z A-Z", marks.path().join("clean").display());
        git(&wt.path, &["config", "filter.late.clean", &clean]);
        fs::write(wt.path.join(".gitattributes"), "*.txt filter=late\n").unwrap();
        fs::write(wt.path.join("a.txt"), "changed\n").unwrap();
        assert_eq!(bring_back(&root, &wt).unwrap(), Back::Applied);
        assert_eq!(read(&root.join("a.txt")), "changed\n");
        assert_eq!(fs::read_dir(marks.path()).unwrap().count(), 0, "the late filter never ran");
    }

    #[test]
    fn one_repo_brings_back_one_at_a_time() {
        let (a, b) = (Path::new("/repo/a"), Path::new("/repo/b"));
        assert!(Arc::ptr_eq(&root_lock(a), &root_lock(a)), "one lock per repository");
        assert!(!Arc::ptr_eq(&root_lock(a), &root_lock(b)), "another repository does not wait");
        let held = root_lock(a);
        let _guard = held.lock().unwrap();
        assert!(root_lock(a).try_lock().is_err(), "a second bring-back in the same repository waits");
        assert!(root_lock(b).try_lock().is_ok());
    }

    #[test]
    fn two_agents_finishing_at_once_both_come_back() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (_dir, root) = repo();
        let (_spot, spot) = worktrees();
        let one = create(&root, &spot, "j1").unwrap();
        let two = create(&root, &spot, "j2").unwrap();
        fs::write(one.path.join("one.txt"), "1\n").unwrap();
        fs::write(two.path.join("two.txt"), "2\n").unwrap();
        let (r1, r2) = (root.clone(), root.clone());
        let first = std::thread::spawn(move || bring_back(&r1, &one).unwrap());
        let second = std::thread::spawn(move || bring_back(&r2, &two).unwrap());
        assert_eq!(first.join().unwrap(), Back::Applied);
        assert_eq!(second.join().unwrap(), Back::Applied);
        assert_eq!(read(&root.join("one.txt")), "1\n");
        assert_eq!(read(&root.join("two.txt")), "2\n");
    }

    #[test]
    #[should_panic(expected = "a job id names one folder")]
    fn an_id_cannot_leave_the_worktrees_folder() {
        // Checked before any git runs, so no repo (and no `SERIAL`, which a panic would poison) is needed.
        let _ = create(Path::new("/nonexistent"), Path::new("/nonexistent"), "../escape");
    }

    #[test]
    fn a_plain_folder_has_no_git_root() {
        let _serial = crate::SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(git_root(dir.path()), None);
    }
}
