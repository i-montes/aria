//! A session's folder kept as it was before each of the person's messages (docs/superpowers/specs/
//! 2026-10-23-terminal-m2-design.md): a git repository of its own in `~/.aria/checkpoints/<session>/` (`--git-dir`
//! there, the folder as the work tree), so the project's own `.git` is never touched. The project's `.gitignore` is
//! respected, `.git` and any file over 5 MB are left out. An instruction's changes are the difference between its
//! checkpoint and the folder now; undo puts back what changed since a checkpoint and nothing else.

use serde::{Deserialize, Serialize};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

/// Files bigger than this are never kept.
const BIG: u64 = 5 * 1024 * 1024;
/// Checkpoints listed (undo can go back this far).
const KEPT: usize = 50;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// The snapshot's commit.
    pub id: String,
    /// The message it was taken before.
    pub text: String,
    /// When (ms).
    pub at: i64,
    /// Files that instruction changed (once its run settled).
    pub files: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Changes {
    pub files: u32,
    pub added: u32,
    pub removed: u32,
}

/// A path as a literal gitignore pattern.
fn escape_pattern(path: &str) -> String {
    path.replace('\\', "\\\\").replace('*', "\\*").replace('?', "\\?").replace('[', "\\[").replace('!', "\\!")
}

/// Was `file` made after `at` (ms)? Its birth time where the system keeps one, else its last change.
fn born_after(file: &Path, at: i64) -> bool {
    let Ok(meta) = std::fs::symlink_metadata(file) else { return false };
    let when = meta.created().or_else(|_| meta.modified());
    when.ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).is_some_and(|d| d.as_millis() as i64 >= at)
}

pub struct Checkpoints {
    git_dir: PathBuf,
    folder: PathBuf,
    /// A global git config to read instead of the person's (tests).
    pub(crate) global_config: Option<PathBuf>,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

impl Checkpoints {
    /// The checkpoints of `session` working in `folder`; none for a folder that is not there or is the home folder.
    pub fn open(root: &Path, session: &str, folder: &Path) -> Option<Checkpoints> {
        let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
        Checkpoints::open_in(root, session, folder, &home)
    }

    pub fn open_in(root: &Path, session: &str, folder: &Path, home: &Path) -> Option<Checkpoints> {
        let folder = folder.canonicalize().ok()?;
        if !folder.is_dir() || home.canonicalize().is_ok_and(|home| home == folder) {
            return None;
        }
        let git_dir = root.join(session);
        let made = Checkpoints { git_dir, folder, global_config: None };
        if !made.git_dir.join("HEAD").exists() {
            std::fs::create_dir_all(&made.git_dir).ok()?;
            made.git(&["init", "-q"]).ok()?;
        }
        Some(made)
    }

    fn git(&self, args: &[&str]) -> io::Result<Output> {
        let mut command = Command::new("git");
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("GIT_") {
                command.env_remove(key);
            }
        }
        command
            .arg("--git-dir")
            .arg(&self.git_dir)
            .arg("--work-tree")
            .arg(&self.folder)
            .args(["-c", "user.name=aria", "-c", "user.email=aria@localhost", "-c", "core.autocrlf=false"])
            // The person's signing and hooks are for their commits, never Aria's own.
            .args(["-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", "-c", "core.hooksPath=/dev/null"])
            .args(args)
            .current_dir(&self.folder)
            .stdin(Stdio::null());
        if let Some(global) = &self.global_config {
            command.env("GIT_CONFIG_GLOBAL", global);
        }
        let output = command.output()?;
        if !output.status.success() {
            return Err(io::Error::other(format!(
                "git {}: {}",
                args.join(" "),
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        Ok(output)
    }

    /// Paths (from `-z` output) of files over the size limit, which never enter a checkpoint.
    fn big_files(&self) -> io::Result<Vec<String>> {
        let out = self.git(&["ls-files", "-z", "--cached", "--others", "--exclude-standard"])?;
        Ok(String::from_utf8_lossy(&out.stdout)
            .split('\0')
            .filter(|path| !path.is_empty())
            .filter(|path| std::fs::metadata(self.folder.join(path)).is_ok_and(|m| m.len() > BIG))
            .map(str::to_string)
            .collect())
    }

    /// What the project's own repository ignores in this folder (its `.gitignore` above the folder, its own excludes):
    /// a package's `node_modules` ignored at a monorepo's root is not the session's to keep. Empty outside a repository.
    fn ignored_by_the_project(&self) -> Vec<String> {
        let mut command = Command::new("git");
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("GIT_") {
                command.env_remove(key);
            }
        }
        let out = command
            .args(["ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory"])
            .current_dir(&self.folder)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output();
        match out {
            Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
                .split('\0')
                .filter(|p| !p.is_empty() && !p.starts_with("../"))
                .map(str::to_string)
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The folder as it is now, as a commit.
    fn snapshot(&self) -> io::Result<String> {
        let big = self.big_files()?;
        let mut exclude = String::from(".git\n");
        for path in self.ignored_by_the_project() {
            exclude.push('/');
            exclude.push_str(&escape_pattern(&path));
            exclude.push('\n');
        }
        for path in &big {
            exclude.push('/');
            exclude.push_str(&escape_pattern(path));
            exclude.push('\n');
        }
        std::fs::create_dir_all(self.git_dir.join("info"))?;
        std::fs::write(self.git_dir.join("info").join("exclude"), exclude)?;
        self.git(&["add", "-A", "."])?;
        if !big.is_empty() {
            let mut args = vec!["rm", "-q", "--cached", "--ignore-unmatch", "--"];
            args.extend(big.iter().map(String::as_str));
            self.git(&args)?;
        }
        self.git(&["commit", "-q", "--allow-empty", "--no-verify", "-m", "checkpoint"])?;
        let head = self.git(&["rev-parse", "HEAD"])?;
        Ok(String::from_utf8_lossy(&head.stdout).trim().to_string())
    }

    fn index_file(&self) -> PathBuf {
        self.git_dir.join("checkpoints.json")
    }

    fn read_index(&self) -> Vec<Checkpoint> {
        std::fs::read(self.index_file()).ok().and_then(|bytes| serde_json::from_slice(&bytes).ok()).unwrap_or_default()
    }

    fn write_index(&self, list: &[Checkpoint]) -> io::Result<()> {
        let start = list.len().saturating_sub(KEPT);
        std::fs::write(self.index_file(), serde_json::to_vec(&list[start..]).map_err(io::Error::other)?)
    }

    /// Keeps the folder as it is now, before the message `text`.
    pub fn take(&self, text: &str) -> io::Result<Checkpoint> {
        // When it starts, not when it ends: what is born after this moment is the turn's.
        let at = now_ms();
        let checkpoint = Checkpoint { id: self.snapshot()?, text: text.to_string(), at, files: 0 };
        let mut list = self.read_index();
        list.push(checkpoint.clone());
        self.write_index(&list)?;
        Ok(checkpoint)
    }

    /// What changed in the folder since `checkpoint`.
    pub fn changes_since(&self, checkpoint: &Checkpoint) -> io::Result<Changes> {
        let now = self.snapshot()?;
        let out = self.git(&["diff", "--numstat", &checkpoint.id, &now])?;
        let mut changes = Changes::default();
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            let mut parts = line.split('\t');
            let added = parts.next().and_then(|n| n.parse::<u32>().ok()).unwrap_or(0);
            let removed = parts.next().and_then(|n| n.parse::<u32>().ok()).unwrap_or(0);
            changes.files += 1;
            changes.added += added;
            changes.removed += removed;
        }
        Ok(changes)
    }

    /// Remembers how many files the instruction after `checkpoint` changed (the undo list shows it).
    pub fn set_files(&self, checkpoint: &Checkpoint, files: u32) -> io::Result<()> {
        let mut list = self.read_index();
        if let Some(entry) = list.iter_mut().find(|c| c.id == checkpoint.id) {
            entry.files = files;
        }
        self.write_index(&list)
    }

    /// Puts the folder back as it was at `checkpoint`: what changed since comes back, what was made since goes, what
    /// was deleted since returns; nothing the checkpoint does not cover moves.
    pub fn restore(&self, checkpoint: &Checkpoint) -> io::Result<()> {
        let now = self.snapshot()?;
        let made = self.git(&["diff", "--name-only", "-z", "--diff-filter=A", &checkpoint.id, &now])?;
        for path in String::from_utf8_lossy(&made.stdout).split('\0').filter(|p| !p.is_empty()) {
            // New to the snapshot is not new to the folder: a file the checkpoint left out (ignored then, or too big)
            // may show up now. Only a file born after the checkpoint goes.
            let file = self.folder.join(path);
            // Filesystems stamp with a coarse clock, a few ms behind: a margin keeps a file made right after the
            // checkpoint from looking older than it.
            if born_after(&file, checkpoint.at - 20) {
                let _ = std::fs::remove_file(file);
            }
        }
        // A checkpoint of an empty folder has nothing to bring back (and git refuses a pathspec that matches nothing).
        let kept = self.git(&["ls-tree", "-r", "--name-only", &checkpoint.id])?;
        if !kept.stdout.is_empty() {
            self.git(&["checkout", "-q", &checkpoint.id, "--", "."])?;
        }
        Ok(())
    }

    /// The checkpoints undo can go back to, newest first.
    pub fn list(&self) -> Vec<Checkpoint> {
        let mut list = self.read_index();
        list.reverse();
        list
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct Place {
        _dir: tempfile::TempDir,
        root: PathBuf,
        folder: PathBuf,
    }

    fn place() -> Place {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("checkpoints");
        let folder = dir.path().join("checkout-widget");
        fs::create_dir_all(folder.join("src")).unwrap();
        fs::write(folder.join("src/coupon.ts"), "a\nb\n").unwrap();
        fs::write(folder.join("notes.txt"), "mine\n").unwrap();
        Place { _dir: dir, root, folder }
    }

    fn open(p: &Place) -> Checkpoints {
        Checkpoints::open_in(&p.root, "s1", &p.folder, Path::new("/home/ana")).unwrap()
    }

    #[test]
    fn a_checkpoint_and_its_changes() {
        let p = place();
        let c = open(&p);
        let cp = c.take("fix the coupon").unwrap();
        assert_eq!(cp.text, "fix the coupon");
        fs::write(p.folder.join("src/coupon.ts"), "a\nb\nc\nd\n").unwrap();
        fs::write(p.folder.join("src/new.ts"), "x\n").unwrap();
        let changes = c.changes_since(&cp).unwrap();
        assert_eq!(changes, Changes { files: 2, added: 3, removed: 0 });
        assert_eq!(c.changes_since(&c.take("again").unwrap()).unwrap(), Changes::default(), "nothing since");
    }

    #[test]
    fn undo_touches_only_what_changed_since_the_checkpoint() {
        let p = place();
        fs::create_dir_all(p.folder.join("build")).unwrap();
        fs::write(p.folder.join("build/out.bin"), "old build").unwrap();
        let c = open(&p);
        let cp = c.take("change things").unwrap();
        fs::write(p.folder.join("src/coupon.ts"), "changed\n").unwrap();
        fs::remove_file(p.folder.join("notes.txt")).unwrap();
        fs::write(p.folder.join("src/new.ts"), "x\n").unwrap();
        c.restore(&cp).unwrap();
        assert_eq!(fs::read_to_string(p.folder.join("src/coupon.ts")).unwrap(), "a\nb\n", "an edited file comes back");
        assert_eq!(fs::read_to_string(p.folder.join("notes.txt")).unwrap(), "mine\n", "a deleted file returns");
        assert!(!p.folder.join("src/new.ts").exists(), "a file made since goes");
        assert_eq!(fs::read_to_string(p.folder.join("build/out.bin")).unwrap(), "old build", "what was there stays");
    }

    #[test]
    fn undo_in_a_folder_that_was_empty() {
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().join("empty");
        fs::create_dir_all(&folder).unwrap();
        let c = Checkpoints::open_in(&dir.path().join("cp"), "s1", &folder, Path::new("/home/ana")).unwrap();
        let cp = c.take("start").unwrap();
        fs::write(folder.join("made.txt"), "x").unwrap();
        c.restore(&cp).unwrap();
        assert!(!folder.join("made.txt").exists());
    }

    #[test]
    fn undo_never_deletes_a_file_the_checkpoint_left_out() {
        let p = place();
        fs::write(p.folder.join(".gitignore"), ".env\n").unwrap();
        fs::write(p.folder.join(".env"), "SECRET=1").unwrap();
        // A file that was there before, as files are: not in the last milliseconds before the message.
        std::thread::sleep(std::time::Duration::from_millis(60));
        let c = open(&p);
        let cp = c.take("drop the ignore line").unwrap();
        fs::write(p.folder.join(".gitignore"), "").unwrap();
        c.restore(&cp).unwrap();
        assert_eq!(fs::read_to_string(p.folder.join(".env")).unwrap(), "SECRET=1", "it was there before: it stays");
        assert_eq!(fs::read_to_string(p.folder.join(".gitignore")).unwrap(), ".env\n");
    }

    #[test]
    fn a_global_git_config_cannot_break_checkpoints() {
        let p = place();
        let global = p._dir.path().join("gitconfig");
        fs::write(
            &global,
            "[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = false\n[core]\n\thooksPath = /nonexistent\n",
        )
        .unwrap();
        let mut c = open(&p);
        c.global_config = Some(global);
        assert!(c.take("signed?").is_ok(), "the user's signing and hooks never apply to Aria's own commits");
    }

    #[test]
    fn what_the_enclosing_repo_ignores_is_left_out() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("monorepo");
        let folder = repo.join("packages").join("checkout-widget");
        fs::create_dir_all(folder.join("node_modules").join("lib")).unwrap();
        fs::write(folder.join("node_modules").join("lib").join("x.js"), "dep").unwrap();
        fs::write(folder.join("index.ts"), "a").unwrap();
        fs::write(repo.join(".gitignore"), "node_modules/\n").unwrap();
        std::process::Command::new("git").args(["init", "-q"]).current_dir(&repo).output().unwrap();
        let c = Checkpoints::open_in(&dir.path().join("cp"), "s1", &folder, Path::new("/home/ana")).unwrap();
        let cp = c.take("x").unwrap();
        fs::write(folder.join("node_modules").join("lib").join("x.js"), "changed dep").unwrap();
        assert_eq!(c.changes_since(&cp).unwrap(), Changes::default(), "node_modules is not the session's to keep");
    }

    #[test]
    fn big_files_are_left_out_of_a_checkpoint() {
        let p = place();
        fs::write(p.folder.join("video.mp4"), vec![0u8; 6 * 1024 * 1024]).unwrap();
        let c = open(&p);
        let cp = c.take("x").unwrap();
        fs::remove_file(p.folder.join("video.mp4")).unwrap();
        c.restore(&cp).unwrap();
        assert!(!p.folder.join("video.mp4").exists(), "a 6 MB file is not in the checkpoint");
        assert_eq!(c.changes_since(&cp).unwrap(), Changes::default());
    }

    #[test]
    fn a_projects_own_git_is_never_touched() {
        let p = place();
        let git = |args: &[&str]| std::process::Command::new("git").args(args).current_dir(&p.folder).output().unwrap();
        git(&["init", "-q"]);
        let before = fs::read_dir(p.folder.join(".git")).unwrap().count();
        let c = open(&p);
        c.take("x").unwrap();
        assert_eq!(fs::read_dir(p.folder.join(".git")).unwrap().count(), before);
        assert!(
            String::from_utf8_lossy(&git(&["status", "--porcelain"]).stdout).contains("notes.txt"),
            "nothing staged in it"
        );
    }

    #[test]
    fn no_checkpoints_in_home() {
        let p = place();
        assert!(Checkpoints::open_in(&p.root, "s1", &p.folder, &p.folder).is_none());
        assert!(Checkpoints::open_in(&p.root, "s1", &p.folder.join("gone"), Path::new("/home/ana")).is_none());
    }

    #[test]
    fn the_list_keeps_fifty_newest_first() {
        let p = place();
        let c = open(&p);
        for n in 0..53 {
            fs::write(p.folder.join("n.txt"), n.to_string()).unwrap();
            c.take(&format!("message {n}")).unwrap();
        }
        let list = c.list();
        assert_eq!(list.len(), 50);
        assert_eq!(list[0].text, "message 52");
        assert_eq!(list[49].text, "message 3");
    }
}
