//! The footer's git part: the branch and how many files changed, from `git status --porcelain=v1 -b` in the session's
//! folder (one second at most; none outside a repository).

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
pub struct GitStatus {
    pub branch: String,
    pub changed: u32,
}

/// The branch line (`## main...origin/main`, `## No commits yet on main`, `## HEAD (no branch)`) and the changed files.
pub fn parse(output: &str) -> Option<GitStatus> {
    let mut lines = output.lines();
    let head = lines.next()?.strip_prefix("## ")?;
    let head = head.strip_prefix("No commits yet on ").unwrap_or(head);
    let branch = head.split("...").next()?.split(' ').next()?.to_string();
    let changed = lines.filter(|l| !l.trim().is_empty()).count() as u32;
    Some(GitStatus { branch, changed })
}

pub fn status(dir: &Path) -> Option<GitStatus> {
    let mut child = Command::new("git")
        .args(["status", "--porcelain=v1", "-b"])
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    // Read while it runs: a large output would otherwise fill the pipe and stop git before it exits.
    let mut stdout = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut output = String::new();
        std::io::Read::read_to_string(&mut stdout, &mut output).ok().map(|_| output)
    });
    let start = Instant::now();
    loop {
        match child.try_wait().ok()? {
            Some(exit) if exit.success() => break,
            Some(_) => return None,
            None if start.elapsed() > Duration::from_secs(1) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    }
    parse(&reader.join().ok()??)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_status_outside_a_repo_is_none() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(status(dir.path()), None);
    }

    #[test]
    fn git_status_reads_the_branch_and_counts_changes() {
        let dir = tempfile::tempdir().unwrap();
        let git = |args: &[&str]| {
            std::process::Command::new("git").args(args).current_dir(dir.path()).output().unwrap();
        };
        git(&["init", "-q", "-b", "main"]);
        std::fs::write(dir.path().join("a.txt"), "a").unwrap();
        std::fs::write(dir.path().join("b.txt"), "b").unwrap();
        assert_eq!(status(dir.path()), Some(GitStatus { branch: "main".into(), changed: 2 }));
    }

    #[test]
    fn the_branch_line_is_read_in_its_forms() {
        assert_eq!(
            parse("## main...origin/main [ahead 1]\n M src/a.ts\n?? b\n"),
            Some(GitStatus { branch: "main".into(), changed: 2 })
        );
        assert_eq!(parse("## No commits yet on main\n"), Some(GitStatus { branch: "main".into(), changed: 0 }));
        assert_eq!(parse("## HEAD (no branch)\n"), Some(GitStatus { branch: "HEAD".into(), changed: 0 }));
    }
}
