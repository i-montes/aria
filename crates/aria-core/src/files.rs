//! The `@` file picker: a plain directory walk under the chat's folder, `.gitignore`-aware, shallowest
//! files first. Symlinked directories are never followed, so a loop cannot hang it.

use std::collections::VecDeque;
use std::path::Path;

/// Directories nobody wants in a picker, whatever `.gitignore` says.
const ALWAYS_SKIPPED: [&str; 4] = [".git", "node_modules", "target", ".venv"];
/// The walk gives up after this many directory entries (a folder of repos can be huge).
const VISIT_CAP: usize = 50_000;

#[derive(Debug, Clone)]
struct Rule {
    /// Directory (relative to the root, `""` for the root) whose `.gitignore` holds this pattern.
    base: String,
    pattern: String,
    dir_only: bool,
    /// The pattern names a path (it had a `/` before its end), not just a file or folder name.
    anchored: bool,
}

fn parse_gitignore(base: &str, text: &str) -> Vec<Rule> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#') && !line.starts_with('!'))
        .map(|line| {
            let dir_only = line.ends_with('/');
            let trimmed = line.trim_end_matches('/');
            let anchored = trimmed.contains('/');
            Rule { base: base.to_string(), pattern: trimmed.trim_start_matches('/').to_string(), dir_only, anchored }
        })
        .collect()
}

/// `*` matches any run of characters (not `/`), `?` one character.
fn glob_match(pattern: &[u8], text: &[u8]) -> bool {
    match (pattern.first(), text.first()) {
        (None, None) => true,
        (Some(b'*'), _) => {
            glob_match(&pattern[1..], text) || (!text.is_empty() && text[0] != b'/' && glob_match(pattern, &text[1..]))
        }
        (Some(b'?'), Some(c)) if *c != b'/' => glob_match(&pattern[1..], &text[1..]),
        (Some(p), Some(c)) if p == c => glob_match(&pattern[1..], &text[1..]),
        _ => false,
    }
}

fn ignored(rules: &[Rule], rel: &str, name: &str, is_dir: bool) -> bool {
    rules.iter().any(|rule| {
        if rule.dir_only && !is_dir {
            return false;
        }
        let below = if rule.base.is_empty() { Some(rel) } else { rel.strip_prefix(&format!("{}/", rule.base)) };
        let Some(below) = below else { return false };
        if rule.anchored {
            glob_match(rule.pattern.as_bytes(), below.as_bytes())
        } else {
            glob_match(rule.pattern.as_bytes(), name.as_bytes())
        }
    })
}

/// Relative paths of files under `root` whose path contains `query` (case-insensitive), shallowest
/// first and alphabetical within a depth, at most `limit`.
pub fn list_files(root: &Path, query: &str, limit: usize) -> Vec<String> {
    let needle = query.to_lowercase();
    let mut found = Vec::new();
    let mut visited = 0usize;
    let mut queue: VecDeque<(String, Vec<Rule>)> = VecDeque::new();
    queue.push_back((String::new(), Vec::new()));
    while let Some((dir, mut rules)) = queue.pop_front() {
        let full = if dir.is_empty() { root.to_path_buf() } else { root.join(&dir) };
        if let Ok(text) = std::fs::read_to_string(full.join(".gitignore")) {
            rules.extend(parse_gitignore(&dir, &text));
        }
        let Ok(read) = std::fs::read_dir(&full) else { continue };
        let mut entries: Vec<_> = read.filter_map(Result::ok).collect();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            visited += 1;
            if visited > VISIT_CAP {
                return found;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let Ok(kind) = entry.file_type() else { continue };
            let rel = if dir.is_empty() { name.clone() } else { format!("{dir}/{name}") };
            if kind.is_dir() {
                if ALWAYS_SKIPPED.contains(&name.as_str()) || ignored(&rules, &rel, &name, true) {
                    continue;
                }
                queue.push_back((rel, rules.clone()));
            } else if kind.is_file() || kind.is_symlink() {
                if kind.is_symlink() && entry.path().is_dir() {
                    continue; // a symlinked directory is not followed
                }
                if ignored(&rules, &rel, &name, false) || !rel.to_lowercase().contains(&needle) {
                    continue;
                }
                found.push(rel);
                if found.len() >= limit {
                    return found;
                }
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(files: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for file in files {
            let path = dir.path().join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "x").unwrap();
        }
        dir
    }

    #[test]
    fn lists_matching_files_shallowest_first() {
        let dir = tree(&["src/deep/main.rs", "main.rs", "src/lib.rs", "README.md"]);
        assert_eq!(list_files(dir.path(), "", 10), vec!["README.md", "main.rs", "src/lib.rs", "src/deep/main.rs"]);
        assert_eq!(list_files(dir.path(), "MAIN", 10), vec!["main.rs", "src/deep/main.rs"]);
        assert_eq!(list_files(dir.path(), "src/", 10), vec!["src/lib.rs", "src/deep/main.rs"]);
    }

    #[test]
    fn respects_gitignore_patterns() {
        let dir = tree(&[
            "keep.rs",
            "debug.log",
            "build/out.o",
            "src/x.log",
            "src/y.rs",
            "docs/a.md",
            "a/docs/b.md",
            "secret.txt",
        ]);
        std::fs::write(dir.path().join(".gitignore"), "# build output\n*.log\nbuild/\n/secret.txt\ndocs/a.md\n")
            .unwrap();
        let files = list_files(dir.path(), "", 50);
        assert!(files.contains(&"keep.rs".to_string()));
        assert!(files.contains(&"src/y.rs".to_string()));
        assert!(files.contains(&"a/docs/b.md".to_string()));
        for hidden in ["debug.log", "build/out.o", "src/x.log", "secret.txt", "docs/a.md"] {
            assert!(!files.contains(&hidden.to_string()), "{hidden} should be ignored: {files:?}");
        }
    }

    #[test]
    fn a_nested_gitignore_applies_below_its_folder_only() {
        let dir = tree(&["proj/a.tmp", "proj/keep.rs", "other/a.tmp"]);
        std::fs::write(dir.path().join("proj/.gitignore"), "*.tmp\n").unwrap();
        let files = list_files(dir.path(), "", 50);
        assert!(!files.contains(&"proj/a.tmp".to_string()));
        assert!(files.contains(&"other/a.tmp".to_string()));
        assert!(files.contains(&"proj/keep.rs".to_string()));
    }

    #[test]
    fn skips_dot_git_and_caps_the_limit() {
        let dir = tree(&[".git/config", "node_modules/x/index.js", "a.rs", "b.rs", "c.rs"]);
        let files = list_files(dir.path(), "", 2);
        assert_eq!(files, vec!["a.rs", "b.rs"]);
        let all = list_files(dir.path(), "", 50);
        assert!(all.iter().all(|f| !f.starts_with(".git/") && !f.starts_with("node_modules/")), "{all:?}");
    }

    #[test]
    fn handles_a_symlink_loop() {
        let dir = tree(&["real/file.txt"]);
        std::os::unix::fs::symlink(dir.path(), dir.path().join("real/loop")).unwrap();
        std::os::unix::fs::symlink(dir.path().join("real"), dir.path().join("linked")).unwrap();
        assert_eq!(list_files(dir.path(), "", 50), vec!["real/file.txt"]);
    }

    #[test]
    fn a_missing_root_lists_nothing() {
        assert!(list_files(Path::new("/nonexistent/aria"), "", 10).is_empty());
    }
}
