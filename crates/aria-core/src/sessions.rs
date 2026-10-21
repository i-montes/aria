//! The dock's tabs: which sessions are open, their titles and the active one, in
//! `~/.aria/sessions.json`. The conversations themselves are Pi's session files (one per id).

use crate::json_file::{read_json, write_json_atomic};
use crate::paths::AriaPaths;
use serde::{Deserialize, Serialize};
use std::io;

/// The design's limit.
pub const MAX_SESSIONS: usize = 8;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionEntry {
    pub id: String,
    pub title: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionList {
    pub sessions: Vec<SessionEntry>,
    pub active: Option<String>,
}

/// Ids become Pi's `--session-id`: letters, digits and dashes, starting and ending with a letter or digit
/// (Pi refuses anything else and exits).
pub fn valid_id(id: &str) -> bool {
    let edge = |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphanumeric());
    id.len() <= 64
        && edge(id.chars().next())
        && edge(id.chars().last())
        && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}

impl SessionList {
    /// At most `MAX_SESSIONS` entries with valid, distinct ids; `active` must be one of them.
    pub fn sanitized(self) -> SessionList {
        let mut seen = std::collections::HashSet::new();
        let sessions: Vec<SessionEntry> = self
            .sessions
            .into_iter()
            .filter(|entry| valid_id(&entry.id) && seen.insert(entry.id.clone()))
            .take(MAX_SESSIONS)
            .collect();
        let active = self.active.filter(|id| sessions.iter().any(|entry| &entry.id == id));
        SessionList { sessions, active }
    }
}

/// A missing file is an empty list. A file that does not parse is moved aside (`sessions.json.broken`) and the list starts
/// empty: the tabs are a convenience, never a reason to fail, and the old file is never overwritten.
pub fn load(paths: &AriaPaths) -> SessionList {
    let parsed = match read_json(&paths.sessions()) {
        Ok(None) => return SessionList::default(),
        Ok(Some(value)) => serde_json::from_value::<SessionList>(value).ok(),
        Err(_) => None,
    };
    match parsed {
        Some(list) => list.sanitized(),
        None => {
            let _ = std::fs::rename(paths.sessions(), paths.root.join("sessions.json.broken"));
            SessionList::default()
        }
    }
}

pub fn save(paths: &AriaPaths, list: &SessionList) -> io::Result<()> {
    let value = serde_json::to_value(list.clone().sanitized()).map_err(io::Error::other)?;
    write_json_atomic(&paths.sessions(), &value, 0o600)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths() -> (tempfile::TempDir, AriaPaths) {
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths::new(dir.path().to_path_buf());
        (dir, paths)
    }

    fn entry(id: &str, title: &str) -> SessionEntry {
        SessionEntry { id: id.into(), title: title.into() }
    }

    #[test]
    fn a_list_round_trips() {
        let (_dir, paths) = paths();
        let list =
            SessionList { sessions: vec![entry("a1", "Fix the build"), entry("b2", "")], active: Some("b2".into()) };
        save(&paths, &list).unwrap();
        assert_eq!(load(&paths), list);
    }

    #[test]
    fn the_file_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let (_dir, paths) = paths();
        save(&paths, &SessionList::default()).unwrap();
        assert_eq!(std::fs::metadata(paths.sessions()).unwrap().permissions().mode() & 0o777, 0o600);
    }

    #[test]
    fn more_than_eight_sessions_are_cut() {
        let (_dir, paths) = paths();
        let sessions = (0..12).map(|i| entry(&format!("s{i}"), "t")).collect();
        save(&paths, &SessionList { sessions, active: Some("s11".into()) }).unwrap();
        let loaded = load(&paths);
        assert_eq!(loaded.sessions.len(), 8);
        assert_eq!(loaded.active, None, "an active id that was cut is dropped");
    }

    #[test]
    fn bad_and_repeated_ids_are_dropped() {
        let list = SessionList {
            sessions: vec![
                entry("ok-1", "a"),
                entry("../etc", "b"),
                entry("", "c"),
                entry("ok-1", "again"),
                entry(&"x".repeat(65), "d"),
                entry("with space", "e"),
            ],
            active: Some("ok-1".into()),
        }
        .sanitized();
        assert_eq!(list.sessions, vec![entry("ok-1", "a")]);
        assert_eq!(list.active.as_deref(), Some("ok-1"));
    }

    #[test]
    fn ids_must_start_and_end_with_a_letter_or_digit_like_pi_requires() {
        for good in ["a", "a1", "d6b71394-02c1-438f-aebd-7d26222c8bea", "a-b"] {
            assert!(valid_id(good), "{good}");
        }
        for bad in ["-a", "a-", "-", "--", ""] {
            assert!(!valid_id(bad), "{bad}");
        }
    }

    #[test]
    fn a_missing_or_broken_file_is_an_empty_list_and_a_broken_one_is_kept_aside() {
        let (_dir, paths) = paths();
        assert_eq!(load(&paths), SessionList::default());
        std::fs::write(paths.sessions(), "{ not json").unwrap();
        assert_eq!(load(&paths), SessionList::default());
        assert_eq!(std::fs::read_to_string(paths.root.join("sessions.json.broken")).unwrap(), "{ not json");
        assert!(!paths.sessions().exists(), "moved, so the next save cannot overwrite it");
        std::fs::write(paths.sessions(), r#"{"sessions": 3}"#).unwrap();
        assert_eq!(load(&paths), SessionList::default());
    }
}
