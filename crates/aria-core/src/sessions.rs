//! The dock's tabs: which sessions are open, their titles and the active one, in
//! `~/.aria/sessions.json`. The conversations themselves are Pi's session files: a tab owns a chain of them, its
//! segments (oldest first), and Pi runs on the last one.

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
    /// The tab's Pi session ids, oldest first. Empty (every tab written before segments existed) means `[id]`.
    #[serde(default)]
    pub segments: Vec<String>,
}

/// The tab's segments, oldest first: `segments`, or the tab's own id when it has none.
pub fn segments_of(entry: &SessionEntry) -> Vec<String> {
    if entry.segments.is_empty() {
        vec![entry.id.clone()]
    } else {
        entry.segments.clone()
    }
}

/// The Pi session id the tab runs on: its last segment.
pub fn current_segment(entry: &SessionEntry) -> String {
    segments_of(entry).pop().unwrap_or_else(|| entry.id.clone())
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
            .map(|mut entry| {
                entry.segments.retain(|segment| {
                    let valid = valid_id(segment);
                    if !valid {
                        note_dropped(&entry.id, segment);
                    }
                    valid
                });
                entry
            })
            .collect();
        let active = self.active.filter(|id| sessions.iter().any(|entry| &entry.id == id));
        SessionList { sessions, active }
    }
}

/// A stored segment id Pi would refuse is dropped from its tab's chain: said in the dock's log, once per process (every
/// read of `sessions.json` sanitizes it again until the next save).
fn note_dropped(tab: &str, segment: &str) {
    static NOTED: std::sync::Mutex<Option<std::collections::HashSet<String>>> = std::sync::Mutex::new(None);
    let mut noted = NOTED.lock().unwrap_or_else(|e| e.into_inner());
    if noted.get_or_insert_with(Default::default).insert(format!("{tab}\0{segment}")) {
        eprintln!(
            "aria: tab {tab}: its stored segment {segment:?} is not a valid session id and was dropped from its chain"
        );
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

/// Record `segment` as the tab's new current segment and save. `false` when the tab is not in the list or `segment` is
/// already its current one (nothing written); an id Pi would refuse is an `InvalidInput` error.
pub fn append_segment(paths: &AriaPaths, tab: &str, segment: &str) -> io::Result<bool> {
    if !valid_id(segment) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "not a valid session id"));
    }
    let mut list = load(paths);
    let Some(entry) = list.sessions.iter_mut().find(|entry| entry.id == tab) else {
        return Ok(false);
    };
    if current_segment(entry) == segment {
        return Ok(false);
    }
    let mut segments = segments_of(entry);
    segments.push(segment.to_string());
    entry.segments = segments;
    save(paths, &list)?;
    Ok(true)
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
        SessionEntry { id: id.into(), title: title.into(), segments: vec![] }
    }

    #[test]
    fn an_entry_without_segments_is_one_segment_its_own_id() {
        let (_dir, paths) = paths();
        std::fs::write(paths.sessions(), r#"{"sessions":[{"id":"a1","title":"Old"}],"active":"a1"}"#).unwrap();
        let list = load(&paths);
        assert_eq!(list.sessions[0].segments, Vec::<String>::new());
        assert_eq!(segments_of(&list.sessions[0]), vec!["a1".to_string()]);
        assert_eq!(current_segment(&list.sessions[0]), "a1");
    }

    #[test]
    fn an_appended_segment_is_saved_and_becomes_the_current_one() {
        let (_dir, paths) = paths();
        save(&paths, &SessionList { sessions: vec![entry("a1", "Fix"), entry("b2", "")], active: None }).unwrap();
        let uuid = "d6b71394-02c1-438f-aebd-7d26222c8bea";
        assert!(append_segment(&paths, "a1", uuid).unwrap());
        let list = load(&paths);
        assert_eq!(list.sessions[0].segments, vec!["a1".to_string(), uuid.to_string()]);
        assert_eq!(current_segment(&list.sessions[0]), uuid);
        assert_eq!(segments_of(&list.sessions[1]), vec!["b2".to_string()], "the other tab is untouched");
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(paths.sessions()).unwrap()).unwrap();
        assert_eq!(written["sessions"][0]["segments"], serde_json::json!(["a1", uuid]));

        assert!(append_segment(&paths, "a1", "c3").unwrap());
        assert_eq!(segments_of(&load(&paths).sessions[0]), vec!["a1".to_string(), uuid.to_string(), "c3".to_string()]);
    }

    #[test]
    fn appending_to_an_unknown_tab_or_the_current_segment_changes_nothing() {
        let (_dir, paths) = paths();
        save(&paths, &SessionList { sessions: vec![entry("a1", "Fix")], active: None }).unwrap();
        assert!(!append_segment(&paths, "zz", "c3").unwrap());
        assert!(!append_segment(&paths, "a1", "a1").unwrap(), "already the current segment");
        assert_eq!(load(&paths).sessions[0].segments, Vec::<String>::new());
    }

    #[test]
    fn segment_ids_follow_pi_session_id_rules() {
        let (_dir, paths) = paths();
        save(&paths, &SessionList { sessions: vec![entry("a1", "Fix")], active: None }).unwrap();
        for bad in ["../x", "a/b", "a\\b", "", "-a", "with space"] {
            assert!(append_segment(&paths, "a1", bad).is_err(), "{bad}");
        }
        assert_eq!(load(&paths).sessions[0].segments, Vec::<String>::new());

        let mut tab = entry("a1", "Fix");
        tab.segments = vec!["a1".into(), "../etc".into(), "b-2".into()];
        let list = SessionList { sessions: vec![tab], active: None }.sanitized();
        assert_eq!(list.sessions[0].segments, vec!["a1".to_string(), "b-2".to_string()], "a bad stored id is dropped");
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
