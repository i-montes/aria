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
    /// The scheduled task this tab runs (`crate::schedules`); such tabs do not count against `MAX_SESSIONS`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedule: Option<String>,
    /// The folder the tab's Pi works in (absolute): a session opened from a terminal works where it was opened. None is
    /// the settings' projects folder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workdir: Option<String>,
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
    /// Entries with valid, distinct ids: at most `MAX_SESSIONS` regular ones, and every scheduled one; `active` must be one
    /// of them.
    pub fn sanitized(self) -> SessionList {
        let mut seen = std::collections::HashSet::new();
        let mut regular = 0;
        let sessions: Vec<SessionEntry> = self
            .sessions
            .into_iter()
            .filter(|entry| valid_id(&entry.id) && seen.insert(entry.id.clone()))
            .filter(|entry| {
                if entry.schedule.is_some() {
                    return true;
                }
                regular += 1;
                regular <= MAX_SESSIONS
            })
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

/// The tabs a page save keeps: the page's, minus scheduled ones whose task is gone, plus the scheduled tabs on disk the page
/// did not send (the dock made them; the page may not know them yet) whose task still exists.
pub fn merge_scheduled(from_page: Vec<SessionEntry>, on_disk: &SessionList, tasks: &[String]) -> Vec<SessionEntry> {
    let alive = |entry: &SessionEntry| entry.schedule.as_ref().is_none_or(|task| tasks.contains(task));
    let mut merged: Vec<SessionEntry> = from_page.into_iter().filter(alive).collect();
    for kept in on_disk.sessions.iter().filter(|e| e.schedule.is_some() && alive(e)) {
        if !merged.iter().any(|entry| entry.id == kept.id) {
            merged.push(kept.clone());
        }
    }
    merged
}

/// Adds a scheduled task's tab (never the active one). `false` when it is already there.
pub fn add_scheduled(paths: &AriaPaths, id: &str, title: &str, schedule: &str) -> io::Result<bool> {
    let mut list = load(paths);
    if list.sessions.iter().any(|entry| entry.id == id) {
        return Ok(false);
    }
    list.sessions.push(SessionEntry {
        id: id.to_string(),
        title: title.to_string(),
        segments: Vec::new(),
        schedule: Some(schedule.to_string()),
        workdir: None,
    });
    save(paths, &list)?;
    Ok(true)
}

#[derive(Debug)]
pub enum AddError {
    /// Already `MAX_SESSIONS` regular tabs.
    TooMany,
    Io(String),
}

/// A new tab's id, like the page's: 16 lowercase hex characters.
pub fn new_id() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..16).map(|_| char::from_digit(rng.gen_range(0..16), 16).unwrap_or('0')).collect()
}

/// Adds a regular tab at the end (never the active one), working in `workdir` when given.
pub fn add(paths: &AriaPaths, id: &str, title: &str, workdir: Option<&str>) -> Result<SessionEntry, AddError> {
    if !valid_id(id) {
        return Err(AddError::Io(format!("{id:?} is not a valid session id")));
    }
    let mut list = load(paths);
    if list.sessions.iter().any(|entry| entry.id == id) {
        return Err(AddError::Io(format!("the tab {id} is already there")));
    }
    if list.sessions.iter().filter(|entry| entry.schedule.is_none()).count() >= MAX_SESSIONS {
        return Err(AddError::TooMany);
    }
    let entry = SessionEntry {
        id: id.to_string(),
        title: title.to_string(),
        segments: Vec::new(),
        schedule: None,
        workdir: workdir.map(str::to_string),
    };
    list.sessions.push(entry.clone());
    save(paths, &list).map_err(|e| AddError::Io(e.to_string()))?;
    Ok(entry)
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

    fn scheduled(id: &str, task: &str) -> SessionEntry {
        SessionEntry { schedule: Some(task.into()), ..entry(id, "Notion tasks") }
    }

    #[test]
    fn scheduled_tabs_do_not_count_against_eight() {
        let mut sessions: Vec<SessionEntry> = (0..8).map(|n| entry(&format!("t{n}"), "x")).collect();
        sessions.push(scheduled("sched-1", "s-aaaaaaaaaaaa"));
        sessions.push(entry("t9", "the ninth regular tab"));
        sessions.push(scheduled("sched-2", "s-bbbbbbbbbbbb"));
        let list = SessionList { sessions, active: None }.sanitized();
        let ids: Vec<&str> = list.sessions.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids.len(), 10);
        assert!(ids.contains(&"sched-1") && ids.contains(&"sched-2") && !ids.contains(&"t9"), "{ids:?}");
    }

    #[test]
    fn a_page_save_keeps_a_scheduled_tab_it_did_not_send() {
        let on_disk =
            SessionList { sessions: vec![entry("aa", "A"), scheduled("sched-1", "s-aaaaaaaaaaaa")], active: None };
        let merged = merge_scheduled(vec![entry("aa", "A")], &on_disk, &["s-aaaaaaaaaaaa".to_string()]);
        assert_eq!(merged.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), vec!["aa", "sched-1"]);
    }

    #[test]
    fn a_scheduled_tab_of_a_deleted_task_is_dropped() {
        let on_disk = SessionList { sessions: vec![scheduled("sched-1", "s-gone00000000")], active: None };
        let from_page = vec![entry("aa", "A"), scheduled("sched-2", "s-gone11111111")];
        let merged = merge_scheduled(from_page, &on_disk, &[]);
        assert_eq!(merged.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), vec!["aa"]);
    }

    #[test]
    fn add_scheduled_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths::new(dir.path().to_path_buf());
        assert!(add_scheduled(&paths, "sched-1", "Notion tasks", "s-aaaaaaaaaaaa").unwrap());
        assert!(!add_scheduled(&paths, "sched-1", "Notion tasks", "s-aaaaaaaaaaaa").unwrap());
        let list = load(&paths);
        assert_eq!(list.sessions.len(), 1);
        assert_eq!(list.sessions[0].schedule.as_deref(), Some("s-aaaaaaaaaaaa"));
        assert_eq!(list.active, None, "a scheduled tab never takes the active place");
    }

    fn entry(id: &str, title: &str) -> SessionEntry {
        SessionEntry { id: id.into(), title: title.into(), segments: vec![], schedule: None, workdir: None }
    }

    #[test]
    fn an_entry_keeps_its_folder() {
        let (_dir, paths) = paths();
        std::fs::write(paths.sessions(), r#"{"sessions":[{"id":"a1","title":"Old"}],"active":"a1"}"#).unwrap();
        assert_eq!(load(&paths).sessions[0].workdir, None);
        let mut list = load(&paths);
        list.sessions.push(SessionEntry {
            workdir: Some("/home/ana/Projects/checkout-widget".into()),
            ..entry("b2", "Coupons")
        });
        save(&paths, &list).unwrap();
        let written = std::fs::read_to_string(paths.sessions()).unwrap();
        assert_eq!(written.matches("workdir").count(), 1, "an entry without a folder is written without the field");
        assert_eq!(load(&paths).sessions[1].workdir.as_deref(), Some("/home/ana/Projects/checkout-widget"));
    }

    #[test]
    fn add_appends_with_a_folder_and_refuses_a_ninth() {
        let (_dir, paths) = paths();
        save(&paths, &SessionList { sessions: vec![entry("a1", "First")], active: Some("a1".into()) }).unwrap();
        let added = add(&paths, "b2", "", Some("/home/ana/Projects/checkout-widget")).unwrap();
        assert_eq!(added.workdir.as_deref(), Some("/home/ana/Projects/checkout-widget"));
        let list = load(&paths);
        assert_eq!(list.sessions.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["a1", "b2"]);
        assert_eq!(list.active.as_deref(), Some("a1"), "adding never changes the active tab");
        for n in 3..=8 {
            add(&paths, &format!("t{n}"), "", None).unwrap();
        }
        assert!(matches!(add(&paths, "t9", "", None), Err(AddError::TooMany)));
        assert!(matches!(add(&paths, "a1", "", None), Err(AddError::Io(_))), "an id already there is refused");
        let id = new_id();
        assert!(
            valid_id(&id) && id.len() == 16 && id.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
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
