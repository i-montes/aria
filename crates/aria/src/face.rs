//! Aria's face: one word for what Aria is doing, written where the face in the bar (Omarchy) reads it,
//! `$XDG_RUNTIME_DIR/aria/face` (`~/.aria/run/face` on macOS, beside the control socket). Every chat reports what its
//! session is doing ([`report`]); the face is the most pressing of them, by [`PRIORITY`], plus two that come with time:
//! `done` for a few seconds after a run ends well, and `sleeping` after ten minutes with nothing going on.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The faces, by the words the widget knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Face {
    Idle,
    Thinking,
    Reading,
    Editing,
    Working,
    Delegating,
    Asking,
    Scheduled,
    Done,
    Error,
    Sleeping,
}

impl Face {
    pub fn word(self) -> &'static str {
        match self {
            Face::Idle => "idle",
            Face::Thinking => "thinking",
            Face::Reading => "reading",
            Face::Editing => "editing",
            Face::Working => "working",
            Face::Delegating => "delegating",
            Face::Asking => "asking",
            Face::Scheduled => "scheduled",
            Face::Done => "done",
            Face::Error => "error",
            Face::Sleeping => "sleeping",
        }
    }
}

/// When sessions disagree, the first of these wins: what needs the person, then what went wrong, then the work going on
/// (the most involved first), then what just ended.
const PRIORITY: [Face; 10] = [
    Face::Asking,
    Face::Error,
    Face::Delegating,
    Face::Editing,
    Face::Working,
    Face::Reading,
    Face::Thinking,
    Face::Scheduled,
    Face::Done,
    Face::Idle,
];

/// How long `done` shows after a run ends well.
pub const DONE_FOR: Duration = Duration::from_secs(8);
/// How long nothing has to happen before the face falls asleep.
pub const SLEEP_AFTER: Duration = Duration::from_secs(600);

/// How a session's last run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Done,
    Failed,
}

/// What one session is doing, as its chat sees it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Activity {
    pub running: bool,
    /// The tools running now, by name.
    pub tools: Vec<String>,
    /// A dialog (permission, question, confirmation) waits for the person.
    pub asking: bool,
    /// This is a scheduled task's run.
    pub scheduled: bool,
    /// How the last run ended, and when; cleared when the next one starts.
    pub outcome: Option<(Outcome, Instant)>,
}

/// The face a tool calls for: reading (files, search, the web, memory), editing (writes), delegating (subagents), and
/// working for everything else (a command, a connector).
pub fn tool_face(name: &str) -> Face {
    let name = name.to_ascii_lowercase();
    let has = |part: &str| name.contains(part);
    if has("parallel") || has("subagent") || name == "agent" || name == "task" {
        Face::Delegating
    } else if has("edit") || has("write") || has("patch") {
        Face::Editing
    } else if matches!(name.as_str(), "read" | "grep" | "find" | "ls" | "glob")
        || has("search")
        || has("fetch")
        || has("web")
        || has("mem_")
        || has("memory")
        || has("browse")
    {
        Face::Reading
    } else {
        Face::Working
    }
}

/// One session's face. A failure stays until the person looks ([`seen`] clears it) or the session runs again.
pub fn session_face(activity: &Activity, now: Instant) -> Face {
    if activity.asking {
        return Face::Asking;
    }
    if activity.running {
        if activity.scheduled {
            return Face::Scheduled;
        }
        let tools = activity.tools.iter().map(|name| tool_face(name));
        return PRIORITY.into_iter().find(|face| tools.clone().any(|tool| tool == *face)).unwrap_or(Face::Thinking);
    }
    match activity.outcome {
        Some((Outcome::Failed, _)) => Face::Error,
        Some((Outcome::Done, at)) if now.duration_since(at) < DONE_FOR => Face::Done,
        _ => Face::Idle,
    }
}

/// The face for all the sessions: the most pressing one; `sleeping` when all are idle and nothing happened for
/// [`SLEEP_AFTER`].
pub fn overall<'a>(sessions: impl IntoIterator<Item = &'a Activity>, last_active: Instant, now: Instant) -> Face {
    let faces: Vec<Face> = sessions.into_iter().map(|activity| session_face(activity, now)).collect();
    let face = PRIORITY.into_iter().find(|face| faces.contains(face)).unwrap_or(Face::Idle);
    if face == Face::Idle && now.duration_since(last_active) >= SLEEP_AFTER {
        Face::Sleeping
    } else {
        face
    }
}

struct State {
    sessions: HashMap<String, Activity>,
    last_active: Instant,
    written: Option<Face>,
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| {
        // The time-based faces (`done` ending, `sleeping`) need a look now and then.
        std::thread::spawn(|| loop {
            std::thread::sleep(Duration::from_secs(1));
            refresh();
        });
        Mutex::new(State { sessions: HashMap::new(), last_active: Instant::now(), written: None })
    })
}

/// The file the face is written to: next to the control socket.
pub fn face_path() -> PathBuf {
    crate::control::control_socket_path().with_file_name("face")
}

/// The dock started: the face is written now (idle), and from here it falls asleep on its own when nothing happens.
pub fn start() {
    refresh();
}

/// A session's chat reports what it is doing; the face is written when that changes it.
pub fn report(session: &str, activity: Activity) {
    let mut state = state().lock().unwrap_or_else(|e| e.into_inner());
    let busy = activity.running || activity.asking;
    if state.sessions.get(session) != Some(&activity) {
        state.last_active = Instant::now();
    }
    if busy {
        state.last_active = Instant::now();
    }
    state.sessions.insert(session.to_string(), activity);
    write_if_changed(&mut state);
}

/// A closed session no longer counts.
pub fn forget(session: &str) {
    let mut state = state().lock().unwrap_or_else(|e| e.into_inner());
    state.sessions.remove(session);
    write_if_changed(&mut state);
}

/// The person opened the dock: failures have been seen, and Aria is awake.
pub fn seen() {
    let mut state = state().lock().unwrap_or_else(|e| e.into_inner());
    for activity in state.sessions.values_mut() {
        if matches!(activity.outcome, Some((Outcome::Failed, _))) {
            activity.outcome = None;
        }
    }
    state.last_active = Instant::now();
    write_if_changed(&mut state);
}

/// One session's face now (the suites check what a chat reported).
#[cfg(test)]
pub fn session_now(session: &str) -> Option<Face> {
    let state = state().lock().unwrap_or_else(|e| e.into_inner());
    state.sessions.get(session).map(|activity| session_face(activity, Instant::now()))
}

fn refresh() {
    let mut state = state().lock().unwrap_or_else(|e| e.into_inner());
    write_if_changed(&mut state);
}

fn write_if_changed(state: &mut State) {
    let face = overall(state.sessions.values(), state.last_active, Instant::now());
    if state.written == Some(face) {
        return;
    }
    if cfg!(test) {
        state.written = Some(face); // the suites' chats must not change the face of the person running them
        return;
    }
    let path = face_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    // Written in place (not renamed over): the bar watches this file, and a rename would leave it watching the old one.
    match std::fs::write(&path, format!("{}\n", face.word())) {
        Ok(()) => state.written = Some(face),
        Err(error) => crate::note(&format!("the face could not be written to {}: {error}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn running(tools: &[&str]) -> Activity {
        Activity { running: true, tools: tools.iter().map(|t| t.to_string()).collect(), ..Activity::default() }
    }

    #[test]
    fn tools_have_their_faces() {
        for (tool, face) in [
            ("read", Face::Reading),
            ("grep", Face::Reading),
            ("web_search", Face::Reading),
            ("mem_search", Face::Reading),
            ("edit", Face::Editing),
            ("write", Face::Editing),
            ("bash", Face::Working),
            ("notion_query", Face::Working),
            ("parallel", Face::Delegating),
            ("subagent", Face::Delegating),
        ] {
            assert_eq!(tool_face(tool), face, "{tool}");
        }
    }

    #[test]
    fn a_session_shows_what_it_does() {
        let now = Instant::now();
        assert_eq!(session_face(&Activity::default(), now), Face::Idle);
        assert_eq!(session_face(&running(&[]), now), Face::Thinking);
        assert_eq!(session_face(&running(&["read"]), now), Face::Reading);
        assert_eq!(session_face(&running(&["read", "bash"]), now), Face::Working, "the more involved tool wins");
        assert_eq!(session_face(&running(&["edit", "parallel"]), now), Face::Delegating);
        let asking = Activity { asking: true, ..running(&["bash"]) };
        assert_eq!(session_face(&asking, now), Face::Asking);
        let scheduled = Activity { scheduled: true, ..running(&["bash"]) };
        assert_eq!(session_face(&scheduled, now), Face::Scheduled);
    }

    #[test]
    fn done_lasts_a_few_seconds_and_a_failure_until_it_is_seen() {
        let start = Instant::now();
        let done = Activity { outcome: Some((Outcome::Done, start)), ..Activity::default() };
        assert_eq!(session_face(&done, start + Duration::from_secs(3)), Face::Done);
        assert_eq!(session_face(&done, start + DONE_FOR), Face::Idle);
        let failed = Activity { outcome: Some((Outcome::Failed, start)), ..Activity::default() };
        assert_eq!(session_face(&failed, start + Duration::from_secs(3600)), Face::Error);
    }

    #[test]
    fn the_most_pressing_session_wins_and_idle_falls_asleep() {
        let now = Instant::now();
        let asking = Activity { asking: true, ..Activity::default() };
        let failed = Activity { outcome: Some((Outcome::Failed, now)), ..Activity::default() };
        assert_eq!(overall([&running(&["bash"]), &asking, &failed], now, now), Face::Asking);
        assert_eq!(overall([&running(&["bash"]), &failed], now, now), Face::Error);
        let scheduled = Activity { scheduled: true, ..running(&[]) };
        assert_eq!(overall([&scheduled, &running(&["read"])], now, now), Face::Reading, "the person's own run first");
        assert_eq!(overall([&Activity::default()], now, now + SLEEP_AFTER - Duration::from_secs(1)), Face::Idle);
        assert_eq!(overall([&Activity::default()], now, now + SLEEP_AFTER), Face::Sleeping);
        assert_eq!(overall(std::iter::empty(), now, now), Face::Idle, "no tabs yet");
        assert_eq!(overall([&running(&[])], now, now + SLEEP_AFTER), Face::Thinking, "a run never sleeps");
    }

    #[test]
    fn every_face_has_a_word_the_widget_knows() {
        let widget = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../install/omarchy/aria.face/Face.qml"));
        for face in PRIORITY.into_iter().chain([Face::Sleeping]) {
            assert!(widget.contains(&format!("{}:", face.word())), "the widget has no '{}'", face.word());
        }
    }
}
