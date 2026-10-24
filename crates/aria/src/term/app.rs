//! The client as a pure state machine: keys, clicks and the dock's messages go in, effects come out (what to send, to
//! attach, to save, to quit). `term::run` does the input and output.

use crate::term::dialog::{saved_answer, Answer, DialogBox};
use crate::term::editor::{bang, Editor, Trigger};
use crate::term::picker::{models_for_picker, Item, Outcome, Picker};
use crate::term::rewind::{self, Rewind};
use crate::term::schedules::{self, Schedules};
use crate::term::signin::{self, SignIn};
use crate::term::state::Session;
use crate::term::texts::t;
use crate::term::Key;
use aria_core::bridge::{DockMessage, PageMessage};
use aria_core::sessions::SessionEntry;
use serde_json::json;
use std::collections::HashMap;

pub struct Config {
    pub cwd: String,
    pub lang: String,
    pub verbose: bool,
    /// The credentials in auth.json (`provider:type`), for `/model`.
    pub connected: Vec<String>,
    pub allow_pay_per_use: bool,
    pub compact_at: u8,
    /// The host `aria connect` is on: sessions there have no folder of this terminal's.
    pub remote: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Press {
    pub key: Key,
    pub ctrl: bool,
    pub alt: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Input {
    Key(Press),
    /// A click on a row of the screen (0 is the top).
    Click {
        row: u16,
    },
    Resize {
        height: u16,
    },
    /// Text pasted into the terminal.
    Paste(String),
    /// The link to the dock was lost and is back (the dock had dropped this client).
    Reconnected,
    /// The link to the dock was lost and the dock does not answer.
    Lost,
    /// The mouse wheel: lines up (negative: down).
    Scroll(i32),
    Focus(bool),
    Dock(DockMessage),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    Send(PageMessage),
    Attach(String),
    Detach,
    Focus(bool),
    SaveVerbose(bool),
    /// Clear the terminal and draw everything again (Ctrl+L).
    Redraw,
    /// Connect to the dock again; answer with `Reconnected` or `Lost`.
    Reconnect,
    Quit,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Screen {
    List,
    Session,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PickerKind {
    Files,
    Commands,
    Models,
}

#[derive(Debug, Clone, PartialEq)]
enum Pending {
    SessionsLoad,
    SessionNew,
    Files,
    Commands,
    Models,
    Bang(String),
    Secret(String),
    Checkpoints,
    /// `/context`'s `get_session_stats`.
    Context,
    /// An undo, with the text that goes back to the editor once it is done (empty when the conversation stays).
    Undo(String),
    /// `/schedules`'s `schedulesList`: its reply opens the panel.
    SchedulesOpen,
    /// A pause or a delete from the panel, or a resume once it was checked: its reply is the new list.
    SchedulesChange,
    /// Enter on a paused task: the fresh list that says whether it may be resumed (it is not waiting for approval).
    SchedulesResume(String),
    /// `/signin`'s `startOauth` or `startMcpOauth` (the service): answered with the URL and then the result.
    SignIn(String),
    /// An address sent to the sign-in: only a refusal answers it.
    SignInCode,
}

/// The client's own commands, before the session's.
const OWN: [(&str, &str); 8] = [
    ("exit", "term.cmd.exit"),
    ("verbose", "term.cmd.verbose"),
    ("model", "term.cmd.model"),
    ("sessions", "term.cmd.sessions"),
    ("undo", "term.cmd.undo"),
    ("context", "term.cmd.context"),
    ("schedules", "term.cmd.schedules"),
    ("signin", "term.cmd.signin"),
];

pub struct App {
    config: Config,
    screen: Screen,
    sessions: Vec<SessionEntry>,
    selected: usize,
    attached: Option<String>,
    session: Session,
    pub editor: Editor,
    picker: Option<(PickerKind, Picker)>,
    dialog: Option<DialogBox>,
    /// The session's commands (`listCommands`), once asked.
    commands: Option<Vec<Item>>,
    pending: HashMap<u64, Pending>,
    next_id: u64,
    note: Option<String>,
    height: u16,
    /// Lines scrolled up from the bottom of the conversation.
    scroll: usize,
    /// When the running turn started (ms), for the working line's timer.
    started: Option<u64>,
    /// The row the suggestions start at on screen.
    overlay_top: u16,
    /// A message Ctrl+Enter sends once the turn it stopped settles.
    after_abort: Option<String>,
    /// Calls whose whole output is shown.
    expanded: std::collections::HashSet<String>,
    /// `/context`'s panel: (tokens used, window), while it is open.
    context: Option<(u64, u64)>,
    /// The task strip shows only its first row (Ctrl+T, a click on it).
    tasks_folded: bool,
    /// The task strip's rows on screen.
    strip_rows: Option<(u16, u16)>,
    /// When Esc was last pressed while idle (ms), for Esc Esc.
    last_esc: Option<u64>,
    /// Rewind (Esc Esc, `/undo`), while it is open.
    rewind: Option<Rewind>,
    /// `/schedules`'s panel, while it is open.
    schedules: Option<Schedules>,
    /// `/signin`'s panel, while it is open.
    signin: Option<SignIn>,
    /// What each open session is doing, and since when (ms).
    statuses: HashMap<String, (String, i64)>,
    /// The time now (ms), from `clock`.
    now_ms: u64,
    /// The `… N more lines` rows on screen, with their call.
    more_rows: Vec<(u16, String)>,
}

fn folder_name(path: &str) -> String {
    path.trim_end_matches('/').rsplit('/').next().unwrap_or(path).to_string()
}

impl App {
    pub fn new(config: Config) -> (App, Vec<Effect>) {
        let mut app = App {
            config,
            screen: Screen::List,
            sessions: Vec::new(),
            selected: 0,
            attached: None,
            session: Session::default(),
            editor: Editor::default(),
            picker: None,
            dialog: None,
            commands: None,
            pending: HashMap::new(),
            next_id: 1,
            note: None,
            height: 24,
            scroll: 0,
            started: None,
            overlay_top: 0,
            after_abort: None,
            expanded: Default::default(),
            more_rows: Vec::new(),
            statuses: HashMap::new(),
            last_esc: None,
            tasks_folded: false,
            context: None,
            strip_rows: None,
            rewind: None,
            schedules: None,
            signin: None,
            now_ms: 0,
        };
        let load = app.request(Pending::SessionsLoad, |id| PageMessage::SessionsLoad { id });
        (app, vec![load])
    }

    pub fn screen(&self) -> Screen {
        self.screen
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    pub fn note(&self) -> Option<String> {
        self.note.clone()
    }

    /// The time now (ms since the epoch): a run that just started is timed from it, one that ended stops being timed.
    pub fn clock(&mut self, now_ms: u64) {
        self.now_ms = now_ms;
        if self.session.running {
            self.started.get_or_insert(now_ms);
        } else {
            self.started = None;
        }
    }

    /// How long the running turn has run, at `now_ms`; none while nothing runs.
    pub fn working_ms(&self, now_ms: u64) -> Option<u64> {
        self.started.filter(|_| self.session.running).map(|start| now_ms.saturating_sub(start))
    }

    pub fn scroll(&self) -> usize {
        self.scroll
    }

    /// Never past the top: `max` is how far the conversation can scroll.
    pub fn clamp_scroll(&mut self, max: usize) {
        self.scroll = self.scroll.min(max);
    }

    fn scroll_by(&mut self, lines: i32) {
        self.scroll = if lines >= 0 {
            self.scroll.saturating_add(lines as usize)
        } else {
            self.scroll.saturating_sub(lines.unsigned_abs() as usize)
        };
    }

    pub fn verbose(&self) -> bool {
        self.config.verbose
    }

    pub fn lang(&self) -> &str {
        &self.config.lang
    }

    pub fn compact_at(&self) -> u8 {
        self.config.compact_at
    }

    fn request(&mut self, pending: Pending, make: impl FnOnce(u64) -> PageMessage) -> Effect {
        let id = self.next_id;
        self.next_id += 1;
        self.pending.insert(id, pending);
        Effect::Send(make(id))
    }

    fn in_session(&self, message: PageMessage) -> Effect {
        Effect::Send(PageMessage::Session {
            session: self.attached.clone().unwrap_or_default(),
            message: Box::new(message),
        })
    }

    fn session_request(&mut self, pending: Pending, make: impl FnOnce(u64) -> PageMessage) -> Effect {
        let session = self.attached.clone().unwrap_or_default();
        self.request(pending, |id| PageMessage::Session { session, message: Box::new(make(id)) })
    }

    // ---- the list ------------------------------------------------------------------------------------------------

    /// The session the client is attached to.
    #[cfg(test)]
    pub fn attached_session(&self) -> Option<&str> {
        self.attached.as_deref()
    }

    /// The sessions as the list shows them: the one whose status changed last first.
    fn ordered(&self) -> Vec<&SessionEntry> {
        let mut sessions: Vec<&SessionEntry> = self.sessions.iter().collect();
        sessions.sort_by_key(|s| std::cmp::Reverse(self.statuses.get(&s.id).map_or(0, |(_, at)| *at)));
        sessions
    }

    /// How long ago `at` (ms) was: `just now`, `3 min ago`, `2 h ago`, `3 d ago`.
    fn ago(&self, at: i64) -> String {
        let seconds = (self.now_ms as i64 - at).max(0) / 1000;
        let (key, n) = match seconds {
            s if s < 60 => ("term.ago.now", 0),
            s if s < 3_600 => ("term.ago.minutes", s / 60),
            s if s < 86_400 => ("term.ago.hours", s / 3_600),
            s => ("term.ago.days", s / 86_400),
        };
        t(&self.config.lang, key, &json!({ "n": n }))
    }

    /// The list's rows: title (or "untitled"), folder, what it is doing and how long ago; the new one last.
    pub fn list_rows(&self) -> Vec<(String, String, String, String)> {
        let lang = &self.config.lang;
        let untitled = t(lang, "term.untitled", &json!({}));
        let mut rows: Vec<(String, String, String, String)> = self
            .ordered()
            .into_iter()
            .map(|s| {
                let title = if s.title.trim().is_empty() { untitled.clone() } else { s.title.clone() };
                let folder = s.workdir.as_deref().map(folder_name).unwrap_or_default();
                let (state, ago) = match self.statuses.get(&s.id) {
                    Some((status, at)) => {
                        let key = match status.as_str() {
                            "working" => "term.working",
                            "waiting" => "term.waiting",
                            "done" => "term.done",
                            _ => "",
                        };
                        (if key.is_empty() { String::new() } else { t(lang, key, &json!({})) }, self.ago(*at))
                    }
                    None => (String::new(), String::new()),
                };
                (title, folder, state, ago)
            })
            .collect();
        let new = match &self.config.remote {
            Some(host) => t(lang, "term.newRemote", &json!({"host": host})),
            None => t(lang, "term.new", &json!({"folder": self.config.cwd})),
        };
        rows.push((new, String::new(), String::new(), String::new()));
        rows
    }

    pub fn selected(&self) -> usize {
        self.selected
    }

    fn open_selected(&mut self) -> Vec<Effect> {
        match self.ordered().get(self.selected).map(|s| s.id.clone()) {
            Some(id) => self.attach(id),
            None => {
                let workdir = if self.config.remote.is_some() { None } else { Some(self.config.cwd.clone()) };
                vec![self.request(Pending::SessionNew, |id| PageMessage::SessionNew { id, workdir })]
            }
        }
    }

    fn attach(&mut self, id: String) -> Vec<Effect> {
        self.attached = Some(id.clone());
        self.session = Session::default();
        self.started = None;
        self.dialog = None;
        self.picker = None;
        self.commands = None;
        self.drop_schedules();
        self.drop_signin();
        self.scroll = 0;
        self.screen = Screen::Session;
        self.note = None;
        let load = self.in_session(PageMessage::ChatLoad { fresh: true });
        vec![Effect::Attach(id), load]
    }

    fn back_to_list(&mut self) -> Vec<Effect> {
        self.attached = None;
        self.screen = Screen::List;
        self.dialog = None;
        self.picker = None;
        self.drop_schedules();
        self.drop_signin();
        let load = self.request(Pending::SessionsLoad, |id| PageMessage::SessionsLoad { id });
        vec![Effect::Detach, load]
    }

    // ---- input ---------------------------------------------------------------------------------------------------

    pub fn update(&mut self, input: Input) -> Vec<Effect> {
        match input {
            Input::Key(press) if press.ctrl && matches!(press.key, Key::Char('c' | 'd')) => vec![Effect::Quit],
            Input::Key(press) => match self.screen {
                Screen::List => self.list_key(press.key),
                Screen::Session => self.session_key(press),
            },
            Input::Click { row } => self.click(row),
            Input::Resize { height } => {
                self.height = height;
                Vec::new()
            }
            Input::Focus(focused) => vec![Effect::Focus(focused)],
            Input::Lost => {
                self.note = Some(t(&self.config.lang, "term.gone", &json!({})));
                vec![Effect::Quit]
            }
            Input::Reconnected => match self.attached.clone() {
                Some(session) => {
                    let effects = self.attach(session);
                    let said = t(&self.config.lang, "term.reconnected", &json!({}));
                    self.session
                        .entries
                        .push(crate::term::state::Entry::Note { text: crate::term::state::Text::Raw(said) });
                    effects
                }
                None => vec![self.request(Pending::SessionsLoad, |id| PageMessage::SessionsLoad { id })],
            },
            Input::Paste(text) if self.signin.is_some() => {
                // The address goes to the panel whole: its newlines are not Enter, and it never reaches the editor.
                if let Some(open) = self.signin.as_mut() {
                    open.paste(&text);
                }
                Vec::new()
            }
            Input::Paste(text) => {
                let text = text.replace("\r\n", "\n").replace('\r', "\n");
                let mut effects = Vec::new();
                for c in text.chars() {
                    let key = if c == '\n' { Key::Enter } else { Key::Char(c) };
                    effects.extend(self.update(Input::Key(Press { key, ctrl: false, alt: c == '\n' })));
                }
                effects
            }
            Input::Scroll(lines) => {
                self.scroll_by(lines);
                Vec::new()
            }
            Input::Dock(message) => self.dock(message),
        }
    }

    fn list_key(&mut self, key: Key) -> Vec<Effect> {
        let last = self.sessions.len();
        match key {
            Key::Up => self.selected = self.selected.saturating_sub(1),
            Key::Down => self.selected = (self.selected + 1).min(last),
            Key::Enter => return self.open_selected(),
            Key::Esc | Key::Char('q') => return vec![Effect::Quit],
            _ => {}
        }
        Vec::new()
    }

    fn session_key(&mut self, press: Press) -> Vec<Effect> {
        if press.ctrl {
            return match press.key {
                Key::Char('o') => self.set_verbose(!self.config.verbose),
                Key::Char('l') => vec![Effect::Redraw],
                Key::Char('t') => {
                    self.tasks_folded = !self.tasks_folded;
                    Vec::new()
                }
                Key::Enter if self.dialog.is_none() => self.interrupt(),
                _ => Vec::new(),
            };
        }
        if let Some(open) = self.rewind.as_mut() {
            let outcome = open.key(press.key);
            return self.rewound(outcome);
        }
        if let Some(open) = self.signin.as_mut() {
            let outcome = open.key(press.key);
            return self.signed_in_outcome(outcome);
        }
        if let Some(open) = self.schedules.as_mut() {
            let outcome = open.key(press.key);
            return self.scheduled_outcome(outcome);
        }
        if self.context.is_some() {
            if press.key == Key::Esc {
                self.context = None;
            }
            return Vec::new();
        }
        if let Some(dialog) = self.dialog.as_mut() {
            let answer = dialog.key(press.key);
            return self.answered(answer);
        }
        if let Some((kind, picker)) = self.picker.as_mut() {
            let kind = *kind;
            match press.key {
                Key::Up | Key::Down | Key::Esc | Key::Tab | Key::Enter => {
                    let outcome = picker.key(press.key);
                    return self.picked(kind, outcome, press.key == Key::Enter);
                }
                _ => {}
            }
        }
        let half = i32::from(self.height / 2).max(1);
        match press.key {
            Key::PageUp => {
                self.scroll_by(half);
                Vec::new()
            }
            Key::PageDown => {
                self.scroll_by(-half);
                Vec::new()
            }
            Key::Enter if press.alt => {
                self.editor.newline();
                Vec::new()
            }
            Key::Enter => self.submit(),
            Key::Esc if self.session.running => vec![self.in_session(PageMessage::ChatAbort)],
            // Esc Esc (two presses within 500 ms) while idle: back to before a message.
            Key::Esc => {
                let double = self.last_esc.is_some_and(|at| self.now_ms.saturating_sub(at) <= 500);
                self.last_esc = if double { None } else { Some(self.now_ms) };
                if double {
                    vec![self.session_request(Pending::Checkpoints, |id| PageMessage::ChatCheckpoints { id })]
                } else {
                    Vec::new()
                }
            }
            Key::Up => {
                self.editor.history_prev();
                Vec::new()
            }
            Key::Down => {
                self.editor.history_next();
                Vec::new()
            }
            Key::Left => {
                self.editor.left();
                Vec::new()
            }
            Key::Right => {
                self.editor.right();
                Vec::new()
            }
            Key::Backspace => {
                self.editor.backspace();
                self.follow_trigger()
            }
            Key::Char(c) => {
                self.editor.insert(c);
                self.follow_trigger()
            }
            _ => Vec::new(),
        }
    }

    /// The text at the cursor opens (or narrows, or closes) the files or commands list.
    fn follow_trigger(&mut self) -> Vec<Effect> {
        match self.editor.trigger() {
            Trigger::Files(query) => {
                let session = self.attached.clone().unwrap_or_default();
                let effect = self.request(Pending::Files, |id| PageMessage::Session {
                    session,
                    message: Box::new(PageMessage::ListFiles { id, query }),
                });
                vec![effect]
            }
            Trigger::Commands(query) => {
                if let Some(items) = self.commands.clone() {
                    let mut picker = Picker::new(items);
                    picker.set_query(&query);
                    self.picker = Some((PickerKind::Commands, picker));
                    return Vec::new();
                }
                vec![self.session_request(Pending::Commands, |id| PageMessage::ListCommands { id })]
            }
            Trigger::None => {
                if matches!(self.picker, Some((PickerKind::Files | PickerKind::Commands, _))) {
                    self.picker = None;
                }
                Vec::new()
            }
        }
    }

    /// `run`: a command picked with Enter runs at once; with Tab or a click it is only completed in the editor (a stray
    /// click never closes a session).
    fn picked(&mut self, kind: PickerKind, outcome: Outcome, run: bool) -> Vec<Effect> {
        match outcome {
            // Enter on a `/` word no command matches: it goes to the session as typed, as in the dock.
            Outcome::None if run && kind == PickerKind::Commands => {
                self.picker = None;
                self.submit()
            }
            Outcome::None => Vec::new(),
            Outcome::Closed => {
                self.picker = None;
                Vec::new()
            }
            Outcome::Picked(value) => {
                self.picker = None;
                match kind {
                    PickerKind::Files => {
                        self.editor.complete(&value);
                        Vec::new()
                    }
                    PickerKind::Commands if run => {
                        self.editor.set(&format!("/{value}"));
                        self.submit()
                    }
                    PickerKind::Commands => {
                        self.editor.set(&format!("/{value} "));
                        Vec::new()
                    }
                    PickerKind::Models => {
                        vec![self.session_request(Pending::Models, |id| PageMessage::SetModel { id, model: value })]
                    }
                }
            }
        }
    }

    fn rewound(&mut self, outcome: rewind::Outcome) -> Vec<Effect> {
        match outcome {
            rewind::Outcome::None => Vec::new(),
            rewind::Outcome::Close => {
                self.rewind = None;
                Vec::new()
            }
            rewind::Outcome::Undo { entry, checkpoint, files, conversation, text } => {
                self.rewind = None;
                let back = if conversation { text } else { String::new() };
                vec![self.session_request(Pending::Undo(back), |id| PageMessage::ChatUndo {
                    id,
                    entry,
                    checkpoint,
                    files,
                    conversation,
                })]
            }
        }
    }

    /// The panel goes, and so do the requests it waits on: a late reply must not open it in another session.
    fn drop_schedules(&mut self) {
        self.schedules = None;
        self.pending.retain(|_, p| {
            !matches!(p, Pending::SchedulesOpen | Pending::SchedulesChange | Pending::SchedulesResume(_))
        });
    }

    fn scheduled_outcome(&mut self, outcome: schedules::Outcome) -> Vec<Effect> {
        match outcome {
            schedules::Outcome::None => Vec::new(),
            schedules::Outcome::Close => {
                self.schedules = None;
                Vec::new()
            }
            schedules::Outcome::Update { schedule, enabled } => {
                vec![self.request(Pending::SchedulesChange, |id| PageMessage::ScheduleUpdate { id, schedule, enabled })]
            }
            schedules::Outcome::Check { schedule } => {
                vec![self.request(Pending::SchedulesResume(schedule), |id| PageMessage::SchedulesList { id })]
            }
            schedules::Outcome::Delete { schedule } => {
                vec![self.request(Pending::SchedulesChange, |id| PageMessage::ScheduleDelete { id, schedule })]
            }
        }
    }

    /// The panel goes, and so do the requests it waits on: a late reply is no one's (the dock's own sign-in times out on
    /// its own).
    fn drop_signin(&mut self) {
        self.signin = None;
        self.pending.retain(|_, p| !matches!(p, Pending::SignIn(_) | Pending::SignInCode));
    }

    fn signed_in_outcome(&mut self, outcome: signin::Outcome) -> Vec<Effect> {
        match outcome {
            signin::Outcome::None => Vec::new(),
            signin::Outcome::Close => {
                self.drop_signin();
                Vec::new()
            }
            signin::Outcome::Send { service, redirect } => {
                vec![self.request(Pending::SignInCode, |id| PageMessage::OAuthCode { id, service, redirect })]
            }
        }
    }

    /// A line in the conversation, in the client's own words.
    fn say(&mut self, tone: crate::term::state::Tone, key: &str, vars: serde_json::Value) {
        self.session.entries.push(crate::term::state::Entry::Notice {
            tone,
            text: crate::term::state::Text::Key { key: key.into(), vars },
        });
    }

    /// The sign-in is over without a result of its own to show: `code` is the dock's error key.
    fn sign_in_failed(&mut self, code: &str) {
        self.drop_signin();
        let lang = self.config.lang.clone();
        let text = t(&lang, &format!("ob.error.{code}"), &json!({}));
        let why = if text.starts_with("ob.error.") { code.to_string() } else { text };
        self.say(crate::term::state::Tone::Warn, "term.signin.failed", json!({ "why": why }));
    }

    /// The most rows a panel in the editor's place takes: two thirds of the screen.
    fn panel_rows(&self) -> usize {
        usize::from(self.height * 2 / 3).max(8)
    }

    fn set_verbose(&mut self, on: bool) -> Vec<Effect> {
        self.config.verbose = on;
        vec![Effect::SaveVerbose(on)]
    }

    fn submit(&mut self) -> Vec<Effect> {
        let text = self.editor.take();
        self.scroll = 0;
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }
        if let Some(rest) = trimmed.strip_prefix('/') {
            let mut words = rest.split_whitespace();
            match words.next().unwrap_or_default() {
                "exit" => {
                    let close =
                        Effect::Send(PageMessage::SessionClose { session: self.attached.clone().unwrap_or_default() });
                    return vec![close, Effect::Quit];
                }
                "verbose" => {
                    let on = match words.next() {
                        Some("on") => true,
                        Some("off") => false,
                        _ => !self.config.verbose,
                    };
                    return self.set_verbose(on);
                }
                "sessions" => return self.back_to_list(),
                "model" => return vec![self.session_request(Pending::Models, |id| PageMessage::ListModels { id })],
                "undo" => {
                    return vec![self.session_request(Pending::Checkpoints, |id| PageMessage::ChatCheckpoints { id })]
                }
                "schedules" => {
                    return vec![self.request(Pending::SchedulesOpen, |id| PageMessage::SchedulesList { id })]
                }
                "signin" => {
                    let services: Vec<&str> = words.collect();
                    return match services[..] {
                        ["anthropic"] => {
                            self.drop_signin();
                            vec![self.request(Pending::SignIn("anthropic".into()), |id| PageMessage::StartOAuth { id })]
                        }
                        ["notion"] => {
                            self.drop_signin();
                            vec![self.request(Pending::SignIn("notion".into()), |id| PageMessage::StartMcpOAuth {
                                id,
                                service: "notion".into(),
                            })]
                        }
                        _ => {
                            self.say(crate::term::state::Tone::Warn, "term.signin.usage", json!({}));
                            Vec::new()
                        }
                    };
                }
                "context" => {
                    return vec![self.session_request(Pending::Context, |id| PageMessage::PiCommand {
                        id,
                        command: "get_session_stats".into(),
                        args: json!({}),
                    })]
                }
                _ => {}
            }
        }
        if let Some(command) = bang(trimmed) {
            let args = json!({ "command": command });
            return vec![self.session_request(Pending::Bang(command), |id| PageMessage::PiCommand {
                id,
                command: "bash".into(),
                args,
            })];
        }
        let behavior = if self.session.running {
            // For the next gap: after the thinking or the tool that runs now (Pi's steering).
            Some("steer".to_string())
        } else {
            self.session.add_user(trimmed);
            None
        };
        vec![self.in_session(PageMessage::ChatSend { text: trimmed.to_string(), behavior, attachments: Vec::new() })]
    }

    fn answered(&mut self, answer: Option<Answer>) -> Vec<Effect> {
        match answer {
            None => Vec::new(),
            Some(Answer::Ui(message)) => {
                self.dialog = None;
                vec![self.in_session(message)]
            }
            Some(Answer::Secret { name, value, purpose }) => {
                let dialog = self.dialog.as_ref().map(|d| d.id.clone()).unwrap_or_default();
                vec![self.request(Pending::Secret(dialog), |id| PageMessage::SecretSave { id, name, value, purpose })]
            }
        }
    }

    /// The row a dialog starts at, as `view` draws it in the editor's place: above the footer and a rule.
    fn panel_top(&self, panel_len: usize) -> u16 {
        self.height.saturating_sub(2 + panel_len as u16)
    }

    /// A dialog, in the editor's place.
    pub fn panel_lines(
        &self,
        palette: &crate::term::theme::Palette,
        width: u16,
    ) -> Option<Vec<ratatui::text::Line<'static>>> {
        if let Some(open) = &self.rewind {
            return Some(open.lines(&self.config.lang, palette, width, self.panel_rows()));
        }
        if let Some(open) = &self.signin {
            return Some(open.lines(&self.config.lang, palette, width));
        }
        if let Some(open) = &self.schedules {
            return Some(open.lines(&self.config.lang, palette, width, self.panel_rows()));
        }
        if let Some((used, window)) = self.context {
            let parts = self
                .session
                .statuses
                .get("aria-context")
                .and_then(|text| serde_json::from_str(text).ok())
                .unwrap_or_else(|| json!({}));
            return Some(crate::term::view::context_lines(
                &parts,
                used,
                window,
                self.config.compact_at,
                palette,
                &self.config.lang,
            ));
        }
        self.dialog.as_ref().map(|dialog| dialog.lines(&self.config.lang, palette, width))
    }

    /// The suggestions (`/`, `@`, `/model`), floating above the editor; none while a dialog is open.
    pub fn overlay_lines(&self, palette: &crate::term::theme::Palette) -> Option<Vec<ratatui::text::Line<'static>>> {
        if self.dialog.is_some() {
            return None;
        }
        self.picker.as_ref().map(|(_, picker)| picker.lines(palette))
    }

    pub fn tasks_folded(&self) -> bool {
        self.tasks_folded
    }

    /// Where the screen drew the task strip.
    pub fn set_strip_rows(&mut self, rows: Option<(u16, u16)>) {
        self.strip_rows = rows;
    }

    pub fn expanded(&self) -> &std::collections::HashSet<String> {
        &self.expanded
    }

    /// Where the screen drew the `… N more lines` rows.
    pub fn set_more_rows(&mut self, rows: Vec<(u16, String)>) {
        self.more_rows = rows;
    }

    /// The row the suggestions start at, as the screen drew them.
    pub fn set_overlay_top(&mut self, row: u16) {
        self.overlay_top = row;
    }

    fn click(&mut self, row: u16) -> Vec<Effect> {
        let mut effects = Vec::new();
        match self.screen {
            Screen::List => {
                // The list starts on the second row.
                let at = usize::from(row.saturating_sub(1));
                if at <= self.sessions.len() {
                    self.selected = at;
                    effects.extend(self.open_selected());
                }
            }
            Screen::Session if self.strip_rows.is_some_and(|(first, last)| row >= first && row <= last) => {
                self.tasks_folded = !self.tasks_folded;
            }
            Screen::Session if self.more_rows.iter().any(|(r, _)| *r == row) => {
                if let Some((_, id)) = self.more_rows.iter().find(|(r, _)| *r == row).cloned() {
                    if !self.expanded.remove(&id) {
                        self.expanded.insert(id);
                    }
                }
            }
            Screen::Session => {
                let palette = crate::term::theme::Palette::truecolor();
                if let Some(lines) = self.panel_lines(&palette, 80) {
                    let top = self.panel_top(lines.len());
                    if row >= top {
                        let rows = self.panel_rows();
                        if let Some(open) = self.rewind.as_mut() {
                            let outcome = open.click(usize::from(row - top), rows);
                            effects.extend(self.rewound(outcome));
                        } else {
                            let answer = self.dialog.as_mut().and_then(|dialog| dialog.click(usize::from(row - top)));
                            effects.extend(self.answered(answer));
                        }
                    }
                } else if let Some(lines) = self.overlay_lines(&palette) {
                    let top = self.overlay_top;
                    if row >= top && usize::from(row - top) < lines.len() {
                        if let Some((kind, picker)) = self.picker.as_ref() {
                            let (kind, outcome) = (*kind, picker.click(usize::from(row - top)));
                            effects.extend(self.picked(kind, outcome, false));
                        }
                    }
                }
            }
        }
        effects
    }

    // ---- the dock ------------------------------------------------------------------------------------------------

    fn dock(&mut self, message: DockMessage) -> Vec<Effect> {
        match message {
            // Lost: the dock may only have dropped this client (it read too slowly); try once.
            DockMessage::Error { id: 0, code } if code == "gone" => vec![Effect::Reconnect],
            DockMessage::Error { id, code } => {
                match self.pending.remove(&id) {
                    // A connected hub refused what was asked (a `!command`, a session it does not have): say so.
                    Some(_) if code == "not-for-remote" || code == "unknown-session" => {
                        self.say(crate::term::state::Tone::Warn, "term.notForRemote", json!({}));
                    }
                    Some(Pending::SessionNew) if code == "too-many-sessions" => {
                        self.note = Some(t(&self.config.lang, "term.tooMany", &json!({})));
                    }
                    Some(Pending::SchedulesOpen) => {
                        self.session.entries.push(crate::term::state::Entry::Notice {
                            tone: crate::term::state::Tone::Warn,
                            text: crate::term::state::Text::Key {
                                key: "term.schedules.failed".into(),
                                vars: json!({}),
                            },
                        });
                    }
                    // The address was not this sign-in's: the panel waits for another.
                    Some(Pending::SignInCode) if code == "bad-redirect" => {
                        if let Some(open) = self.signin.as_mut() {
                            open.refuse();
                        }
                    }
                    Some(Pending::SignInCode) if code == "no-sign-in" => {
                        self.drop_signin();
                        self.say(crate::term::state::Tone::Warn, "term.signin.nothing", json!({}));
                    }
                    Some(Pending::SignInCode | Pending::SignIn(_)) => self.sign_in_failed(&code),
                    _ => {}
                }
                Vec::new()
            }
            DockMessage::OAuthUrl { id, url } => {
                if let Some(Pending::SignIn(service)) = self.pending.get(&id) {
                    if self.screen == Screen::Session {
                        self.rewind = None;
                        self.context = None;
                        self.schedules = None;
                        self.signin = Some(SignIn::new(service, &url));
                    }
                }
                Vec::new()
            }
            DockMessage::OAuthResult { id, ok, error } => {
                if let Some(Pending::SignIn(_)) = self.pending.get(&id) {
                    if ok {
                        self.drop_signin();
                        self.say(crate::term::state::Tone::Info, "term.signin.done", json!({}));
                    } else {
                        self.sign_in_failed(&error.unwrap_or_else(|| "oauth-failed".into()));
                    }
                }
                Vec::new()
            }
            DockMessage::Schedules { id, tasks } => {
                match self.pending.remove(&id) {
                    Some(Pending::SchedulesOpen) if self.screen == Screen::Session => {
                        self.rewind = None;
                        self.context = None;
                        self.schedules = Some(Schedules::new(&tasks));
                    }
                    Some(Pending::SchedulesChange) => {
                        if let Some(open) = self.schedules.as_mut() {
                            open.replace(&tasks);
                        }
                    }
                    // The resume goes only if the fresh list still shows the task paused and not waiting for approval
                    // (the dock approves a waiting task when it is resumed). What can still happen between this reply
                    // and the dock reading the update is accepted: it is a few milliseconds, not a stale screen.
                    Some(Pending::SchedulesResume(task)) => {
                        let Some(open) = self.schedules.as_mut() else { return Vec::new() };
                        open.replace(&tasks);
                        if open.can_resume(&task) {
                            return vec![self.request(Pending::SchedulesChange, |id| PageMessage::ScheduleUpdate {
                                id,
                                schedule: task,
                                enabled: true,
                            })];
                        }
                    }
                    _ => {}
                }
                Vec::new()
            }
            DockMessage::ScheduleAwaiting { schedule } => {
                if let Some(open) = self.schedules.as_mut() {
                    open.mark_awaiting(&schedule);
                }
                Vec::new()
            }
            DockMessage::Sessions { id, sessions, .. } if self.pending.remove(&id).is_some() => {
                self.sessions = sessions.into_iter().filter(|s| s.schedule.is_none()).collect();
                self.selected = self.selected.min(self.sessions.len());
                Vec::new()
            }
            DockMessage::SessionAdded { id: Some(id), session } if self.pending.remove(&id).is_some() => {
                self.attach(session.id)
            }
            DockMessage::SessionAdded { .. } | DockMessage::SessionsChanged if self.screen == Screen::List => {
                vec![self.request(Pending::SessionsLoad, |id| PageMessage::SessionsLoad { id })]
            }
            DockMessage::SessionRemoved { session } if self.attached.as_deref() == Some(session.as_str()) => {
                let effects = self.back_to_list();
                self.note = Some(t(&self.config.lang, "term.closedElsewhere", &json!({})));
                effects
            }
            DockMessage::SessionRemoved { .. } if self.screen == Screen::List => {
                vec![self.request(Pending::SessionsLoad, |id| PageMessage::SessionsLoad { id })]
            }
            DockMessage::SecretResult { id, ok, name, error, .. } => match self.pending.remove(&id) {
                Some(Pending::Secret(dialog)) if ok => {
                    self.dialog = None;
                    match saved_answer(&dialog, name.as_deref().unwrap_or_default()) {
                        Answer::Ui(message) => vec![self.in_session(message)],
                        Answer::Secret { .. } => Vec::new(),
                    }
                }
                Some(Pending::Secret(_)) => {
                    self.note = error;
                    Vec::new()
                }
                _ => Vec::new(),
            },
            DockMessage::Session { message, .. } if matches!(*message, DockMessage::SessionStatus { .. }) => {
                if let DockMessage::SessionStatus { session, status, at } = *message {
                    self.statuses.insert(session, (status, at));
                }
                Vec::new()
            }
            DockMessage::Session { session, message } if self.attached.as_deref() == Some(session.as_str()) => {
                self.on_session(*message)
            }
            _ => Vec::new(),
        }
    }

    fn on_session(&mut self, message: DockMessage) -> Vec<Effect> {
        match message {
            DockMessage::Files { id, files } if self.pending.remove(&id).is_some() => {
                let items = files
                    .into_iter()
                    .map(|f| Item { label: f.clone(), detail: String::new(), value: f, group: None })
                    .collect();
                self.picker = Some((PickerKind::Files, Picker::new(items)));
            }
            DockMessage::Commands { id, commands } if self.pending.remove(&id).is_some() => {
                let lang = self.config.lang.clone();
                let mut items: Vec<Item> = OWN
                    .iter()
                    .map(|(name, key)| Item {
                        label: format!("/{name}"),
                        detail: t(&lang, key, &json!({})),
                        value: name.to_string(),
                        group: None,
                    })
                    .collect();
                items.extend(commands.as_array().into_iter().flatten().filter_map(|c| {
                    let name = c["name"].as_str()?;
                    Some(Item {
                        label: format!("/{name}"),
                        detail: c["description"].as_str().unwrap_or_default().to_string(),
                        value: name.to_string(),
                        group: None,
                    })
                }));
                self.commands = Some(items.clone());
                let mut picker = Picker::new(items);
                if let Trigger::Commands(query) = self.editor.trigger() {
                    picker.set_query(&query);
                    self.picker = Some((PickerKind::Commands, picker));
                }
            }
            DockMessage::Models { id, models } if self.pending.remove(&id).is_some() => {
                let current = self.session.model.clone().unwrap_or_default();
                let items = models_for_picker(&models, &self.config.connected, &current, self.config.allow_pay_per_use);
                self.picker = Some((PickerKind::Models, Picker::new(items)));
            }
            DockMessage::PiReply { id, ok, data, .. } if self.pending.get(&id) == Some(&Pending::Context) => {
                self.pending.remove(&id);
                if ok {
                    let usage = &data["contextUsage"];
                    self.context =
                        Some((usage["tokens"].as_u64().unwrap_or(0), usage["contextWindow"].as_u64().unwrap_or(0)));
                }
            }
            DockMessage::PiReply { id, ok, data, error } => {
                if let Some(Pending::Bang(command)) = self.pending.remove(&id) {
                    let output = if ok {
                        data["output"].as_str().unwrap_or_default().to_string()
                    } else {
                        error.unwrap_or_default()
                    };
                    let exit = if ok { data["exitCode"].as_i64().unwrap_or(0) } else { 1 };
                    self.session.add_bang(&command, &output, exit);
                }
            }
            DockMessage::Checkpoints { id, list, off } if self.pending.get(&id) == Some(&Pending::Checkpoints) => {
                self.pending.remove(&id);
                self.picker = None;
                self.rewind = Some(Rewind::new(&list, off));
            }
            DockMessage::Saved { id } if matches!(self.pending.get(&id), Some(Pending::Undo(_))) => {
                if let Some(Pending::Undo(text)) = self.pending.remove(&id) {
                    if !text.is_empty() {
                        self.editor.set(&text);
                    }
                }
            }
            DockMessage::Error { id, code } if matches!(self.pending.get(&id), Some(Pending::Undo(_))) => {
                self.pending.remove(&id);
                let key = if code == "busy" { "term.undo.busy" } else { "term.undo.failed" };
                self.session.entries.push(crate::term::state::Entry::Notice {
                    tone: crate::term::state::Tone::Warn,
                    text: crate::term::state::Text::Key { key: key.into(), vars: json!({}) },
                });
            }
            DockMessage::Saved { id } | DockMessage::Error { id, .. } if self.pending.contains_key(&id) => {
                self.pending.remove(&id);
            }
            other => {
                self.session.apply(&other);
                if let Some(text) = self.session.give_back.take() {
                    self.editor.set(&text);
                }
            }
        }
        self.follow_dialogs();
        // A Ctrl+Enter stopped the turn: its message goes now that the turn settled.
        if !self.session.running {
            if let Some(text) = self.after_abort.take() {
                self.session.add_user(&text);
                return vec![self.in_session(PageMessage::ChatSend { text, behavior: None, attachments: Vec::new() })];
            }
        }
        Vec::new()
    }

    /// Ctrl+Enter: while a turn runs, stop it and send the message once it settles; otherwise send it.
    fn interrupt(&mut self) -> Vec<Effect> {
        if !self.session.running {
            return self.submit();
        }
        let text = self.editor.take().trim().to_string();
        if text.is_empty() {
            return Vec::new();
        }
        self.scroll = 0;
        self.after_abort = Some(text);
        vec![self.in_session(PageMessage::ChatAbort)]
    }

    /// The first open dialog takes the editor's place; one closed elsewhere goes away.
    fn follow_dialogs(&mut self) {
        if let Some(open) = &self.dialog {
            if !self.session.dialogs.iter().any(|d| d.id == open.id) {
                self.dialog = None;
            }
        }
        if self.dialog.is_none() {
            if let Some(first) = self.session.dialogs.first() {
                self.dialog = Some(DialogBox::from(first));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_core::bridge::{DockMessage, PageMessage};
    use aria_core::sessions::SessionEntry;
    use serde_json::json;

    fn config() -> Config {
        Config {
            cwd: "/home/ana/Projects/checkout-widget".into(),
            lang: "en".into(),
            verbose: true,
            connected: vec![],
            allow_pay_per_use: false,
            compact_at: 40,
            remote: None,
        }
    }

    fn entry(id: &str, title: &str) -> SessionEntry {
        SessionEntry { id: id.into(), title: title.into(), segments: vec![], schedule: None, workdir: None }
    }

    fn key(app: &mut App, key: Key) -> Vec<Effect> {
        app.update(Input::Key(Press { key, ctrl: false, alt: false }))
    }

    fn ctrl(app: &mut App, c: char) -> Vec<Effect> {
        app.update(Input::Key(Press { key: Key::Char(c), ctrl: true, alt: false }))
    }

    fn typed(app: &mut App, text: &str) {
        for c in text.chars() {
            key(app, Key::Char(c));
        }
    }

    /// An app on the list with sessions `aa` and `bb`, answered.
    fn listed() -> App {
        let (mut app, effects) = App::new(config());
        let Some(Effect::Send(PageMessage::SessionsLoad { id })) = effects.first().cloned() else {
            panic!("{effects:?}")
        };
        app.update(Input::Dock(DockMessage::Sessions {
            id,
            sessions: vec![entry("aa", "Coupons"), entry("bb", "")],
            active: Some("aa".into()),
        }));
        app
    }

    fn attached() -> App {
        let mut app = listed();
        key(&mut app, Key::Enter);
        app
    }

    fn in_session(message: DockMessage) -> Input {
        Input::Dock(DockMessage::Session { session: "aa".into(), message: Box::new(message) })
    }

    #[test]
    fn starts_on_the_list_and_attaches_on_enter() {
        let mut app = listed();
        assert_eq!(app.screen(), Screen::List);
        let effects = key(&mut app, Key::Enter);
        assert_eq!(effects[0], Effect::Attach("aa".into()));
        assert!(matches!(&effects[1], Effect::Send(PageMessage::Session { session, message })
            if session == "aa" && **message == PageMessage::ChatLoad { fresh: true }));
        assert_eq!(app.screen(), Screen::Session);
    }

    #[test]
    fn a_new_session_uses_the_current_folder() {
        let mut app = listed();
        key(&mut app, Key::Down);
        key(&mut app, Key::Down);
        let effects = key(&mut app, Key::Enter);
        let Some(Effect::Send(PageMessage::SessionNew { id, workdir })) = effects.first().cloned() else {
            panic!("{effects:?}")
        };
        assert_eq!(workdir.as_deref(), Some("/home/ana/Projects/checkout-widget"));
        let added = entry("cc", "");
        let effects = app.update(Input::Dock(DockMessage::SessionAdded { id: Some(id), session: added }));
        assert_eq!(effects[0], Effect::Attach("cc".into()));
    }

    #[test]
    fn ctrl_c_quits_without_closing_anything() {
        let mut app = attached();
        app.update(in_session(DockMessage::ChatState {
            workdir: "/w".into(),
            running: true,
            compacting: false,
            model: None,
            thinking: None,
            context: None,
            problem: None,
        }));
        let effects = ctrl(&mut app, 'c');
        assert_eq!(effects, vec![Effect::Quit]);
        let mut again = attached();
        assert_eq!(ctrl(&mut again, 'd'), vec![Effect::Quit]);
    }

    #[test]
    fn exit_closes_the_session_then_quits() {
        let mut app = attached();
        typed(&mut app, "/exit");
        let effects = key(&mut app, Key::Enter);
        assert_eq!(effects, vec![Effect::Send(PageMessage::SessionClose { session: "aa".into() }), Effect::Quit]);
    }

    #[test]
    fn esc_stops_a_running_turn() {
        let mut app = attached();
        app.update(in_session(DockMessage::ChatEvent { event: json!({"type": "agent_start"}) }));
        let effects = key(&mut app, Key::Esc);
        assert!(
            matches!(&effects[0], Effect::Send(PageMessage::Session { message, .. }) if **message == PageMessage::ChatAbort)
        );
    }

    #[test]
    fn a_message_is_sent_and_queued_while_running() {
        let mut app = attached();
        typed(&mut app, "fix it");
        let effects = key(&mut app, Key::Enter);
        assert!(matches!(&effects[0], Effect::Send(PageMessage::Session { message, .. })
            if **message == PageMessage::ChatSend { text: "fix it".into(), behavior: None, attachments: vec![] }));
        app.update(in_session(DockMessage::ChatEvent { event: json!({"type": "agent_start"}) }));
        typed(&mut app, "and the tests");
        let effects = key(&mut app, Key::Enter);
        assert!(
            matches!(&effects[0], Effect::Send(PageMessage::Session { message, .. })
            if matches!(&**message, PageMessage::ChatSend { behavior: Some(b), .. } if b == "steer")),
            "while it works, Enter queues the message for the next gap (after thinking or a tool)"
        );
    }

    #[test]
    fn a_session_closed_elsewhere_returns_to_the_list() {
        let mut app = attached();
        let effects = app.update(Input::Dock(DockMessage::SessionRemoved { session: "aa".into() }));
        assert_eq!(app.screen(), Screen::List);
        assert!(effects.contains(&Effect::Detach));
        assert!(effects.iter().any(|e| matches!(e, Effect::Send(PageMessage::SessionsLoad { .. }))));
        assert_eq!(app.note().as_deref(), Some("This session was closed."));
    }

    #[test]
    fn verbose_flips_and_saves() {
        let mut app = attached();
        assert_eq!(ctrl(&mut app, 'o'), vec![Effect::SaveVerbose(false)]);
        typed(&mut app, "/verbose on");
        assert_eq!(key(&mut app, Key::Enter), vec![Effect::SaveVerbose(true)]);
    }

    #[test]
    fn a_bang_runs_and_shows_its_output() {
        let mut app = attached();
        typed(&mut app, "!pwd");
        let effects = key(&mut app, Key::Enter);
        let Some(Effect::Send(PageMessage::Session { message, .. })) = effects.first() else { panic!("{effects:?}") };
        let PageMessage::PiCommand { id, command, args } = &**message else { panic!("{message:?}") };
        assert_eq!((command.as_str(), args["command"].as_str()), ("bash", Some("pwd")));
        app.update(in_session(DockMessage::PiReply {
            id: *id,
            ok: true,
            data: json!({"output": "/w\n", "exitCode": 0}),
            error: None,
        }));
        assert!(
            matches!(app.session().entries.last(), Some(crate::term::state::Entry::Tools { calls }) if calls[0].output == "/w\n")
        );
    }

    #[test]
    fn what_a_connected_hub_refuses_is_said_not_dropped() {
        // A `!command` over `aria connect` is answered `not-for-remote`; a made-up session, `unknown-session`.
        for code in ["not-for-remote", "unknown-session"] {
            let mut app = attached();
            typed(&mut app, "!pwd");
            let effects = key(&mut app, Key::Enter);
            let Some(Effect::Send(PageMessage::Session { message, .. })) = effects.first() else {
                panic!("{effects:?}")
            };
            let PageMessage::PiCommand { id, .. } = &**message else { panic!("{message:?}") };
            app.update(Input::Dock(DockMessage::Error { id: *id, code: code.into() }));
            let shown = app.session().entries.last();
            assert!(
                matches!(shown, Some(crate::term::state::Entry::Notice { text: crate::term::state::Text::Key { key, .. }, .. }) if key == "term.notForRemote"),
                "{code}: {shown:?}"
            );
        }
        assert_eq!(t("en", "term.notForRemote", &json!({})), "not available on a connected hub");
        assert_eq!(t("es", "term.notForRemote", &json!({})), "no disponible en un hub conectado");
        // An error nobody asked for says nothing.
        let mut app = attached();
        let before = app.session().entries.len();
        app.update(Input::Dock(DockMessage::Error { id: 999, code: "not-for-remote".into() }));
        assert_eq!(app.session().entries.len(), before);
    }

    #[test]
    fn a_dialog_takes_the_keys_and_its_answer_goes_out() {
        let mut app = attached();
        app.update(in_session(DockMessage::ChatDialog {
            id: "u1".into(),
            method: "confirm".into(),
            request: json!({"title": "Sure?"}),
            hidden: false,
        }));
        let effects = key(&mut app, Key::Char('y'));
        assert!(matches!(&effects[0], Effect::Send(PageMessage::Session { message, .. })
            if **message == PageMessage::UiAnswer { id: "u1".into(), value: None, confirmed: Some(true), cancelled: None }));
    }

    #[test]
    fn a_secret_is_kept_by_the_dock_before_the_answer() {
        let mut app = attached();
        let title = json!({"aria": 1, "kind": "secret", "name": "GH_TOKEN", "purpose": "push"}).to_string();
        app.update(in_session(DockMessage::ChatDialog {
            id: "u2".into(),
            method: "input".into(),
            request: json!({"title": title}),
            hidden: false,
        }));
        typed(&mut app, "s3cret");
        let effects = key(&mut app, Key::Enter);
        let Some(Effect::Send(PageMessage::SecretSave { id, name, value, .. })) = effects.first().cloned() else {
            panic!("{effects:?}")
        };
        assert_eq!((name.as_str(), value.as_str()), ("GH_TOKEN", "s3cret"));
        let effects = app.update(Input::Dock(DockMessage::SecretResult {
            id,
            ok: true,
            name: Some("GH_TOKEN".into()),
            env: None,
            error: None,
        }));
        assert!(matches!(&effects[0], Effect::Send(PageMessage::Session { message, .. })
            if matches!(&**message, PageMessage::UiAnswer { value: Some(v), .. } if v == "saved:GH_TOKEN")));
    }

    #[test]
    fn page_up_scrolls_and_sending_returns_to_the_bottom() {
        let mut app = attached();
        app.update(Input::Resize { height: 30 });
        key(&mut app, Key::PageUp);
        assert_eq!(app.scroll(), 15);
        app.update(Input::Scroll(3));
        assert_eq!(app.scroll(), 18);
        key(&mut app, Key::PageDown);
        assert_eq!(app.scroll(), 3);
        app.clamp_scroll(2);
        assert_eq!(app.scroll(), 2, "never past the top");
        typed(&mut app, "hi");
        key(&mut app, Key::Enter);
        assert_eq!(app.scroll(), 0);
    }

    #[test]
    fn a_run_is_timed_from_when_it_starts() {
        let mut app = attached();
        app.clock(1_000);
        assert_eq!(app.working_ms(5_000), None);
        app.update(in_session(DockMessage::ChatEvent { event: json!({"type": "agent_start"}) }));
        app.clock(2_000);
        assert_eq!(app.working_ms(5_400), Some(3_400));
        app.update(in_session(DockMessage::ChatEvent { event: json!({"type": "agent_settled"}) }));
        app.clock(6_000);
        assert_eq!(app.working_ms(6_000), None);
    }

    #[test]
    fn a_click_on_a_suggestion_picks_it_and_the_editor_stays() {
        let mut app = attached();
        let effects = key(&mut app, Key::Char('/'));
        let Some(Effect::Send(PageMessage::Session { message, .. })) = effects.first() else { panic!("{effects:?}") };
        let PageMessage::ListCommands { id } = **message else { panic!("{message:?}") };
        app.update(in_session(DockMessage::Commands { id, commands: json!([]) }));
        let palette = crate::term::theme::Palette::truecolor();
        assert!(app.panel_lines(&palette, 80).is_none(), "the editor keeps its place");
        assert!(app.overlay_lines(&palette).is_some());
        app.set_overlay_top(10);
        app.update(Input::Click { row: 10 });
        assert_eq!(app.editor.text(), "/exit ");
    }

    #[test]
    fn ctrl_enter_stops_the_turn_and_sends_once_it_settles() {
        let mut app = attached();
        app.update(in_session(DockMessage::ChatEvent { event: json!({"type": "agent_start"}) }));
        typed(&mut app, "stop, do this instead");
        let effects = app.update(Input::Key(Press { key: Key::Enter, ctrl: true, alt: false }));
        assert!(
            matches!(&effects[..], [Effect::Send(PageMessage::Session { message, .. })] if **message == PageMessage::ChatAbort)
        );
        assert_eq!(app.editor.text(), "");
        let effects =
            app.update(in_session(DockMessage::ChatEvent { event: json!({"type": "agent_settled", "aborted": true}) }));
        assert!(matches!(&effects[..], [Effect::Send(PageMessage::Session { message, .. })]
            if **message == PageMessage::ChatSend { text: "stop, do this instead".into(), behavior: None, attachments: vec![] }));
        assert!(app
            .session()
            .entries
            .iter()
            .any(|e| matches!(e, crate::term::state::Entry::User { text } if text == "stop, do this instead")));
    }

    #[test]
    fn ctrl_enter_while_idle_just_sends() {
        let mut app = attached();
        typed(&mut app, "hi");
        let effects = app.update(Input::Key(Press { key: Key::Enter, ctrl: true, alt: false }));
        assert!(matches!(&effects[..], [Effect::Send(PageMessage::Session { message, .. })]
            if matches!(&**message, PageMessage::ChatSend { behavior: None, .. })));
    }

    #[test]
    fn enter_on_a_command_suggestion_runs_it_and_tab_completes_it() {
        let mut app = attached();
        typed(&mut app, "/");
        let id = app.pending.iter().find(|(_, p)| **p == Pending::Commands).map(|(id, _)| *id).unwrap();
        app.update(in_session(DockMessage::Commands { id, commands: json!([]) }));
        typed(&mut app, "exit");
        assert_eq!(
            key(&mut app, Key::Enter),
            vec![Effect::Send(PageMessage::SessionClose { session: "aa".into() }), Effect::Quit]
        );
        let mut other = attached();
        typed(&mut other, "/");
        let id = other.pending.iter().find(|(_, p)| **p == Pending::Commands).map(|(id, _)| *id).unwrap();
        other.update(in_session(DockMessage::Commands { id, commands: json!([]) }));
        typed(&mut other, "verb");
        key(&mut other, Key::Tab);
        assert_eq!(other.editor.text(), "/verbose ");
    }

    #[test]
    fn enter_on_a_word_no_command_matches_sends_it_as_a_message() {
        let mut app = attached();
        typed(&mut app, "/");
        let id = app.pending.iter().find(|(_, p)| **p == Pending::Commands).map(|(id, _)| *id).unwrap();
        app.update(in_session(DockMessage::Commands { id, commands: json!([]) }));
        typed(&mut app, "zzz");
        let effects = key(&mut app, Key::Enter);
        assert!(
            matches!(&effects[..], [Effect::Send(PageMessage::Session { message, .. })]
            if **message == PageMessage::ChatSend { text: "/zzz".into(), behavior: None, attachments: vec![] }),
            "{effects:?}"
        );
        assert_eq!(app.editor.text(), "");
        assert!(app.picker.is_none(), "the list closes with the message");
    }

    #[test]
    fn pasted_line_breaks_are_newlines() {
        let mut app = attached();
        app.update(Input::Paste("a\r\nb\rc".into()));
        assert_eq!(app.editor.text(), "a\nb\nc");
    }

    #[test]
    fn ctrl_l_redraws() {
        let mut app = attached();
        assert_eq!(ctrl(&mut app, 'l'), vec![Effect::Redraw]);
    }

    #[test]
    fn a_click_on_more_lines_shows_the_whole_output() {
        let mut app = attached();
        app.set_more_rows(vec![(7, "c1".into())]);
        app.update(Input::Click { row: 7 });
        assert!(app.expanded().contains("c1"));
        app.update(Input::Click { row: 7 });
        assert!(!app.expanded().contains("c1"), "a second click folds it again");
    }

    #[test]
    fn the_list_says_what_each_session_does_newest_first() {
        let mut app = listed();
        app.clock(10_000_000);
        let status = |session: &str, status: &str, at: i64| {
            Input::Dock(DockMessage::Session {
                session: session.into(),
                message: Box::new(DockMessage::SessionStatus { session: session.into(), status: status.into(), at }),
            })
        };
        app.update(status("aa", "done", 10_000_000 - 3_600_000));
        app.update(status("bb", "working", 10_000_000 - 180_000));
        let rows = app.list_rows();
        assert_eq!(rows[0].0, "untitled");
        assert_eq!((rows[0].2.as_str(), rows[0].3.as_str()), ("working", "3 min ago"));
        assert_eq!((rows[1].0.as_str(), rows[1].2.as_str(), rows[1].3.as_str()), ("Coupons", "done", "1 h ago"));
        assert!(rows[2].0.starts_with("+ new in"));
        key(&mut app, Key::Enter);
        assert_eq!(app.attached_session(), Some("bb"), "Enter opens the row shown first");
    }

    #[test]
    fn esc_esc_opens_the_undo_list() {
        let mut app = attached();
        app.clock(1_000);
        assert!(key(&mut app, Key::Esc).is_empty());
        app.clock(1_300);
        let effects = key(&mut app, Key::Esc);
        assert!(
            matches!(&effects[..], [Effect::Send(PageMessage::Session { message, .. })] if matches!(**message, PageMessage::ChatCheckpoints { .. }))
        );
    }

    #[test]
    fn a_single_esc_while_idle_does_nothing() {
        let mut app = attached();
        app.clock(1_000);
        assert!(key(&mut app, Key::Esc).is_empty());
        app.clock(2_000);
        assert!(key(&mut app, Key::Esc).is_empty(), "too far apart: two single presses");
    }

    #[test]
    fn picking_a_checkpoint_undoes_and_gives_the_text_back() {
        let mut app = attached();
        app.clock(10_000_000);
        typed(&mut app, "/undo");
        let effects = key(&mut app, Key::Enter);
        let Some(Effect::Send(PageMessage::Session { message, .. })) = effects.first() else { panic!("{effects:?}") };
        let PageMessage::ChatCheckpoints { id } = **message else { panic!("{message:?}") };
        app.update(in_session(DockMessage::Checkpoints {
            id,
            list: json!([{"entry": "e1", "text": "fix the coupon", "checkpoint": "c9", "files": 3, "added": 4, "removed": 0}]),
            off: None,
        }));
        key(&mut app, Key::Up);
        key(&mut app, Key::Enter);
        let effects = key(&mut app, Key::Enter);
        let Some(Effect::Send(PageMessage::Session { message, .. })) = effects.first() else { panic!("{effects:?}") };
        let PageMessage::ChatUndo { id, ref entry, ref checkpoint, files, conversation } = **message else {
            panic!("{message:?}")
        };
        assert_eq!((entry.as_str(), checkpoint.as_deref(), files, conversation), ("e1", Some("c9"), true, true));
        app.update(in_session(DockMessage::Saved { id }));
        assert_eq!(app.editor.text(), "fix the coupon");
    }

    #[test]
    fn the_changes_of_an_instruction_show_after_it() {
        let mut app = attached();
        app.update(in_session(DockMessage::ChatEvent {
            event: json!({"type": "changes", "files": 3, "added": 40, "removed": 12}),
        }));
        assert!(matches!(
            app.session().entries.last(),
            Some(crate::term::state::Entry::Changes { files: 3, added: 40, removed: 12 })
        ));
    }

    #[test]
    fn ctrl_t_folds_the_strip() {
        let mut app = attached();
        assert!(!app.tasks_folded());
        ctrl(&mut app, 't');
        assert!(app.tasks_folded());
        ctrl(&mut app, 't');
        assert!(!app.tasks_folded());
    }

    #[test]
    fn slash_context_asks_pi_and_opens_the_panel() {
        let mut app = attached();
        typed(&mut app, "/context");
        let effects = key(&mut app, Key::Enter);
        let Some(Effect::Send(PageMessage::Session { message, .. })) = effects.first() else { panic!("{effects:?}") };
        let PageMessage::PiCommand { id, ref command, .. } = **message else { panic!("{message:?}") };
        assert_eq!(command, "get_session_stats");
        app.update(in_session(DockMessage::PiReply {
            id,
            ok: true,
            data: json!({"contextUsage": {"tokens": 20000, "contextWindow": 200000, "percent": 10}}),
            error: None,
        }));
        let palette = crate::term::theme::Palette::truecolor();
        assert!(app.panel_lines(&palette, 80).is_some());
        key(&mut app, Key::Esc);
        assert!(app.panel_lines(&palette, 80).is_none(), "Esc closes it");
    }

    #[test]
    fn esc_esc_opens_rewind_in_the_editors_place_and_leaves_the_chat_alone() {
        let mut app = attached();
        key(&mut app, Key::Esc);
        let effects = key(&mut app, Key::Esc);
        let Some(Effect::Send(PageMessage::Session { message, .. })) = effects.first() else { panic!("{effects:?}") };
        let PageMessage::ChatCheckpoints { id } = **message else { panic!("{message:?}") };
        let before = app.session().entries.len();
        app.update(in_session(DockMessage::Checkpoints { id, list: json!([]), off: Some("projects".into()) }));
        assert_eq!(app.session().entries.len(), before, "nothing is written in the chat");
        let palette = crate::term::theme::Palette::truecolor();
        assert!(app.panel_lines(&palette, 80).is_some());
        key(&mut app, Key::Esc);
        assert!(app.panel_lines(&palette, 80).is_none(), "Esc closes it");
    }

    #[test]
    fn an_undo_that_could_not_be_done_says_so() {
        let mut app = attached();
        typed(&mut app, "/undo");
        let effects = key(&mut app, Key::Enter);
        let Some(Effect::Send(PageMessage::Session { message, .. })) = effects.first() else { panic!("{effects:?}") };
        let PageMessage::ChatCheckpoints { id } = **message else { panic!("{message:?}") };
        app.update(in_session(DockMessage::Checkpoints {
            id,
            list: json!([{"entry": "e1", "text": "x", "checkpoint": "c9", "files": 1, "added": 1, "removed": 0}]),
            off: None,
        }));
        key(&mut app, Key::Up);
        key(&mut app, Key::Enter);
        let effects = key(&mut app, Key::Enter);
        let Some(Effect::Send(PageMessage::Session { message, .. })) = effects.first() else { panic!("{effects:?}") };
        let PageMessage::ChatUndo { id, .. } = **message else { panic!("{message:?}") };
        app.update(in_session(DockMessage::Error { id, code: "busy".into() }));
        assert!(
            matches!(app.session().entries.last(), Some(crate::term::state::Entry::Notice { .. })),
            "{:?}",
            app.session().entries
        );
    }

    #[test]
    fn the_dock_going_away_quits_with_a_note() {
        let mut app = attached();
        let effects = app.update(Input::Dock(DockMessage::Error { id: 0, code: "gone".into() }));
        assert_eq!(effects, vec![Effect::Reconnect], "first, try again: the dock may only have dropped this client");
        assert_eq!(app.update(Input::Lost), vec![Effect::Quit]);
        assert!(app.note().unwrap().contains("dock stopped"));
    }

    #[test]
    fn a_dropped_client_reconnects_and_reattaches() {
        let mut app = attached();
        app.update(Input::Dock(DockMessage::Error { id: 0, code: "gone".into() }));
        let effects = app.update(Input::Reconnected);
        assert_eq!(effects[0], Effect::Attach("aa".into()));
        assert!(
            matches!(&effects[1], Effect::Send(PageMessage::Session { message, .. }) if **message == PageMessage::ChatLoad { fresh: true })
        );
        assert!(
            app.session().entries.iter().any(|e| matches!(e, crate::term::state::Entry::Note { .. })),
            "it says so"
        );
        let effects = app.update(Input::Dock(DockMessage::Error { id: 0, code: "gone".into() }));
        assert_eq!(effects, vec![Effect::Reconnect], "every loss tries once more");
    }

    #[test]
    fn over_connect_a_new_session_has_no_folder() {
        let (mut app, effects) = App::new(Config { remote: Some("hub.example".into()), ..config() });
        let Some(Effect::Send(PageMessage::SessionsLoad { id })) = effects.first().cloned() else {
            panic!("{effects:?}")
        };
        app.update(Input::Dock(DockMessage::Sessions { id, sessions: vec![], active: None }));
        let rows = app.list_rows();
        assert_eq!(rows.last().unwrap().0, "+ new on hub.example");
        let effects = key(&mut app, Key::Enter);
        assert!(
            effects.iter().any(|e| matches!(e, Effect::Send(PageMessage::SessionNew { workdir: None, .. }))),
            "{effects:?}"
        );
    }

    fn scheduled(id: &str, title: &str, enabled: bool) -> serde_json::Value {
        json!({
            "id": id, "title": title, "prompt": "p", "enabled": enabled, "awaiting": false, "created": 1,
            "when": {"at": "07:30", "days": ["mon", "tue", "wed", "thu", "fri"]},
            "grants": {}, "lastRun": null, "lastResult": null, "next": null
        })
    }

    fn panel_text(app: &App) -> String {
        let palette = crate::term::theme::Palette::truecolor();
        let lines = app.panel_lines(&palette, 80).unwrap_or_default();
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// An attached app whose `/schedules` panel is open on `tasks`.
    fn with_schedules(tasks: serde_json::Value) -> App {
        let mut app = attached();
        typed(&mut app, "/schedules");
        let effects = key(&mut app, Key::Enter);
        let [Effect::Send(PageMessage::SchedulesList { id })] = effects[..] else { panic!("{effects:?}") };
        app.update(Input::Dock(DockMessage::Schedules { id, tasks }));
        app
    }

    #[test]
    fn schedules_asks_the_dock_for_its_list() {
        let mut app = attached();
        typed(&mut app, "/schedules");
        let effects = key(&mut app, Key::Enter);
        assert!(
            matches!(&effects[..], [Effect::Send(PageMessage::SchedulesList { .. })]),
            "a request of its own, not a message to the session: {effects:?}"
        );
        assert_eq!(app.editor.text(), "");
        assert!(app.panel_lines(&crate::term::theme::Palette::truecolor(), 80).is_none(), "it opens with the reply");
    }

    #[test]
    fn a_list_reply_opens_the_panel_with_a_row_per_task() {
        let app =
            with_schedules(json!([scheduled("s-1", "Notion tasks", true), scheduled("s-2", "Backup notes", false)]));
        let shown = panel_text(&app);
        assert!(shown.contains("Notion tasks") && shown.contains("Backup notes"), "{shown}");
        assert!(shown.contains("Weekdays at 07:30"), "{shown}");
        assert_eq!(shown.matches("paused").count(), 1, "only the paused one says so: {shown}");
    }

    #[test]
    fn enter_pauses_a_running_task_without_asking_the_dock_again() {
        let mut app =
            with_schedules(json!([scheduled("s-1", "Notion tasks", true), scheduled("s-2", "Backup notes", false)]));
        let effects = key(&mut app, Key::Enter);
        let [Effect::Send(PageMessage::ScheduleUpdate { id, schedule, enabled })] = &effects[..] else {
            panic!("{effects:?}")
        };
        assert_eq!((schedule.as_str(), *enabled), ("s-1", false));
        let reply = json!([scheduled("s-1", "Notion tasks", false), scheduled("s-2", "Backup notes", false)]);
        app.update(Input::Dock(DockMessage::Schedules { id: *id, tasks: reply }));
        assert_eq!(panel_text(&app).matches("paused").count(), 2, "the panel redraws from the reply");
    }

    /// A panel open on a running task `s-1` and a paused one `s-2`, with the highlight on `s-2`.
    fn on_a_paused_task() -> App {
        let mut app =
            with_schedules(json!([scheduled("s-1", "Notion tasks", true), scheduled("s-2", "Backup notes", false)]));
        key(&mut app, Key::Down);
        app
    }

    fn the_fresh_list_id(effects: &[Effect]) -> u64 {
        let [Effect::Send(PageMessage::SchedulesList { id })] = effects else { panic!("{effects:?}") };
        *id
    }

    #[test]
    fn enter_on_a_paused_task_asks_for_a_fresh_list_and_resumes_only_if_it_is_still_paused() {
        let mut app = on_a_paused_task();
        let effects = key(&mut app, Key::Enter);
        let id = the_fresh_list_id(&effects);
        let effects = app.update(Input::Dock(DockMessage::Schedules {
            id,
            tasks: json!([scheduled("s-1", "Notion tasks", true), scheduled("s-2", "Backup notes", false)]),
        }));
        let [Effect::Send(PageMessage::ScheduleUpdate { id, schedule, enabled })] = &effects[..] else {
            panic!("{effects:?}")
        };
        assert_eq!((schedule.as_str(), *enabled), ("s-2", true));
        app.update(Input::Dock(DockMessage::Schedules {
            id: *id,
            tasks: json!([scheduled("s-1", "Notion tasks", true), scheduled("s-2", "Backup notes", true)]),
        }));
        assert_eq!(panel_text(&app).matches("paused").count(), 0, "the panel redraws from the reply");
    }

    #[test]
    fn a_fresh_list_that_shows_the_task_awaiting_or_gone_sends_nothing() {
        let mut waiting = scheduled("s-2", "Backup notes", false);
        waiting["awaiting"] = json!(true);
        for tasks in [
            json!([scheduled("s-1", "Notion tasks", true), waiting]),
            json!([scheduled("s-1", "Notion tasks", true)]),
            json!([scheduled("s-1", "Notion tasks", true), scheduled("s-2", "Backup notes", true)]),
            json!("nope"),
        ] {
            let mut app = on_a_paused_task();
            let id = the_fresh_list_id(&key(&mut app, Key::Enter));
            let effects = app.update(Input::Dock(DockMessage::Schedules { id, tasks }));
            assert!(effects.is_empty(), "{effects:?}");
            assert!(app.panel_lines(&crate::term::theme::Palette::truecolor(), 80).is_some(), "the panel stays");
        }
        let mut app = on_a_paused_task();
        let id = the_fresh_list_id(&key(&mut app, Key::Enter));
        let mut waiting = scheduled("s-2", "Backup notes", false);
        waiting["awaiting"] = json!(true);
        app.update(Input::Dock(DockMessage::Schedules { id, tasks: json!([waiting]) }));
        assert!(panel_text(&app).contains("waiting for approval in the dock"), "it redraws from the fresh list");
        assert!(key(&mut app, Key::Enter).is_empty());
    }

    #[test]
    fn a_resume_check_whose_panel_was_closed_sends_nothing() {
        let mut app = on_a_paused_task();
        let id = the_fresh_list_id(&key(&mut app, Key::Enter));
        key(&mut app, Key::Esc);
        let effects = app.update(Input::Dock(DockMessage::Schedules {
            id,
            tasks: json!([scheduled("s-2", "Backup notes", false)]),
        }));
        assert!(effects.is_empty(), "{effects:?}");
    }

    #[test]
    fn a_live_awaiting_message_stops_the_row_before_any_request() {
        let mut app = on_a_paused_task();
        let effects = app.update(Input::Dock(DockMessage::ScheduleAwaiting { schedule: "s-2".into() }));
        assert!(effects.is_empty());
        assert!(panel_text(&app).contains("waiting for approval in the dock"));
        assert!(key(&mut app, Key::Enter).is_empty(), "not even the refetch");
        // With no panel open the message is ignored.
        let mut closed = attached();
        assert!(closed.update(Input::Dock(DockMessage::ScheduleAwaiting { schedule: "s-2".into() })).is_empty());
        assert!(closed.panel_lines(&crate::term::theme::Palette::truecolor(), 80).is_none());
    }

    #[test]
    fn a_row_without_a_boolean_awaiting_cannot_be_resumed() {
        for awaiting in [None, Some(json!("false")), Some(json!(1)), Some(json!(null))] {
            let mut task = scheduled("s-2", "Backup notes", false);
            match awaiting {
                None => task.as_object_mut().unwrap().remove("awaiting"),
                Some(value) => task.as_object_mut().unwrap().insert("awaiting".into(), value),
            };
            let mut app = with_schedules(json!([task]));
            assert!(key(&mut app, Key::Enter).is_empty());
        }
    }

    #[test]
    fn a_late_reply_after_switching_session_opens_no_panel() {
        let mut app = attached();
        typed(&mut app, "/schedules");
        let id = the_fresh_list_id(&key(&mut app, Key::Enter));
        key(&mut app, Key::Esc);
        typed(&mut app, "/sessions");
        key(&mut app, Key::Enter);
        key(&mut app, Key::Down);
        key(&mut app, Key::Enter);
        assert_eq!(app.attached_session(), Some("bb"));
        app.update(Input::Dock(DockMessage::Schedules { id, tasks: json!([scheduled("s-1", "Notion tasks", true)]) }));
        assert!(app.panel_lines(&crate::term::theme::Palette::truecolor(), 80).is_none());
        // The same through the list: the reply of a resume check does not reach a panel opened later.
        let mut app = on_a_paused_task();
        let check = the_fresh_list_id(&key(&mut app, Key::Enter));
        app.update(Input::Dock(DockMessage::SessionRemoved { session: "aa".into() }));
        let effects = app.update(Input::Dock(DockMessage::Schedules { id: check, tasks: json!([]) }));
        assert!(effects.is_empty());
        assert!(app.panel_lines(&crate::term::theme::Palette::truecolor(), 80).is_none());
    }

    #[test]
    fn delete_needs_a_confirmation() {
        let mut app = with_schedules(json!([scheduled("s-1", "Notion tasks", true)]));
        assert!(key(&mut app, Key::Char('d')).is_empty(), "d alone sends nothing");
        assert!(panel_text(&app).contains("Delete"), "it asks first");
        let effects = key(&mut app, Key::Char('y'));
        let [Effect::Send(PageMessage::ScheduleDelete { id, schedule })] = &effects[..] else { panic!("{effects:?}") };
        assert_eq!(schedule, "s-1");
        app.update(Input::Dock(DockMessage::Schedules { id: *id, tasks: json!([]) }));
        assert!(panel_text(&app).contains("No scheduled tasks"));
        for no in [Key::Char('n'), Key::Esc, Key::Char('x')] {
            let mut app = with_schedules(json!([scheduled("s-1", "Notion tasks", true)]));
            key(&mut app, Key::Char('d'));
            assert!(key(&mut app, no).is_empty(), "{no:?} cancels");
            assert!(key(&mut app, Key::Char('y')).is_empty(), "the question is over");
            assert!(app.panel_lines(&crate::term::theme::Palette::truecolor(), 80).is_some(), "the panel stays");
        }
    }

    #[test]
    fn an_empty_list_says_so() {
        let mut app = with_schedules(json!([]));
        assert!(panel_text(&app).contains("No scheduled tasks"));
        assert!(key(&mut app, Key::Enter).is_empty());
        key(&mut app, Key::Esc);
        assert!(app.panel_lines(&crate::term::theme::Palette::truecolor(), 80).is_none(), "Esc closes it");
    }

    #[test]
    fn a_task_awaiting_approval_is_never_approved_from_the_terminal() {
        let mut waiting = scheduled("s-9", "Writes notes", false);
        waiting["awaiting"] = json!(true);
        let mut app = with_schedules(json!([waiting]));
        assert!(panel_text(&app).contains("waiting for approval in the dock"));
        assert!(key(&mut app, Key::Enter).is_empty(), "Enter does nothing on it");
    }

    #[test]
    fn a_reply_with_something_unexpected_does_not_break_the_panel() {
        for tasks in [json!("nope"), json!(null), json!([1, null, {"title": "no id"}, {"id": "s-1"}])] {
            let mut app = with_schedules(tasks);
            assert!(app.panel_lines(&crate::term::theme::Palette::truecolor(), 80).is_some());
            key(&mut app, Key::Down);
            key(&mut app, Key::Char('d'));
            key(&mut app, Key::Esc);
        }
    }

    #[test]
    fn a_reply_after_the_panel_was_closed_does_not_open_it_again() {
        let mut app = with_schedules(json!([scheduled("s-1", "Notion tasks", true)]));
        let effects = key(&mut app, Key::Enter);
        let [Effect::Send(PageMessage::ScheduleUpdate { id, .. })] = &effects[..] else { panic!("{effects:?}") };
        key(&mut app, Key::Esc);
        app.update(Input::Dock(DockMessage::Schedules {
            id: *id,
            tasks: json!([scheduled("s-1", "Notion tasks", false)]),
        }));
        assert!(app.panel_lines(&crate::term::theme::Palette::truecolor(), 80).is_none());
    }

    #[test]
    fn schedules_is_one_of_the_clients_own_commands() {
        let mut app = attached();
        typed(&mut app, "/");
        let id = app.pending.iter().find(|(_, p)| **p == Pending::Commands).map(|(id, _)| *id).unwrap();
        app.update(in_session(DockMessage::Commands { id, commands: json!([]) }));
        typed(&mut app, "sched");
        let effects = key(&mut app, Key::Enter);
        assert!(matches!(&effects[..], [Effect::Send(PageMessage::SchedulesList { .. })]), "{effects:?}");
    }

    #[test]
    fn a_request_the_dock_refuses_says_so_in_the_chat() {
        let mut app = attached();
        typed(&mut app, "/schedules");
        let effects = key(&mut app, Key::Enter);
        let [Effect::Send(PageMessage::SchedulesList { id })] = effects[..] else { panic!("{effects:?}") };
        app.update(Input::Dock(DockMessage::Error { id, code: "not-allowed".into() }));
        assert!(matches!(app.session().entries.last(), Some(crate::term::state::Entry::Notice { .. })));
    }

    // ---- /signin ---------------------------------------------------------------------------------------------

    const REDIRECT: &str = "http://localhost:53692/callback?code=SECRETCODE&state=st";

    fn last_notice(app: &App) -> Option<(crate::term::state::Tone, String, String)> {
        match app.session().entries.last() {
            Some(crate::term::state::Entry::Notice { tone, text: crate::term::state::Text::Key { key, vars } }) => {
                Some((*tone, key.clone(), vars.to_string()))
            }
            _ => None,
        }
    }

    /// An app whose `/signin <service>` has been answered with `url`: the panel is open. Returns it with the id of the
    /// request that started the sign-in.
    fn signing_in(service: &str, url: &str) -> (App, u64) {
        let mut app = attached();
        typed(&mut app, &format!("/signin {service}"));
        let effects = key(&mut app, Key::Enter);
        let id = match &effects[..] {
            [Effect::Send(PageMessage::StartOAuth { id })] if service == "anthropic" => *id,
            [Effect::Send(PageMessage::StartMcpOAuth { id, service: s })] if s == service => *id,
            other => panic!("{other:?}"),
        };
        app.update(Input::Dock(DockMessage::OAuthUrl { id, url: url.into() }));
        (app, id)
    }

    #[test]
    fn signin_shows_the_url_then_sends_the_pasted_address() {
        let url = "https://auth.example/authorize?client_id=abc&redirect_uri=http%3A%2F%2F127.0.0.1%3A5%2Fcallback&state=st&code_challenge=0123456789abcdef";
        let mut app = attached();
        typed(&mut app, "/signin notion");
        let effects = key(&mut app, Key::Enter);
        let [Effect::Send(PageMessage::StartMcpOAuth { id, service })] = &effects[..] else { panic!("{effects:?}") };
        assert_eq!(service, "notion");
        let id = *id;
        assert_eq!(app.editor.text(), "");
        assert!(app.panel_lines(&crate::term::theme::Palette::truecolor(), 80).is_none(), "it opens with the URL");

        app.update(Input::Dock(DockMessage::OAuthUrl { id, url: url.into() }));
        let shown = panel_text(&app);
        // wrapped to the screen, every character of the address still there in order
        assert!(shown.replace('\n', "").contains(url), "{shown}");
        assert!(shown.lines().all(|l| l.chars().count() <= 80), "{shown}");
        assert!(shown.to_lowercase().contains("browser"), "{shown}");

        let entries = app.session().entries.len();
        let effects = app.update(Input::Paste(format!("{REDIRECT}\n")));
        assert!(effects.is_empty(), "a paste does not send by itself: {effects:?}");
        assert_eq!(app.editor.text(), "", "the editor is not where it goes");
        assert_eq!(app.session().entries.len(), entries, "and the conversation never hears of it");
        assert!(!panel_text(&app).contains("SECRETCODE"), "it is not echoed back");

        let effects = key(&mut app, Key::Enter);
        let [Effect::Send(PageMessage::OAuthCode { id: code_id, service, redirect })] = &effects[..] else {
            panic!("{effects:?}")
        };
        assert_ne!(*code_id, id);
        assert_eq!((service.as_str(), redirect.as_str()), ("notion", REDIRECT));

        app.update(Input::Dock(DockMessage::OAuthResult { id, ok: true, error: None }));
        assert!(app.panel_lines(&crate::term::theme::Palette::truecolor(), 80).is_none(), "the sign-in is over");
        let (tone, key, _) = last_notice(&app).expect("a notice");
        assert_eq!((tone, key.as_str()), (crate::term::state::Tone::Info, "term.signin.done"));
        assert!(t("en", "term.signin.done", &json!({})).to_lowercase().contains("signed in"));
    }

    #[test]
    fn signin_anthropic_starts_claudes_sign_in_and_takes_typed_text_too() {
        let (mut app, id) = signing_in("anthropic", "https://claude.example/oauth");
        typed(&mut app, "code=a&state=b");
        key(&mut app, Key::Backspace);
        typed(&mut app, "c");
        let effects = key(&mut app, Key::Enter);
        let [Effect::Send(PageMessage::OAuthCode { service, redirect, .. })] = &effects[..] else {
            panic!("{effects:?}")
        };
        assert_eq!((service.as_str(), redirect.as_str()), ("anthropic", "code=a&state=c"));
        assert_eq!(app.editor.text(), "");
        let _ = id;
    }

    #[test]
    fn signin_without_a_known_service_says_how_to_use_it() {
        for line in ["/signin", "/signin figma", "/signin anthropic notion"] {
            let mut app = attached();
            typed(&mut app, line);
            let effects = key(&mut app, Key::Enter);
            assert!(effects.is_empty(), "{line}: {effects:?}");
            let (_, key, _) = last_notice(&app).expect("the usage");
            assert_eq!(key, "term.signin.usage", "{line}");
            assert!(t("en", "term.signin.usage", &json!({})).contains("anthropic"));
            assert!(t("es", "term.signin.usage", &json!({})).contains("notion"));
        }
    }

    #[test]
    fn a_refused_address_is_said_and_the_panel_keeps_waiting() {
        let (mut app, _) = signing_in("notion", "https://auth.example/a");
        app.update(Input::Paste("not an address".into()));
        let effects = key(&mut app, Key::Enter);
        let [Effect::Send(PageMessage::OAuthCode { id, .. })] = &effects[..] else { panic!("{effects:?}") };
        let before = panel_text(&app);
        assert!(!before.contains("not an address"), "{before}");
        app.update(Input::Dock(DockMessage::Error { id: *id, code: "bad-redirect".into() }));
        let shown = panel_text(&app).split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(shown.contains(&t("en", "term.signin.refused", &json!({}))), "{shown}");
        assert!(last_notice(&app).is_none(), "it is the panel that says it");
        // the person pastes it again
        app.update(Input::Paste(REDIRECT.into()));
        let effects = key(&mut app, Key::Enter);
        assert!(
            matches!(&effects[..], [Effect::Send(PageMessage::OAuthCode { redirect, .. })] if redirect == REDIRECT)
        );
    }

    #[test]
    fn an_empty_address_is_not_sent() {
        let (mut app, _) = signing_in("notion", "https://auth.example/a");
        assert!(key(&mut app, Key::Enter).is_empty());
    }

    #[test]
    fn a_sign_in_nobody_waits_for_says_so_and_closes() {
        let (mut app, _) = signing_in("notion", "https://auth.example/a");
        app.update(Input::Paste(REDIRECT.into()));
        let effects = key(&mut app, Key::Enter);
        let [Effect::Send(PageMessage::OAuthCode { id, .. })] = &effects[..] else { panic!("{effects:?}") };
        app.update(Input::Dock(DockMessage::Error { id: *id, code: "no-sign-in".into() }));
        assert!(app.panel_lines(&crate::term::theme::Palette::truecolor(), 80).is_none());
        let (tone, key, _) = last_notice(&app).expect("a notice");
        assert_eq!((tone, key.as_str()), (crate::term::state::Tone::Warn, "term.signin.nothing"));
    }

    #[test]
    fn a_failed_sign_in_says_why_in_the_dock_s_words() {
        let (mut app, id) = signing_in("anthropic", "https://claude.example/oauth");
        app.update(Input::Dock(DockMessage::OAuthResult { id, ok: false, error: Some("oauth-timeout".into()) }));
        assert!(app.panel_lines(&crate::term::theme::Palette::truecolor(), 80).is_none());
        let (tone, key, vars) = last_notice(&app).expect("a notice");
        assert_eq!((tone, key.as_str()), (crate::term::state::Tone::Warn, "term.signin.failed"));
        assert!(vars.contains(&t("en", "ob.error.oauth-timeout", &json!({}))), "{vars}");
        assert!(!vars.contains("ob.error"), "the key was found: {vars}");
    }

    #[test]
    fn a_sign_in_the_dock_cannot_start_says_so_without_a_panel() {
        let mut app = attached();
        typed(&mut app, "/signin notion");
        let effects = key(&mut app, Key::Enter);
        let [Effect::Send(PageMessage::StartMcpOAuth { id, .. })] = &effects[..] else { panic!("{effects:?}") };
        app.update(Input::Dock(DockMessage::OAuthResult {
            id: *id,
            ok: false,
            error: Some("network-unavailable".into()),
        }));
        assert!(app.panel_lines(&crate::term::theme::Palette::truecolor(), 80).is_none());
        assert_eq!(last_notice(&app).map(|n| n.1).as_deref(), Some("term.signin.failed"));
    }

    #[test]
    fn esc_closes_the_panel_and_a_late_result_is_ignored() {
        let (mut app, id) = signing_in("notion", "https://auth.example/a");
        app.update(Input::Paste(REDIRECT.into()));
        assert!(key(&mut app, Key::Esc).is_empty());
        assert!(app.panel_lines(&crate::term::theme::Palette::truecolor(), 80).is_none());
        let entries = app.session().entries.len();
        app.update(Input::Dock(DockMessage::OAuthResult { id, ok: true, error: None }));
        assert_eq!(app.session().entries.len(), entries);
        // and what was pasted is gone: the next panel starts empty
        let (mut again, _) = signing_in("notion", "https://auth.example/b");
        assert!(key(&mut again, Key::Enter).is_empty());
    }

    #[test]
    fn the_pasted_address_is_not_in_the_editor_history() {
        let (mut app, _) = signing_in("notion", "https://auth.example/a");
        app.update(Input::Paste(REDIRECT.into()));
        key(&mut app, Key::Enter);
        key(&mut app, Key::Esc);
        key(&mut app, Key::Up);
        assert!(!app.editor.text().contains("SECRETCODE"), "{}", app.editor.text());
    }

    #[test]
    fn signin_is_one_of_the_clients_own_commands() {
        assert!(OWN.iter().any(|(name, key)| *name == "signin" && !t("es", key, &json!({})).starts_with("term.")));
    }

    #[test]
    fn a_dropped_ssh_reconnects_and_attaches_again() {
        let mut app = attached();
        let effects = app.update(Input::Dock(DockMessage::Error { id: 0, code: "gone".into() }));
        assert_eq!(effects, vec![Effect::Reconnect]);
        let effects = app.update(Input::Reconnected);
        assert_eq!(effects[0], Effect::Attach("aa".into()));
    }
}
