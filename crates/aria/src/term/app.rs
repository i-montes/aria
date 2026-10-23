//! The client as a pure state machine: keys, clicks and the dock's messages go in, effects come out (what to send, to
//! attach, to save, to quit). `term::run` does the input and output.

use crate::term::dialog::{saved_answer, Answer, DialogBox};
use crate::term::editor::{bang, Editor, Trigger};
use crate::term::picker::{models_for_picker, Item, Outcome, Picker};
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
    /// The checkpoints to go back to.
    Undo,
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
    /// An undo, with the text that goes back to the editor once it is done.
    Undo(String),
}

/// The client's own commands, before the session's.
const OWN: [(&str, &str); 6] = [
    ("exit", "term.cmd.exit"),
    ("verbose", "term.cmd.verbose"),
    ("model", "term.cmd.model"),
    ("sessions", "term.cmd.sessions"),
    ("undo", "term.cmd.undo"),
    ("context", "term.cmd.context"),
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
    /// The undo list's messages, by checkpoint.
    undo_texts: HashMap<String, String>,
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
            undo_texts: HashMap::new(),
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
        rows.push((
            t(lang, "term.new", &json!({"folder": self.config.cwd})),
            String::new(),
            String::new(),
            String::new(),
        ));
        rows
    }

    pub fn selected(&self) -> usize {
        self.selected
    }

    fn open_selected(&mut self) -> Vec<Effect> {
        match self.ordered().get(self.selected).map(|s| s.id.clone()) {
            Some(id) => self.attach(id),
            None => {
                let workdir = Some(self.config.cwd.clone());
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
                    PickerKind::Undo => {
                        let text = self.undo_texts.get(&value).cloned().unwrap_or_default();
                        vec![self
                            .session_request(Pending::Undo(text), |id| PageMessage::ChatUndo { id, checkpoint: value })]
                    }
                }
            }
        }
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
                        let answer = self.dialog.as_mut().and_then(|dialog| dialog.click(usize::from(row - top)));
                        effects.extend(self.answered(answer));
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
                if self.pending.remove(&id) == Some(Pending::SessionNew) && code == "too-many-sessions" {
                    self.note = Some(t(&self.config.lang, "term.tooMany", &json!({})));
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
                self.undo_texts.clear();
                let lang = self.config.lang.clone();
                let items: Vec<Item> = list
                    .as_array()
                    .into_iter()
                    .flatten()
                    .take(10)
                    .filter_map(|c| {
                        let id = c["id"].as_str()?.to_string();
                        let text = c["text"].as_str().unwrap_or_default().to_string();
                        let files = c["files"].as_u64().unwrap_or(0);
                        let mut detail = self.ago(c["at"].as_i64().unwrap_or(0));
                        if files > 0 {
                            detail.push_str(&format!(" · {}", t(&lang, "term.files", &json!({ "n": files }))));
                        }
                        self.undo_texts.insert(id.clone(), text.clone());
                        let label: String = text.lines().next().unwrap_or_default().chars().take(60).collect();
                        Some(Item { label, detail, value: id, group: None })
                    })
                    .collect();
                if !items.is_empty() {
                    self.picker = Some((PickerKind::Undo, Picker::new(items)));
                } else {
                    // Say why there is nothing to pick, never nothing at all.
                    let key = match off.as_deref() {
                        Some("projects") => "term.undo.projects",
                        Some("system") => "term.undo.system",
                        Some("big") => "term.undo.big",
                        Some(_) => "term.undo.folder",
                        None => "term.undo.nothing",
                    };
                    self.session.entries.push(crate::term::state::Entry::Notice {
                        tone: crate::term::state::Tone::Info,
                        text: crate::term::state::Text::Key { key: key.into(), vars: json!({}) },
                    });
                }
            }
            DockMessage::Saved { id } if matches!(self.pending.get(&id), Some(Pending::Undo(_))) => {
                if let Some(Pending::Undo(text)) = self.pending.remove(&id) {
                    self.editor.set(&text);
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
            list: json!([{"id": "c9", "text": "fix the coupon", "at": 10_000_000 - 120_000, "files": 3}]),
            off: None,
        }));
        let effects = key(&mut app, Key::Enter);
        let Some(Effect::Send(PageMessage::Session { message, .. })) = effects.first() else { panic!("{effects:?}") };
        let PageMessage::ChatUndo { id, ref checkpoint } = **message else { panic!("{message:?}") };
        assert_eq!(checkpoint, "c9");
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

    /// The notice an Esc Esc answered by `list` and `off` leaves, as its text key.
    fn undo_notice(list: serde_json::Value, off: Option<&str>) -> Option<String> {
        let mut app = attached();
        key(&mut app, Key::Esc);
        let effects = key(&mut app, Key::Esc);
        let Some(Effect::Send(PageMessage::Session { message, .. })) = effects.first() else { panic!("{effects:?}") };
        let PageMessage::ChatCheckpoints { id } = **message else { panic!("{message:?}") };
        app.update(in_session(DockMessage::Checkpoints { id, list, off: off.map(String::from) }));
        match app.session().entries.last() {
            Some(crate::term::state::Entry::Notice { text: crate::term::state::Text::Key { key, .. }, .. }) => {
                Some(key.clone())
            }
            _ => None,
        }
    }

    #[test]
    fn esc_esc_with_nothing_to_undo_says_so() {
        assert_eq!(undo_notice(json!([]), None).as_deref(), Some("term.undo.nothing"));
    }

    #[test]
    fn esc_esc_in_a_folder_without_checkpoints_says_why() {
        assert_eq!(undo_notice(json!([]), Some("projects")).as_deref(), Some("term.undo.projects"));
        assert_eq!(undo_notice(json!([]), Some("folder")).as_deref(), Some("term.undo.folder"));
        assert_eq!(undo_notice(json!([]), Some("system")).as_deref(), Some("term.undo.system"));
        assert_eq!(undo_notice(json!([]), Some("big")).as_deref(), Some("term.undo.big"));
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
            list: json!([{"id": "c9", "text": "x", "at": 0, "files": 1}]),
            off: None,
        }));
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
}
