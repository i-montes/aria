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
    /// The hint about selecting text with the mouse was shown once already.
    pub hint_seen: bool,
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
    /// The mouse hint was shown: remember it.
    SaveHint,
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
}

/// The client's own commands, before the session's.
const OWN: [(&str, &str); 4] = [
    ("exit", "term.cmd.exit"),
    ("verbose", "term.cmd.verbose"),
    ("model", "term.cmd.model"),
    ("sessions", "term.cmd.sessions"),
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

    /// The list's rows: each session's title (or "untitled") and folder, then the new one.
    pub fn list_rows(&self) -> Vec<(String, String)> {
        let untitled = t(&self.config.lang, "term.untitled", &json!({}));
        let mut rows: Vec<(String, String)> = self
            .sessions
            .iter()
            .map(|s| {
                let title = if s.title.trim().is_empty() { untitled.clone() } else { s.title.clone() };
                (title, s.workdir.as_deref().map(folder_name).unwrap_or_default())
            })
            .collect();
        rows.push((t(&self.config.lang, "term.new", &json!({"folder": self.config.cwd})), String::new()));
        rows
    }

    pub fn selected(&self) -> usize {
        self.selected
    }

    fn open_selected(&mut self) -> Vec<Effect> {
        match self.sessions.get(self.selected) {
            Some(entry) => {
                let id = entry.id.clone();
                self.attach(id)
            }
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
                _ => Vec::new(),
            };
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
                    return self.picked(kind, outcome);
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

    fn picked(&mut self, kind: PickerKind, outcome: Outcome) -> Vec<Effect> {
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
            Some("followUp".to_string())
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

    /// The rows the bottom panel (a dialog or a picker) starts at, as `view` draws it: the footer, a rule, the panel.
    fn panel_top(&self, panel_len: usize) -> u16 {
        self.height.saturating_sub(2 + panel_len as u16)
    }

    pub fn panel_lines(
        &self,
        palette: &crate::term::theme::Palette,
        width: u16,
    ) -> Option<Vec<ratatui::text::Line<'static>>> {
        if let Some(dialog) = &self.dialog {
            return Some(dialog.lines(&self.config.lang, palette, width));
        }
        self.picker.as_ref().map(|(_, picker)| picker.lines(palette))
    }

    fn click(&mut self, row: u16) -> Vec<Effect> {
        let mut effects = Vec::new();
        if !self.config.hint_seen {
            self.config.hint_seen = true;
            let hint = t(&self.config.lang, "term.selectHint", &json!({}));
            if self.screen == Screen::Session {
                self.session
                    .entries
                    .push(crate::term::state::Entry::Note { text: crate::term::state::Text::Raw(hint) });
            } else {
                self.note = Some(hint);
            }
            effects.push(Effect::SaveHint);
        }
        match self.screen {
            Screen::List => {
                // The list starts on the second row.
                let at = usize::from(row.saturating_sub(1));
                if at <= self.sessions.len() {
                    self.selected = at;
                    effects.extend(self.open_selected());
                }
            }
            Screen::Session => {
                let palette = crate::term::theme::Palette::truecolor();
                let Some(lines) = self.panel_lines(&palette, 80) else { return effects };
                let top = self.panel_top(lines.len());
                if row < top {
                    return effects;
                }
                let at = usize::from(row - top);
                if let Some(dialog) = self.dialog.as_mut() {
                    let answer = dialog.click(at);
                    effects.extend(self.answered(answer));
                } else if let Some((kind, picker)) = self.picker.as_ref() {
                    let (kind, outcome) = (*kind, picker.click(at));
                    effects.extend(self.picked(kind, outcome));
                }
            }
        }
        effects
    }

    // ---- the dock ------------------------------------------------------------------------------------------------

    fn dock(&mut self, message: DockMessage) -> Vec<Effect> {
        match message {
            DockMessage::Error { id: 0, code } if code == "gone" => {
                self.note = Some(t(&self.config.lang, "term.gone", &json!({})));
                vec![Effect::Quit]
            }
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
        Vec::new()
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
            hint_seen: true,
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
        assert!(matches!(&effects[0], Effect::Send(PageMessage::Session { message, .. })
            if matches!(&**message, PageMessage::ChatSend { behavior: Some(b), .. } if b == "followUp")));
    }

    #[test]
    fn a_session_closed_elsewhere_returns_to_the_list() {
        let mut app = attached();
        let effects = app.update(Input::Dock(DockMessage::SessionRemoved { session: "aa".into() }));
        assert_eq!(app.screen(), Screen::List);
        assert!(effects.contains(&Effect::Detach));
        assert!(effects.iter().any(|e| matches!(e, Effect::Send(PageMessage::SessionsLoad { .. }))));
        assert_eq!(app.note().as_deref(), Some("This session was closed in the dock."));
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
    fn the_dock_going_away_quits_with_a_note() {
        let mut app = attached();
        let effects = app.update(Input::Dock(DockMessage::Error { id: 0, code: "gone".into() }));
        assert_eq!(effects, vec![Effect::Quit]);
        assert!(app.note().unwrap().contains("dock stopped"));
    }
}
