//! Rewind (Esc Esc, `/undo`), in the editor's place as Claude Code draws it: the person's messages, oldest first, each
//! with what its turn changed, then `(current)`; Enter on a message asks what goes back — code and conversation, the
//! conversation, the code — or nothing. The list is the dock's `checkpoints` reply; nothing here talks to the dock.

use crate::term::texts::t;
use crate::term::theme::Palette;
use crate::term::Key;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
struct Message {
    entry: String,
    text: String,
    checkpoint: Option<String>,
    files: u64,
    added: u64,
    removed: u64,
}

/// What the person chose.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    None,
    Close,
    /// Go back to before the message `entry`: its files (to `checkpoint`), its conversation, or both; `text` is the
    /// message, for the editor once the conversation goes back.
    Undo {
        entry: String,
        checkpoint: Option<String>,
        files: bool,
        conversation: bool,
        text: String,
    },
}

/// The choices for a message, in order.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Choice {
    Both,
    Conversation,
    Code,
    Never,
}

#[derive(Debug, Clone)]
pub struct Rewind {
    messages: Vec<Message>,
    /// Why the folder keeps no code (the dock's `off`).
    off: Option<String>,
    /// The highlighted row: a message, or `messages.len()` for `(current)`.
    selected: usize,
    /// The message whose choices are shown, and the highlighted choice.
    choosing: Option<(usize, usize)>,
}

/// What a row of the drawn panel does when clicked.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Target {
    Row(usize),
    Choice(usize),
}

impl Rewind {
    /// From the dock's list (`[{ entry, text, checkpoint, files, added, removed }]`, oldest first) and its `off`.
    pub fn new(list: &Value, off: Option<String>) -> Rewind {
        let messages: Vec<Message> = list
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|m| {
                Some(Message {
                    entry: m["entry"].as_str()?.to_string(),
                    text: m["text"].as_str().unwrap_or_default().to_string(),
                    checkpoint: m["checkpoint"].as_str().map(str::to_string),
                    files: m["files"].as_u64().unwrap_or(0),
                    added: m["added"].as_u64().unwrap_or(0),
                    removed: m["removed"].as_u64().unwrap_or(0),
                })
            })
            .collect();
        let selected = messages.len();
        Rewind { messages, off, selected, choosing: None }
    }

    /// What changed in the code from message `at` on: (files, added, removed).
    fn changed_from(&self, at: usize) -> (u64, u64, u64) {
        self.messages[at..].iter().fold((0, 0, 0), |(f, a, r), m| (f + m.files, a + m.added, r + m.removed))
    }

    /// The choices for message `at`: the code's only when its checkpoint is kept and code changed from there on.
    fn choices(&self, at: usize) -> Vec<Choice> {
        if self.messages[at].checkpoint.is_some() && self.changed_from(at).0 > 0 {
            vec![Choice::Both, Choice::Conversation, Choice::Code, Choice::Never]
        } else {
            vec![Choice::Conversation, Choice::Never]
        }
    }

    fn pick(&self, at: usize, choice: Choice) -> Outcome {
        let message = &self.messages[at];
        let (files, conversation) = match choice {
            Choice::Both => (true, true),
            Choice::Conversation => (false, true),
            Choice::Code => (true, false),
            Choice::Never => return Outcome::Close,
        };
        Outcome::Undo {
            entry: message.entry.clone(),
            checkpoint: message.checkpoint.clone(),
            files,
            conversation,
            text: message.text.clone(),
        }
    }

    /// Opens the choices of the highlighted row; `(current)` closes.
    fn open(&mut self, row: usize) -> Outcome {
        if row >= self.messages.len() {
            return Outcome::Close;
        }
        self.selected = row;
        self.choosing = Some((row, 0));
        Outcome::None
    }

    pub fn key(&mut self, key: Key) -> Outcome {
        if let Some((at, choice)) = self.choosing {
            let choices = self.choices(at);
            match key {
                Key::Up => self.choosing = Some((at, choice.saturating_sub(1))),
                Key::Down => self.choosing = Some((at, (choice + 1).min(choices.len() - 1))),
                Key::Enter => return self.pick(at, choices[choice]),
                Key::Char(c) => {
                    let n = c.to_digit(10).unwrap_or(0) as usize;
                    if (1..=choices.len()).contains(&n) {
                        return self.pick(at, choices[n - 1]);
                    }
                }
                Key::Esc => self.choosing = None,
                _ => {}
            }
            return Outcome::None;
        }
        match key {
            Key::Up => self.selected = self.selected.saturating_sub(1),
            Key::Down => self.selected = (self.selected + 1).min(self.messages.len()),
            Key::Enter => return self.open(self.selected),
            Key::Esc => return Outcome::Close,
            _ => {}
        }
        Outcome::None
    }

    /// A click on row `row` of the panel drawn for `rows` screen rows.
    pub fn click(&mut self, row: usize, rows: usize) -> Outcome {
        let (_, targets) = self.layout("en", &Palette::truecolor(), 80, rows);
        match targets.get(row).copied().flatten() {
            Some(Target::Row(at)) => self.open(at),
            Some(Target::Choice(k)) => match self.choosing {
                Some((at, _)) => {
                    let choice = self.choices(at)[k];
                    self.pick(at, choice)
                }
                None => Outcome::None,
            },
            None => Outcome::None,
        }
    }

    /// The panel, at most `rows` lines tall.
    pub fn lines(&self, lang: &str, p: &Palette, width: u16, rows: usize) -> Vec<Line<'static>> {
        self.layout(lang, p, width, rows).0
    }

    /// The panel's lines and what a click on each does.
    fn layout(&self, lang: &str, p: &Palette, width: u16, rows: usize) -> (Vec<Line<'static>>, Vec<Option<Target>>) {
        let cut = |s: &str| -> String {
            s.lines().next().unwrap_or_default().chars().take(usize::from(width).saturating_sub(6)).collect()
        };
        let say = |key: &str, vars: Value| t(lang, key, &vars);
        let mut out: Vec<(Line<'static>, Option<Target>)> = Vec::new();
        let blank = || (Line::default(), None);
        let faint = Style::default().fg(p.faint).add_modifier(Modifier::ITALIC);
        let marker = |on: bool| Span::styled(if on { "❯ " } else { "  " }, Style::default().fg(p.accent));
        let code_kept = self.off.is_none();
        out.push((
            Line::from(Span::styled(
                say("term.rewind.title", json!({})),
                Style::default().fg(p.accent).add_modifier(Modifier::BOLD),
            )),
            None,
        ));
        out.push(blank());
        let changes = |files: u64, added: u64, removed: u64| -> Vec<Span<'static>> {
            if files == 0 {
                vec![Span::styled(say("term.rewind.none", json!({})), Style::default().fg(p.grey))]
            } else {
                crate::term::view::changes_spans(lang, p, files, added, removed)
            }
        };
        if let Some((at, choice)) = self.choosing {
            let choices = self.choices(at);
            let lead =
                if choices.contains(&Choice::Code) { "term.rewind.lead" } else { "term.rewind.leadConversation" };
            out.push((Line::from(Span::styled(say(lead, json!({})), Style::default().fg(p.text))), None));
            out.push(blank());
            let message = &self.messages[at];
            out.push((
                Line::from(Span::styled(format!("  {}", cut(&message.text)), Style::default().fg(p.text))),
                None,
            ));
            if choices.contains(&Choice::Code) {
                let (files, added, removed) = self.changed_from(at);
                let mut spans = vec![Span::styled(
                    format!("  {} ", say("term.rewind.from", json!({}))),
                    Style::default().fg(p.grey),
                )];
                spans.extend(changes(files, added, removed));
                out.push((Line::from(spans), None));
            }
            out.push(blank());
            for (k, kind) in choices.iter().enumerate() {
                let key = match kind {
                    Choice::Both => "term.rewind.both",
                    Choice::Conversation => "term.rewind.conversation",
                    Choice::Code => "term.rewind.code",
                    Choice::Never => "term.rewind.never",
                };
                let on = k == choice;
                out.push((
                    Line::from(vec![
                        marker(on),
                        Span::styled(format!("{}. ", k + 1), Style::default().fg(p.grey)),
                        Span::styled(say(key, json!({})), Style::default().fg(if on { p.accent } else { p.text })),
                    ]),
                    Some(Target::Choice(k)),
                ));
            }
            out.push(blank());
            out.push((Line::from(Span::styled(say("term.rewind.keysBack", json!({})), faint)), None));
            return out.into_iter().unzip();
        }
        let lead = if code_kept { "term.rewind.lead" } else { "term.rewind.leadConversation" };
        out.push((Line::from(Span::styled(say(lead, json!({})), Style::default().fg(p.text))), None));
        if let Some(off) = &self.off {
            let key = match off.as_str() {
                "projects" => "term.rewind.off.projects",
                "system" => "term.rewind.off.system",
                "big" => "term.rewind.off.big",
                _ => "term.rewind.off.folder",
            };
            out.push((Line::from(Span::styled(say(key, json!({})), Style::default().fg(p.grey))), None));
        }
        out.push(blank());
        // Title, lead (and why), blanks, `(current)`, the keys, and room for the two `… more` rows.
        let fixed = out.len() + 3 + 2;
        let n = self.messages.len();
        let fit = (rows.saturating_sub(fixed) / 3).max(1);
        let start = if n <= fit { 0 } else { self.selected.min(n - fit) };
        let end = (start + fit).min(n);
        if start > 0 {
            out.push((
                Line::from(Span::styled(format!("  {}", say("term.rewind.above", json!({"n": start}))), faint)),
                None,
            ));
        }
        for (at, message) in self.messages.iter().enumerate().take(end).skip(start) {
            let on = at == self.selected;
            out.push((
                Line::from(vec![
                    marker(on),
                    Span::styled(cut(&message.text), Style::default().fg(if on { p.accent } else { p.text })),
                ]),
                Some(Target::Row(at)),
            ));
            let mut spans = vec![Span::raw("  ")];
            spans.extend(changes(message.files, message.added, message.removed));
            out.push((Line::from(spans), Some(Target::Row(at))));
            out.push(blank());
        }
        if end < n {
            out.push((
                Line::from(Span::styled(format!("  {}", say("term.rewind.below", json!({"n": n - end}))), faint)),
                None,
            ));
        }
        let on = self.selected == n;
        let current = Style::default().fg(if on { p.accent } else { p.grey }).add_modifier(Modifier::ITALIC);
        out.push((
            Line::from(vec![marker(on), Span::styled(say("term.rewind.current", json!({})), current)]),
            Some(Target::Row(n)),
        ));
        out.push(blank());
        out.push((Line::from(Span::styled(say("term.rewind.keys", json!({})), faint)), None));
        out.into_iter().unzip()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list() -> Value {
        json!([
            {"entry": "e1", "text": "fix the coupon", "checkpoint": "c1", "files": 2, "added": 10, "removed": 3},
            {"entry": "e2", "text": "and the tests", "checkpoint": "c2", "files": 0, "added": 0, "removed": 0},
            {"entry": "e3", "text": "explain it", "checkpoint": null, "files": 0, "added": 0, "removed": 0},
        ])
    }

    fn text_of(lines: &[Line<'static>]) -> String {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn it_opens_on_current_below_the_messages_oldest_first() {
        let r = Rewind::new(&list(), None);
        let drawn = text_of(&r.lines("en", &Palette::truecolor(), 100, 40));
        assert!(drawn.starts_with("Rewind"), "{drawn}");
        assert!(drawn.contains("Restore the code and/or conversation to the point before…"), "{drawn}");
        let (first, last) = (drawn.find("fix the coupon").unwrap(), drawn.find("explain it").unwrap());
        assert!(first < last, "oldest first: {drawn}");
        assert!(drawn.contains("2 files changed  +10 −3"), "{drawn}");
        assert!(drawn.contains("No code changes"), "{drawn}");
        assert!(drawn.contains("❯ (current)"), "{drawn}");
        assert!(drawn.ends_with("Enter to continue · Esc to cancel"), "{drawn}");
    }

    #[test]
    fn enter_on_current_or_esc_closes() {
        let mut r = Rewind::new(&list(), None);
        assert_eq!(r.key(Key::Enter), Outcome::Close);
        let mut r = Rewind::new(&list(), None);
        assert_eq!(r.key(Key::Esc), Outcome::Close);
    }

    #[test]
    fn a_message_with_code_changes_offers_code_and_conversation() {
        let mut r = Rewind::new(&list(), None);
        for _ in 0..3 {
            r.key(Key::Up);
        }
        assert_eq!(r.key(Key::Enter), Outcome::None, "first, what goes back");
        let drawn = text_of(&r.lines("en", &Palette::truecolor(), 100, 40));
        for choice in ["Restore code and conversation", "Restore conversation", "Restore code", "Never mind"] {
            assert!(drawn.contains(choice), "{drawn}");
        }
        assert_eq!(
            r.key(Key::Enter),
            Outcome::Undo {
                entry: "e1".into(),
                checkpoint: Some("c1".into()),
                files: true,
                conversation: true,
                text: "fix the coupon".into()
            }
        );
    }

    #[test]
    fn code_is_offered_only_where_code_changed_from_there_on() {
        let mut r = Rewind::new(&list(), None);
        r.key(Key::Up);
        r.key(Key::Enter);
        let drawn = text_of(&r.lines("en", &Palette::truecolor(), 100, 40));
        assert!(!drawn.contains("Restore code"), "the last message changed no code: {drawn}");
        assert_eq!(
            r.key(Key::Enter),
            Outcome::Undo {
                entry: "e3".into(),
                checkpoint: None,
                files: false,
                conversation: true,
                text: "explain it".into()
            }
        );
    }

    #[test]
    fn a_number_picks_a_choice_and_esc_goes_back_to_the_list() {
        let mut r = Rewind::new(&list(), None);
        for _ in 0..3 {
            r.key(Key::Up);
        }
        r.key(Key::Enter);
        assert_eq!(r.key(Key::Esc), Outcome::None, "back to the list");
        assert!(text_of(&r.lines("en", &Palette::truecolor(), 100, 40)).contains("(current)"));
        r.key(Key::Enter);
        match r.key(Key::Char('3')) {
            Outcome::Undo { files: true, conversation: false, .. } => {}
            other => panic!("{other:?}"),
        }
        let mut r = Rewind::new(&list(), None);
        for _ in 0..3 {
            r.key(Key::Up);
        }
        r.key(Key::Enter);
        assert_eq!(r.key(Key::Char('4')), Outcome::Close, "never mind");
    }

    #[test]
    fn a_folder_without_checkpoints_says_why_inside_the_panel() {
        let r = Rewind::new(&json!([]), Some("projects".into()));
        let drawn = text_of(&r.lines("en", &Palette::truecolor(), 100, 40));
        assert!(drawn.contains("Restore the conversation to the point before…"), "{drawn}");
        assert!(drawn.contains("Code is not kept in the projects folder."), "{drawn}");
        assert!(drawn.contains("❯ (current)"), "{drawn}");
    }

    #[test]
    fn a_long_list_shows_how_many_are_above() {
        let many: Vec<Value> = (0..30)
            .map(|n| json!({"entry": format!("e{n}"), "text": format!("message {n}"), "checkpoint": null}))
            .collect();
        let r = Rewind::new(&Value::Array(many), None);
        let lines = r.lines("en", &Palette::truecolor(), 100, 20);
        assert!(lines.len() <= 20, "{}", lines.len());
        let drawn = text_of(&lines);
        assert!(drawn.contains("↑ "), "{drawn}");
        assert!(drawn.contains("message 29") && !drawn.contains("message 0\n"), "{drawn}");
    }

    #[test]
    fn a_click_on_a_message_then_on_a_choice() {
        let mut r = Rewind::new(&list(), None);
        let lines = r.lines("en", &Palette::truecolor(), 100, 40);
        let row = lines.iter().position(|l| text_of(std::slice::from_ref(l)).contains("fix the coupon")).unwrap();
        assert_eq!(r.click(row, 40), Outcome::None);
        let lines = r.lines("en", &Palette::truecolor(), 100, 40);
        let row = lines.iter().position(|l| text_of(std::slice::from_ref(l)).contains("Restore conversation")).unwrap();
        match r.click(row, 40) {
            Outcome::Undo { files: false, conversation: true, .. } => {}
            other => panic!("{other:?}"),
        }
    }
}
