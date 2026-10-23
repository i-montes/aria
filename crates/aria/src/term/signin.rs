//! `/signin anthropic|notion`, in the editor's place: the address the person opens in a browser, and a line that takes
//! what that browser could not load. On a computer with no browser of its own (a terminal on another machine) the
//! sign-in's redirect to `localhost` fails to load, but its address bar holds the whole address, which has the
//! authorization code: it is pasted here and the dock takes the code from it.
//!
//! What is pasted or typed lives only in `input`, is never drawn (the panel says how many characters, not which), is
//! sent once and forgotten, and never goes through the editor, so neither the conversation nor the editor's history
//! hears of it.

use crate::term::texts::t;
use crate::term::theme::Palette;
use crate::term::Key;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use serde_json::json;

/// What the person did.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    None,
    Close,
    /// Send the dock what was pasted (an `oauthCode`).
    Send {
        service: String,
        redirect: String,
    },
}

pub struct SignIn {
    service: String,
    url: String,
    input: String,
    /// The dock did not accept the last address.
    refused: bool,
}

impl SignIn {
    pub fn new(service: &str, url: &str) -> SignIn {
        SignIn { service: service.to_string(), url: url.to_string(), input: String::new(), refused: false }
    }

    /// Text pasted into the terminal. An address has no whitespace, and a long one a terminal wrapped may have gained
    /// some, so none is kept.
    pub fn paste(&mut self, text: &str) {
        self.input.extend(text.chars().filter(|c| !c.is_whitespace()));
        self.refused = false;
    }

    /// The dock did not accept the address that was sent: the person pastes another.
    pub fn refuse(&mut self) {
        self.refused = true;
    }

    pub fn key(&mut self, key: Key) -> Outcome {
        match key {
            Key::Esc => return Outcome::Close,
            Key::Enter if !self.input.is_empty() => {
                return Outcome::Send { service: self.service.clone(), redirect: std::mem::take(&mut self.input) };
            }
            Key::Char(c) if !c.is_whitespace() => {
                self.input.push(c);
                self.refused = false;
            }
            Key::Backspace => {
                self.input.pop();
            }
            _ => {}
        }
        Outcome::None
    }

    pub fn lines(&self, lang: &str, p: &Palette, width: u16) -> Vec<Line<'static>> {
        let say = |key: &str, vars: serde_json::Value| t(lang, key, &vars);
        let width = usize::from(width).max(20);
        let faint = Style::default().fg(p.faint).add_modifier(Modifier::ITALIC);
        let name = if self.service == "anthropic" { "Claude" } else { "Notion" };
        let mut out = vec![
            Line::from(Span::styled(
                say("term.signin.title", json!({ "service": name })),
                Style::default().fg(p.accent).add_modifier(Modifier::BOLD),
            )),
            Line::default(),
        ];
        for line in wrap_words(&say("term.signin.steps", json!({})), width) {
            out.push(Line::from(Span::styled(line, Style::default().fg(p.grey))));
        }
        out.push(Line::default());
        // The address on rows of its own, with nothing beside it, cut only where the screen ends.
        let chars: Vec<char> = self.url.chars().collect();
        for row in chars.chunks(width) {
            out.push(Line::from(Span::styled(row.iter().collect::<String>(), Style::default().fg(p.text))));
        }
        out.push(Line::default());
        let typed = self.input.chars().count();
        out.push(Line::from(vec![
            Span::styled("❯ ", Style::default().fg(p.accent)),
            if typed == 0 {
                Span::styled(say("term.signin.empty", json!({})), faint)
            } else {
                Span::styled(say("term.signin.pasted", json!({ "n": typed })), Style::default().fg(p.text))
            },
        ]));
        if self.refused {
            for line in wrap_words(&say("term.signin.refused", json!({})), width) {
                out.push(Line::from(Span::styled(line, Style::default().fg(p.amber))));
            }
        }
        for line in wrap_words(&say("term.signin.keys", json!({})), width) {
            out.push(Line::from(Span::styled(line, faint)));
        }
        out
    }
}

/// `text` cut at spaces into lines of at most `width` characters (a longer word takes a line of its own).
fn wrap_words(text: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        match lines.last_mut() {
            Some(line) if line.chars().count() + 1 + word.chars().count() <= width => {
                line.push(' ');
                line.push_str(word);
            }
            _ => lines.push(word.to_string()),
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(lines: &[Line]) -> String {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn what_is_pasted_is_counted_never_shown() {
        let mut open = SignIn::new("notion", "https://auth.example/a");
        open.paste("http://localhost:1/callback?code=SECRETCODE\n&state=st\n");
        let shown = text(&open.lines("en", &Palette::truecolor(), 60));
        assert!(!shown.contains("SECRETCODE") && !shown.contains("callback"), "{shown}");
        assert!(shown.contains(&format!("{} characters", "http://localhost:1/callback?code=SECRETCODE&state=st".len())));
        assert_eq!(
            open.key(Key::Enter),
            Outcome::Send {
                service: "notion".into(),
                redirect: "http://localhost:1/callback?code=SECRETCODE&state=st".into()
            }
        );
        assert!(
            text(&open.lines("en", &Palette::truecolor(), 60)).contains("Paste the address"),
            "forgotten once sent"
        );
    }

    #[test]
    fn a_narrow_screen_still_shows_the_whole_address() {
        let url = "https://auth.example/authorize?state=0123456789012345678901234567890123456789";
        let open = SignIn::new("anthropic", url);
        let lines = open.lines("es", &Palette::truecolor(), 30);
        let shown = text(&lines);
        assert!(shown.lines().all(|l| l.chars().count() <= 30), "{shown}");
        assert!(shown.replace('\n', "").contains(url), "{shown}");
    }
}
