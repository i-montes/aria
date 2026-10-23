//! The editor: multi-line text with a cursor, the history of what was sent (↑ on an empty editor), and what the text at
//! the cursor asks for: `@…` a file, a leading `/…` a command. A line that starts with `!` runs as a command.

/// What the text at the cursor opens.
#[derive(Debug, Clone, PartialEq)]
pub enum Trigger {
    Files(String),
    Commands(String),
    None,
}

#[derive(Debug, Default)]
pub struct Editor {
    text: String,
    /// A byte index on a character boundary.
    cursor: usize,
    history: Vec<String>,
    /// Which history entry is shown (from the newest), while browsing it.
    browsing: Option<usize>,
}

/// `!command` → the command (none for an empty one or a `!` later in the line).
pub fn bang(text: &str) -> Option<String> {
    let command = text.trim_start().strip_prefix('!')?.trim();
    (!command.is_empty()).then(|| command.to_string())
}

impl Editor {
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn set(&mut self, text: &str) {
        self.text = text.to_string();
        self.cursor = self.text.len();
        self.browsing = None;
    }

    pub fn insert(&mut self, c: char) {
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
        self.browsing = None;
    }

    pub fn newline(&mut self) {
        self.insert('\n');
    }

    pub fn backspace(&mut self) {
        if let Some((at, _)) = self.text[..self.cursor].char_indices().next_back() {
            self.text.remove(at);
            self.cursor = at;
        }
    }

    pub fn left(&mut self) {
        if let Some((at, _)) = self.text[..self.cursor].char_indices().next_back() {
            self.cursor = at;
        }
    }

    pub fn right(&mut self) {
        if let Some(c) = self.text[self.cursor..].chars().next() {
            self.cursor += c.len_utf8();
        }
    }

    /// The text to send; the editor is empty again and the text joins the history.
    pub fn take(&mut self) -> String {
        let text = std::mem::take(&mut self.text);
        self.cursor = 0;
        self.browsing = None;
        if !text.trim().is_empty() && self.history.last() != Some(&text) {
            self.history.push(text.clone());
        }
        text
    }

    /// An older message, only from an empty editor or while already browsing (a draft is never replaced).
    pub fn history_prev(&mut self) {
        if self.history.is_empty() || (self.browsing.is_none() && !self.text.is_empty()) {
            return;
        }
        let next = self.browsing.map_or(0, |b| (b + 1).min(self.history.len() - 1));
        self.show(next);
    }

    pub fn history_next(&mut self) {
        match self.browsing {
            Some(0) => {
                self.text.clear();
                self.cursor = 0;
                self.browsing = None;
            }
            Some(b) => self.show(b - 1),
            None => {}
        }
    }

    fn show(&mut self, from_newest: usize) {
        self.text = self.history[self.history.len() - 1 - from_newest].clone();
        self.cursor = self.text.len();
        self.browsing = Some(from_newest);
    }

    /// The word at the cursor.
    fn word(&self) -> &str {
        let before = &self.text[..self.cursor];
        let start = before.rfind(char::is_whitespace).map_or(0, |i| i + 1);
        &before[start..]
    }

    pub fn trigger(&self) -> Trigger {
        if let Some(query) = self.word().strip_prefix('@') {
            return Trigger::Files(query.to_string());
        }
        if let Some(query) = self.text.strip_prefix('/') {
            if !query.contains(char::is_whitespace) {
                return Trigger::Commands(query.to_string());
            }
        }
        Trigger::None
    }

    /// The `@query` at the cursor becomes `@<path> `.
    pub fn complete(&mut self, path: &str) {
        let word = self.word().len();
        let start = self.cursor - word;
        let replacement = format!("@{path} ");
        self.text.replace_range(start..self.cursor, &replacement);
        self.cursor = start + replacement.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed(text: &str) -> Editor {
        let mut e = Editor::default();
        for c in text.chars() {
            e.insert(c);
        }
        e
    }

    #[test]
    fn history_walks_back_only_from_an_empty_editor() {
        let mut e = Editor::default();
        for sent in ["first", "second"] {
            for c in sent.chars() {
                e.insert(c);
            }
            assert_eq!(e.take(), sent);
        }
        e.history_prev();
        assert_eq!(e.text(), "second");
        e.history_prev();
        assert_eq!(e.text(), "first");
        e.history_next();
        assert_eq!(e.text(), "second");
        e.history_next();
        assert_eq!(e.text(), "", "past the newest: empty again");
        let mut busy = typed("half");
        busy.history_prev();
        assert_eq!(busy.text(), "half", "a draft is never replaced by the history");
    }

    #[test]
    fn at_opens_files_and_slash_opens_commands() {
        assert_eq!(typed("look at @src/co").trigger(), Trigger::Files("src/co".into()));
        assert_eq!(typed("/mo").trigger(), Trigger::Commands("mo".into()));
        assert_eq!(typed("/model x").trigger(), Trigger::None);
        assert_eq!(typed("a / b").trigger(), Trigger::None);
        let mut e = typed("look at @src/co");
        e.complete("src/coupon.ts");
        assert_eq!(e.text(), "look at @src/coupon.ts ");
    }

    #[test]
    fn bang_lines_become_chat_bash() {
        assert_eq!(bang("!git status"), Some("git status".into()));
        assert_eq!(bang("  !ls"), Some("ls".into()));
        assert_eq!(bang("!"), None);
        assert_eq!(bang("say !hi"), None);
    }

    #[test]
    fn editing_moves_by_characters() {
        let mut e = typed("añb");
        e.left();
        e.backspace();
        assert_eq!(e.text(), "ab");
        e.newline();
        assert_eq!(e.text(), "a\nb");
    }
}
