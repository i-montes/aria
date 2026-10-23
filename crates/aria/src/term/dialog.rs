//! A dialog of the session's Pi, in place of the editor: an `ask_user` question card (options by number 1–9 or a
//! click, or a written answer; several questions one after the other), a permission card (`y` once, `a` always, `n` no,
//! or its number, or a click), a plain confirm, select or input, and a secret (typed hidden; its value goes to the dock
//! as `secretSave`, never to Pi). The cards are the ones `ui/chat/parts/cards.js` reads from a dialog's title.

use crate::term::state::Dialog;
use crate::term::texts::t;
use crate::term::theme::Palette;
use crate::term::Key;
use aria_core::bridge::PageMessage;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// The `uiAnswer` to send.
    Ui(PageMessage),
    /// Save the secret first (`secretSave`); once saved, answer with `saved_answer`.
    Secret { name: String, value: String, purpose: String },
}

/// The answer of a secret card once the dock kept it.
pub fn saved_answer(id: &str, name: &str) -> Answer {
    Answer::Ui(PageMessage::UiAnswer {
        id: id.into(),
        value: Some(format!("saved:{name}")),
        confirmed: None,
        cancelled: None,
    })
}

#[derive(Debug, Clone)]
struct Question {
    question: String,
    header: String,
    options: Vec<String>,
    multi: bool,
}

#[derive(Debug, Clone)]
enum Kind {
    Question(Vec<Question>),
    /// A permission or quota card, or a plain select: (value, label) options; `keys` when y/a/n answer it.
    Choice {
        title: String,
        detail: String,
        options: Vec<(String, String)>,
        keys: bool,
    },
    Confirm {
        title: String,
        message: String,
    },
    Input {
        title: String,
    },
    Secret {
        name: String,
        purpose: String,
    },
}

#[derive(Debug, Clone)]
pub struct DialogBox {
    pub id: String,
    kind: Kind,
    step: usize,
    picked: Vec<Vec<usize>>,
    typed: Vec<String>,
    /// The answer this dialog gave, once it gave one.
    pub last: Option<Answer>,
}

fn card(request: &Value) -> Option<Value> {
    let card: Value = serde_json::from_str(request["title"].as_str()?).ok()?;
    (card["aria"] == json!(1)).then_some(card)
}

fn text(value: &Value) -> String {
    value.as_str().unwrap_or_default().to_string()
}

impl DialogBox {
    pub fn from(dialog: &Dialog) -> DialogBox {
        let request = &dialog.request;
        let accepted: Vec<String> = request["options"].as_array().map_or(Vec::new(), |o| o.iter().map(text).collect());
        let kind = match card(request) {
            Some(card) if card["kind"] == "question" => Kind::Question(
                card["questions"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|q| q.is_object())
                    .take(4)
                    .map(|q| Question {
                        question: text(&q["question"]),
                        header: text(&q["header"]),
                        options: q["options"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .map(|o| o.as_str().map(str::to_string).unwrap_or_else(|| text(&o["label"])))
                            .take(9)
                            .collect(),
                        multi: q["multi"] == json!(true) || q["type"] == "check",
                    })
                    .collect(),
            ),
            Some(card) if card["kind"] == "secret" => {
                Kind::Secret { name: text(&card["name"]), purpose: text(&card["purpose"]) }
            }
            Some(card) => {
                let options: Vec<(String, String)> = card["options"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|o| (text(&o["value"]), text(&o["label"])))
                    .filter(|(value, _)| accepted.is_empty() || accepted.contains(value))
                    .collect();
                let options = if options.is_empty() {
                    accepted.iter().map(|o| (o.clone(), o.clone())).collect()
                } else {
                    options
                };
                let title = [&card["summary"], &card["path"]]
                    .iter()
                    .map(|v| text(v))
                    .find(|s| !s.is_empty())
                    .unwrap_or_default();
                let keys = matches!(card["kind"].as_str(), Some("sandbox" | "write" | "connector"));
                Kind::Choice { title, detail: text(&card["detail"]), options, keys }
            }
            None => match dialog.method.as_str() {
                "confirm" => Kind::Confirm { title: text(&request["title"]), message: text(&request["message"]) },
                "input" | "editor" => Kind::Input { title: text(&request["title"]) },
                _ => Kind::Choice {
                    title: text(&request["title"]),
                    detail: String::new(),
                    options: accepted.iter().map(|o| (o.clone(), o.clone())).collect(),
                    keys: false,
                },
            },
        };
        let n = match &kind {
            Kind::Question(questions) => questions.len().max(1),
            _ => 1,
        };
        DialogBox {
            id: dialog.id.clone(),
            kind,
            step: 0,
            picked: vec![Vec::new(); n],
            typed: vec![String::new(); n],
            last: None,
        }
    }

    fn ui(&mut self, value: Option<String>, confirmed: Option<bool>, cancelled: Option<bool>) -> Option<Answer> {
        let answer = Answer::Ui(PageMessage::UiAnswer { id: self.id.clone(), value, confirmed, cancelled });
        self.last = Some(answer.clone());
        Some(answer)
    }

    fn cancel(&mut self) -> Option<Answer> {
        self.ui(None, None, Some(true))
    }

    /// The question card's answers, as the page sends them: the answered questions with their picks and written text.
    fn submit(&mut self) -> Option<Answer> {
        let Kind::Question(questions) = &self.kind else { return None };
        let given: Vec<Value> = questions
            .iter()
            .enumerate()
            .filter_map(|(i, q)| {
                let mut answers: Vec<String> =
                    self.picked[i].iter().filter_map(|k| q.options.get(*k).cloned()).collect();
                let other = self.typed[i].trim();
                if !other.is_empty() {
                    answers.push(other.to_string());
                }
                (!answers.is_empty()).then(|| json!({"question": q.question, "answers": answers}))
            })
            .collect();
        self.ui(Some(Value::Array(given).to_string()), None, None)
    }

    /// The current question is answered: the next one, or the answers.
    fn advance(&mut self) -> Option<Answer> {
        let Kind::Question(questions) = &self.kind else { return None };
        if self.step + 1 < questions.len() {
            self.step += 1;
            None
        } else {
            self.submit()
        }
    }

    fn pick(&mut self, k: usize) -> Option<Answer> {
        if self.last.is_some() {
            return None;
        }
        match &self.kind {
            Kind::Question(questions) => {
                let q = questions.get(self.step)?;
                if k >= q.options.len() {
                    return None;
                }
                if q.multi {
                    let picked = &mut self.picked[self.step];
                    if let Some(at) = picked.iter().position(|p| *p == k) {
                        picked.remove(at);
                    } else {
                        picked.push(k);
                        picked.sort_unstable();
                    }
                    None
                } else {
                    self.picked[self.step] = vec![k];
                    self.typed[self.step].clear();
                    self.advance()
                }
            }
            Kind::Choice { options, .. } => {
                let value = options.get(k)?.0.clone();
                self.ui(Some(value), None, None)
            }
            _ => None,
        }
    }

    pub fn key(&mut self, key: Key) -> Option<Answer> {
        if self.last.is_some() {
            return None;
        }
        if key == Key::Esc {
            return self.cancel();
        }
        match self.kind.clone() {
            Kind::Question(_) => {
                let typing = !self.typed[self.step].is_empty();
                match key {
                    Key::Char(c) if !typing && ('1'..='9').contains(&c) => self.pick(c as usize - '1' as usize),
                    Key::Char(c) => {
                        self.typed[self.step].push(c);
                        None
                    }
                    Key::Backspace => {
                        self.typed[self.step].pop();
                        None
                    }
                    Key::Enter if typing || !self.picked[self.step].is_empty() => self.advance(),
                    _ => None,
                }
            }
            Kind::Choice { options, keys, .. } => match key {
                Key::Char(c) if ('1'..='9').contains(&c) => self.pick(c as usize - '1' as usize),
                Key::Char(c) if keys => {
                    let wanted = match c {
                        'y' => "allow",
                        'a' => "always",
                        'n' => "deny",
                        _ => return None,
                    };
                    let k = options.iter().position(|(value, _)| value == wanted)?;
                    self.pick(k)
                }
                _ => None,
            },
            Kind::Confirm { .. } => match key {
                Key::Char('y') | Key::Enter => self.ui(None, Some(true), None),
                Key::Char('n') => self.ui(None, Some(false), None),
                _ => None,
            },
            Kind::Input { .. } | Kind::Secret { .. } => match key {
                Key::Char(c) => {
                    self.typed[0].push(c);
                    None
                }
                Key::Backspace => {
                    self.typed[0].pop();
                    None
                }
                Key::Enter if !self.typed[0].trim().is_empty() => {
                    let value = self.typed[0].clone();
                    if let Kind::Secret { name, purpose } = &self.kind {
                        let answer = Answer::Secret { name: name.clone(), value, purpose: purpose.clone() };
                        self.last = Some(answer.clone());
                        return Some(answer);
                    }
                    self.ui(Some(value), None, None)
                }
                _ => None,
            },
        }
    }

    /// The rows before the first option.
    fn heading_rows(&self) -> usize {
        match &self.kind {
            Kind::Question(_) => 1,
            Kind::Choice { detail, .. } => 1 + usize::from(!detail.is_empty()),
            _ => 0,
        }
    }

    /// A click on row `row` of `lines`.
    pub fn click(&mut self, row: usize) -> Option<Answer> {
        let first = self.heading_rows();
        if row < first {
            return None;
        }
        self.pick(row - first)
    }

    pub fn lines(&self, lang: &str, p: &Palette, width: u16) -> Vec<Line<'static>> {
        let width = usize::from(width);
        let cut = |s: &str| -> String { s.chars().take(width.saturating_sub(4)).collect() };
        let title =
            |s: &str| Line::from(Span::styled(cut(s), Style::default().fg(p.text).add_modifier(Modifier::BOLD)));
        let grey = Style::default().fg(p.grey);
        let option = |n: usize, label: &str, on: bool| {
            Line::from(vec![
                Span::styled(format!("  {n} "), Style::default().fg(p.accent)),
                Span::styled(cut(label), Style::default().fg(if on { p.accent } else { p.text })),
            ])
        };
        let mut out = Vec::new();
        match &self.kind {
            Kind::Question(questions) => {
                let Some(q) = questions.get(self.step) else { return out };
                let head =
                    if q.header.is_empty() { q.question.clone() } else { format!("{} · {}", q.header, q.question) };
                let count = if questions.len() > 1 {
                    format!("  {}/{}", self.step + 1, questions.len())
                } else {
                    String::new()
                };
                out.push(Line::from(vec![
                    Span::styled(cut(&head), Style::default().fg(p.text).add_modifier(Modifier::BOLD)),
                    Span::styled(count, grey),
                ]));
                for (k, label) in q.options.iter().enumerate() {
                    out.push(option(k + 1, label, self.picked[self.step].contains(&k)));
                }
                let typed = &self.typed[self.step];
                out.push(if typed.is_empty() {
                    Line::from(Span::styled(
                        format!("  › {}", t(lang, "term.answer.placeholder", &json!({}))),
                        Style::default().fg(p.faint),
                    ))
                } else {
                    Line::from(vec![
                        Span::styled("  › ", Style::default().fg(p.accent)),
                        Span::styled(cut(typed), Style::default().fg(p.text)),
                    ])
                });
            }
            Kind::Choice { title: heading, detail, options, keys } => {
                out.push(title(heading));
                if !detail.is_empty() {
                    out.push(Line::from(Span::styled(format!("  {}", cut(detail)), grey)));
                }
                for (k, (_, label)) in options.iter().enumerate() {
                    out.push(option(k + 1, label, false));
                }
                if *keys {
                    out.push(Line::from(Span::styled(
                        format!("  {}", t(lang, "term.permission.keys", &json!({}))),
                        Style::default().fg(p.faint),
                    )));
                }
            }
            Kind::Confirm { title: heading, message } => {
                out.push(title(heading));
                if !message.is_empty() {
                    out.push(Line::from(Span::styled(format!("  {}", cut(message)), grey)));
                }
                out.push(Line::from(Span::styled("  y / n", Style::default().fg(p.faint))));
            }
            Kind::Input { title: heading } => {
                out.push(title(heading));
                out.push(Line::from(vec![
                    Span::styled("  › ", Style::default().fg(p.accent)),
                    Span::styled(cut(&self.typed[0]), Style::default().fg(p.text)),
                ]));
            }
            Kind::Secret { name, purpose } => {
                out.push(title(if purpose.is_empty() { name } else { purpose }));
                let shown = if self.typed[0].is_empty() {
                    Span::styled(t(lang, "term.secret.placeholder", &json!({})), Style::default().fg(p.faint))
                } else {
                    Span::styled("•".repeat(self.typed[0].chars().count()), Style::default().fg(p.text))
                };
                out.push(Line::from(vec![Span::styled(format!("  {name} › "), Style::default().fg(p.accent)), shown]));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::state::Dialog;
    use aria_core::bridge::PageMessage;
    use serde_json::json;

    fn card(card: serde_json::Value, options: serde_json::Value) -> DialogBox {
        DialogBox::from(&Dialog {
            id: "u1".into(),
            method: "select".into(),
            request: json!({"title": card.to_string(), "options": options}),
        })
    }

    fn answer(value: &str) -> Answer {
        Answer::Ui(PageMessage::UiAnswer {
            id: "u1".into(),
            value: Some(value.into()),
            confirmed: None,
            cancelled: None,
        })
    }

    fn question() -> DialogBox {
        card(
            json!({"aria": 1, "kind": "question", "questions": [
                {"question": "Which database?", "header": "DB", "options": [{"label": "Postgres"}, {"label": "SQLite"}]},
                {"question": "Tests?", "options": [{"label": "Yes"}, {"label": "No"}]}]}),
            json!([]),
        )
    }

    #[test]
    fn a_question_is_answered_by_number_and_by_click() {
        let mut d = question();
        assert_eq!(d.key(Key::Char('2')), None, "the first of two questions moves on");
        let rows = d.lines("en", &crate::term::theme::Palette::truecolor(), 80);
        let row = rows.iter().position(|l| l.spans.iter().any(|s| s.content.contains("Yes"))).unwrap();
        let got = d.click(row).unwrap();
        assert_eq!(
            got,
            answer(&json!([{"question": "Which database?", "answers": ["SQLite"]}, {"question": "Tests?", "answers": ["Yes"]}]).to_string())
        );
    }

    #[test]
    fn a_question_takes_a_written_answer() {
        let mut d = question();
        for c in "MySQL".chars() {
            d.key(Key::Char(c));
        }
        d.key(Key::Enter);
        d.key(Key::Char('1'));
        let Some(Answer::Ui(PageMessage::UiAnswer { value: Some(value), .. })) = d.key(Key::Enter).or(d.last.clone())
        else {
            panic!("no answer")
        };
        assert!(value.contains("MySQL"), "{value}");
    }

    #[test]
    fn a_permission_takes_y_a_n() {
        let options = json!([{"value": "allow", "label": "Allow once"}, {"value": "always", "label": "Always"}, {"value": "deny", "label": "No"}]);
        let make = || {
            card(
                json!({"aria": 1, "kind": "sandbox", "summary": "Push the branch", "detail": "git push", "options": options}),
                json!(["allow", "always", "deny"]),
            )
        };
        assert_eq!(make().key(Key::Char('y')), Some(answer("allow")));
        assert_eq!(make().key(Key::Char('a')), Some(answer("always")));
        assert_eq!(make().key(Key::Char('n')), Some(answer("deny")));
        assert_eq!(make().key(Key::Char('2')), Some(answer("always")));
        assert_eq!(
            make().key(Key::Esc),
            Some(Answer::Ui(PageMessage::UiAnswer {
                id: "u1".into(),
                value: None,
                confirmed: None,
                cancelled: Some(true)
            }))
        );
    }

    #[test]
    fn a_secret_is_masked_and_sent_as_secret_save() {
        let mut d = card(json!({"aria": 1, "kind": "secret", "name": "GH_TOKEN", "purpose": "push"}), json!([]));
        for c in "s3cret".chars() {
            d.key(Key::Char(c));
        }
        let shown: String = d
            .lines("en", &crate::term::theme::Palette::truecolor(), 80)
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.to_string()))
            .collect();
        assert!(!shown.contains("s3cret") && shown.contains("••••••"), "{shown}");
        assert_eq!(
            d.key(Key::Enter),
            Some(Answer::Secret { name: "GH_TOKEN".into(), value: "s3cret".into(), purpose: "push".into() })
        );
        assert_eq!(saved_answer("u1", "GH_TOKEN"), answer("saved:GH_TOKEN"));
    }

    #[test]
    fn a_plain_confirm_and_select_work() {
        let mut confirm =
            DialogBox::from(&Dialog { id: "u1".into(), method: "confirm".into(), request: json!({"title": "Sure?"}) });
        assert_eq!(
            confirm.key(Key::Char('y')),
            Some(Answer::Ui(PageMessage::UiAnswer {
                id: "u1".into(),
                value: None,
                confirmed: Some(true),
                cancelled: None
            }))
        );
        let mut select = DialogBox::from(&Dialog {
            id: "u1".into(),
            method: "select".into(),
            request: json!({"title": "Pick", "options": ["one", "two"]}),
        });
        assert_eq!(select.key(Key::Char('2')), Some(answer("two")));
    }
}
