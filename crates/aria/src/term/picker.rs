//! A list to pick from, filtered as you type (files after `@`, commands after `/`, the models of `/model`): ↑↓ move,
//! Enter picks, Esc closes, a click picks the row under it. Rows of a group (a provider) come under its name.

use crate::term::theme::Palette;
use crate::term::Key;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use serde_json::Value;
use std::cmp::Ordering;

/// Rows shown at once.
const ROWS: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub label: String,
    pub detail: String,
    /// What picking it gives back.
    pub value: String,
    pub group: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum Outcome {
    Picked(String),
    Closed,
    None,
}

pub struct Picker {
    items: Vec<Item>,
    query: String,
    active: usize,
}

/// Where `query` falls in `text` (lowercase): a run of it (rank by its position), or its letters in order (after every
/// run), or nowhere.
fn rank(text: &str, query: &str) -> Option<usize> {
    if query.is_empty() {
        return Some(0);
    }
    let text = text.to_lowercase();
    let query = query.to_lowercase();
    if let Some(at) = text.find(&query) {
        return Some(at);
    }
    let mut letters = text.char_indices();
    let mut first = None;
    for q in query.chars() {
        let (at, _) = letters.find(|(_, c)| *c == q)?;
        first.get_or_insert(at);
    }
    Some(10_000 + first.unwrap_or(0))
}

impl Picker {
    pub fn new(items: Vec<Item>) -> Picker {
        Picker { items, query: String::new(), active: 0 }
    }

    pub fn set_query(&mut self, query: &str) {
        self.query = query.to_string();
        self.active = 0;
    }

    pub fn shown(&self) -> Vec<&Item> {
        let mut ranked: Vec<(usize, usize, &Item)> = self
            .items
            .iter()
            .enumerate()
            .filter_map(|(order, item)| {
                rank(&format!("{} {}", item.label, item.detail), &self.query).map(|r| (r, order, item))
            })
            .collect();
        ranked.sort_by_key(|(r, order, _)| (*r, *order));
        ranked.into_iter().map(|(_, _, item)| item).collect()
    }

    /// Each screen row: a group's name, or a shown item's index.
    fn rows(&self) -> Vec<Row> {
        let mut rows = Vec::new();
        let mut group: Option<&str> = None;
        for (i, item) in self.shown().into_iter().enumerate() {
            if item.group.is_some() && item.group.as_deref() != group {
                group = item.group.as_deref();
                rows.push(Row::Group(group.unwrap_or_default().to_string()));
            }
            rows.push(Row::Item(i));
        }
        rows
    }

    pub fn key(&mut self, key: Key) -> Outcome {
        let n = self.shown().len();
        match key {
            Key::Up => self.active = self.active.saturating_sub(1),
            Key::Down if n > 0 => self.active = (self.active + 1).min(n - 1),
            Key::Enter | Key::Tab => {
                return self.shown().get(self.active).map_or(Outcome::None, |item| Outcome::Picked(item.value.clone()))
            }
            Key::Esc => return Outcome::Closed,
            _ => {}
        }
        Outcome::None
    }

    /// `row` counts from the first line `lines` draws.
    pub fn click(&self, row: usize) -> Outcome {
        match self.rows().into_iter().skip(self.window()).nth(row) {
            Some(Row::Item(i)) => Outcome::Picked(self.shown()[i].value.clone()),
            _ => Outcome::None,
        }
    }

    /// The first row shown, so the active one stays in view.
    fn window(&self) -> usize {
        let at = self.rows().iter().position(|r| *r == Row::Item(self.active)).unwrap_or(0);
        at.saturating_sub(ROWS - 1)
    }

    pub fn lines(&self, palette: &Palette) -> Vec<Line<'static>> {
        let shown = self.shown();
        self.rows()
            .into_iter()
            .skip(self.window())
            .take(ROWS)
            .map(|row| match row {
                Row::Group(name) => Line::styled(format!("  {name}"), Style::default().fg(palette.faint)),
                Row::Item(i) => {
                    let item = shown[i];
                    let on = i == self.active;
                    let style = Style::default().fg(if on { palette.accent } else { palette.text });
                    let mut spans = vec![
                        Span::styled(if on { "› " } else { "  " }, style),
                        Span::styled(item.label.clone(), style),
                    ];
                    if !item.detail.is_empty() {
                        spans.push(Span::styled(format!("   {}", item.detail), Style::default().fg(palette.grey)));
                    }
                    Line::from(spans)
                }
            })
            .collect()
    }
}

#[derive(Debug, PartialEq)]
enum Row {
    Group(String),
    Item(usize),
}

const PROVIDERS: [(&str, &str); 4] = [
    ("opencode-go", "OpenCode Go"),
    ("opencode", "OpenCode Zen"),
    ("anthropic", "Anthropic"),
    ("claude-bridge", "Claude (sign-in)"),
];

/// The credential each of Pi's providers is used with (`connected`'s names): as `ui/chat/parts/models.js`.
const CREDENTIAL: [(&str, &str); 3] =
    [("opencode-go", "opencode-go:api_key"), ("anthropic", "anthropic:api_key"), ("claude-bridge", "anthropic:oauth")];

/// Names compared as people read versions: `Opus 5.5` after `Opus 4.5`, `GLM 5.10` after `GLM 5.9`.
fn natural(a: &str, b: &str) -> Ordering {
    let chunks = |s: &str| -> Vec<(bool, String)> {
        let mut out: Vec<(bool, String)> = Vec::new();
        for c in s.to_lowercase().chars() {
            let digit = c.is_ascii_digit();
            match out.last_mut() {
                Some((d, run)) if *d == digit => run.push(c),
                _ => out.push((digit, c.to_string())),
            }
        }
        out
    };
    for (x, y) in chunks(a).iter().zip(chunks(b).iter()) {
        let order = match (x.0 && y.0, x.1.parse::<u64>(), y.1.parse::<u64>()) {
            (true, Ok(m), Ok(n)) => m.cmp(&n),
            _ => x.1.cmp(&y.1),
        };
        if order != Ordering::Equal {
            return order;
        }
    }
    a.len().cmp(&b.len())
}

/// Pi's models for `/model`: only connected providers (the model in use always stays), OpenCode Zen only when pay per
/// use is allowed, each provider's models newest first, providers in the order Pi lists them.
pub fn models_for_picker(models: &Value, connected: &[String], current: &str, allow_pay_per_use: bool) -> Vec<Item> {
    let mut groups: Vec<(String, Vec<Item>)> = Vec::new();
    for model in models.as_array().into_iter().flatten() {
        let provider = model["provider"].as_str().unwrap_or_default();
        let id = model["id"].as_str().unwrap_or_default();
        let value = format!("{provider}/{id}");
        let needs = CREDENTIAL.iter().find(|(p, _)| *p == provider).map(|(_, c)| *c);
        let wanted =
            needs.is_none_or(|c| connected.iter().any(|k| k == c)) && (provider != "opencode" || allow_pay_per_use);
        if !wanted && value != current {
            continue;
        }
        let label = PROVIDERS.iter().find(|(p, _)| *p == provider).map_or(provider.to_string(), |(_, l)| l.to_string());
        let item = Item {
            label: model["name"].as_str().filter(|n| !n.is_empty()).unwrap_or(id).to_string(),
            detail: if value == current { "●".into() } else { String::new() },
            value,
            group: Some(label.clone()),
        };
        match groups.iter_mut().find(|(name, _)| *name == label) {
            Some((_, items)) => items.push(item),
            None => groups.push((label, vec![item])),
        }
    }
    groups
        .into_iter()
        .flat_map(|(_, mut items)| {
            items.sort_by(|a, b| natural(&b.label, &a.label));
            items
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn item(label: &str) -> Item {
        Item { label: label.into(), detail: String::new(), value: label.into(), group: None }
    }

    #[test]
    fn the_filter_is_fuzzy_and_keys_move_and_pick() {
        let mut p = Picker::new(vec![item("src/coupon.ts"), item("src/checkout.ts"), item("README.md")]);
        p.set_query("cou");
        assert_eq!(
            p.shown().iter().map(|i| i.label.as_str()).collect::<Vec<_>>(),
            ["src/coupon.ts", "src/checkout.ts"]
        );
        p.key(Key::Down);
        assert_eq!(p.key(Key::Enter), Outcome::Picked("src/checkout.ts".into()));
        assert_eq!(p.key(Key::Esc), Outcome::Closed);
    }

    #[test]
    fn a_click_picks_the_row_under_it() {
        let p = Picker::new(vec![item("a"), item("b")]);
        assert_eq!(p.click(1), Outcome::Picked("b".into()));
        assert_eq!(p.click(9), Outcome::None);
    }

    #[test]
    fn the_model_picker_lists_connected_providers_newest_first() {
        let models = json!([
            {"provider": "anthropic", "id": "claude-opus-5-5", "name": "Claude Opus 5.5"},
            {"provider": "claude-bridge", "id": "claude-opus-4-5", "name": "Claude Opus 4.5"},
            {"provider": "claude-bridge", "id": "claude-opus-5-5", "name": "Claude Opus 5.5"},
            {"provider": "opencode-go", "id": "glm-5.3", "name": "GLM 5.3"},
            {"provider": "opencode-go", "id": "kimi-k3", "name": "Kimi K3"},
            {"provider": "opencode", "id": "gpt-6-sol", "name": "GPT 6 Sol"},
        ]);
        let connected = vec!["anthropic:oauth".to_string(), "opencode-go:api_key".to_string()];
        let items = models_for_picker(&models, &connected, "", false);
        let values: Vec<&str> = items.iter().map(|i| i.value.as_str()).collect();
        assert_eq!(
            values,
            [
                "claude-bridge/claude-opus-5-5",
                "claude-bridge/claude-opus-4-5",
                "opencode-go/kimi-k3",
                "opencode-go/glm-5.3"
            ]
        );
        assert_eq!(items[0].group.as_deref(), Some("Claude (sign-in)"));
        let with_current = models_for_picker(&models, &connected, "anthropic/claude-opus-5-5", false);
        assert!(with_current.iter().any(|i| i.value == "anthropic/claude-opus-5-5"), "the model in use stays");
    }
}
