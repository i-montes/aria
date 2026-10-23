//! `/schedules`, in the editor's place: the dock's scheduled tasks, one row each with when it runs and whether it is
//! paused. Enter pauses a running task or resumes a paused one, `d` then `y` deletes, Esc closes. The list is the
//! dock's `schedules` reply; nothing here talks to the dock.
//!
//! A task the model made or widened to write (`awaiting`) is approved in the dock, never here: it is shown as waiting
//! and Enter does nothing on it.

use crate::term::texts::t;
use crate::term::theme::Palette;
use crate::term::Key;
use aria_core::schedules::{Clock, SystemClock, DAYS};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
struct Row {
    id: String,
    title: String,
    enabled: bool,
    awaiting: bool,
    at: String,
    days: Vec<String>,
    date: Option<String>,
    next: Option<i64>,
}

/// What the person chose.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    None,
    Close,
    /// Pause (`enabled` false) or resume (true) the task.
    Update {
        schedule: String,
        enabled: bool,
    },
    Delete {
        schedule: String,
    },
}

#[derive(Debug, Clone)]
pub struct Schedules {
    rows: Vec<Row>,
    selected: usize,
    /// The task a `d` asked to delete, until `y` or any other key answers.
    confirming: Option<String>,
}

impl Schedules {
    /// From the dock's `tasks` (`[Task + { next }]`, camelCase). Anything that is not a task with an id is skipped.
    pub fn new(tasks: &Value) -> Schedules {
        Schedules { rows: parse(tasks), selected: 0, confirming: None }
    }

    /// A newer list from the dock: the highlighted task stays highlighted if it is still there.
    pub fn replace(&mut self, tasks: &Value) {
        let kept = self.rows.get(self.selected).map(|row| row.id.clone());
        self.rows = parse(tasks);
        self.selected = kept
            .and_then(|id| self.rows.iter().position(|row| row.id == id))
            .unwrap_or(self.selected)
            .min(self.rows.len().saturating_sub(1));
        self.confirming = None;
    }

    pub fn key(&mut self, key: Key) -> Outcome {
        if let Some(id) = self.confirming.take() {
            // Only `y` deletes; Esc and every other key take the question back.
            return if matches!(key, Key::Char('y' | 'Y')) { Outcome::Delete { schedule: id } } else { Outcome::None };
        }
        match key {
            Key::Up => self.selected = self.selected.saturating_sub(1),
            Key::Down => self.selected = (self.selected + 1).min(self.rows.len().saturating_sub(1)),
            Key::Esc => return Outcome::Close,
            Key::Enter => {
                if let Some(row) = self.rows.get(self.selected).filter(|row| !row.awaiting) {
                    return Outcome::Update { schedule: row.id.clone(), enabled: !row.enabled };
                }
            }
            Key::Char('d') => self.confirming = self.rows.get(self.selected).map(|row| row.id.clone()),
            _ => {}
        }
        Outcome::None
    }

    /// The panel, at most `rows` lines tall.
    pub fn lines(&self, lang: &str, p: &Palette, width: u16, rows: usize) -> Vec<Line<'static>> {
        self.lines_with(&SystemClock, lang, p, width, rows)
    }

    fn lines_with(&self, clock: &dyn Clock, lang: &str, p: &Palette, width: u16, rows: usize) -> Vec<Line<'static>> {
        let say = |key: &str, vars: Value| t(lang, key, &vars);
        let cut = |s: &str| -> String { s.chars().take(usize::from(width).saturating_sub(6)).collect() };
        let faint = Style::default().fg(p.faint).add_modifier(Modifier::ITALIC);
        let mut out = vec![Line::from(Span::styled(
            say("term.schedules.title", json!({})),
            Style::default().fg(p.accent).add_modifier(Modifier::BOLD),
        ))];
        out.push(Line::default());
        if self.rows.is_empty() {
            out.push(Line::from(Span::styled(say("term.schedules.empty", json!({})), Style::default().fg(p.grey))));
            return out;
        }
        // Title, blank, blank, the keys and the two `… more` rows are fixed; each task takes two lines.
        let n = self.rows.len();
        let fit = (rows.saturating_sub(6) / 2).max(1);
        let start = (self.selected + 1).saturating_sub(fit);
        let end = (start + fit).min(n);
        if start > 0 {
            out.push(Line::from(Span::styled(format!("  {}", say("term.rewind.above", json!({"n": start}))), faint)));
        }
        for (at, row) in self.rows.iter().enumerate().take(end).skip(start) {
            let on = at == self.selected;
            let title = if row.title.trim().is_empty() { say("term.untitled", json!({})) } else { row.title.clone() };
            let mut head = vec![
                Span::styled(if on { "❯ " } else { "  " }, Style::default().fg(p.accent)),
                Span::styled(cut(&title), Style::default().fg(if on { p.accent } else { p.text })),
            ];
            if row.awaiting {
                // Shown in the second line.
            } else if !row.enabled {
                head.push(Span::styled(format!("  {}", say("term.schedules.paused", json!({}))), faint));
            }
            out.push(Line::from(head));
            let mut spans = vec![Span::styled(format!("  {}", when_label(lang, row)), Style::default().fg(p.grey))];
            if row.awaiting {
                spans.push(Span::styled(
                    format!(" · {}", say("term.schedules.awaiting", json!({}))),
                    Style::default().fg(p.amber),
                ));
            } else if let (true, Some(next)) = (row.enabled, row.next) {
                let when = next_label(lang, clock, next);
                spans.push(Span::styled(
                    format!(" · {}", say("term.schedules.next", json!({ "when": when }))),
                    Style::default().fg(p.grey),
                ));
            }
            out.push(Line::from(spans));
        }
        if end < n {
            out.push(Line::from(Span::styled(format!("  {}", say("term.rewind.below", json!({"n": n - end}))), faint)));
        }
        out.push(Line::default());
        match self.confirming.as_ref().and_then(|id| self.rows.iter().find(|row| &row.id == id)) {
            Some(row) => {
                let title =
                    if row.title.trim().is_empty() { say("term.untitled", json!({})) } else { row.title.clone() };
                out.push(Line::from(Span::styled(
                    say("term.schedules.confirm", json!({ "title": cut(&title) })),
                    Style::default().fg(p.amber),
                )));
            }
            None => out.push(Line::from(Span::styled(say("term.schedules.keys", json!({})), faint))),
        }
        out
    }
}

/// The tasks a dock `schedules` reply carries; what is not an object with an `id` is left out, a missing field takes a
/// quiet default.
fn parse(tasks: &Value) -> Vec<Row> {
    tasks
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|task| {
            let id = task["id"].as_str().filter(|id| !id.is_empty())?;
            let when = &task["when"];
            Some(Row {
                id: id.to_string(),
                title: task["title"].as_str().unwrap_or_default().to_string(),
                enabled: task["enabled"].as_bool().unwrap_or(true),
                awaiting: task["awaiting"].as_bool().unwrap_or(false),
                at: when["at"].as_str().unwrap_or_default().to_string(),
                days: when["days"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|d| d.as_str())
                    .map(str::to_string)
                    .collect(),
                date: when["date"].as_str().map(str::to_string),
                next: task["next"].as_i64(),
            })
        })
        .collect()
}

/// When a task runs, in words: `Every day at 07:30`, `Weekdays at 07:30`, `Mon, Wed at 07:30`, `On 2026-11-02 at 07:30`.
fn when_label(lang: &str, row: &Row) -> String {
    let say = |key: &str, vars: Value| t(lang, key, &vars);
    let at = &row.at;
    if let Some(date) = &row.date {
        return say("term.schedules.once", json!({ "at": at, "date": date }));
    }
    let has = |d: &str| row.days.iter().any(|day| day == d);
    if row.days.is_empty() && at.is_empty() {
        return String::new();
    }
    if DAYS.iter().all(|d| has(d)) {
        return say("term.schedules.everyDay", json!({ "at": at }));
    }
    if row.days.len() == 5 && DAYS[..5].iter().all(|d| has(d)) {
        return say("term.schedules.weekdays", json!({ "at": at }));
    }
    let named: Vec<String> =
        DAYS.iter().filter(|d| has(d)).map(|d| say(&format!("term.schedules.day.{d}"), json!({}))).collect();
    say("term.schedules.days", json!({ "days": named.join(", "), "at": at }))
}

/// `Tue 2026-10-27 07:30`, local.
fn next_label(lang: &str, clock: &dyn Clock, next: i64) -> String {
    let local = clock.local(next);
    let day = DAYS.get(local.weekday as usize).map(|d| t(lang, &format!("term.schedules.day.{d}"), &json!({})));
    format!(
        "{}{:04}-{:02}-{:02} {:02}:{:02}",
        day.map(|d| format!("{d} ")).unwrap_or_default(),
        local.y,
        local.m,
        local.d,
        local.h,
        local.min
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_core::schedules::Local;

    /// Every instant is Tuesday 2026-10-27 07:30.
    struct Fixed;

    impl Clock for Fixed {
        fn local(&self, _: i64) -> Local {
            Local { y: 2026, m: 10, d: 27, h: 7, min: 30, weekday: 1 }
        }
        fn epoch(&self, _: i32, _: u32, _: u32, _: u32, _: u32) -> i64 {
            0
        }
    }

    fn task(id: &str, title: &str, enabled: bool) -> Value {
        json!({
            "id": id, "title": title, "prompt": "p", "enabled": enabled, "awaiting": false, "created": 1,
            "when": {"at": "07:30", "days": ["mon", "tue", "wed", "thu", "fri"]},
            "grants": {}, "lastRun": null, "lastResult": null, "next": 1_000
        })
    }

    fn text(lines: &[Line<'static>]) -> String {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn render(panel: &Schedules, lang: &str) -> String {
        text(&panel.lines_with(&Fixed, lang, &Palette::truecolor(), 80, 20))
    }

    #[test]
    fn a_row_says_its_title_when_it_runs_and_its_next_time() {
        let panel = Schedules::new(&json!([task("s-1", "Notion tasks", true)]));
        let shown = render(&panel, "en");
        assert!(shown.contains("Notion tasks"), "{shown}");
        assert!(shown.contains("Weekdays at 07:30"), "{shown}");
        assert!(shown.contains("Tue 2026-10-27 07:30"), "{shown}");
        assert!(!shown.contains("paused"), "{shown}");
    }

    #[test]
    fn a_paused_task_has_a_marker_and_no_next_time() {
        let panel = Schedules::new(&json!([task("s-1", "Notion tasks", false)]));
        let shown = render(&panel, "en");
        assert!(shown.contains("paused"), "{shown}");
        assert!(!shown.contains("2026-10-27"), "{shown}");
        assert!(render(&panel, "es").contains("en pausa"));
    }

    #[test]
    fn the_schedule_is_said_as_every_day_weekdays_some_days_or_a_date() {
        let row = |when: Value| parse(&json!([{"id": "s", "when": when}])).remove(0);
        let all = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];
        assert_eq!(when_label("en", &row(json!({"at": "07:30", "days": all}))), "Every day at 07:30");
        assert_eq!(when_label("en", &row(json!({"at": "07:30", "days": ["fri", "mon"]}))), "Mon, Fri at 07:30");
        assert_eq!(when_label("en", &row(json!({"at": "18:00", "date": "2026-11-02"}))), "On 2026-11-02 at 18:00");
        assert_eq!(when_label("es", &row(json!({"at": "07:30", "days": ["sat"]}))), "Sáb a las 07:30");
        assert_eq!(when_label("en", &row(json!({}))), "");
    }

    #[test]
    fn enter_flips_enabled_and_awaiting_tasks_are_left_to_the_dock() {
        let mut waiting = task("s-3", "Writes notes", false);
        waiting["awaiting"] = json!(true);
        let mut panel = Schedules::new(&json!([task("s-1", "A", true), task("s-2", "B", false), waiting]));
        assert_eq!(panel.key(Key::Enter), Outcome::Update { schedule: "s-1".into(), enabled: false });
        panel.key(Key::Down);
        assert_eq!(panel.key(Key::Enter), Outcome::Update { schedule: "s-2".into(), enabled: true });
        panel.key(Key::Down);
        assert_eq!(panel.key(Key::Enter), Outcome::None, "never approved from here");
        let shown = render(&panel, "en");
        assert!(shown.contains("waiting for approval in the dock"), "{shown}");
        let mut alone =
            Schedules::new(&json!([{"id": "s-3", "title": "Writes notes", "enabled": false, "awaiting": true}]));
        assert!(!render(&alone, "en").contains("paused"), "waiting, not paused");
        assert_eq!(alone.key(Key::Char('d')), Outcome::None);
        assert_eq!(alone.key(Key::Char('y')), Outcome::Delete { schedule: "s-3".into() }, "it can be deleted");
    }

    #[test]
    fn only_y_confirms_a_delete_and_esc_only_takes_the_question_back() {
        let mut panel = Schedules::new(&json!([task("s-1", "A", true)]));
        assert_eq!(panel.key(Key::Char('d')), Outcome::None);
        assert!(render(&panel, "en").contains("Delete"), "it asks");
        assert_eq!(panel.key(Key::Char('y')), Outcome::Delete { schedule: "s-1".into() });
        for no in [Key::Char('n'), Key::Esc, Key::Enter, Key::Down, Key::Char('x')] {
            panel.key(Key::Char('d'));
            assert_eq!(panel.key(no), Outcome::None, "{no:?}");
            assert_eq!(panel.key(Key::Char('y')), Outcome::None, "the question is gone after {no:?}");
        }
        assert_eq!(panel.key(Key::Esc), Outcome::Close);
    }

    #[test]
    fn a_new_list_keeps_the_highlighted_task_and_drops_a_pending_question() {
        let mut panel = Schedules::new(&json!([task("s-1", "A", true), task("s-2", "B", true)]));
        panel.key(Key::Down);
        panel.key(Key::Char('d'));
        panel.replace(&json!([task("s-2", "B", false), task("s-1", "A", true)]));
        assert_eq!(panel.key(Key::Enter), Outcome::Update { schedule: "s-2".into(), enabled: true });
        panel.replace(&json!([task("s-1", "A", true)]));
        assert_eq!(panel.key(Key::Enter), Outcome::Update { schedule: "s-1".into(), enabled: false });
    }

    #[test]
    fn an_empty_list_says_so() {
        let mut panel = Schedules::new(&json!([]));
        assert!(render(&panel, "en").contains("No scheduled tasks"));
        assert!(render(&panel, "es").contains("No hay tareas programadas"));
        assert_eq!(panel.key(Key::Enter), Outcome::None);
        assert_eq!(panel.key(Key::Char('d')), Outcome::None);
        assert_eq!(panel.key(Key::Char('y')), Outcome::None);
        assert_eq!(panel.key(Key::Esc), Outcome::Close);
    }

    #[test]
    fn what_cannot_be_read_is_skipped() {
        for tasks in
            [json!("nope"), json!(null), json!({"id": "s"}), json!([1, null, "x", [], {"title": "no id"}, {"id": 4}])]
        {
            let panel = Schedules::new(&tasks);
            assert!(panel.rows.is_empty(), "{tasks}");
            assert!(render(&panel, "en").contains("No scheduled tasks"));
        }
        let mut panel =
            Schedules::new(&json!([{"id": "s-1"}, {"id": "s-2", "when": 5, "next": "soon", "enabled": "yes"}]));
        assert_eq!(panel.rows.len(), 2);
        render(&panel, "en");
        assert_eq!(panel.key(Key::Enter), Outcome::Update { schedule: "s-1".into(), enabled: false });
    }

    #[test]
    fn a_long_list_scrolls_with_the_highlight() {
        let tasks: Vec<Value> = (0..10).map(|i| task(&format!("s-{i}"), &format!("Task {i}"), true)).collect();
        let mut panel = Schedules::new(&json!(tasks));
        for _ in 0..7 {
            panel.key(Key::Down);
        }
        let lines = panel.lines_with(&Fixed, "en", &Palette::truecolor(), 80, 11);
        assert!(lines.len() <= 11, "{}", lines.len());
        let shown = text(&lines);
        assert!(shown.contains("❯ Task 7"), "{shown}");
        assert!(!shown.contains("Task 0"), "{shown}");
        assert!(shown.contains("more"), "{shown}");
    }
}
