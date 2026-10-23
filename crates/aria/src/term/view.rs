//! The session view (the canvas "Aria Terminal"): the conversation from the top, the editor between two thin rules at
//! the bottom (or a panel in its place: a picker, a dialog), and one footer line. Nothing here reads or writes anything
//! but the frame.

use crate::term::git::GitStatus;
use crate::term::markdown;
use crate::term::state::{Call, Entry, Session, Status, Text, Tone};
use crate::term::texts::t;
use crate::term::theme::Palette;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;
use serde_json::json;
use unicode_width::UnicodeWidthStr;

/// A call's output lines shown under it before `… N more`.
const OUTPUT_LINES: usize = 12;
/// At most this many lines of editor.
const EDITOR_LINES: usize = 6;

pub struct SessionView<'a> {
    pub session: &'a Session,
    pub verbose: bool,
    /// The editor's text.
    pub input: &'a str,
    /// A picker or a dialog, drawn where the editor goes.
    pub panel: Option<Vec<Line<'static>>>,
    pub git: Option<GitStatus>,
    /// The folder's last component.
    pub folder: String,
    pub compact_at: u8,
    pub lang: String,
    pub palette: Palette,
    /// Seconds since the epoch, for the time to a quota's reset.
    pub now: i64,
}

/// `claude-bridge/claude-opus-5-5` → `opus 5.5`, `opencode-go/mimo-v2.6-pro` → `mimo v2.6 pro`.
pub fn short_model(model: &str) -> String {
    let id = model.rsplit('/').next().unwrap_or(model);
    let id = id.strip_prefix("claude-").unwrap_or(id);
    let mut parts: Vec<String> = id.split('-').map(str::to_string).collect();
    let digits = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    if parts.len() >= 3 && digits(&parts[parts.len() - 1]) && digits(&parts[parts.len() - 2]) {
        let minor = parts.pop().unwrap_or_default();
        let major = parts.pop().unwrap_or_default();
        parts.push(format!("{major}.{minor}"));
    }
    parts.join(" ")
}

fn say(lang: &str, text: &Text) -> String {
    match text {
        Text::Key { key, vars } => t(lang, key, vars),
        Text::Raw(raw) => raw.clone(),
    }
}

/// Plain text wrapped at `width`: `first` before the first line, `rest` before the others.
fn wrap(text: &str, width: usize, first: Vec<Span<'static>>, rest: &str, style: Style) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let mut prefix = first;
    for paragraph in text.split('\n') {
        let mut line = prefix.clone();
        let mut used: usize = line.iter().map(|s| s.content.width()).sum();
        let start = used;
        for word in paragraph.split(' ') {
            let need = word.width() + usize::from(used > start);
            if used > start && used + need > width {
                out.push(Line::from(std::mem::take(&mut line)));
                line.push(Span::raw(rest.to_string()));
                used = rest.width();
            }
            let piece = if used > start { format!(" {word}") } else { word.to_string() };
            used += piece.width();
            line.push(Span::styled(piece, style));
        }
        out.push(Line::from(line));
        prefix = vec![Span::raw(rest.to_string())];
    }
    out
}

fn mark(call: &Call, p: &Palette) -> Span<'static> {
    match call.status {
        Status::Running => Span::styled("●", Style::default().fg(p.amber)),
        Status::Done => Span::styled("✓", Style::default().fg(p.green)),
        Status::Error => Span::styled("✗", Style::default().fg(p.coral)),
    }
}

fn call_lines(call: &Call, v: &SessionView, width: usize) -> Vec<Line<'static>> {
    let p = &v.palette;
    let grey = Style::default().fg(p.grey);
    if call.think {
        return wrap(&call.output, width, vec![Span::raw("  ")], "  ", grey);
    }
    let mut head = vec![
        Span::raw("  "),
        mark(call, p),
        Span::raw(" "),
        Span::styled(format!("{:<6}", call.name), Style::default().fg(p.text)),
    ];
    if !call.summary.is_empty() && call.summary != call.name {
        head.push(Span::styled(format!(" {}", call.summary), grey));
    }
    if call.added > 0 || call.removed > 0 {
        head.push(Span::styled(format!("  +{}", call.added), Style::default().fg(p.green)));
        head.push(Span::styled(format!(" −{}", call.removed), Style::default().fg(p.coral)));
    }
    let mut out = vec![Line::from(head)];
    let output = call.output.trim_end();
    if !output.is_empty() {
        let all: Vec<&str> = output.lines().collect();
        for line in all.iter().take(OUTPUT_LINES) {
            let cut: String = line.chars().take(width.saturating_sub(11)).collect();
            out.push(Line::from(vec![Span::raw(" ".repeat(11)), Span::styled(cut, grey)]));
        }
        if all.len() > OUTPUT_LINES {
            let more = t(&v.lang, "term.more", &json!({"n": all.len() - OUTPUT_LINES}));
            out.push(Line::from(vec![Span::raw(" ".repeat(11)), Span::styled(more, Style::default().fg(p.faint))]));
        }
    }
    out
}

/// One line for a whole turn (verbose off): `✓ 3 reads · 2 edits · tests 43/43`.
fn summary_line(v: &SessionView, turn: usize, running: bool) -> Line<'static> {
    let p = &v.palette;
    let s = v.session.turn_summary(turn);
    let count = |n: u32, one: &str, many: &str| -> Option<String> {
        match n {
            0 => None,
            1 => Some(t(&v.lang, one, &json!({}))),
            n => Some(t(&v.lang, many, &json!({ "n": n }))),
        }
    };
    let mut parts: Vec<String> = [
        count(s.reads, "term.summary.read", "term.summary.reads"),
        count(s.edits, "term.summary.edit", "term.summary.edits"),
        count(s.writes, "term.summary.write", "term.summary.writes"),
        count(s.commands, "term.summary.command", "term.summary.commands"),
    ]
    .into_iter()
    .flatten()
    .collect();
    if let Some((passed, total)) = s.tests {
        parts.push(t(&v.lang, "term.summary.tests", &json!({"passed": passed, "total": total})));
    }
    let mark = if running {
        Span::styled("●", Style::default().fg(p.amber))
    } else {
        Span::styled("✓", Style::default().fg(p.green))
    };
    Line::from(vec![
        Span::raw("  "),
        mark,
        Span::styled(format!(" {}", parts.join(" · ")), Style::default().fg(p.grey)),
    ])
}

/// The conversation as lines, a blank line between entries.
pub fn conversation(v: &SessionView, width: u16) -> Vec<Line<'static>> {
    let width = usize::from(width.max(1));
    let p = &v.palette;
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut turn: usize = 0;
    let mut summarized: Option<usize> = None;
    let entries = &v.session.entries;
    for (i, entry) in entries.iter().enumerate() {
        let lines: Vec<Line<'static>> = match entry {
            Entry::User { text } => {
                turn = i;
                wrap(
                    text,
                    width,
                    vec![Span::styled("› ", Style::default().fg(p.accent))],
                    "  ",
                    Style::default().fg(p.text),
                )
            }
            Entry::Assistant { text, thinking, .. } => {
                let mut lines = Vec::new();
                if v.verbose && !thinking.is_empty() {
                    lines.extend(wrap(thinking, width, vec![Span::raw("  ")], "  ", Style::default().fg(p.grey)));
                }
                for line in markdown::lines(text, width.saturating_sub(2) as u16, p) {
                    let mut spans = vec![Span::raw("  ")];
                    spans.extend(line.spans);
                    lines.push(Line::from(spans));
                }
                lines
            }
            Entry::Tools { calls } if v.verbose => calls.iter().flat_map(|call| call_lines(call, v, width)).collect(),
            Entry::Tools { .. } => {
                if summarized == Some(turn) {
                    continue;
                }
                summarized = Some(turn);
                let last_turn = !entries[i + 1..].iter().any(|e| matches!(e, Entry::User { .. }));
                vec![summary_line(v, turn, v.session.running && last_turn)]
            }
            Entry::Notice { tone, text } => {
                let colour = match tone {
                    Tone::Info => p.text,
                    Tone::Warn => p.amber,
                    Tone::Error => p.coral,
                };
                wrap(&say(&v.lang, text), width, vec![Span::raw("  ")], "  ", Style::default().fg(colour))
            }
            Entry::Note { text } => {
                vec![Line::from(vec![Span::styled(format!("  · {}", say(&v.lang, text)), Style::default().fg(p.grey))])]
            }
            Entry::Divider { text } => {
                vec![Line::from(Span::styled(format!("  ── {} ──", say(&v.lang, text)), Style::default().fg(p.faint)))]
            }
        };
        if lines.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(Line::default());
        }
        out.extend(lines);
    }
    out
}

/// Seconds since the epoch of `YYYY-MM-DDTHH:MM:SSZ` (or with a fraction or an offset of Z only).
fn epoch(stamp: &str) -> Option<i64> {
    let (date, time) = stamp.split_once('T')?;
    let mut d = date.split('-').map(|x| x.parse::<i64>().ok());
    let (y, m, day) = (d.next()??, d.next()??, d.next()??);
    let time = time.trim_end_matches('Z');
    let mut tm = time.split(':');
    let (h, mi) = (tm.next()?.parse::<i64>().ok()?, tm.next()?.parse::<i64>().ok()?);
    let sec = tm.next().and_then(|s| s.split('.').next()?.parse::<i64>().ok()).unwrap_or(0);
    // Days from civil (Howard Hinnant's algorithm).
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + h * 3600 + mi * 60 + sec)
}

/// `1h12`, `0h41`.
fn left(seconds: i64) -> String {
    let minutes = seconds.max(0) / 60;
    format!("{}h{:02}", minutes / 60, minutes % 60)
}

fn footer(v: &SessionView, width: u16) -> Line<'static> {
    let p = &v.palette;
    let s = v.session;
    let grey = Style::default().fg(p.grey);
    let mut left_side: Vec<Span<'static>> = Vec::new();
    if let Some(model) = &s.model {
        left_side.push(Span::styled(short_model(model), Style::default().fg(p.text)));
    }
    if let Some(thinking) = &s.thinking {
        left_side.push(Span::styled(format!(" {thinking}"), grey));
    }
    if let Some(context) = s.context {
        left_side.push(Span::styled("    ctx ", grey));
        left_side.push(Span::styled(
            format!("{:.0} / {}", context, v.compact_at),
            Style::default().fg(p.for_context(context, v.compact_at)),
        ));
    }
    for (n, window) in s.quota.iter().enumerate() {
        let label = match window.id.as_str() {
            "5h" => "5h".to_string(),
            "week" => t(&v.lang, "term.week", &json!({})),
            "month" => t(&v.lang, "term.month", &json!({})),
            _ => window.label.to_lowercase(),
        };
        left_side.push(Span::styled(format!("{}{label} ", if n == 0 { "    " } else { "   " }), grey));
        left_side
            .push(Span::styled(format!("{:.0}%", window.percent), Style::default().fg(p.for_quota(window.percent))));
        if n == 0 {
            if let Some(reset) = window.resets_at.as_deref().and_then(epoch) {
                left_side.push(Span::styled(format!(" {}", left(reset - v.now)), Style::default().fg(p.faint)));
            }
        }
    }
    let mut right_side: Vec<Span<'static>> = Vec::new();
    if let Some(git) = &v.git {
        let changed = if git.changed > 0 { format!(" ~{}", git.changed) } else { String::new() };
        right_side.push(Span::styled(format!("{}{changed}    ", git.branch), grey));
    }
    right_side.push(Span::styled(v.folder.clone(), grey));
    let width = usize::from(width);
    let size = |spans: &[Span]| spans.iter().map(|s| s.content.width()).sum::<usize>();
    let (l, r) = (size(&left_side), size(&right_side));
    let mut spans = left_side;
    if l + r + 2 <= width {
        spans.push(Span::raw(" ".repeat(width - l - r)));
        spans.extend(right_side);
    }
    Line::from(spans)
}

fn editor_lines(v: &SessionView, width: usize) -> Vec<Line<'static>> {
    let p = &v.palette;
    if v.input.is_empty() {
        return vec![Line::from(vec![
            Span::styled("› ", Style::default().fg(p.accent)),
            Span::styled(t(&v.lang, "term.input", &json!({})), Style::default().fg(p.faint)),
        ])];
    }
    let lines = wrap(
        v.input,
        width,
        vec![Span::styled("› ", Style::default().fg(p.accent))],
        "  ",
        Style::default().fg(p.text),
    );
    let skip = lines.len().saturating_sub(EDITOR_LINES);
    lines.into_iter().skip(skip).collect()
}

/// The start screen: the open sessions (title, folder) and the new one in the current folder last; `note` under them.
pub fn draw_list(frame: &mut Frame, rows: &[(String, String)], selected: usize, note: Option<&str>, p: &Palette) {
    let area = frame.area();
    if area.width < 4 || area.height < 2 {
        return;
    }
    let inner = Rect { x: area.x + 2, y: area.y + 1, width: area.width - 4, height: area.height - 1 };
    let title_width = rows.iter().map(|(title, _)| title.width()).max().unwrap_or(0).min(usize::from(inner.width) / 2);
    let mut lines: Vec<Line<'static>> = rows
        .iter()
        .enumerate()
        .map(|(i, (title, folder))| {
            let on = i == selected;
            let style = Style::default().fg(if on { p.accent } else { p.text });
            let cut: String = title.chars().take(title_width).collect();
            let pad = title_width.saturating_sub(cut.width());
            Line::from(vec![
                Span::styled(if on { "› " } else { "  " }, style),
                Span::styled(format!("{cut}{}", " ".repeat(pad)), style),
                Span::styled(format!("    {folder}"), Style::default().fg(p.grey)),
            ])
        })
        .collect();
    if let Some(note) = note {
        lines.push(Line::default());
        lines.push(Line::styled(format!("  {note}"), Style::default().fg(p.amber)));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

pub fn draw_session(frame: &mut Frame, v: &SessionView) {
    let area = frame.area();
    if area.width == 0 || area.height == 0 {
        return;
    }
    let pad = if area.width > 20 { 2 } else { 0 };
    let inner = Rect { x: area.x + pad, width: area.width - 2 * pad, ..area };
    let width = usize::from(inner.width.max(1));
    let rule = Line::styled("─".repeat(width), Style::default().fg(v.palette.rule));
    let middle: Vec<Line<'static>> = match &v.panel {
        Some(panel) => panel.clone(),
        None => editor_lines(v, width),
    };
    let mut bottom: Vec<Line<'static>> = vec![rule.clone()];
    let room = usize::from(area.height).saturating_sub(4).max(1);
    bottom.extend(middle.into_iter().take(room));
    bottom.push(rule);
    bottom.push(footer(v, inner.width));
    let bottom_height = (bottom.len() as u16).min(area.height);
    let skip = bottom.len().saturating_sub(usize::from(bottom_height));
    let bottom_area = Rect { y: area.y + area.height - bottom_height, height: bottom_height, ..inner };
    frame.render_widget(Paragraph::new(bottom.into_iter().skip(skip).collect::<Vec<_>>()), bottom_area);
    let room = area.height - bottom_height;
    if room == 0 {
        return;
    }
    let lines = conversation(v, inner.width);
    let skip = lines.len().saturating_sub(usize::from(room.saturating_sub(1)));
    let top = Rect { y: area.y + 1.min(room), height: room.saturating_sub(1).max(1), ..inner };
    let shown: Vec<Line<'static>> = lines.into_iter().skip(skip).collect();
    frame.render_widget(Paragraph::new(shown).style(Style::default().add_modifier(Modifier::empty())), top);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::git::GitStatus;
    use crate::term::state::Session;
    use crate::term::theme::Palette;
    use aria_core::bridge::DockMessage;
    use ratatui::backend::TestBackend;
    use ratatui::style::Color;
    use ratatui::Terminal;
    use serde_json::json;

    fn session(lines: Vec<serde_json::Value>) -> Session {
        let mut s = Session::default();
        for line in lines {
            s.apply(&serde_json::from_value::<DockMessage>(line).unwrap());
        }
        s
    }

    fn event(e: serde_json::Value) -> serde_json::Value {
        json!({"type": "chatEvent", "event": e})
    }

    fn worked() -> Session {
        let mut s = session(vec![
            json!({"type": "chatState", "workdir": "/home/ana/Projects/checkout-widget", "running": false, "compacting": false,
                "model": "claude-bridge/claude-opus-5-5", "thinking": "high", "context": 31.0, "problem": null}),
            json!({"type": "chatQuota", "windows": [
                {"id": "5h", "label": "5h", "percent": 84.0, "resetsAt": "2026-10-23T14:12:00Z"},
                {"id": "week", "label": "Week", "percent": 38.0, "resetsAt": null},
                {"id": "fable", "label": "Fable", "percent": 96.0, "resetsAt": null}]}),
        ]);
        s.add_user("fix the coupon");
        for line in [
            event(json!({"type": "agent_start"})),
            event(json!({"type": "message_start", "message": {"role": "assistant", "content": []}})),
            event(
                json!({"type": "message_end", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "Read first."}]}}),
            ),
            event(
                json!({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "read", "args": {"path": "src/coupon.ts"}}),
            ),
            event(json!({"type": "tool_execution_end", "toolCallId": "c1", "toolName": "read", "isError": false,
                "result": {"content": [{"type": "text", "text": "export const x = 1"}]}})),
            event(
                json!({"type": "tool_execution_start", "toolCallId": "c3", "toolName": "bash", "args": {"command": "bun test coupon"}}),
            ),
            event(json!({"type": "tool_execution_end", "toolCallId": "c3", "toolName": "bash", "isError": false,
                "result": {"content": [{"type": "text", "text": "43 pass\n0 fail"}]}})),
            event(json!({"type": "message_start", "message": {"role": "assistant", "content": []}})),
            event(
                json!({"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "Listo."}]}}),
            ),
            event(json!({"type": "agent_settled"})),
        ] {
            s.apply(&serde_json::from_value::<DockMessage>(line).unwrap());
        }
        s
    }

    fn view(session: &Session, verbose: bool) -> SessionView<'_> {
        SessionView {
            session,
            verbose,
            input: "",
            panel: None,
            git: Some(GitStatus { branch: "main".into(), changed: 3 }),
            folder: "checkout-widget".into(),
            compact_at: 40,
            lang: "es".into(),
            palette: Palette::truecolor(),
            // 2026-10-23T13:00:00Z: the 5-hour window resets in 1h12.
            now: 1_792_760_400,
        }
    }

    fn render(v: &SessionView, w: u16, h: u16) -> ratatui::buffer::Buffer {
        let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
        terminal.draw(|frame| draw_session(frame, v)).unwrap();
        terminal.backend().buffer().clone()
    }

    fn text(buffer: &ratatui::buffer::Buffer) -> String {
        let area = buffer.area;
        (0..area.height)
            .map(|y| (0..area.width).map(|x| buffer[(x, y)].symbol()).collect::<String>().trim_end().to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The colour of the first cell of `needle` on the screen.
    fn colour_of(buffer: &ratatui::buffer::Buffer, needle: &str) -> Color {
        let screen = text(buffer);
        let (y, line) = screen.lines().enumerate().find(|(_, l)| l.contains(needle)).expect(needle);
        let x = line[..line.find(needle).unwrap()].chars().count() as u16;
        buffer[(x, y as u16)].fg
    }

    #[test]
    fn verbose_on_shows_thinking_and_output() {
        let s = worked();
        let screen = text(&render(&view(&s, true), 100, 30));
        assert!(screen.contains("› fix the coupon"), "{screen}");
        assert!(screen.contains("Read first."), "{screen}");
        assert!(screen.contains("read   src/coupon.ts") || screen.contains("read  src/coupon.ts"), "{screen}");
        assert!(screen.contains("export const x = 1"), "{screen}");
        assert!(screen.contains("Listo."), "{screen}");
    }

    #[test]
    fn verbose_off_shows_one_line_per_turn() {
        let s = worked();
        let screen = text(&render(&view(&s, false), 100, 30));
        assert!(screen.contains("✓ 1 lectura · 1 comando · tests 43/43"), "{screen}");
        assert!(!screen.contains("export const x = 1"), "{screen}");
        assert!(!screen.contains("Read first."), "{screen}");
        assert!(screen.contains("Listo."), "{screen}");
    }

    #[test]
    fn quota_turns_amber_at_80_and_coral_at_95() {
        let s = worked();
        let b = render(&view(&s, true), 120, 30);
        let p = Palette::truecolor();
        assert_eq!(colour_of(&b, "84%"), p.amber);
        assert_eq!(colour_of(&b, "38%"), p.grey);
        assert_eq!(colour_of(&b, "96%"), p.coral);
        assert!(text(&b).contains("1h12"), "{}", text(&b));
    }

    #[test]
    fn context_turns_amber_five_points_before_compacting() {
        let mut s = worked();
        s.context = Some(36.0);
        let b = render(&view(&s, true), 120, 30);
        assert_eq!(colour_of(&b, "36 / 40"), Palette::truecolor().amber);
    }

    #[test]
    fn api_key_footer_has_no_quota() {
        let mut s = worked();
        s.quota.clear();
        s.model = Some("anthropic/claude-sonnet-5-5".into());
        let screen = text(&render(&view(&s, true), 120, 30));
        let footer = screen.lines().last().unwrap().trim_start();
        assert!(footer.starts_with("sonnet 5.5"), "{footer}");
        assert!(!footer.contains("5h"), "{footer}");
        assert!(footer.contains("main ~3") && footer.ends_with("checkout-widget"), "{footer}");
    }

    #[test]
    fn draws_in_a_tiny_terminal_without_panicking() {
        let s = worked();
        render(&view(&s, true), 20, 5);
        render(&view(&s, false), 1, 1);
        let wide = text(&render(&view(&s, true), 200, 60));
        assert!(wide.contains("Listo."));
        let narrow = text(&render(&view(&s, true), 40, 12));
        let footer = narrow.lines().last().unwrap().trim_start();
        assert!(footer.starts_with("opus 5.5"), "the footer keeps its left side: {footer}");
    }

    #[test]
    fn a_panel_takes_the_editor_place() {
        let s = worked();
        let mut v = view(&s, true);
        v.panel = Some(vec![ratatui::text::Line::from("volver a antes de")]);
        let screen = text(&render(&v, 100, 30));
        assert!(screen.contains("volver a antes de"), "{screen}");
    }

    #[test]
    fn model_names_are_short() {
        assert_eq!(short_model("claude-bridge/claude-opus-5-5"), "opus 5.5");
        assert_eq!(short_model("opencode-go/mimo-v2.6-pro"), "mimo v2.6 pro");
        assert_eq!(short_model("opencode-go/glm-5.3"), "glm 5.3");
        assert_eq!(short_model("x"), "x");
    }
}
