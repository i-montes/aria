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
    /// Lines scrolled up from the bottom of the conversation.
    pub scroll: usize,
    /// How long the running turn has run (ms); none while nothing runs.
    pub working_ms: Option<u64>,
    /// A clock in ms that turns the spinner.
    pub tick: u64,
    /// Suggestions (`/`, `@`, `/model`) floating over the conversation, right above the editor.
    pub overlay: Option<Vec<Line<'static>>>,
    /// Calls whose whole output is shown (a click on their `… N more lines`).
    pub expanded: std::collections::HashSet<String>,
}

/// What a frame drew that the input needs: how far the conversation can scroll up, and the screen rows of the
/// `… N more lines` rows with their call.
#[derive(Debug, Default, PartialEq)]
pub struct Drawn {
    pub above: usize,
    pub more: Vec<(u16, String)>,
}

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// `3.4s`, or `2m 5.0s` from a minute on (as the dock's working line).
fn elapsed(ms: u64) -> String {
    let tenths = ms / 100;
    let seconds = tenths as f64 / 10.0;
    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else {
        format!("{}m {:.1}s", tenths / 600, (tenths % 600) as f64 / 10.0)
    }
}

/// The messages waiting for the next gap of the running turn, oldest first, right above the editor.
fn queue_lines(v: &SessionView, width: usize) -> Vec<Line<'static>> {
    let p = &v.palette;
    let label = t(&v.lang, "term.queued", &json!({}));
    let indent = label.width() + 4;
    v.session
        .queue
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let first = text.lines().next().unwrap_or_default();
            let cut: String = first.chars().take(width.saturating_sub(indent + 1)).collect();
            let head = if i == 0 { format!("  {label}  ") } else { " ".repeat(indent) };
            Line::from(vec![
                Span::styled(head, Style::default().fg(p.faint)),
                Span::styled(cut, Style::default().fg(p.grey)),
            ])
        })
        .collect()
}

/// The working line: the spinner, what the run is doing and for how long.
fn working_line(v: &SessionView, ms: u64) -> Line<'static> {
    let p = &v.palette;
    let phase = if v.session.phase.is_empty() { "working" } else { v.session.phase.as_str() };
    let key = format!("chat.working.phase.{}", phase.replace(' ', "_"));
    let label = t(&v.lang, &key, &json!({}));
    let label = if label == key { phase.to_string() } else { label };
    Line::from(vec![
        Span::styled(format!("{} ", SPINNER[(v.tick / 80) as usize % SPINNER.len()]), Style::default().fg(p.accent)),
        Span::styled(label, Style::default().fg(p.text)),
        Span::styled(format!(" · {}", elapsed(ms)), Style::default().fg(p.faint)),
    ])
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

/// A call's lines, and which of them (if any) is its `… N more lines` row.
fn call_lines(call: &Call, v: &SessionView, width: usize) -> (Vec<Line<'static>>, Option<usize>) {
    let p = &v.palette;
    let grey = Style::default().fg(p.grey);
    if call.think {
        return (wrap(&call.output, width, vec![Span::raw("  ")], "  ", grey), None);
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
    let mut more_at = None;
    let output = call.output.trim_end();
    if !output.is_empty() {
        let all: Vec<&str> = output.lines().collect();
        // A click on the `… N more lines` row shows the rest (`expanded`), and another folds it again.
        let shown = if v.expanded.contains(&call.id) { all.len() } else { OUTPUT_LINES };
        for line in all.iter().take(shown) {
            let cut: String = line.chars().take(width.saturating_sub(11)).collect();
            out.push(Line::from(vec![Span::raw(" ".repeat(11)), Span::styled(cut, grey)]));
        }
        if all.len() > shown {
            let more = t(&v.lang, "term.more", &json!({"n": all.len() - shown}));
            more_at = Some(out.len());
            out.push(Line::from(vec![Span::raw(" ".repeat(11)), Span::styled(more, Style::default().fg(p.faint))]));
        }
    }
    (out, more_at)
}

/// A step's title, as the dock's accordion says it: what the tool does ("Leer archivo", "Escribir 3 líneas"), or a
/// command's reason.
fn step_title(call: &Call, lang: &str) -> String {
    let say = |key: &str, vars: serde_json::Value| t(lang, key, &vars);
    match call.name.as_str() {
        "bash" if !call.reason.is_empty() => call.reason.clone(),
        "bash" => say("chat.tools.run", json!({})),
        "read" => say("chat.tools.read", json!({})),
        "edit" => say("chat.tools.edit", json!({})),
        "write" => match call.added {
            0 => say("chat.tools.write_file", json!({})),
            1 => say("chat.tools.write_one", json!({})),
            n => say("chat.tools.write_n", json!({ "n": n })),
        },
        "grep" => say("chat.tools.grep", json!({})),
        "find" => say("chat.tools.find", json!({})),
        "ls" => say("chat.tools.ls", json!({})),
        name => {
            let words = name.replace(['_', '-'], " ");
            let mut chars = words.chars();
            chars.next().map_or_else(|| say("chat.tools.tool", json!({})), |c| c.to_uppercase().chain(chars).collect())
        }
    }
}

/// What a step works on: a file by its name, a command, a pattern.
fn step_chip(call: &Call) -> String {
    if matches!(call.name.as_str(), "read" | "write" | "edit") {
        return call.summary.rsplit('/').next().unwrap_or_default().to_string();
    }
    if call.summary == call.name {
        String::new()
    } else {
        call.summary.clone()
    }
}

/// One line per step (verbose off), as the dock's accordion rows: `✓ Leer archivo  coupon.ts`.
fn step_line(call: &Call, v: &SessionView, width: usize) -> Line<'static> {
    let p = &v.palette;
    let mut spans = vec![
        Span::raw("  "),
        mark(call, p),
        Span::styled(format!(" {}", step_title(call, &v.lang)), Style::default().fg(p.text)),
    ];
    let chip = step_chip(call);
    if !chip.is_empty() {
        let used: usize = spans.iter().map(|s| s.content.width()).sum();
        let cut: String = chip.chars().take(width.saturating_sub(used + 2)).collect();
        spans.push(Span::styled(format!("  {cut}"), Style::default().fg(p.grey)));
    }
    if call.added > 0 || call.removed > 0 {
        spans.push(Span::styled(format!("  +{}", call.added), Style::default().fg(p.green)));
        spans.push(Span::styled(format!(" −{}", call.removed), Style::default().fg(p.coral)));
    }
    Line::from(spans)
}

/// The conversation as lines, a blank line between entries.
#[cfg(test)]
pub fn conversation(v: &SessionView, width: u16) -> Vec<Line<'static>> {
    conversation_marked(v, width).0
}

/// The conversation's lines, with the `… N more lines` rows (their line index and call id).
fn conversation_marked(v: &SessionView, width: u16) -> (Vec<Line<'static>>, Vec<(usize, String)>) {
    let width = usize::from(width.max(1));
    let p = &v.palette;
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut marks: Vec<(usize, String)> = Vec::new();
    let entries = &v.session.entries;
    for entry in entries {
        let mut entry_marks: Vec<(usize, String)> = Vec::new();
        let lines: Vec<Line<'static>> = match entry {
            Entry::User { text } => wrap(
                text,
                width,
                vec![Span::styled("› ", Style::default().fg(p.accent))],
                "  ",
                Style::default().fg(p.text),
            ),
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
            Entry::Tools { calls } if v.verbose => {
                let mut lines = Vec::new();
                for call in calls {
                    let (call_lines, more_at) = call_lines(call, v, width);
                    if let Some(at) = more_at {
                        entry_marks.push((lines.len() + at, call.id.clone()));
                    }
                    lines.extend(call_lines);
                }
                lines
            }
            Entry::Tools { calls } => calls.iter().filter(|c| !c.think).map(|call| step_line(call, v, width)).collect(),
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
            Entry::Changes { files, added, removed } => {
                let key = if *files == 1 { "term.changes.one" } else { "term.changes.many" };
                vec![Line::from(vec![
                    Span::styled(format!("  {}", t(&v.lang, key, &json!({ "n": files }))), Style::default().fg(p.grey)),
                    Span::styled(format!("  +{added}"), Style::default().fg(p.green)),
                    Span::styled(format!(" −{removed}"), Style::default().fg(p.coral)),
                ])]
            }
        };
        if lines.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(Line::default());
        }
        marks.extend(entry_marks.into_iter().map(|(at, id)| (out.len() + at, id)));
        out.extend(lines);
    }
    (out, marks)
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
pub fn draw_list(
    frame: &mut Frame,
    rows: &[(String, String, String, String)],
    selected: usize,
    note: Option<&str>,
    p: &Palette,
) {
    let area = frame.area();
    if area.width < 4 || area.height < 2 {
        return;
    }
    let inner = Rect { x: area.x + 2, y: area.y + 1, width: area.width - 4, height: area.height - 1 };
    let title_width = rows.iter().map(|row| row.0.width()).max().unwrap_or(0).min(usize::from(inner.width) / 2);
    let folder_width = rows.iter().map(|row| row.1.width()).max().unwrap_or(0).min(24);
    let state_width = rows.iter().map(|row| row.2.width()).max().unwrap_or(0);
    let mut lines: Vec<Line<'static>> = rows
        .iter()
        .enumerate()
        .map(|(i, (title, folder, state, ago))| {
            let on = i == selected;
            let style = Style::default().fg(if on { p.accent } else { p.text });
            let cut: String = title.chars().take(title_width).collect();
            let pad = title_width.saturating_sub(cut.width());
            let folder: String = folder.chars().take(folder_width).collect();
            let folder_pad = folder_width.saturating_sub(folder.width());
            let state_pad = state_width.saturating_sub(state.width());
            Line::from(vec![
                Span::styled(if on { "› " } else { "  " }, style),
                Span::styled(format!("{cut}{}", " ".repeat(pad)), style),
                Span::styled(format!("    {folder}{}", " ".repeat(folder_pad)), Style::default().fg(p.grey)),
                Span::styled(format!("    {state}{}", " ".repeat(state_pad)), Style::default().fg(p.text)),
                Span::styled(format!("    {ago}"), Style::default().fg(p.faint)),
            ])
        })
        .collect();
    if let Some(note) = note {
        lines.push(Line::default());
        lines.push(Line::styled(format!("  {note}"), Style::default().fg(p.amber)));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

/// The rows the editor's block takes at the bottom (the working line, the rules, the editor or a dialog, the footer):
/// suggestions float right above it.
pub fn bottom_height(v: &SessionView, width: u16, height: u16) -> u16 {
    let pad = if width > 20 { 2 } else { 0 };
    let width = usize::from(width.saturating_sub(2 * pad).max(1));
    let middle = v.panel.as_ref().map_or_else(|| editor_lines(v, width).len(), Vec::len);
    let room = usize::from(height).saturating_sub(4).max(1);
    let rows = usize::from(v.working_ms.is_some()) + v.session.queue.len() + 1 + middle.min(room) + 1 + 1;
    (rows as u16).min(height)
}

/// How far the conversation can scroll up in a `width` × `height` screen (its lines above the ones shown).
#[cfg(test)]
pub fn max_scroll(v: &SessionView, width: u16, height: u16) -> usize {
    let pad = if width > 20 { 2 } else { 0 };
    let lines = conversation(v, width.saturating_sub(2 * pad)).len();
    let room = height.saturating_sub(bottom_height(v, width, height));
    lines.saturating_sub(usize::from(conversation_rows(room)))
}

/// The rows the conversation shows in `room` rows: one blank above it and, room permitting, one below.
fn conversation_rows(room: u16) -> u16 {
    if room >= 4 {
        room - 2
    } else {
        room.saturating_sub(1)
    }
}

/// Whether to draw a frame now: something changed, or a turn runs and the spinner is due (every 80 ms).
pub fn should_draw(changed: bool, running: bool, since_last_ms: u64) -> bool {
    changed || (running && since_last_ms >= 80)
}

/// Where the terminal's cursor goes for the editor's `cursor` (a byte index into `text`): its line, counted from the
/// bottom rule up, and its column after the `› ` and the margin. Long lines are not wrapped here.
pub fn cursor_position(text: &str, cursor: usize, width: u16, height: u16) -> (u16, u16) {
    let pad = if width > 20 { 2 } else { 0 };
    let before = &text[..cursor.min(text.len())];
    let line = before.matches('\n').count();
    let lines = text.matches('\n').count() + 1;
    let column = before.rsplit('\n').next().unwrap_or_default().width();
    let below = (lines - 1 - line).min(EDITOR_LINES - 1) as u16;
    let x = (pad + 2 + column).min(usize::from(width.saturating_sub(1))) as u16;
    (x, height.saturating_sub(3 + below))
}

/// Draws the session; returns how far the conversation can scroll up and where its `… N more lines` rows are.
pub fn draw_session(frame: &mut Frame, v: &SessionView) -> Drawn {
    let area = frame.area();
    if area.width == 0 || area.height == 0 {
        return Drawn::default();
    }
    let pad = if area.width > 20 { 2 } else { 0 };
    let inner = Rect { x: area.x + pad, width: area.width - 2 * pad, ..area };
    let width = usize::from(inner.width.max(1));
    let rule = Line::styled("─".repeat(width), Style::default().fg(v.palette.rule));
    let middle: Vec<Line<'static>> = match &v.panel {
        Some(panel) => panel.clone(),
        None => editor_lines(v, width),
    };
    let mut bottom: Vec<Line<'static>> = Vec::new();
    if let Some(ms) = v.working_ms {
        bottom.push(working_line(v, ms));
    }
    bottom.extend(queue_lines(v, width));
    bottom.push(rule.clone());
    let room = usize::from(area.height).saturating_sub(4).max(1);
    bottom.extend(middle.into_iter().take(room));
    bottom.push(rule);
    bottom.push(footer(v, inner.width));
    let bottom_height = (bottom.len() as u16).min(area.height);
    let skip = bottom.len().saturating_sub(usize::from(bottom_height));
    let bottom_area = Rect { y: area.y + area.height - bottom_height, height: bottom_height, ..inner };
    let overlay_area = v.overlay.as_ref().map(|lines| {
        let height = (lines.len() as u16).min(bottom_area.y.saturating_sub(area.y));
        Rect { y: bottom_area.y - height, height, ..inner }
    });
    frame.render_widget(Paragraph::new(bottom.into_iter().skip(skip).collect::<Vec<_>>()), bottom_area);
    let room = area.height - bottom_height;
    if room == 0 {
        return Drawn::default();
    }
    // A row of air above and below the conversation: it never touches the working line or the editor.
    let shown_rows = conversation_rows(room);
    let (lines, marks) = conversation_marked(v, inner.width);
    let above = lines.len().saturating_sub(usize::from(shown_rows));
    let skip = above.saturating_sub(v.scroll);
    let top = Rect { y: area.y + 1.min(room), height: shown_rows.max(1), ..inner };
    let shown: Vec<Line<'static>> = lines.into_iter().skip(skip).collect();
    frame.render_widget(Paragraph::new(shown).style(Style::default().add_modifier(Modifier::empty())), top);
    // Over what is there: the conversation does not move.
    if let (Some(lines), Some(overlay)) = (&v.overlay, overlay_area) {
        frame.render_widget(ratatui::widgets::Clear, overlay);
        frame.render_widget(Paragraph::new(lines.clone()).style(Style::default().bg(v.palette.ground)), overlay);
    }
    let more = marks
        .into_iter()
        .filter(|(at, _)| *at >= skip && *at < skip + usize::from(shown_rows))
        .map(|(at, id)| (top.y + (at - skip) as u16, id))
        .collect();
    Drawn { above, more }
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
            scroll: 0,
            working_ms: None,
            tick: 0,
            overlay: None,
            expanded: Default::default(),
        }
    }

    fn render(v: &SessionView, w: u16, h: u16) -> ratatui::buffer::Buffer {
        let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
        terminal
            .draw(|frame| {
                draw_session(frame, v);
            })
            .unwrap();
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
    fn verbose_off_lists_each_step_as_the_docks_accordion() {
        let mut s = worked();
        s.apply(
            &serde_json::from_value::<DockMessage>(event(json!({"type": "tool_execution_start", "toolCallId": "c4",
            "toolName": "bash", "args": {"command": "git push", "reason": "Push the branch"}})))
            .unwrap(),
        );
        let screen = text(&render(&view(&s, false), 100, 30));
        assert!(screen.contains("✓ Leer archivo"), "{screen}");
        assert!(screen.contains("coupon.ts") && !screen.contains("src/coupon.ts"), "a file by its name: {screen}");
        assert!(screen.contains("✓ Ejecutar comando") && screen.contains("bun test coupon"), "{screen}");
        assert!(
            screen.contains("● Push the branch") && screen.contains("git push"),
            "the command's reason is its title: {screen}"
        );
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
    fn scrolling_shows_earlier_lines() {
        let mut s = Session::default();
        for n in 0..60 {
            s.add_user(&format!("message {n}"));
        }
        let mut v = view(&s, true);
        let bottom = text(&render(&v, 80, 20));
        assert!(bottom.contains("message 59") && !bottom.contains("message 40"), "{bottom}");
        v.scroll = 20;
        let up = text(&render(&v, 80, 20));
        assert!(up.contains("message 49") && !up.contains("message 59"), "each message takes two lines: {up}");
        v.scroll = 10_000;
        let top = text(&render(&v, 80, 20));
        assert!(top.contains("message 0"), "scrolled past the top shows the top: {top}");
    }

    #[test]
    fn the_cursor_follows_the_editor() {
        assert_eq!(cursor_position("ab\ncd", 1, 80, 20), (5, 16), "after a, on the first of two lines");
        assert_eq!(cursor_position("ab\ncd", 6, 80, 20), (6, 17), "the end of the last line");
        assert_eq!(cursor_position("", 0, 10, 5), (2, 2), "no margin below 21 columns");
    }

    #[test]
    fn a_running_turn_shows_what_it_does_and_for_how_long() {
        let mut s = worked();
        s.running = true;
        s.phase = "reading files".into();
        let mut v = view(&s, true);
        v.working_ms = Some(3_400);
        let screen = text(&render(&v, 100, 30));
        assert!(screen.contains("Leyendo archivos · 3.4s"), "{screen}");
        v.working_ms = Some(125_000);
        assert!(text(&render(&v, 100, 30)).contains("2m 5.0s"));
        let mut idle = view(&s, true);
        idle.working_ms = None;
        assert!(!text(&render(&idle, 100, 30)).contains("Leyendo archivos"));
    }

    #[test]
    fn suggestions_float_above_the_editor_without_pushing_the_chat() {
        let s = worked();
        let mut plain = view(&s, true);
        plain.input = "/mo";
        let before = text(&render(&plain, 100, 30));
        let mut with = view(&s, true);
        with.input = "/mo";
        with.overlay =
            Some(vec![ratatui::text::Line::from("› /model   change the model"), ratatui::text::Line::from("  /exit")]);
        let after = text(&render(&with, 100, 30));
        let (b, a): (Vec<&str>, Vec<&str>) = (before.lines().collect(), after.lines().collect());
        let editor = a.iter().position(|l| l.trim() == "› /mo").expect("the editor stays");
        // Right above the editor's top rule.
        assert!(a[editor - 3].contains("/model") && a[editor - 2].contains("/exit"), "{after}");
        assert_eq!(&b[..editor - 3], &a[..editor - 3], "the conversation above did not move");
        assert_eq!(&b[editor - 1..], &a[editor - 1..], "the editor, its rules and the footer are unchanged");
    }

    #[test]
    fn the_queue_waits_above_the_editor() {
        let mut s = worked();
        s.queue = vec!["and run the tests".into(), "then commit".into()];
        let screen = text(&render(&view(&s, true), 100, 30));
        let rows: Vec<&str> = screen.lines().collect();
        let rule = rows.iter().rposition(|l| l.trim_start().starts_with("──") && l.len() > 40).unwrap();
        let editor_top = rows[..rule].iter().rposition(|l| l.trim_start().starts_with("──")).unwrap();
        assert!(
            rows[editor_top - 2].contains("and run the tests") && rows[editor_top - 1].contains("then commit"),
            "{screen}"
        );
        assert!(rows[editor_top - 2].contains("en cola"), "{screen}");
    }

    #[test]
    fn drawing_says_how_far_it_can_scroll() {
        let mut s = Session::default();
        for n in 0..60 {
            s.add_user(&format!("message {n}"));
        }
        let v = view(&s, true);
        let mut terminal = Terminal::new(TestBackend::new(80, 20)).unwrap();
        let mut drawn = Drawn::default();
        terminal.draw(|frame| drawn = draw_session(frame, &v)).unwrap();
        assert_eq!(drawn.above, max_scroll(&v, 80, 20));
        assert!(drawn.above > 100);
    }

    #[test]
    fn a_frame_is_drawn_only_when_something_changed_or_the_spinner_turns() {
        assert!(should_draw(true, false, 0));
        assert!(!should_draw(false, false, 5_000), "idle: nothing to draw");
        assert!(!should_draw(false, true, 40), "the spinner turns every 80 ms");
        assert!(should_draw(false, true, 80));
    }

    #[test]
    fn the_conversation_breathes_above_the_editor() {
        let mut s = worked();
        s.running = true;
        s.phase = "working".into();
        let mut v = view(&s, true);
        v.working_ms = Some(8_700);
        // Short enough that the conversation fills the screen, as in a real one.
        let screen = text(&render(&v, 100, 14));
        let rows: Vec<&str> = screen.lines().collect();
        let working = rows.iter().position(|l| l.contains("Trabajando")).unwrap();
        assert_eq!(rows[working - 1].trim(), "", "a blank row before the working line: {screen}");
        assert!(rows[working - 2].contains("Listo."), "{screen}");
        let idle = text(&render(&view(&s, true), 100, 13));
        let rows: Vec<&str> = idle.lines().collect();
        let rule = rows.iter().position(|l| l.trim_start().starts_with("──")).unwrap();
        assert_eq!(rows[rule - 1].trim(), "", "a blank row before the editor: {idle}");
        assert!(rows[rule - 2].contains("Listo."), "{idle}");
    }

    #[test]
    fn a_long_output_says_how_much_more_and_where() {
        let mut s = Session::default();
        let output: String = (0..20).map(|n| format!("line {n}\n")).collect();
        for line in [
            event(
                json!({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "bash", "args": {"command": "seq 20"}}),
            ),
            event(json!({"type": "tool_execution_end", "toolCallId": "c1", "toolName": "bash", "isError": false,
                "result": {"content": [{"type": "text", "text": output}]}})),
        ] {
            s.apply(&serde_json::from_value::<DockMessage>(line).unwrap());
        }
        let mut v = view(&s, true);
        let mut terminal = Terminal::new(TestBackend::new(80, 30)).unwrap();
        let mut drawn = Drawn::default();
        terminal.draw(|frame| drawn = draw_session(frame, &v)).unwrap();
        let screen = text(terminal.backend().buffer());
        let row = screen.lines().position(|l| l.contains("8") && l.contains("…")).expect("the more row");
        assert_eq!(drawn.more, vec![(row as u16, "c1".to_string())]);
        assert!(!screen.contains("line 19"));
        v.expanded.insert("c1".into());
        assert!(text(&render(&v, 80, 40)).contains("line 19"));
    }

    #[test]
    fn model_names_are_short() {
        assert_eq!(short_model("claude-bridge/claude-opus-5-5"), "opus 5.5");
        assert_eq!(short_model("opencode-go/mimo-v2.6-pro"), "mimo v2.6 pro");
        assert_eq!(short_model("opencode-go/glm-5.3"), "glm 5.3");
        assert_eq!(short_model("x"), "x");
    }
}
