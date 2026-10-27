//! The full-screen front end of `aria hub init`: a ratatui panel in an inline viewport, the second implementation of
//! [`HubUi`]. It shows the steps of a run with a mark each (waiting, a spinner, done, failed), the sentence of what the
//! current one is for, a progress bar, and at the end a panel that says what is ready and what to type.
//!
//! It is chosen only where it can work ([`wants_tui`]: stdin and stdout are terminals, `TERM` is not dumb, `NO_COLOR` is unset,
//! `--plain` is not given, and the terminal is at least [`MIN_COLS`] by [`MIN_ROWS`]); everywhere else (a pipe, a log, a script,
//! the tests) [`PlainUi`] is. The viewport is inline, so the last screen stays in the scrollback when the run ends, and what is
//! permanent (the notes, the technical plan) is printed above it. Raw mode is on only while a key is awaited. The panel is
//! redrawn from a small thread every ~100 ms from shared state, because the work behind each step is a blocking call in
//! `hub_admin.rs`, which does not know about the screen.
//!
//! A step that gives the terminal to another program (`ssh-keygen`'s passphrase, the installer on the hub over `ssh -tt`) is
//! wrapped by `hand_terminal_over` and `take_terminal_back`: the viewport is cleared and dropped, the program has the terminal,
//! and a new viewport is drawn under what it printed.
//!
//! When a run fails (`failed`) or ends, the panel is printed once into the scrollback (`finish`) and the cursor goes back under
//! it, so what `hub_admin.rs` writes to stderr next lands below it and the rest of the run goes through the text version.

use crate::hub_ui::{wrap, HubUi, Note, NoteLine, Outcome, PlainUi, Plan, WIDTH};
use crate::term::theme::Palette;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::backend::CrosstermBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph, Widget};
use ratatui::{Terminal, TerminalOptions, Viewport};
use std::io::{self, IsTerminal};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

/// The narrowest and shortest terminal the full-screen version is used on.
pub const MIN_COLS: u16 = 60;
pub const MIN_ROWS: u16 = 16;
/// The widest the panel gets, so it does not stretch across a wide terminal.
const PANEL_WIDTH: u16 = 80;
/// The rows the inline viewport asks for (the panel and one empty row under it where the cursor waits).
const VIEWPORT_ROWS: u16 = 22;
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Whether this run gets the full-screen version, from what the terminal says. A pure function of its arguments.
pub fn wants_tui(
    stdin_tty: bool,
    stdout_tty: bool,
    term: &str,
    no_color: bool,
    force_plain: bool,
    cols: u16,
    rows: u16,
) -> bool {
    stdin_tty
        && stdout_tty
        && !term.is_empty()
        && term != "dumb"
        && !no_color
        && !force_plain
        && cols >= MIN_COLS
        && rows >= MIN_ROWS
}

/// The full-screen version for this terminal, or None where [`wants_tui`] says no (or the terminal cannot say how big it is).
pub fn for_terminal(force_plain: bool) -> Option<TuiUi> {
    let (cols, rows) = crossterm::terminal::size().unwrap_or((0, 0));
    let no_color = std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty());
    let term = std::env::var("TERM").unwrap_or_default();
    if wants_tui(io::stdin().is_terminal(), io::stdout().is_terminal(), &term, no_color, force_plain, cols, rows) {
        Some(TuiUi::new())
    } else {
        None
    }
}

// ---- what is on the screen: data, and a pure function from it to lines --------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum RowState {
    Waiting,
    Running,
    Done,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    /// The hub is being read.
    Looking,
    /// The plan is shown.
    Plan,
    /// The steps run.
    Running,
    /// Over: what is ready, or what failed.
    Finished,
}

#[derive(Debug, Clone, PartialEq)]
enum Check {
    Ok(String),
    Info(String),
}

/// Everything the panel shows. The events of a run change it; [`panel`] turns it into lines.
#[derive(Debug, Clone)]
pub(crate) struct View {
    title: String,
    host: String,
    facts: Vec<String>,
    phase: Phase,
    plan: Option<Plan>,
    rows: Vec<(String, RowState)>,
    why: String,
    detail: Vec<Check>,
    failure: Option<String>,
    outcome: Option<Outcome>,
    tick: usize,
    /// Show "Enter to start" (the plan is waiting for an answer).
    footer: bool,
    /// False in the copy that goes into the scrollback: no spinner, no footer.
    live: bool,
}

impl View {
    pub(crate) fn new() -> View {
        View {
            title: String::new(),
            host: String::new(),
            facts: Vec::new(),
            phase: Phase::Looking,
            plan: None,
            rows: Vec::new(),
            why: String::new(),
            detail: Vec::new(),
            failure: None,
            outcome: None,
            tick: 0,
            footer: false,
            live: true,
        }
    }

    fn animating(&self) -> bool {
        self.live
            && match self.phase {
                Phase::Looking => true,
                Phase::Running => self.rows.iter().any(|(_, state)| *state == RowState::Running),
                _ => false,
            }
    }

    fn set_plan(&mut self, plan: &Plan) {
        self.host = plan.host.clone();
        self.rows = plan.steps.iter().map(|step| (step.clone(), RowState::Waiting)).collect();
        self.plan = Some(plan.clone());
        self.phase = Phase::Plan;
    }

    fn step_start(&mut self, k: usize, n: usize, title: &str, why: &str) {
        while self.rows.len() < n.max(k) {
            let at = self.rows.len() + 1;
            self.rows.push((format!("Step {at}"), RowState::Waiting));
        }
        if self.rows[k - 1].0.is_empty() {
            self.rows[k - 1].0 = title.to_string();
        }
        // A step that was skipped over, or finished without a word, is done.
        for row in self.rows.iter_mut().take(k - 1) {
            if row.1 == RowState::Running || row.1 == RowState::Waiting {
                row.1 = RowState::Done;
            }
        }
        self.rows[k - 1].1 = RowState::Running;
        self.why = why.to_string();
        self.detail.clear();
        self.phase = Phase::Running;
        self.footer = false;
    }

    fn running(&self) -> Option<usize> {
        self.rows.iter().position(|(_, state)| *state == RowState::Running)
    }

    fn fail(&mut self, line: &str) {
        if let Some(at) = self.running() {
            self.rows[at].1 = RowState::Failed;
        }
        self.failure = Some(line.to_string());
        self.phase = Phase::Finished;
        self.footer = false;
    }

    fn finish(&mut self, outcome: &Outcome) {
        for row in self.rows.iter_mut() {
            if row.1 != RowState::Failed {
                row.1 = RowState::Done;
            }
        }
        self.outcome = Some(outcome.clone());
        self.phase = Phase::Finished;
        self.footer = false;
    }
}

fn style(color: ratatui::style::Color) -> Style {
    Style::new().fg(color)
}

fn bold(color: ratatui::style::Color) -> Style {
    Style::new().fg(color).add_modifier(Modifier::BOLD)
}

/// `lead` and then `text` wrapped to the width, the continuation lines indented under the text.
fn hang(lead: Vec<Span<'static>>, cols: usize, text: &str, text_style: Style, width: usize) -> Vec<Line<'static>> {
    let lines = wrap(text, width.saturating_sub(cols).max(10));
    if lines.is_empty() {
        return vec![Line::from(lead)];
    }
    lines
        .into_iter()
        .enumerate()
        .map(|(at, text)| {
            let mut spans = if at == 0 { lead.clone() } else { vec![Span::raw(" ".repeat(cols))] };
            spans.push(Span::styled(text, text_style));
            Line::from(spans)
        })
        .collect()
}

fn blank() -> Line<'static> {
    Line::from("")
}

fn glyph(state: RowState, tick: usize, live: bool, p: &Palette) -> Span<'static> {
    match state {
        RowState::Waiting => Span::styled("○", style(p.faint)),
        RowState::Running if live => Span::styled(SPINNER[tick % SPINNER.len()], style(p.accent)),
        RowState::Running => Span::styled("◌", style(p.grey)),
        RowState::Done => Span::styled("✓", style(p.green)),
        RowState::Failed => Span::styled("✗", style(p.coral)),
    }
}

/// The thin progress line: `━━━╸━━  2 of 5`.
fn progress(v: &View, p: &Palette, width: usize) -> Line<'static> {
    let n = v.rows.len().max(1);
    let done = v.rows.iter().filter(|(_, state)| *state == RowState::Done).count();
    let running = v.running();
    let label = format!("{} of {n}", running.map(|at| at + 1).unwrap_or(done));
    let bar = width.saturating_sub(label.chars().count() + 6).max(8);
    let fraction = (done as f64 + if running.is_some() { 0.5 } else { 0.0 }) / n as f64;
    let filled = ((fraction * bar as f64).round() as usize).min(bar);
    let all_done = done == n;
    let (head, rest) = if all_done { (p.green, p.green) } else { (p.accent, p.faint) };
    let mut spans = vec![Span::raw("  ")];
    if all_done {
        spans.push(Span::styled("━".repeat(bar), style(head)));
    } else {
        spans.push(Span::styled("━".repeat(filled.saturating_sub(1)), style(head)));
        spans.push(Span::styled(if filled > 0 { "╸" } else { "" }, style(head)));
        spans.push(Span::styled("━".repeat(bar - filled.max(1) + usize::from(filled == 0)), style(rest)));
    }
    spans.push(Span::styled(format!("  {label}"), style(p.grey)));
    Line::from(spans)
}

/// The note as lines: `!` and its sentences, a command set apart.
fn note_lines(note: &Note, p: &Palette, width: usize) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    for (at, line) in note.lines.iter().enumerate() {
        match line {
            NoteLine::Text(text) => {
                let lead = if at == 0 { Span::styled("  ! ", bold(p.amber)) } else { Span::raw("    ") };
                out.extend(hang(vec![lead], 4, text, style(p.text), width));
            }
            NoteLine::Command(command) => {
                out.push(Line::from(vec![Span::raw("        "), Span::styled(command.clone(), bold(p.text))]));
            }
        }
    }
    out
}

fn header_line(v: &View, p: &Palette) -> Line<'static> {
    let mut spans = vec![Span::raw(" "), Span::styled(v.host.clone(), bold(p.text))];
    if let Some(first) = v.facts.first() {
        spans.push(Span::styled(format!("  ·  {first}"), style(p.grey)));
    }
    Line::from(spans)
}

/// What the Plan phase shows, with `facts` ("✓ ..." lines) and the first `bullets` of "Good to know" kept.
fn plan_lines(v: &View, p: &Palette, width: usize, facts: bool, bullets: usize) -> Vec<Line<'static>> {
    let mut out = vec![header_line(v, p), blank()];
    if facts && v.facts.len() > 1 {
        for fact in &v.facts[1..] {
            out.extend(hang(vec![Span::raw("  "), Span::styled("✓ ", style(p.green))], 4, fact, style(p.text), width));
        }
        out.push(blank());
    }
    out.push(Line::from(Span::styled("  What I will do", bold(p.text))));
    for (at, (title, _)) in v.rows.iter().enumerate() {
        let lead = vec![Span::raw("    "), Span::styled(format!("{}  ", at + 1), bold(p.accent))];
        out.extend(hang(lead, 7, title, style(p.text), width));
    }
    if let Some(plan) = &v.plan {
        if !plan.good_to_know.is_empty() {
            out.push(blank());
            out.push(Line::from(Span::styled("  Good to know", bold(p.text))));
            for item in plan.good_to_know.iter().take(bullets) {
                out.extend(hang(
                    vec![Span::raw("    "), Span::styled("· ", style(p.grey))],
                    6,
                    item,
                    style(p.grey),
                    width,
                ));
            }
            if bullets < plan.good_to_know.len() {
                out.push(Line::from(Span::styled("    · …", style(p.grey))));
            }
        }
    }
    if v.footer {
        out.push(blank());
        out.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("Enter", bold(p.accent)),
            Span::styled(" to start  ·  ", style(p.grey)),
            Span::styled("Ctrl+C", bold(p.accent)),
            Span::styled(" to cancel", style(p.grey)),
        ]));
    }
    out
}

/// The body of the panel, at most `budget` lines where something can be left out (the plan's extras, the current step's
/// check lines).
fn body(v: &View, p: &Palette, width: usize, budget: usize) -> Vec<Line<'static>> {
    match v.phase {
        Phase::Looking => {
            let mut out = vec![header_line(v, p), blank()];
            if v.facts.len() > 1 {
                for fact in &v.facts[1..] {
                    out.extend(hang(
                        vec![Span::raw("  "), Span::styled("✓ ", style(p.green))],
                        4,
                        fact,
                        style(p.text),
                        width,
                    ));
                }
            }
            let frame = if v.live { SPINNER[v.tick % SPINNER.len()] } else { "·" };
            out.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("{frame} "), style(p.accent)),
                Span::styled(format!("Looking at {}...", v.host), style(p.grey)),
            ]));
            out
        }
        Phase::Plan => {
            let total = v.plan.as_ref().map_or(0, |plan| plan.good_to_know.len());
            let mut bullets = total;
            let mut facts = true;
            loop {
                let lines = plan_lines(v, p, width, facts, bullets);
                if lines.len() <= budget || (!facts && bullets == 0) {
                    return lines;
                }
                if facts {
                    facts = false;
                } else {
                    bullets -= 1;
                }
            }
        }
        Phase::Running | Phase::Finished => {
            let mut out = vec![header_line(v, p), blank()];
            for (at, (title, state)) in v.rows.iter().enumerate() {
                let title_style = match state {
                    RowState::Running => bold(p.text),
                    RowState::Waiting => style(p.grey),
                    RowState::Failed => style(p.coral),
                    RowState::Done => style(p.text),
                };
                let lead = vec![
                    Span::raw("  "),
                    glyph(*state, v.tick, v.live, p),
                    Span::styled(format!(" {}  ", at + 1), style(p.faint)),
                ];
                out.extend(hang(lead, 7, title, title_style, width));
            }
            // A run with no steps (every server already current) has no bar to show.
            let mut tail = if v.rows.is_empty() { Vec::new() } else { vec![blank(), progress(v, p, width)] };
            if v.phase == Phase::Finished {
                tail.push(blank());
                match (&v.failure, &v.outcome) {
                    (Some(failure), _) => {
                        tail.extend(hang(
                            vec![Span::raw("  "), Span::styled("✗ ", bold(p.coral))],
                            4,
                            failure,
                            bold(p.coral),
                            width,
                        ));
                    }
                    (None, Some(Outcome::Ready { connect })) => {
                        tail.push(Line::from(vec![
                            Span::raw("  "),
                            Span::styled("✓ ", bold(p.green)),
                            Span::styled("Your server is ready", bold(p.green)),
                        ]));
                        tail.push(blank());
                        tail.push(Line::from(Span::styled("  To use it, type:", style(p.grey))));
                        tail.push(blank());
                        tail.push(Line::from(vec![
                            Span::raw("      "),
                            Span::styled(
                                format!("  {connect}  "),
                                Style::new().fg(p.ground).bg(p.green).add_modifier(Modifier::BOLD),
                            ),
                        ]));
                    }
                    (None, Some(Outcome::NotReady { why })) => {
                        tail.extend(hang(
                            vec![Span::raw("  "), Span::styled("! ", bold(p.amber))],
                            4,
                            &format!("The server is installed, but not ready: {why}"),
                            bold(p.amber),
                            width,
                        ));
                    }
                    (None, Some(Outcome::Updated { runs })) => {
                        for (host, version) in runs {
                            tail.extend(hang(
                                vec![Span::raw("  "), Span::styled("✓ ", bold(p.green))],
                                4,
                                &format!("{host} runs Aria {version}"),
                                bold(p.green),
                                width,
                            ));
                        }
                    }
                    (None, None) => {}
                }
            }
            // What the current step says goes between the list and the bar, as much as fits.
            let mut detail: Vec<Line<'static>> = Vec::new();
            if v.phase == Phase::Running {
                if !v.why.is_empty() {
                    detail.extend(hang(vec![Span::raw("    ")], 4, &v.why, style(p.grey), width));
                }
                let mut lines: Vec<Line<'static>> = Vec::new();
                for line in &v.detail {
                    match line {
                        Check::Ok(text) => lines.extend(hang(
                            vec![Span::raw("    "), Span::styled("✓ ", style(p.green))],
                            6,
                            text,
                            style(p.text),
                            width,
                        )),
                        Check::Info(text) => lines.extend(hang(vec![Span::raw("    ")], 4, text, style(p.grey), width)),
                    }
                }
                let room = budget.saturating_sub(out.len() + tail.len() + 1);
                // The newest check lines win over the oldest, and over the sentence of what the step is for.
                if detail.len() + lines.len() > room {
                    let keep = room.min(lines.len());
                    lines = lines.split_off(lines.len() - keep);
                    detail.truncate(room - keep);
                }
                detail.extend(lines);
                if !detail.is_empty() {
                    out.push(blank());
                    out.extend(detail);
                }
            }
            out.extend(tail);
            out
        }
    }
}

fn border_color(v: &View, p: &Palette) -> ratatui::style::Color {
    match (&v.failure, &v.outcome) {
        (Some(_), _) => p.coral,
        (None, Some(Outcome::Ready { .. })) => p.green,
        (None, Some(Outcome::NotReady { .. })) => p.amber,
        (None, Some(Outcome::Updated { .. })) => p.green,
        _ => p.accent,
    }
}

/// The panel: its lines, and how tall it is with the border. At most `max_height` rows where the content can be cut.
fn panel(v: &View, p: &Palette, width: u16, max_height: u16) -> (Vec<Line<'static>>, u16) {
    let inner = width.saturating_sub(2) as usize;
    let budget = max_height.saturating_sub(2) as usize;
    let lines = body(v, p, inner, budget);
    let height = (lines.len() as u16).saturating_add(2);
    (lines, height)
}

fn render_panel(v: &View, p: &Palette, lines: Vec<Line<'static>>, area: Rect, buf: &mut Buffer) {
    let title = Line::from(vec![
        Span::raw(" "),
        Span::styled("Aria", bold(p.accent)),
        Span::styled(format!(" · {} ", v.title), style(p.grey)),
    ]);
    let block = Block::bordered().border_type(BorderType::Rounded).border_style(style(border_color(v, p))).title(title);
    Paragraph::new(lines).block(block).render(area, buf);
}

/// The panel of `v` drawn on a buffer as wide as `width`, as tall as it needs (at most `max_height`).
fn draw_view(v: &View, p: &Palette, width: u16, max_height: u16, buf_area: Rect, buf: &mut Buffer) {
    let width = width.min(PANEL_WIDTH).min(buf_area.width);
    let (lines, height) = panel(v, p, width, max_height.min(buf_area.height));
    render_panel(v, p, lines, Rect::new(buf_area.x, buf_area.y, width, height.min(buf_area.height)), buf);
}

/// Permanent text above the panel: lines that are printed once and stay.
fn permanent(lines: Vec<Line<'static>>) -> (u16, Paragraph<'static>) {
    (lines.len().min(u16::MAX as usize) as u16, Paragraph::new(lines))
}

// ---- the thing on the terminal ------------------------------------------------------------------------------------------

type Term = Terminal<CrosstermBackend<io::Stdout>>;

struct Inner {
    view: View,
    palette: Palette,
    term: Option<Term>,
    /// The terminal could not be set up, or the run ended: the text version takes over.
    finished: bool,
    started: bool,
}

fn new_terminal() -> io::Result<Term> {
    let rows = crossterm::terminal::size().map(|(_, rows)| rows).unwrap_or(VIEWPORT_ROWS);
    let height = VIEWPORT_ROWS.min(rows.saturating_sub(1)).max(MIN_ROWS - 1);
    Terminal::with_options(CrosstermBackend::new(io::stdout()), TerminalOptions { viewport: Viewport::Inline(height) })
}

impl Inner {
    fn draw(&mut self) {
        let Inner { view, palette, term, .. } = self;
        let Some(term) = term.as_mut() else {
            return;
        };
        // The viewport, as the frame sees it (the completed frame's own area is the whole terminal).
        let mut viewport = Rect::default();
        let drawn = term.draw(|frame| {
            let area = frame.area();
            viewport = area;
            // The last row stays empty: the cursor waits there, so what is written to stderr lands under the panel.
            let max = area.height.saturating_sub(1);
            draw_view(view, palette, area.width, max, area, frame.buffer_mut());
        });
        if drawn.is_ok() {
            let _ = term.set_cursor_position(Position::new(viewport.x, viewport.bottom().saturating_sub(1)));
        }
    }

    /// Prints `lines` once above the panel.
    fn print_above(&mut self, lines: Vec<Line<'static>>) {
        if lines.is_empty() {
            return;
        }
        let (height, paragraph) = permanent(lines);
        if let Some(term) = self.term.as_mut() {
            let _ = term.insert_before(height, |buf| {
                let area = buf.area;
                paragraph.render(area, buf);
            });
        }
        self.draw();
    }

    /// Puts the panel into the scrollback as it is, gives the terminal back (the cursor under it, visible) and hands the rest
    /// of the run to the text version.
    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.view.live = false;
        self.view.footer = false;
        if let Some(term) = self.term.as_mut() {
            let width = term.size().map(|size| size.width).unwrap_or(PANEL_WIDTH);
            let rows = term.size().map(|size| size.height).unwrap_or(VIEWPORT_ROWS);
            let width = width.min(PANEL_WIDTH);
            let (lines, height) = panel(&self.view, &self.palette, width, rows.saturating_sub(1).max(MIN_ROWS));
            let (view, palette) = (&self.view, &self.palette);
            let _ = term.insert_before(height, |buf| {
                let area = Rect::new(0, 0, width, height.min(buf.area.height));
                render_panel(view, palette, lines, area, buf);
            });
            let _ = term.clear();
        }
        self.term = None; // dropping it shows the cursor again
        self.finished = true;
        uninstall_interrupt();
    }
}

extern "C" fn on_interrupt(_signal: libc::c_int) {
    // Only what a signal handler may do: give the cursor back and leave.
    let show = b"\x1b[?25h";
    unsafe {
        libc::write(1, show.as_ptr().cast(), show.len());
        libc::_exit(130);
    }
}

/// Ctrl+C while the panel is up (outside the question, where raw mode makes it a key): show the cursor, which the panel hides,
/// and exit with the status a shell gives it.
fn install_interrupt() {
    unsafe {
        libc::signal(libc::SIGINT, on_interrupt as *const () as libc::sighandler_t);
    }
}

fn uninstall_interrupt() {
    unsafe {
        libc::signal(libc::SIGINT, libc::SIG_DFL);
    }
}

/// A panic gives the terminal back before the message is printed.
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = io::Write::write_all(&mut io::stdout(), b"\x1b[?25h");
        previous(info);
    }));
}

/// What a key means to the question: Some(true) to go on, Some(false) to cancel, None for any other.
pub(crate) fn key_answer(key: &KeyEvent) -> Option<bool> {
    if key.kind == KeyEventKind::Release {
        return None;
    }
    match (key.code, key.modifiers) {
        (KeyCode::Enter, _) => Some(true),
        (KeyCode::Char('c'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => Some(false),
        (KeyCode::Char('q' | 'n' | 'Q' | 'N'), _) | (KeyCode::Esc, _) => Some(false),
        _ => None,
    }
}

/// Waits for Enter or a way out, in raw mode only while it waits. The panel keeps being redrawn by its own thread.
fn read_answer() -> bool {
    if enable_raw_mode().is_err() {
        return false;
    }
    let answer = loop {
        match event::poll(Duration::from_millis(100)) {
            Ok(true) => match event::read() {
                Ok(Event::Key(key)) => {
                    if let Some(answer) = key_answer(&key) {
                        break answer;
                    }
                }
                Ok(_) => {}
                Err(_) => break false,
            },
            Ok(false) => {}
            Err(_) => break false,
        }
    };
    let _ = disable_raw_mode();
    answer
}

/// The full-screen screen of a run. The text version is kept for what comes after the panel is printed for good.
pub struct TuiUi {
    inner: Arc<Mutex<Inner>>,
    stop: Arc<AtomicBool>,
    ticker: Option<JoinHandle<()>>,
    plain: PlainUi,
}

impl TuiUi {
    pub fn new() -> TuiUi {
        let inner = Arc::new(Mutex::new(Inner {
            view: View::new(),
            palette: Palette::detect(),
            term: None,
            finished: false,
            started: false,
        }));
        TuiUi { inner, stop: Arc::new(AtomicBool::new(false)), ticker: None, plain: PlainUi::stdio() }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Sets the terminal up on the first event: the viewport, the interrupt, the thread that animates the panel.
    fn start(&mut self) {
        let mut inner = self.lock();
        if inner.started {
            return;
        }
        inner.started = true;
        match new_terminal() {
            Ok(term) => {
                inner.term = Some(term);
                install_panic_hook();
                install_interrupt();
            }
            Err(_) => {
                inner.finished = true;
                return;
            }
        }
        drop(inner);
        let (shared, stop) = (self.inner.clone(), self.stop.clone());
        self.ticker = Some(std::thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(100));
                let mut inner = shared.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                if inner.finished {
                    break;
                }
                if inner.term.is_some() && inner.view.animating() {
                    inner.view.tick += 1;
                    inner.draw();
                }
            }
        }));
    }

    /// Runs `change` on the view and draws; false when the text version has the screen now.
    fn update(&mut self, change: impl FnOnce(&mut View)) -> bool {
        self.start();
        let mut inner = self.lock();
        if inner.finished {
            return false;
        }
        change(&mut inner.view);
        inner.draw();
        true
    }

    fn done_with_panel(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.lock().finish();
        if let Some(handle) = self.ticker.take() {
            let _ = handle.join();
        }
    }
}

impl Default for TuiUi {
    fn default() -> TuiUi {
        TuiUi::new()
    }
}

impl Drop for TuiUi {
    fn drop(&mut self) {
        self.done_with_panel();
    }
}

impl HubUi for TuiUi {
    fn title(&mut self, text: &str) {
        let text = text.to_string();
        if !self.update(|view| view.title = text.clone()) {
            self.plain.title(&text);
        }
    }

    fn looking(&mut self, host: &str) {
        let host = host.to_string();
        if !self.update(|view| view.host = host.clone()) {
            self.plain.looking(&host);
        }
    }

    fn probe_ok(&mut self, line: &str) {
        let line = line.to_string();
        if !self.update(|view| view.facts.push(line.clone())) {
            self.plain.probe_ok(&line);
        }
    }

    fn probe_info(&mut self, line: &str) {
        // Said where it happens: a line above the panel would be gone by the time it matters.
        let mut inner = self.lock();
        if inner.finished {
            drop(inner);
            self.plain.probe_info(line);
            return;
        }
        let p = inner.palette;
        inner.print_above(vec![Line::from(vec![Span::raw("  "), Span::styled(line.to_string(), style(p.grey))])]);
    }

    fn probe_note(&mut self, note: &Note) {
        self.start();
        let mut inner = self.lock();
        if inner.finished {
            drop(inner);
            self.plain.probe_note(note);
            return;
        }
        let p = inner.palette;
        let mut lines = vec![blank()];
        lines.extend(note_lines(note, &p, WIDTH));
        inner.print_above(lines);
    }

    fn details(&mut self, paragraphs: &[String]) {
        self.start();
        let mut inner = self.lock();
        if inner.finished {
            drop(inner);
            self.plain.details(paragraphs);
            return;
        }
        let mut lines = Vec::new();
        for paragraph in paragraphs {
            lines.extend(wrap(paragraph, WIDTH).into_iter().map(Line::from));
        }
        inner.print_above(lines);
    }

    fn plan(&mut self, plan: &Plan) {
        if !self.update(|view| view.set_plan(plan)) {
            self.plain.plan(plan);
        }
    }

    fn confirm(&mut self) -> bool {
        if !self.update(|view| view.footer = true) {
            return self.plain.confirm();
        }
        let answer = read_answer();
        let mut inner = self.lock();
        inner.view.footer = false;
        inner.draw();
        answer
    }

    fn declined(&mut self) {
        self.done_with_panel();
        self.plain.declined();
    }

    fn step_start(&mut self, k: usize, n: usize, title: &str, why: &str) {
        let (title, why) = (title.to_string(), why.to_string());
        if !self.update(|view| view.step_start(k, n, &title, &why)) {
            self.plain.step_start(k, n, &title, &why);
        }
    }

    fn step_ok(&mut self, line: &str) {
        let line = line.to_string();
        if !self.update(|view| view.detail.push(Check::Ok(line.clone()))) {
            self.plain.step_ok(&line);
        }
    }

    fn step_info(&mut self, line: &str) {
        let line = line.to_string();
        if !self.update(|view| view.detail.push(Check::Info(line.clone()))) {
            self.plain.step_info(&line);
        }
    }

    fn failed(&mut self, line: &str) {
        // The error text that follows goes to stderr, under the panel: the panel is printed for good first.
        let finished = self.lock().finished;
        if finished {
            self.plain.failed(line);
            return;
        }
        let line_owned = line.to_string();
        self.update(|view| view.fail(&line_owned));
        self.done_with_panel();
    }

    fn heads_up(&mut self, notes: &[Note]) {
        if notes.is_empty() {
            return;
        }
        self.start();
        let mut inner = self.lock();
        if inner.finished {
            drop(inner);
            self.plain.heads_up(notes);
            return;
        }
        let p = inner.palette;
        let mut lines = vec![blank(), Line::from(Span::styled("  Heads up", bold(p.text)))];
        for note in notes {
            lines.extend(note_lines(note, &p, WIDTH));
        }
        inner.print_above(lines);
    }

    fn done(&mut self, outcome: &Outcome) {
        if self.lock().finished {
            self.plain.done(outcome);
            return;
        }
        let outcome = outcome.clone();
        self.update(|view| view.finish(&outcome));
        self.done_with_panel();
    }

    fn hand_terminal_over(&mut self) {
        let mut inner = self.lock();
        if inner.finished {
            return;
        }
        // What a step said about the program that takes the terminal ("its steps are printed below") is stale once it ran.
        inner.view.detail.retain(|line| !matches!(line, Check::Info(_)));
        if let Some(mut term) = inner.term.take() {
            let _ = term.clear();
            // Dropping it shows the cursor, which is where the viewport started.
        }
    }

    fn take_terminal_back(&mut self) {
        let mut inner = self.lock();
        if inner.finished || inner.term.is_some() {
            return;
        }
        match new_terminal() {
            Ok(term) => {
                inner.term = Some(term);
                inner.draw();
            }
            Err(_) => inner.finished = true,
        }
    }
}

// ---- the demo -----------------------------------------------------------------------------------------------------------

/// `aria hub ui-demo [--fail]`: this screen driven through a made-up run, with no ssh and nothing changed, so it can be looked
/// at (`--fail` ends in the failed one). Development only.
pub fn demo(args: &[String]) -> i32 {
    let fail = args.iter().any(|arg| arg == "--fail");
    let plain = args.iter().any(|arg| arg == "--plain");
    let mut ui = crate::hub_ui::ui_for_terminal(plain);
    let pause = |ms: u64| std::thread::sleep(Duration::from_millis(ms));
    let host = "ana@hub.example";
    ui.title("linking this computer to your server");
    ui.looking(host);
    pause(700);
    ui.probe_ok("Ubuntu 24.04.4 LTS, aarch64");
    pause(500);
    ui.probe_ok("Aria 0.21.0 already runs there");
    pause(500);
    ui.probe_ok("Your settings and sign-ins are ready to send");
    pause(500);
    let steps = [
        "Make a key only for Aria",
        "Install Aria on the server",
        "Copy your settings and sign-ins",
        "Allow that key on the server, for Aria's terminal only",
        "Test the connection",
    ];
    ui.plan(&Plan {
        host: host.to_string(),
        steps: steps.iter().map(|step| step.to_string()).collect(),
        good_to_know: vec![
            "Your sign-ins are copied to the server, which keeps and uses them.".to_string(),
            "The key can open Aria and nothing else: it is not a login.".to_string(),
            "The server's sudo installs the sandbox tools and turns on linger. Nothing else outside its home folder changes."
                .to_string(),
        ],
        blocked: Vec::new(),
    });
    if !ui.confirm() {
        ui.declined();
        return 0;
    }
    let n = steps.len();
    ui.step_start(
        1,
        n,
        "Making a key only for Aria",
        "It asks for a passphrase: it protects the key if someone copies it.",
    );
    ui.hand_terminal_over();
    println!("  (the passphrase question would be here)");
    pause(900);
    ui.take_terminal_back();
    ui.step_ok("The key is ~/.ssh/aria-hub");
    pause(600);
    ui.step_start(
        2,
        n,
        "Installing Aria on the server",
        "Uploads the build you gave me, then runs the installer there.",
    );
    pause(700);
    ui.step_ok("Uploaded the build");
    ui.step_info("The server's installer prints its own steps below.");
    ui.hand_terminal_over();
    for line in
        ["  - Checking this server", "  - Installing what Aria needs", "  - Downloading Aria", "  - Installing Aria"]
    {
        println!("{line}");
        pause(350);
    }
    ui.take_terminal_back();
    ui.step_ok("Aria is installed on the server");
    pause(600);
    ui.step_start(
        3,
        n,
        "Copying your settings and sign-ins",
        "So the server can use the same models without you typing a key twice.",
    );
    pause(800);
    ui.step_ok("Copied: settings and sign-ins");
    ui.step_ok("The server runs Aria 0.21.0");
    pause(600);
    ui.step_start(
        4,
        n,
        "Allowing the key on the server",
        "The key can open Aria's terminal and nothing else. This computer remembers the server.",
    );
    pause(800);
    if fail {
        ui.failed("The key was not allowed: the server's sandbox does not work.");
        eprintln!("aria hub init: step 4 (key) skipped: the sandbox does not work on the hub");
        return 1;
    }
    ui.step_ok("The key is allowed on the server, for Aria only");
    ui.step_ok("This computer remembers the server (~/.aria/hubs.json)");
    pause(600);
    ui.step_start(5, n, "Testing the connection", "Opens a connection the way `aria connect` does, then closes it.");
    pause(900);
    ui.step_ok("It works");
    pause(400);
    ui.heads_up(&[Note::text(
        "ssh on hub.example (port 22) is open to the internet, and it stays open until you close it yourself.",
    )
    .then_text("Aria never edits sshd_config, the firewall or Tailscale.")]);
    ui.done(&Outcome::Ready { connect: "aria connect".to_string() });
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    fn palette() -> Palette {
        Palette::truecolor()
    }

    fn sample_plan() -> Plan {
        Plan {
            host: "ana@hub.example".to_string(),
            steps: vec![
                "Make a key only for Aria".to_string(),
                "Install Aria on the server".to_string(),
                "Copy your settings and sign-ins".to_string(),
                "Allow that key on the server".to_string(),
                "Test the connection".to_string(),
            ],
            good_to_know: vec![
                "Your sign-ins are copied to the server, which keeps and uses them.".to_string(),
                "The key can open Aria and nothing else: it is not a login.".to_string(),
                "The server's sudo installs the sandbox tools and turns on linger. Nothing else outside its home folder changes."
                    .to_string(),
            ],
            blocked: Vec::new(),
        }
    }

    fn planned() -> View {
        let mut view = View::new();
        view.title = "linking this computer to your server".to_string();
        view.host = "ana@hub.example".to_string();
        view.facts =
            vec!["Ubuntu 24.04.4 LTS, aarch64".to_string(), "Your settings and sign-ins are ready to send".to_string()];
        view.set_plan(&sample_plan());
        view
    }

    fn running_step_two() -> View {
        let mut view = planned();
        view.step_start(1, 5, "Making a key only for Aria", "It asks for a passphrase.");
        view.detail.push(Check::Ok("The key is ~/.ssh/aria-hub".to_string()));
        view.step_start(
            2,
            5,
            "Installing Aria on the server",
            "Uploads the build you gave me, then runs the installer there.",
        );
        view.detail.push(Check::Ok("Uploaded the build".to_string()));
        view
    }

    /// The panel of `view` on a `cols` by `rows` test terminal, one string per row, right-trimmed.
    fn render(view: &View, cols: u16, rows: u16) -> Vec<String> {
        let mut terminal = Terminal::new(TestBackend::new(cols, rows)).unwrap();
        let p = palette();
        terminal
            .draw(|frame| {
                let area = frame.area();
                // One row less, as the live panel has: the cursor waits on the last one.
                draw_view(view, &p, area.width, area.height.saturating_sub(1), area, frame.buffer_mut());
            })
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        (0..rows)
            .map(|y| (0..cols).map(|x| buffer[(x, y)].symbol().to_string()).collect::<String>().trim_end().to_string())
            .collect()
    }

    fn panel_rows(rows: &[String]) -> Vec<&String> {
        rows.iter().filter(|row| !row.is_empty()).collect()
    }

    fn text_of(rows: &[String]) -> String {
        rows.join("\n")
    }

    #[test]
    fn the_tui_is_chosen_only_on_a_terminal_that_can_hold_it() {
        let ok = |stdin, stdout, term: &str, no_color, plain, cols, rows| {
            wants_tui(stdin, stdout, term, no_color, plain, cols, rows)
        };
        assert!(ok(true, true, "xterm-256color", false, false, 80, 24));
        assert!(ok(true, true, "xterm", false, false, MIN_COLS, MIN_ROWS), "the smallest it accepts");
        assert!(!ok(false, true, "xterm", false, false, 80, 24), "no terminal on stdin (a script, a pipe in)");
        assert!(!ok(true, false, "xterm", false, false, 80, 24), "no terminal on stdout (a pipe out, a log)");
        assert!(!ok(true, true, "dumb", false, false, 80, 24));
        assert!(!ok(true, true, "", false, false, 80, 24), "TERM unset");
        assert!(!ok(true, true, "xterm", true, false, 80, 24), "NO_COLOR");
        assert!(!ok(true, true, "xterm", false, true, 80, 24), "--plain");
        assert!(!ok(true, true, "xterm", false, false, MIN_COLS - 1, 24), "too narrow");
        assert!(!ok(true, true, "xterm", false, false, 80, MIN_ROWS - 1), "too short");
    }

    #[test]
    fn a_test_run_never_gets_the_tui() {
        // The tests have no terminal on stdin or stdout.
        assert!(for_terminal(false).is_none());
    }

    #[test]
    fn keys_mean_go_on_or_cancel() {
        let key = |code, modifiers| KeyEvent::new(code, modifiers);
        assert_eq!(key_answer(&key(KeyCode::Enter, KeyModifiers::NONE)), Some(true));
        assert_eq!(key_answer(&key(KeyCode::Char('c'), KeyModifiers::CONTROL)), Some(false));
        assert_eq!(key_answer(&key(KeyCode::Char('q'), KeyModifiers::NONE)), Some(false));
        assert_eq!(key_answer(&key(KeyCode::Char('n'), KeyModifiers::NONE)), Some(false));
        assert_eq!(key_answer(&key(KeyCode::Esc, KeyModifiers::NONE)), Some(false));
        assert_eq!(key_answer(&key(KeyCode::Char('x'), KeyModifiers::NONE)), None);
        assert_eq!(key_answer(&key(KeyCode::Char('c'), KeyModifiers::NONE)), None, "a c without Ctrl is not a way out");
        let mut released = key(KeyCode::Enter, KeyModifiers::NONE);
        released.kind = KeyEventKind::Release;
        assert_eq!(key_answer(&released), None, "a key coming back up is not another press");
    }

    #[test]
    fn the_plan_lists_the_steps_what_to_know_and_how_to_go_on() {
        let mut view = planned();
        view.footer = true;
        let rows = render(&view, 80, 24);
        let text = text_of(&rows);
        for want in [
            "Aria · linking this computer to your server",
            "ana@hub.example  ·  Ubuntu 24.04.4 LTS, aarch64",
            "What I will do",
            "1  Make a key only for Aria",
            "5  Test the connection",
            "Good to know",
            "Your sign-ins are copied to the server",
            "it is not a login.",
            "Enter to start",
            "Ctrl+C to cancel",
        ] {
            assert!(text.contains(want), "{want:?} is not in:\n{text}");
        }
        // The panel is closed: a border on both sides of every row of it, so nothing is cut at the edge.
        let panel = panel_rows(&rows);
        assert!(panel.first().unwrap().starts_with('╭') && panel.first().unwrap().ends_with('╮'));
        assert!(panel.last().unwrap().starts_with('╰') && panel.last().unwrap().ends_with('╯'));
        for row in &panel[1..panel.len() - 1] {
            assert!(row.starts_with('│') && row.ends_with('│'), "an open row: {row:?}");
        }
    }

    #[test]
    fn the_plan_has_no_footer_once_it_is_answered() {
        let rows = render(&planned(), 80, 24);
        assert!(!text_of(&rows).contains("Enter to start"));
    }

    #[test]
    fn the_run_shows_a_mark_per_step_the_sentence_and_the_progress() {
        let mut view = running_step_two();
        view.tick = 3;
        let rows = render(&view, 80, 24);
        let text = text_of(&rows);
        let line_of = |needle: &str| {
            rows.iter().find(|row| row.contains(needle)).unwrap_or_else(|| panic!("no {needle:?} in:\n{text}"))
        };
        assert!(line_of("Make a key only for Aria").contains('✓'));
        assert!(line_of("Install Aria on the server").contains(SPINNER[3]), "the running step shows the spinner frame");
        assert!(line_of("Copy your settings").contains('○'));
        assert!(line_of("Test the connection").contains('○'));
        assert!(text.contains("Uploads the build you gave me, then runs the installer there."));
        assert!(line_of("Uploaded the build").contains('✓'));
        assert!(text.contains("2 of 5"), "{text}");
        assert!(text.contains('━') && text.contains('╸'), "the bar is drawn with thin lines");
    }

    #[test]
    fn the_spinner_turns_with_the_tick() {
        let mut view = running_step_two();
        view.tick = 0;
        let first = text_of(&render(&view, 80, 24));
        view.tick = 1;
        let second = text_of(&render(&view, 80, 24));
        assert_ne!(first, second);
        assert!(first.contains(SPINNER[0]) && second.contains(SPINNER[1]));
        assert!(view.animating());
        view.live = false;
        assert!(!view.animating(), "the copy for the scrollback does not animate");
    }

    #[test]
    fn a_finished_run_says_the_server_is_ready_and_what_to_type() {
        let mut view = running_step_two();
        view.finish(&Outcome::Ready { connect: "aria connect".to_string() });
        view.live = false;
        let rows = render(&view, 80, 24);
        let text = text_of(&rows);
        assert!(text.contains("Your server is ready"), "{text}");
        assert!(text.contains("To use it, type:"));
        assert!(text.contains("aria connect"));
        assert!(text.contains("5 of 5"));
        assert_eq!(text.matches('✓').count(), 5 + 1, "five steps and the ready line: {text}");
        assert!(!text.contains('○'));
    }

    #[test]
    fn an_update_ends_with_each_server_and_the_version_it_runs() {
        let mut view = running_step_two();
        view.finish(&Outcome::Updated {
            runs: vec![
                ("ana@hub.example".to_string(), "0.22.0".to_string()),
                ("b@two".to_string(), "0.22.0".to_string()),
            ],
        });
        view.live = false;
        let text = text_of(&render(&view, 80, 24));
        assert!(text.contains("ana@hub.example runs Aria 0.22.0"), "{text}");
        assert!(text.contains("b@two runs Aria 0.22.0"), "{text}");
        assert!(!text.contains("Your server is ready"), "that is init's closing: {text}");
    }

    #[test]
    fn an_update_with_nothing_to_do_has_no_progress_bar() {
        let mut view = View::new();
        view.title = "updating your server".to_string();
        view.finish(&Outcome::Updated { runs: vec![("ana@hub.example".to_string(), "0.22.0".to_string())] });
        view.live = false;
        let text = text_of(&render(&view, 80, 24));
        assert!(text.contains("ana@hub.example runs Aria 0.22.0"), "{text}");
        assert!(!text.contains(" of "), "no step, no bar: {text}");
    }

    #[test]
    fn a_hub_that_is_not_ready_says_what_to_do() {
        let mut view = running_step_two();
        view.finish(&Outcome::NotReady { why: "turn linger on (see the note above)".to_string() });
        let text = text_of(&render(&view, 80, 24));
        assert!(text.contains("installed, but not ready"), "{text}");
        assert!(text.contains("turn linger on"));
    }

    #[test]
    fn a_failed_step_is_marked_and_its_sentence_is_in_the_panel() {
        let mut view = running_step_two();
        view.step_start(4, 5, "Allowing the key on the server", "");
        view.fail("The key was not allowed: the server's sandbox does not work.");
        view.live = false;
        let rows = render(&view, 80, 24);
        let text = text_of(&rows);
        let failed = rows.iter().find(|row| row.contains("Allow that key on the server")).unwrap();
        assert!(failed.contains('✗'), "{failed:?}");
        assert!(text.contains("The key was not allowed: the server's sandbox does not work."), "{text}");
        assert!(text.contains("3 of 5") || text.contains("4 of 5") || text.contains("of 5"));
        assert!(!text.contains("Your server is ready"));
    }

    #[test]
    fn a_narrow_terminal_gets_a_panel_that_fits_it() {
        for view in [
            {
                let mut view = planned();
                view.footer = true;
                view
            },
            running_step_two(),
            {
                let mut view = running_step_two();
                view.finish(&Outcome::Ready { connect: "aria connect --host ana@hub.example".to_string() });
                view
            },
        ] {
            let rows = render(&view, 60, 24);
            for row in panel_rows(&rows) {
                assert!(row.chars().count() <= 60, "a row is wider than 60 columns: {row:?}");
                assert!(row.starts_with(['│', '╭', '╰']) && row.ends_with(['│', '╮', '╯']), "a cut row: {row:?}");
            }
        }
        let text = text_of(&render(&planned(), 60, 24));
        assert!(text.contains("login."), "wrapped, not cut: {text}");
    }

    #[test]
    fn a_wide_terminal_does_not_stretch_the_panel() {
        let rows = render(&planned(), 200, 24);
        assert!(panel_rows(&rows).iter().all(|row| row.chars().count() <= PANEL_WIDTH as usize));
    }

    #[test]
    fn a_short_terminal_keeps_the_progress_and_drops_what_can_wait() {
        let mut view = running_step_two();
        for at in 0..8 {
            view.detail.push(Check::Ok(format!("a check line number {at}")));
        }
        let rows = render(&view, 80, 16);
        let text = text_of(&rows);
        assert!(text.contains("2 of 5"), "the bar survives a short terminal: {text}");
        assert!(text.contains("a check line number 7"), "the newest line is kept");
        assert!(!text.contains("a check line number 0"), "the oldest goes first");
        assert!(rows.iter().filter(|row| !row.is_empty()).count() <= 15);
    }

    #[test]
    fn looking_shows_the_spinner_and_what_was_found() {
        let mut view = View::new();
        view.title = "linking this computer to your server".to_string();
        view.host = "ana@hub.example".to_string();
        view.facts.push("Ubuntu 24.04.4 LTS, aarch64".to_string());
        view.facts.push("Aria 0.21.0 already runs there".to_string());
        let text = text_of(&render(&view, 80, 24));
        assert!(text.contains("Looking at ana@hub.example..."), "{text}");
        assert!(text.contains("Ubuntu 24.04.4 LTS, aarch64"));
        assert!(text.contains("✓ Aria 0.21.0 already runs there"));
    }

    #[test]
    fn steps_that_are_skipped_over_are_done_and_the_list_grows_if_the_plan_was_short() {
        let mut view = View::new();
        view.step_start(3, 3, "Third", "");
        assert_eq!(view.rows.len(), 3);
        assert_eq!(view.rows[0].1, RowState::Done);
        assert_eq!(view.rows[1].1, RowState::Done);
        assert_eq!(view.rows[2].1, RowState::Running);
    }

    #[test]
    fn the_numbers_of_the_plan_are_the_numbers_of_the_steps() {
        let view = running_step_two();
        let text = text_of(&render(&view, 80, 24));
        let step_two = text.lines().find(|line| line.contains("Install Aria on the server")).unwrap();
        assert!(step_two.contains(" 2 "), "{step_two:?}");
    }
}
