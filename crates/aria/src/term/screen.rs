//! The client's input and output: the terminal (raw mode, the alternate screen, the mouse, focus changes, pasted text)
//! and the dock's link. Everything it decides is `App`'s; this file only moves events and effects, and always gives
//! the terminal back as it found it, a panic included.

use crate::term::app::{App, Config, Effect, Input, Press, Screen};
use crate::term::git::{self, GitStatus};
use crate::term::link::Link;
use crate::term::select::{copy, Selection};
use crate::term::theme::Palette;
use crate::term::view::{bottom_height, cursor_position, draw_list, draw_session, should_draw, SessionView};
use crate::term::Key;
use crossterm::event::{
    self, DisableBracketedPaste, DisableFocusChange, DisableMouseCapture, EnableBracketedPaste, EnableFocusChange,
    EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags, MouseButton,
    MouseEventKind, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Position;
use ratatui::Terminal;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// A thread of the client panicked: the loop ends (the hook already gave the terminal back).
pub static PANICKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn should_quit() -> bool {
    PANICKED.load(std::sync::atomic::Ordering::SeqCst)
}

fn restore() {
    // Harmless where it was never pushed.
    let _ = execute!(std::io::stdout(), PopKeyboardEnhancementFlags);
    let _ = disable_raw_mode();
    let _ = execute!(
        std::io::stdout(),
        DisableMouseCapture,
        DisableFocusChange,
        DisableBracketedPaste,
        LeaveAlternateScreen
    );
}

/// Gives the terminal back when it goes out of scope: every return, an error's `?` included.
struct Guard<F: FnMut()>(F);

impl<F: FnMut()> Guard<F> {
    fn new(restore: F) -> Guard<F> {
        Guard(restore)
    }
}

impl<F: FnMut()> Drop for Guard<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}

fn key_of(code: KeyCode) -> Option<Key> {
    Some(match code {
        KeyCode::Char(c) => Key::Char(c),
        KeyCode::Enter => Key::Enter,
        KeyCode::Esc => Key::Esc,
        KeyCode::Up => Key::Up,
        KeyCode::Down => Key::Down,
        KeyCode::Left => Key::Left,
        KeyCode::Right => Key::Right,
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Tab => Key::Tab,
        KeyCode::PageUp => Key::PageUp,
        KeyCode::PageDown => Key::PageDown,
        _ => return None,
    })
}

/// A terminal event as the app's inputs (pasted text is one key per character).
fn inputs(event: Event) -> Vec<Input> {
    match event {
        Event::Key(key) if key.kind != KeyEventKind::Release => key_of(key.code)
            .map(|k| {
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                let alt = key.modifiers.intersects(KeyModifiers::ALT | KeyModifiers::SHIFT) && k == Key::Enter
                    || key.modifiers.contains(KeyModifiers::ALT);
                vec![Input::Key(Press { key: k, ctrl, alt })]
            })
            .unwrap_or_default(),
        Event::Paste(text) => vec![Input::Paste(text)],
        Event::Mouse(mouse) => match mouse.kind {
            MouseEventKind::ScrollUp => vec![Input::Scroll(3)],
            MouseEventKind::ScrollDown => vec![Input::Scroll(-3)],
            _ => Vec::new(),
        },
        Event::Resize(_, height) => vec![Input::Resize { height }],
        Event::FocusGained => vec![Input::Focus(true)],
        Event::FocusLost => vec![Input::Focus(false)],
        _ => Vec::new(),
    }
}

fn save_verbose(paths: &aria_core::paths::AriaPaths, on: bool) {
    if let Ok(Some(mut settings)) = aria_core::settings::load(paths) {
        settings.display.verbose = on;
        let _ = aria_core::settings::save(paths, &settings);
    }
}

/// The folder whose git status the footer shows: none over ssh, where the session's folder is on the other computer
/// and the same path here (if there is one) is another repository.
fn git_folder(remote: bool, workdir: &str) -> String {
    if remote {
        String::new()
    } else {
        workdir.to_string()
    }
}

/// The git status of the attached session's folder, read every 2 s on its own thread.
fn watch_git(folder: Arc<Mutex<String>>) -> Arc<Mutex<Option<GitStatus>>> {
    let status = Arc::new(Mutex::new(None));
    let shared = status.clone();
    std::thread::spawn(move || loop {
        let dir = folder.lock().unwrap().clone();
        let read = if dir.is_empty() { None } else { git::status(Path::new(&dir)) };
        *shared.lock().unwrap() = read;
        std::thread::sleep(Duration::from_secs(2));
    });
    status
}

/// Runs until the app quits; the note to print after the terminal is back.
pub fn run(
    link: Link,
    config: Config,
    paths: &aria_core::paths::AriaPaths,
    reconnect: &dyn Fn() -> std::io::Result<Link>,
) -> std::io::Result<Option<String>> {
    let mut link = link;
    let palette = Palette::detect();
    let mut events = link.events();
    let mut tags = link.tags();
    let mut roster = link.roster();
    let remote = config.remote.is_some();
    let (mut app, first) = App::new(config);
    app.hub_version(link.hub_version());
    for effect in first {
        match effect {
            Effect::Send(message) => link.send(&message),
            Effect::SendOn { message, machine } => link.send_on(&message, &machine),
            Effect::MachinesList { id } => link.machines_list(id),
            _ => {}
        }
    }
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        PANICKED.store(true, std::sync::atomic::Ordering::SeqCst);
        restore();
        previous(info);
    }));
    enable_raw_mode()?;
    let _guard = Guard::new(restore);
    execute!(std::io::stdout(), EnterAlternateScreen, EnableMouseCapture, EnableFocusChange, EnableBracketedPaste)?;
    // Where the terminal speaks the extended keyboard protocol (kitty, ghostty, foot, WezTerm, recent Alacritty),
    // Ctrl+Enter and Shift+Enter are told apart from Enter.
    if crossterm::terminal::supports_keyboard_enhancement().unwrap_or(false) {
        execute!(std::io::stdout(), PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES))?;
    }
    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;
    let size = terminal.size()?;
    app.update(Input::Resize { height: size.height });
    let folder = Arc::new(Mutex::new(String::new()));
    let git_status = watch_git(folder.clone());
    let mut quit = false;
    let mut max_scroll = 0;
    let mut more_rows: Vec<(u16, String)> = Vec::new();
    let mut strip_rows: Option<(u16, u16)> = None;
    let mut overlay_top = 0;
    // A frame is drawn only when something changed (a key, the mouse, the dock, the size, git) or the spinner is due.
    let mut changed = true;
    let mut last_draw = std::time::Instant::now();
    let mut last_git: Option<GitStatus> = None;
    // The mouse's selection, and the last frame (what a selection copies).
    let mut selection: Option<Selection> = None;
    let mut last_frame: Option<ratatui::buffer::Buffer> = None;
    while !quit {
        if should_quit() {
            return Err(std::io::Error::other("a part of aria stopped (its message is above)"));
        }
        app.clamp_scroll(max_scroll);
        app.set_more_rows(std::mem::take(&mut more_rows));
        app.set_strip_rows(strip_rows);
        app.set_overlay_top(overlay_top);
        let now_ms =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
        app.clock(now_ms);
        *folder.lock().unwrap() = git_folder(remote, &app.session().workdir);
        let git = git_status.lock().unwrap().clone();
        if git != last_git {
            last_git = git.clone();
            changed = true;
        }
        let since = last_draw.elapsed().as_millis() as u64;
        if should_draw(changed, app.session().running && app.screen() == Screen::Session, since) {
            changed = false;
            last_draw = std::time::Instant::now();
            let completed = terminal.draw(|frame| {
                match app.screen() {
                    Screen::List => draw_list(
                        frame,
                        &app.list_rows(),
                        &app.list_places(),
                        app.selected(),
                        app.note().as_deref(),
                        &palette,
                    ),
                    Screen::Session => {
                        let area = frame.area();
                        let width = area.width.saturating_sub(4);
                        let panel = app.panel_lines(&palette, width);
                        let has_panel = panel.is_some();
                        let workdir = app.session().workdir.clone();
                        let view = SessionView {
                            session: app.session(),
                            verbose: app.verbose(),
                            input: app.editor.text(),
                            panel,
                            git: git.clone(),
                            folder: workdir.trim_end_matches('/').rsplit('/').next().unwrap_or_default().to_string(),
                            compact_at: app.compact_at(),
                            lang: app.lang().to_string(),
                            palette,
                            now: (now_ms / 1000) as i64,
                            scroll: app.scroll(),
                            working_ms: app.working_ms(now_ms),
                            tick: now_ms,
                            overlay: app.overlay_lines(&palette),
                            expanded: app.expanded().clone(),
                            tasks_folded: app.tasks_folded(),
                        };
                        let floating = view.overlay.as_ref().map_or(0, Vec::len) as u16;
                        overlay_top =
                            area.height.saturating_sub(bottom_height(&view, area.width, area.height) + floating);
                        let drawn = draw_session(frame, &view);
                        max_scroll = drawn.above;
                        more_rows = drawn.more;
                        strip_rows = drawn.strip;
                        if !has_panel && area.height > 3 {
                            let (x, y) =
                                cursor_position(app.editor.text(), app.editor.cursor(), area.width, area.height);
                            frame.set_cursor_position(Position { x, y });
                        }
                    }
                }
                if let Some(selection) = selection.filter(|s| !s.is_click()) {
                    selection.paint(frame.buffer_mut());
                }
            })?;
            last_frame = Some(completed.buffer.clone());
        }
        let mut effects = Vec::new();
        if event::poll(Duration::from_millis(40))? {
            let event = event::read()?;
            changed = true;
            let mut handled = false;
            if let Event::Mouse(mouse) = &event {
                match mouse.kind {
                    MouseEventKind::Down(MouseButton::Left) => {
                        selection = Some(Selection::new(mouse.column, mouse.row));
                        handled = true;
                    }
                    MouseEventKind::Drag(MouseButton::Left) => {
                        if let Some(selection) = selection.as_mut() {
                            selection.extend(mouse.column, mouse.row);
                        }
                        handled = true;
                    }
                    MouseEventKind::Up(MouseButton::Left) => {
                        match selection {
                            // A press without a drag is a click.
                            Some(s) if s.is_click() => {
                                selection = None;
                                effects.extend(app.update(Input::Click { row: mouse.row }));
                            }
                            // What was selected goes to the clipboard at once; it stays shown until the next key.
                            Some(s) => {
                                if let Some(frame) = &last_frame {
                                    copy(&s.text(frame));
                                }
                            }
                            None => {}
                        }
                        handled = true;
                    }
                    _ => {}
                }
            }
            if !handled {
                if matches!(event, Event::Key(_)) {
                    selection = None;
                }
                for input in inputs(event) {
                    effects.extend(app.update(input));
                }
            }
        }
        // The tags of a list come just before the list: both are taken in this turn of the loop, the tags first.
        while let Ok(found) = tags.try_recv() {
            changed = true;
            app.machine_tags(found);
        }
        while let Ok(found) = roster.try_recv() {
            changed = true;
            app.roster(found);
        }
        while let Ok(message) = events.try_recv() {
            changed = true;
            effects.extend(app.update(Input::Dock(message)));
        }
        for effect in effects {
            match effect {
                Effect::Send(message) => link.send(&message),
                Effect::SendOn { message, machine } => link.send_on(&message, &machine),
                Effect::MachinesList { id } => link.machines_list(id),
                Effect::Attach(session) => link.attach(&session),
                Effect::Detach => link.detach(),
                Effect::Focus(focused) => link.focus(focused),
                Effect::SaveVerbose(on) => save_verbose(paths, on),
                Effect::Reconnect => {
                    let mut reconnected = false;
                    let back = match reconnect() {
                        Ok(new) => {
                            events = new.events();
                            tags = new.tags();
                            roster = new.roster();
                            link = new;
                            reconnected = true;
                            Input::Reconnected
                        }
                        Err(_) => Input::Lost,
                    };
                    for effect in app.update(back) {
                        match effect {
                            Effect::Send(message) => link.send(&message),
                            Effect::SendOn { message, machine } => link.send_on(&message, &machine),
                            Effect::MachinesList { id } => link.machines_list(id),
                            Effect::Attach(session) => link.attach(&session),
                            Effect::Quit => quit = true,
                            _ => {}
                        }
                    }
                    // A new connection is one more connection: the version is said again if it still differs.
                    if reconnected {
                        app.hub_version(link.hub_version());
                    }
                }
                Effect::Redraw => {
                    terminal.clear()?;
                    changed = true;
                }
                Effect::Quit => quit = true,
            }
        }
    }
    Ok(app.note())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn a_remote_session_has_no_local_git_to_watch() {
        assert_eq!(git_folder(true, "/home/ana/Projects/checkout-widget"), "");
        assert_eq!(git_folder(false, "/home/ana/Projects/checkout-widget"), "/home/ana/Projects/checkout-widget");
    }

    #[test]
    fn a_panicked_thread_ends_the_loop() {
        assert!(!should_quit());
        PANICKED.store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(should_quit());
        PANICKED.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    #[test]
    fn the_terminal_is_given_back_on_an_error_exit() {
        static RESTORED: AtomicU32 = AtomicU32::new(0);
        fn fails() -> std::io::Result<()> {
            let _guard = Guard::new(|| {
                RESTORED.fetch_add(1, Ordering::SeqCst);
            });
            Err(std::io::Error::other("the terminal went away"))?;
            Ok(())
        }
        assert!(fails().is_err());
        assert_eq!(RESTORED.load(Ordering::SeqCst), 1);
    }
}
