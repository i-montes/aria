//! The client's input and output: the terminal (raw mode, the alternate screen, the mouse, focus changes, pasted text)
//! and the dock's link. Everything it decides is `App`'s; this file only moves events and effects, and always gives
//! the terminal back as it found it, a panic included.

use crate::term::app::{App, Config, Effect, Input, Press, Screen};
use crate::term::git::{self, GitStatus};
use crate::term::link::Link;
use crate::term::theme::Palette;
use crate::term::view::{cursor_position, draw_list, draw_session, max_scroll as max_scroll_of, SessionView};
use crate::term::Key;
use crossterm::event::{
    self, DisableBracketedPaste, DisableFocusChange, DisableMouseCapture, EnableBracketedPaste, EnableFocusChange,
    EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Position;
use ratatui::Terminal;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn restore() {
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
        Event::Paste(text) => text
            .chars()
            .map(|c| {
                let key = if c == '\n' { Key::Enter } else { Key::Char(c) };
                Input::Key(Press { key, ctrl: false, alt: c == '\n' })
            })
            .collect(),
        Event::Mouse(mouse) => match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => vec![Input::Click { row: mouse.row }],
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
    hint_file: &PathBuf,
) -> std::io::Result<Option<String>> {
    let palette = Palette::detect();
    let events = link.events();
    let (mut app, first) = App::new(config);
    for effect in first {
        if let Effect::Send(message) = effect {
            link.send(&message);
        }
    }
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore();
        previous(info);
    }));
    enable_raw_mode()?;
    let _guard = Guard::new(restore);
    execute!(std::io::stdout(), EnterAlternateScreen, EnableMouseCapture, EnableFocusChange, EnableBracketedPaste)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;
    let size = terminal.size()?;
    app.update(Input::Resize { height: size.height });
    let folder = Arc::new(Mutex::new(String::new()));
    let git_status = watch_git(folder.clone());
    let mut quit = false;
    let mut max_scroll = 0;
    while !quit {
        app.clamp_scroll(max_scroll);
        *folder.lock().unwrap() = app.session().workdir.clone();
        terminal.draw(|frame| match app.screen() {
            Screen::List => draw_list(frame, &app.list_rows(), app.selected(), app.note().as_deref(), &palette),
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
                    git: git_status.lock().unwrap().clone(),
                    folder: workdir.trim_end_matches('/').rsplit('/').next().unwrap_or_default().to_string(),
                    compact_at: app.compact_at(),
                    lang: app.lang().to_string(),
                    palette,
                    now: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |d| d.as_secs() as i64),
                    scroll: app.scroll(),
                };
                max_scroll = max_scroll_of(&view, area.width, area.height);
                draw_session(frame, &view);
                if !has_panel && area.height > 3 {
                    let (x, y) = cursor_position(app.editor.text(), app.editor.cursor(), area.width, area.height);
                    frame.set_cursor_position(Position { x, y });
                }
            }
        })?;
        let mut effects = Vec::new();
        if event::poll(Duration::from_millis(40))? {
            for input in inputs(event::read()?) {
                effects.extend(app.update(input));
            }
        }
        while let Ok(message) = events.try_recv() {
            effects.extend(app.update(Input::Dock(message)));
        }
        for effect in effects {
            match effect {
                Effect::Send(message) => link.send(&message),
                Effect::Attach(session) => link.attach(&session),
                Effect::Detach => link.detach(),
                Effect::Focus(focused) => link.focus(focused),
                Effect::SaveVerbose(on) => save_verbose(paths, on),
                Effect::SaveHint => {
                    let _ = std::fs::write(hint_file, "{\"selectHint\":true}\n");
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
