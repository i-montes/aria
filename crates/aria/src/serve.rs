//! `aria serve`: the service with no window, for a computer with no desktop. It is what the dock runs (`Service`) on a
//! plain loop: no GTK, WebKit, Hyprland, whisper or audio, so a build without the `gui` feature links none of them.
//! Terminals reach it as they reach a dock (the clients socket, or `aria connect` over ssh). It binds the dock's
//! control socket, so a dock and a serve on one computer exclude each other, and `aria quit` and `aria update-ready`
//! talk to it the same way.

use crate::chat::NullNotifier;
use crate::service::{self, Service};
use crate::{control, install_quit_handlers, note, request_quit};
use std::sync::atomic::AtomicBool;
use std::sync::{mpsc, Arc};
use std::time::Duration;

/// The service's loop runs at the dock's rate, so the same ticks (a settings change within a second, the watchdog every
/// 30 s) mean the same time in both.
const TICK: Duration = Duration::from_millis(16);

/// What a line on the control socket asks of a service with no window.
#[derive(Debug, PartialEq)]
enum Command {
    Quit,
    UpdateReady,
    /// `toggle` and `show` are for a window.
    NoWindow,
    Ignored,
}

fn command(line: &str) -> Command {
    match line {
        "quit" => Command::Quit,
        "update-ready" => Command::UpdateReady,
        "toggle" | "show" | "hotkey" => Command::NoWindow,
        _ => Command::Ignored,
    }
}

/// Runs until `aria quit` or a quit signal; the exit code is the process's.
// `Service::tick` can end the process itself (`std::process::exit`) when an update takes over: so can this loop.
pub fn run() -> i32 {
    let listener = match control::bind_single_instance(&control::control_socket_path()) {
        Ok(listener) => listener,
        Err(control::ControlError::AlreadyRunning) => {
            eprintln!("aria: Aria is already running");
            return 0;
        }
        Err(error) => {
            eprintln!("aria: {error}");
            return 1;
        }
    };
    let (sender, commands) = mpsc::channel::<String>();
    control::answer_commands(listener, sender);
    // Stopping the service (systemd, a terminal's Ctrl+C) ends its background work and Pis, never leaves them.
    install_quit_handlers();
    let mut service = Service::start(service::Options {
        notifier: Arc::new(NullNotifier),
        dock_visible: Arc::new(AtomicBool::new(false)),
    });
    // A service nobody can reach is not a service: say so and fail, so the unit is not "active" and useless.
    if service.clients_error.is_some() {
        service.shutdown();
        eprintln!("aria: not serving: terminals have no way in (see above)");
        return 1;
    }
    note("serving with no window (`aria quit` stops it)");
    if service.first_run {
        note("no settings yet: this is a first run, sign in from a terminal (`aria connect`)");
    }
    loop {
        // No window: nothing is on screen, and the page's replies have no page to go to.
        let tick = service.tick(false);
        if tick.quit {
            return 0;
        }
        if tick.settings_changed {
            note("settings.json changed");
        }
        drop(tick.to_page);
        while let Ok(line) = commands.try_recv() {
            match command(&line) {
                Command::Quit => {
                    note("asked to quit");
                    request_quit();
                }
                Command::UpdateReady => service.update_ready(),
                Command::NoWindow => note(&format!("no window here: `{line}` is ignored")),
                Command::Ignored => {}
            }
        }
        std::thread::sleep(TICK);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_service_with_no_window_takes_quit_and_the_update_and_ignores_the_window_commands() {
        assert_eq!(command("quit"), Command::Quit);
        assert_eq!(command("update-ready"), Command::UpdateReady);
        for line in ["toggle", "show", "hotkey"] {
            assert_eq!(command(line), Command::NoWindow);
        }
        assert_eq!(command("anything else"), Command::Ignored);
    }
}
