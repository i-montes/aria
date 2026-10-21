//! `aria` runs the dock (resident, hidden until you press the hotkey); `aria toggle` shows or hides
//! it, and `aria show` only shows it (a notification's click uses that), starting it first when it
//! is not running. `aria update-ready` (run by the installer after it replaced the files) asks the running dock to
//! restart into the new binary as soon as it is hidden and no session has a run going; it fails when no dock answers,
//! and the installer then restarts the service itself.

mod chat;
mod control;
mod dock;
mod handlers;
mod ui;
mod visibility;
mod watch;

use aria_core::bridge::{DockMessage, PageMessage};
use aria_core::paths::AriaPaths;
#[cfg(target_os = "linux")]
use aria_hyprland::Registration;
use chat::{ChatCtx, PlatformNotifier, Sessions};
use dock::{Dock, HOTKEY_LABEL};
use std::sync::atomic::AtomicBool;
use std::sync::{mpsc, Arc};
use std::time::Duration;
use visibility::Event;

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some(command @ ("toggle" | "show")) => {
            let exe = std::env::current_exe().expect("own path");
            if let Err(error) = control::command_or_start(&control::control_socket_path(), &exe, command) {
                eprintln!("aria: {error}");
                std::process::exit(1);
            }
        }
        Some("update-ready") => {
            let answer = control::ask(&control::control_socket_path(), "update-ready", Duration::from_secs(2));
            if let Err(error) = answer {
                eprintln!("aria: no dock took the update: {error}");
                std::process::exit(1);
            }
        }
        None => run_dock(),
        Some(other) => {
            eprintln!("aria: unknown command '{other}'\nusage: aria [toggle|show|update-ready]");
            std::process::exit(2);
        }
    }
}

fn run_dock() {
    let listener = match control::bind_single_instance(&control::control_socket_path()) {
        Ok(listener) => listener,
        Err(control::ControlError::AlreadyRunning) => {
            eprintln!("aria: the dock is already running");
            return;
        }
        Err(error) => {
            eprintln!("aria: {error}");
            std::process::exit(1);
        }
    };

    // WebKitGTK needs its renderer flag before GTK starts; on Mac this creates the app object.
    ui::init();

    // Replies from any thread come back to the UI thread through this channel.
    let (reply_sender, reply_receiver) = mpsc::channel::<DockMessage>();
    let paths = AriaPaths::from_env();
    let first_run = !paths.settings().exists();
    let mut settings_watch = chat::SettingsWatch::new(paths.settings());
    let ctx = Arc::new(handlers::Ctx::real(paths.clone()));

    // The chat owns Pi. Its messages go through one worker thread, in order, so a send is never
    // overtaken by the abort that follows it.
    let dock_visible = Arc::new(AtomicBool::new(false));
    let visible_now = dock_visible.clone();
    let chat = {
        let sender = reply_sender.clone();
        Sessions::new(
            ChatCtx {
                paths,
                notifier: Arc::new(PlatformNotifier),
                dock_visible: dock_visible.clone(),
                chat_shown: Arc::new(AtomicBool::new(true)),
                run_dir: control::control_socket_path()
                    .parent()
                    .map(std::path::Path::to_path_buf)
                    .unwrap_or_else(std::env::temp_dir),
                path_var: std::env::var("PATH").unwrap_or_default(),
                quota_fetcher: Arc::new(aria_core::quota::real_fetcher),
            },
            Arc::new(move |message| {
                let _ = sender.send(message);
            }),
        )
    };
    let (chat_sender, chat_receiver) = mpsc::channel::<PageMessage>();
    let chat_for_visibility = chat.clone();
    let chat_for_settings = chat.clone();
    std::thread::spawn(move || {
        for message in chat_receiver {
            chat.handle(message);
        }
    });

    let dock = {
        let reply_sender = reply_sender.clone();
        Dock::new(
            dock_visible,
            Box::new(move |message, _dock| {
                if handlers::is_chat(&message) {
                    let _ = chat_sender.send(message);
                    return;
                }
                let (ctx, sender) = (ctx.clone(), reply_sender.clone());
                let emit = move |reply: DockMessage| {
                    let _ = sender.send(reply);
                };
                if handlers::is_blocking(&message) {
                    std::thread::spawn(move || handlers::handle(message, &ctx, &emit));
                } else {
                    handlers::handle(message, &ctx, &emit);
                }
            }),
        )
    };
    // The page needs the shortcut for its notice and its "you are set up" screen: Super + A on Omarchy,
    // Option + Space on macOS.
    dock.send(&DockMessage::Platform { hotkey: HOTKEY_LABEL.to_string() });
    if first_run {
        dock.show_notice();
    }

    // Commands arrive on a socket thread and are handled on the UI thread. The hotkey sends the same
    // "toggle" line, so there is one path from both.
    let (sender, receiver) = mpsc::channel::<String>();
    let listener_sender = sender.clone();
    std::thread::spawn(move || {
        use std::io::{BufRead, BufReader, Write};
        for stream in listener.incoming().map_while(Result::ok) {
            if let Some(Ok(line)) = BufReader::new(&stream).lines().next() {
                // The installer waits for this answer; without it (an older dock) it restarts the service itself.
                if line == "update-ready" {
                    let _ = (&stream).write_all(b"ok\n");
                }
                let _ = listener_sender.send(line);
            }
        }
    });

    // The hotkey. On Linux it is registered as soon as Hyprland can take it (it may not be up when a
    // service starts), and again after every config reload; the thread reports how it is going and the
    // problem banner follows it. On Mac Carbon registers it once, without permissions, for the life of
    // the app; a shortcut another app already owns is reported the same way.
    #[cfg(target_os = "linux")]
    let (registration_sender, registration_receiver) = mpsc::channel::<Registration>();
    #[cfg(target_os = "linux")]
    {
        let exe = std::env::current_exe().expect("own path");
        aria_hyprland::keep_hotkey_registered(exe, Duration::from_secs(2), move |state| {
            let _ = registration_sender.send(state);
        });
    }
    #[cfg(target_os = "macos")]
    {
        let hotkey_sender = sender.clone();
        if let Err(error) = aria_mac::install_hotkey(move || {
            let _ = hotkey_sender.send("hotkey".to_string());
        }) {
            eprintln!("aria: could not register Option + Space: {error}");
            dock.set_problem("hotkey-unavailable");
        } else {
            eprintln!("aria: registered Option + Space");
            dock.clear_problem();
        }
    }

    let mut was_visible = false;
    let mut update_ready = false;
    let mut ticks: u32 = 0;
    ui::every(16, move || {
        // About once a second: a settings change (the settings screen, `/effort`, `/autocompact`, an editor) reaches
        // every session's Pi when its run settles, and the page reads the settings again.
        ticks = ticks.wrapping_add(1);
        if ticks.is_multiple_of(60) && settings_watch.changed() {
            // Off the UI thread: preparing the agent directory reads and writes files.
            let sessions = chat_for_settings.clone();
            std::thread::spawn(move || sessions.settings_changed());
            dock.send(&DockMessage::SettingsChanged);
        }
        // A dock that just hid with a question still open needs its notification.
        let visible = visible_now.load(std::sync::atomic::Ordering::SeqCst);
        if was_visible && !visible {
            chat_for_visibility.dock_visibility_changed();
        }
        was_visible = visible;
        while let Ok(command) = receiver.try_recv() {
            match command.as_str() {
                "hotkey" => {
                    note("the shortcut was pressed");
                    dock.feed(Event::Toggle)
                }
                "toggle" => {
                    note("toggle from the socket");
                    dock.feed(Event::Toggle)
                }
                "show" => dock.show(),
                "update-ready" => {
                    eprintln!("aria: a new version is installed; restarting once the dock is hidden and idle");
                    update_ready = true;
                }
                _ => {}
            }
        }
        // A new binary is installed: it takes over once nobody is looking and no run would be cut. The tabs and their
        // conversations come back as after any restart.
        if update_ready && ticks.is_multiple_of(60) && !visible && !chat_for_visibility.busy() {
            eprintln!("aria: restarting into the new version");
            chat_for_visibility.stop_for_update();
            let error = control::reexec();
            eprintln!("aria: could not start the new version: {error}");
            std::process::exit(75); // not a clean exit: systemd and launchd start the dock again
        }
        while let Ok(reply) = reply_receiver.try_recv() {
            dock.send(&reply);
        }
        #[cfg(target_os = "linux")]
        while let Ok(state) = registration_receiver.try_recv() {
            match state {
                Registration::Registered => dock.clear_problem(),
                Registration::WaitingForHyprland => dock.set_problem("hyprland-not-running"),
                Registration::ClassicConfig => dock.set_problem("hyprland-classic-config"),
            }
        }
    });

    eprintln!("aria: ready, press {HOTKEY_HINT}");
    ui::run();
}

/// A line in the dock's log with the time (UTC), for what happens between a keypress and the dock on screen: when the
/// dock does not open, the log says whether the shortcut arrived, what state the dock was in and whether the page
/// answered.
pub(crate) fn note(message: &str) {
    let seconds = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (h, m, s) = (seconds / 3600 % 24, seconds / 60 % 60, seconds % 60);
    eprintln!("[{h:02}:{m:02}:{s:02} UTC] aria: {message}");
}

#[cfg(target_os = "linux")]
const HOTKEY_HINT: &str = "SUPER + A";
#[cfg(target_os = "macos")]
const HOTKEY_HINT: &str = "Option + Space";

/// Tests that start child processes or hold a listening socket take turns: a fork in one test
/// leaves copies of every open descriptor in the child until it execs, which keeps a dropped socket
/// "listening" and a freshly written script "busy" for a moment.
#[cfg(test)]
pub(crate) static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
