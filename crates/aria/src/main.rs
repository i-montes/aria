//! `aria` runs the dock (resident, hidden until you press the hotkey); `aria toggle` shows or hides
//! it, and `aria show` only shows it (a notification's click uses that), starting it first when it
//! is not running.

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
#[cfg(target_os = "linux")]
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
        None => run_dock(),
        Some(other) => {
            eprintln!("aria: unknown command '{other}'\nusage: aria [toggle|show]");
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
                run_dir: control::control_socket_path().parent().map(std::path::Path::to_path_buf).unwrap_or_else(std::env::temp_dir),
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
    std::thread::spawn(move || {
        for message in chat_receiver {
            chat.handle(message);
        }
    });

    let dock = {
        let reply_sender = reply_sender.clone();
        Dock::new(dock_visible, Box::new(move |message, _dock| {
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
        }))
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
        use std::io::{BufRead, BufReader};
        for stream in listener.incoming().map_while(Result::ok) {
            if let Some(Ok(line)) = BufReader::new(stream).lines().next() {
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
            let _ = hotkey_sender.send("toggle".to_string());
        }) {
            eprintln!("aria: could not register Option + Space: {error}");
            dock.set_problem("hotkey-unavailable");
        } else {
            eprintln!("aria: registered Option + Space");
            dock.clear_problem();
        }
    }

    let mut was_visible = false;
    ui::every(16, move || {
        // A dock that just hid with a question still open needs its notification.
        let visible = visible_now.load(std::sync::atomic::Ordering::SeqCst);
        if was_visible && !visible {
            chat_for_visibility.dock_visibility_changed();
        }
        was_visible = visible;
        while let Ok(command) = receiver.try_recv() {
            match command.as_str() {
                "toggle" => dock.feed(Event::Toggle),
                "show" => dock.show(),
                _ => {}
            }
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

#[cfg(target_os = "linux")]
const HOTKEY_HINT: &str = "SUPER + A";
#[cfg(target_os = "macos")]
const HOTKEY_HINT: &str = "Option + Space";

/// Tests that start child processes or hold a listening socket take turns: a fork in one test
/// leaves copies of every open descriptor in the child until it execs, which keeps a dropped socket
/// "listening" and a freshly written script "busy" for a moment.
#[cfg(test)]
pub(crate) static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
