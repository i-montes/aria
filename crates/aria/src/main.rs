//! `aria` runs the dock (resident, hidden until you press SUPER + A); `aria toggle` shows or hides
//! it, and `aria show` only shows it (a notification's click uses that), starting it first when it
//! is not running.

mod chat;
mod control;
mod dock;
mod handlers;
mod visibility;

/// Tests that start child processes or hold a listening socket take turns: a fork in one test
/// leaves copies of every open descriptor in the child until it execs, which keeps a dropped socket
/// "listening" and a freshly written script "busy" for a moment.
#[cfg(test)]
pub(crate) static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

use aria_core::bridge::{DockMessage, PageMessage};
use aria_core::paths::AriaPaths;
use aria_hyprland::Registration;
use chat::{Chat, ChatCtx, OmarchyNotifier};
use dock::Dock;
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

    // WebKitGTK's DMA-BUF renderer draws nothing in a layer-shell surface on this setup.
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }
    gtk::init().expect("gtk");

    // Replies from any thread come back to the GTK thread through this channel.
    let (reply_sender, reply_receiver) = mpsc::channel::<DockMessage>();
    let paths = AriaPaths::from_env();
    let first_run = !paths.settings().exists();
    let ctx = Arc::new(handlers::Ctx::real(paths.clone()));

    // The chat owns Pi. Its messages go through one worker thread, in order, so a send is never
    // overtaken by the abort that follows it.
    let dock_visible = Arc::new(AtomicBool::new(false));
    let chat = {
        let sender = reply_sender.clone();
        Chat::new(
            ChatCtx {
                paths,
                notifier: Arc::new(OmarchyNotifier),
                dock_visible: dock_visible.clone(),
                path_var: std::env::var("PATH").unwrap_or_default(),
            },
            Arc::new(move |message| {
                let _ = sender.send(message);
            }),
        )
    };
    let (chat_sender, chat_receiver) = mpsc::channel::<PageMessage>();
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
    if first_run {
        dock.show_notice();
    }

    // The hotkey: registered as soon as Hyprland can take it (it may not be up when a service starts),
    // and again after every config reload. The thread reports how it is going; the problem banner
    // follows it.
    let (registration_sender, registration_receiver) = mpsc::channel::<Registration>();
    let exe = std::env::current_exe().expect("own path");
    aria_hyprland::keep_hotkey_registered(exe, Duration::from_secs(2), move |state| {
        let _ = registration_sender.send(state);
    });

    // Commands arrive on a socket thread and are handled on the GTK thread.
    let (sender, receiver) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        use std::io::{BufRead, BufReader};
        for stream in listener.incoming().map_while(Result::ok) {
            if let Some(Ok(line)) = BufReader::new(stream).lines().next() {
                let _ = sender.send(line);
            }
        }
    });
    glib::timeout_add_local(Duration::from_millis(16), move || {
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
        while let Ok(state) = registration_receiver.try_recv() {
            match state {
                Registration::Registered => dock.clear_problem(),
                Registration::WaitingForHyprland => dock.set_problem("hyprland-not-running"),
                Registration::ClassicConfig => dock.set_problem("hyprland-classic-config"),
            }
        }
        glib::ControlFlow::Continue
    });

    eprintln!("aria: ready, press SUPER + A");
    gtk::main();
}
