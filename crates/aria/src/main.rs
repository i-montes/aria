//! `aria` runs the dock (resident, hidden until you press SUPER + A); `aria toggle` shows or hides
//! it, starting it first when it is not running.

mod control;
mod dock;
mod handlers;
mod visibility;

use aria_core::bridge::DockMessage;
use aria_core::paths::AriaPaths;
use aria_hyprland::Registration;
use dock::Dock;
use std::sync::{mpsc, Arc};
use std::time::Duration;
use visibility::Event;

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("toggle") => {
            let exe = std::env::current_exe().expect("own path");
            if let Err(error) = control::toggle_or_start(&control::control_socket_path(), &exe) {
                eprintln!("aria: {error}");
                std::process::exit(1);
            }
        }
        None => run_dock(),
        Some(other) => {
            eprintln!("aria: unknown command '{other}'\nusage: aria [toggle]");
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
    let ctx = Arc::new(handlers::Ctx::real(paths));
    let dock = {
        let reply_sender = reply_sender.clone();
        Dock::new(Box::new(move |message, _dock| {
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
            if command == "toggle" {
                dock.feed(Event::Toggle);
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
