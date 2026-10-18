//! `aria` runs the dock (resident, hidden until you press SUPER + A); `aria toggle` shows or hides
//! it, starting it first when it is not running.

mod control;
mod dock;
mod visibility;

use aria_hyprland::{ConfigKind, HyprError};
use dock::Dock;
use std::sync::mpsc;
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
    let dock = Dock::new(Box::new(|_message, _dock| {}));

    let exe = std::env::current_exe().expect("own path");
    match aria_hyprland::detect_config() {
        Ok(ConfigKind::Lua) => {
            if let Err(error) = aria_hyprland::register_hotkey(&exe) {
                eprintln!("aria: could not register SUPER + A: {error}");
            }
            let exe = exe.clone();
            aria_hyprland::watch_events(move || {
                if let Err(error) = aria_hyprland::register_hotkey(&exe) {
                    eprintln!("aria: could not register SUPER + A after a reload: {error}");
                }
            });
        }
        Ok(ConfigKind::Classic) => dock.set_problem("hyprland-classic-config"),
        Err(HyprError::NotRunning) => dock.set_problem("hyprland-not-running"),
        Err(error) => {
            eprintln!("aria: {error}");
            dock.set_problem("hyprland-error");
        }
    }

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
        glib::ControlFlow::Continue
    });

    eprintln!("aria: ready, press SUPER + A");
    gtk::main();
}
