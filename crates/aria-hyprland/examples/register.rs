//! Manual check: binds the shortcut given (default `SUPER + A`, or the first free fallback when it is taken), pointing
//! at /bin/true, then binds it again on every config reload for the number of seconds given (default 15). Run it, then
//! `hyprctl reload`.
//!
//!   cargo run -p aria-hyprland --example register -- 15 "SUPER + SHIFT + A"
use aria_core::shortcut::{candidates, Platform};
use aria_hyprland::Hotkey;
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn main() {
    let seconds: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(15);
    let chosen = std::env::args().nth(2).unwrap_or_default();
    let list = candidates(&chosen, Platform::Linux);
    let hotkey = Arc::new(Mutex::new(Hotkey::new("/bin/true".into())));
    println!("config kind: {:?}", aria_hyprland::detect_config());
    println!("register: {:?}", hotkey.lock().unwrap().register(&list));
    println!("focused monitor: {:?}", aria_hyprland::focused_monitor());
    let _watcher = aria_hyprland::watch_events(move || {
        println!("reloaded -> register: {:?}", hotkey.lock().unwrap().register(&list))
    });
    std::thread::sleep(Duration::from_secs(seconds));
}
