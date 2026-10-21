//! Manual check: registers SUPER + A (pointing at /bin/true), then re-registers on every config
//! reload for the number of seconds given (default 15). Run it, then `hyprctl reload`.
use std::path::Path;
use std::time::Duration;

fn main() {
    let seconds: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(15);
    let exe = Path::new("/bin/true");
    println!("config kind: {:?}", aria_hyprland::detect_config());
    println!("register: {:?}", aria_hyprland::register_hotkey(exe));
    println!("focused monitor: {:?}", aria_hyprland::focused_monitor());
    let _watcher = aria_hyprland::watch_events(move || {
        println!("reloaded -> register: {:?}", aria_hyprland::register_hotkey(exe))
    });
    std::thread::sleep(Duration::from_secs(seconds));
}
