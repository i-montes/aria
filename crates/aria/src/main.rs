//! `aria` runs the dock (resident, hidden until you press the hotkey); `aria toggle` shows or hides
//! it, and `aria show` only shows it (a notification's click uses that), starting it first when it
//! is not running. `aria update-ready` (run by the installer after it replaced the files) asks the running dock to
//! restart into the new binary as soon as it is hidden and no session has a run going; it fails when no dock answers,
//! and the installer then restarts the service itself.
//! `aria serve` is the same service with no window at all (a build without the `gui` feature has nothing else): for a
//! computer with no desktop, which terminals reach over ssh (`aria connect`). It answers `aria quit` and
//! `aria update-ready` as the dock does; a dock and a serve on one computer exclude each other.
//! `aria bar install|remove` puts Aria's face in the Omarchy bar or takes it out (the installer runs it).
//! `aria client-stdio` is the bridge `ssh <host> aria client-stdio` runs: the client's lines on stdin and stdout, the
//! machine's clients socket on the other side.
//! `aria hub-import` is what the owner's computer runs on a hub (`ssh <host> aria hub-import`): it reads one line from
//! stdin, a bundle of settings and credentials, writes its files into `~/.aria` (mode 600, the old ones kept as
//! `.bak`) and prints the names it wrote.
//! `aria hub init <user@host>` (English only: an admin command) sets a server up as a hub over ssh: it probes the host,
//! prints a plan, and only after a yes uploads the tree and the binary, runs the installer, sends the bundle to
//! `hub-import`, restarts the service and, with `--client-key`, allows a restricted key.
//! `aria --version` prints the version.

mod attachments;
mod chat;
mod checkpoints;
mod clients;
mod connect;
mod control;
#[cfg(feature = "gui")]
mod dictation;
#[cfg(feature = "gui")]
mod dock;
mod face;
mod face_motion;
mod handlers;
mod hub_admin;
mod hub_tui;
mod hub_ui;
mod jobs;
mod jobs_socket;
mod keyring_cli;
mod lines;
mod omarchy_bar;
mod schedule_store;
mod serve;
mod service;
mod stdio_bridge;
mod term;
mod trace;
#[cfg(feature = "gui")]
mod ui;
#[cfg(feature = "gui")]
mod visibility;
mod watch;
mod worktree;

// A Mac has no build without a window: its dock is the only thing the Mac code is written for.
#[cfg(all(target_os = "macos", not(feature = "gui")))]
compile_error!("the `gui` feature is required on macOS: only Linux has a build without a window (`aria serve`)");

#[cfg(feature = "gui")]
use aria_core::bridge::DockMessage;
#[cfg(feature = "gui")]
use aria_core::paths::AriaPaths;
#[cfg(feature = "gui")]
use aria_core::shortcut::{default_shortcut, InUse, Platform as ShortcutPlatform};
#[cfg(all(feature = "gui", target_os = "linux"))]
use aria_hyprland::Registration;
#[cfg(feature = "gui")]
use chat::PlatformNotifier;
#[cfg(feature = "gui")]
use dock::Dock;
#[cfg(feature = "gui")]
use service::Service;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "gui")]
use std::sync::{mpsc, Arc};
use std::time::Duration;
#[cfg(feature = "gui")]
use visibility::Event;

/// What this run of `aria` is.
#[derive(Debug, PartialEq)]
enum Mode {
    /// The terminal client (`aria` in a terminal).
    Client,
    /// The service: the dock and every session (`aria dock`, or plain `aria` with no terminal, as an old unit runs it).
    /// A build without a window runs it as `Serve`.
    Dock,
    /// The service with no window (`aria serve`).
    Serve,
    /// One of the small commands (`toggle`, `quit`, `secret`…).
    Command,
}

fn mode(arg: Option<&str>, terminal: bool) -> Mode {
    match arg {
        None if terminal => Mode::Client,
        None | Some("dock") => Mode::Dock,
        Some("serve") => Mode::Serve,
        Some(_) => Mode::Command,
    }
}

fn main() {
    use std::io::IsTerminal;
    let arg = std::env::args().nth(1);
    match mode(arg.as_deref(), std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        Mode::Client => std::process::exit(term::run()),
        Mode::Dock => return run_dock(),
        Mode::Serve => std::process::exit(serve::run()),
        Mode::Command => {}
    }
    match arg.as_deref() {
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
        Some("quit") => match control::ask(&control::control_socket_path(), "quit", Duration::from_secs(2)) {
            Ok(_) => println!("Aria is stopping (`aria start` starts it again)"),
            Err(error)
                if error.kind() == std::io::ErrorKind::NotFound
                    || error.kind() == std::io::ErrorKind::ConnectionRefused =>
            {
                println!("Aria is not running")
            }
            Err(error) => {
                eprintln!("aria: the dock did not take the quit: {error}");
                std::process::exit(1);
            }
        },
        Some("connect") => match std::env::args().nth(2).or_else(connect::default_hub) {
            Some(host) => std::process::exit(term::run_remote(&host)),
            None => {
                eprintln!(
                    "aria: no hub is set up on this computer: run `aria hub init user@host` to set one up, or give the host: aria connect [user@]host"
                );
                std::process::exit(2);
            }
        },
        Some("client-stdio") => std::process::exit(stdio_bridge::command()),
        Some("hub-import") => std::process::exit(hub_import()),
        Some("hub") => {
            let args: Vec<String> = std::env::args().skip(2).collect();
            std::process::exit(hub_admin::hub_command(&args));
        }
        Some("start") => start_service(),
        Some("secret") => {
            let args: Vec<String> = std::env::args().skip(2).collect();
            std::process::exit(keyring_cli::secret_command(&args));
        }
        Some("keyring") => std::process::exit(keyring_cli::keyring_command(std::env::args().nth(2).as_deref())),
        Some("bar") => std::process::exit(omarchy_bar::bar_command(std::env::args().nth(2).as_deref())),
        Some("--version" | "-V" | "version") => println!("aria {}", env!("CARGO_PKG_VERSION")),
        Some(other) => {
            eprintln!("aria: unknown command '{other}'\nusage: aria [dock|serve|toggle|show|start|quit|keyring|secret|bar|update-ready|connect|client-stdio|hub-import|hub|--version]");
            std::process::exit(2);
        }
        None => unreachable!("plain aria is the client or the dock"),
    }
}

/// `aria hub-import`: one line of stdin (at most 16 MiB) is the bundle; each file it wrote is printed on its own line.
fn hub_import() -> i32 {
    let result = hub_admin::read_bundle_line(std::io::stdin().lock(), hub_admin::MAX_BUNDLE_BYTES)
        .and_then(|line| hub_admin::import(&aria_core::paths::AriaPaths::from_env(), &line));
    match result {
        Ok(written) => {
            for name in written {
                println!("{name}");
            }
            0
        }
        Err(why) => {
            eprintln!("aria hub-import: {why}");
            1
        }
    }
}

/// A build without a window has no dock to run: `aria dock` (what a terminal's `aria` and `aria toggle` start when
/// nothing answers, and what an old unit runs) is the service with no window, so they all keep working.
#[cfg(not(feature = "gui"))]
fn run_dock() {
    std::process::exit(serve::run());
}

#[cfg(feature = "gui")]
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
    face::start();

    // A restart into a new version keeps this process and its children: the old page's WebKit processes are reaped as
    // they end, before the new page starts its own.
    #[cfg(target_os = "linux")]
    control::reap_inherited(Duration::from_secs(120));

    // WebKitGTK needs its renderer flag before GTK starts; on Mac this creates the app object.
    ui::init();

    // Everything that is not the window: the sessions, the background work, the scheduled tasks, the terminals.
    let dock_visible = Arc::new(AtomicBool::new(false));
    let visible_now = dock_visible.clone();
    let mut service =
        Service::start(service::Options { notifier: Arc::new(PlatformNotifier), dock_visible: dock_visible.clone() });
    let first_run = service.first_run;

    let dock = {
        let inlet = service.page_inlet();
        Dock::new(dock_visible, Box::new(move |message, _dock| inlet.handle(message)))
    };
    // The page needs the shortcut for its notice and its "you are set up" screen: the default (Super + A on Omarchy,
    // Option + Space on macOS) until the one in use is known, and again whenever it changes.
    let default_hotkey = InUse { shortcut: default_shortcut(ShortcutPlatform::current()), taken: Vec::new() };
    dock.send(&DockMessage::platform(&default_hotkey, ShortcutPlatform::current()));
    dock.send(&DockMessage::Version { version: env!("CARGO_PKG_VERSION").into() });
    if first_run {
        dock.show_notice();
    }

    // Commands arrive on a socket thread and are handled on the UI thread. The hotkey sends the same
    // "toggle" line, so there is one path from both.
    let (sender, receiver) = mpsc::channel::<String>();
    control::answer_commands(listener, sender.clone());

    // The hotkey. On Linux it is registered as soon as Hyprland can take it (it may not be up when a
    // service starts), and again after every config reload and every change of the `shortcut` setting (a taken one
    // falls back); the thread reports how it is going, the problem banner follows it and the page hears which shortcut
    // is in use. On Mac Carbon registers it without permissions (the chosen one or the first free fallback), and again
    // when the `shortcut` setting changes; when another app holds every candidate (or Carbon's handler cannot be
    // installed) the problem banner says so.
    #[cfg(target_os = "linux")]
    let (registration_sender, registration_receiver) = mpsc::channel::<Registration>();
    #[cfg(target_os = "linux")]
    let hotkey = {
        let exe = std::env::current_exe().expect("own path");
        let paths = AriaPaths::from_env();
        let source = move || shortcut_setting(&paths);
        aria_hyprland::keep_hotkey_registered(exe, source, Duration::from_secs(2), move |state| {
            let _ = registration_sender.send(state);
        })
    };
    // The setting the Mac's shortcut was registered for and the shortcut in use (`None` while nothing could be): a
    // change of the settings file re-registers only when it moved; every 10 s a failed registration (or a handler that
    // could not be installed) is tried again, and so is the chosen one while a fallback stands in for it
    // (`aria_mac::carbon::pass_due`).
    #[cfg(target_os = "macos")]
    let mut mac_held = {
        let chosen = shortcut_setting(&AriaPaths::from_env());
        let hotkey_sender = sender.clone();
        let registered = aria_mac::install_hotkey(&chosen, move || {
            let _ = hotkey_sender.send("hotkey".to_string());
        });
        report_mac_hotkey(&dock, None, &chosen, registered, true)
    };

    // Stopping the dock (systemd, launchd, a terminal's Ctrl+C) ends its background work and Pis, never leaves them.
    install_quit_handlers();
    // Quit Aria (Cmd+Q) exits through AppKit without a signal: the same stop runs first, or a dev server Aria started would
    // live on unseen (macOS has no PR_SET_PDEATHSIG to end it with the dock).
    #[cfg(target_os = "macos")]
    {
        let stop = service.stopper();
        ui::on_terminate(stop);
    }
    // The UI ticks since the start, for the Mac's shortcut retries (the service counts its own).
    #[cfg(target_os = "macos")]
    let mut ticks: u32 = 0;
    // macOS: Aria's face beside the notch; a click on it does what the shortcut does.
    #[cfg(target_os = "macos")]
    let face_wing = {
        let click = sender.clone();
        let wing = objc2_foundation::MainThreadMarker::new()
            .and_then(|mtm| dock::face_wing::FaceWing::new(mtm, Box::new(move || drop(click.send("toggle".into())))));
        if wing.is_none() {
            note("no screen for Aria's face");
        }
        wing
    };
    ui::every(16, move || {
        #[cfg(target_os = "macos")]
        if let Some(wing) = &face_wing {
            wing.tick();
        }
        let tick = service.tick(visible_now.load(Ordering::SeqCst));
        if tick.quit {
            std::process::exit(0);
        }
        #[cfg(target_os = "macos")]
        {
            ticks = ticks.wrapping_add(1);
        }
        if tick.settings_changed {
            dock.send(&DockMessage::SettingsChanged);
            #[cfg(target_os = "linux")]
            hotkey.settings_changed();
        }
        // On the main thread, as Carbon wants: after a change of the setting, while nothing is registered and while a
        // fallback is in use (the retries are quiet: only a change of the shortcut in use is reported).
        #[cfg(target_os = "macos")]
        {
            let (settings_changed, retry_tick) = (tick.settings_changed, ticks.is_multiple_of(MAC_HOTKEY_RETRY_TICKS));
            if settings_changed || retry_tick {
                let chosen = shortcut_setting(&AriaPaths::from_env());
                if aria_mac::carbon::pass_due(mac_held.as_ref(), &chosen, settings_changed, retry_tick) {
                    let registered = aria_mac::change_hotkey(&chosen);
                    let loud = settings_changed && mac_held.as_ref().is_none_or(|held| held.setting != chosen);
                    // A failure keeps nothing, so the next tick tries again.
                    mac_held = report_mac_hotkey(&dock, mac_held.as_ref(), &chosen, registered, loud);
                }
            }
        }
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
                "quit" => {
                    note("asked to quit");
                    request_quit()
                }
                "update-ready" => service.update_ready(),
                _ => {}
            }
        }
        for reply in &tick.to_page {
            dock.send(reply);
        }
        #[cfg(target_os = "linux")]
        while let Ok(state) = registration_receiver.try_recv() {
            match state {
                Registration::Registered(in_use) => {
                    dock.clear_problem();
                    eprintln!("{}", ready_line(&in_use));
                    dock.send(&DockMessage::platform(&in_use, ShortcutPlatform::Linux));
                }
                Registration::WaitingForHyprland => dock.set_problem("hyprland-not-running"),
                Registration::ClassicConfig => dock.set_problem("hyprland-classic-config"),
            }
        }
    });

    // The shortcut in use is logged once it is bound (above).
    ui::run();
}

/// The `shortcut` setting; an unreadable settings file means the default, as an empty setting does.
#[cfg(feature = "gui")]
fn shortcut_setting(paths: &AriaPaths) -> String {
    aria_core::settings::load(paths).ok().flatten().map(|s| s.shortcut).unwrap_or_default()
}

/// The log's line once the shortcut is registered: `aria: ready, press SUPER + SHIFT + A (SUPER + A taken)`.
#[cfg(feature = "gui")]
fn ready_line(in_use: &InUse) -> String {
    let taken: Vec<String> = in_use.taken.iter().map(|s| s.label()).collect();
    if taken.is_empty() {
        format!("aria: ready, press {}", in_use.shortcut.label())
    } else {
        format!("aria: ready, press {} ({} taken)", in_use.shortcut.label(), taken.join(", "))
    }
}

/// A Carbon registration's result reaches the log, the problem banner and the page (which shortcut is in use, or none:
/// an empty `hotkey`, with every candidate in `taken` when they are all taken, or none when Carbon's handler could not
/// be installed). A success is reported when the shortcut in use differs from what `before` held (a retry that keeps
/// the same fallback says nothing); a failure only when `loud` (the first pass, a change of the setting). What the
/// dock holds now: `None` after a failure, so the next retry tick tries again.
#[cfg(target_os = "macos")]
fn report_mac_hotkey(
    dock: &Dock,
    before: Option<&aria_mac::carbon::Held>,
    chosen: &str,
    registered: Result<InUse, aria_mac::carbon::HotkeyError>,
    loud: bool,
) -> Option<aria_mac::carbon::Held> {
    match registered {
        Ok(in_use) => {
            if loud || before.is_none_or(|held| held.in_use != in_use) {
                eprintln!("{}", ready_line(&in_use));
                dock.clear_problem();
                dock.send(&DockMessage::platform(&in_use, ShortcutPlatform::Mac));
            }
            Some(aria_mac::carbon::Held { setting: chosen.to_string(), in_use })
        }
        Err(error) => {
            if loud {
                eprintln!("aria: could not register the shortcut: {error}");
                dock.set_problem(error.problem_code());
                dock.send(&DockMessage::Platform {
                    hotkey: String::new(),
                    taken: error.taken().iter().map(|s| s.display()).collect(),
                    os: ShortcutPlatform::Mac.name().into(),
                    animation: aria_core::bridge::animation_of(ShortcutPlatform::Mac).into(),
                });
            }
            None
        }
    }
}

/// How often a Mac whose shortcut could not be registered (or that uses a fallback) tries again, in UI ticks of 16 ms:
/// every 10 s.
#[cfg(target_os = "macos")]
const MAC_HOTKEY_RETRY_TICKS: u32 = 10_000 / 16;

/// A line in the dock's log with the time (UTC), for what happens between a keypress and the dock on screen: when the
/// dock does not open, the log says whether the shortcut arrived, what state the dock was in and whether the page
/// answered.
pub(crate) fn note(message: &str) {
    let seconds = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (h, m, s) = (seconds / 3600 % 24, seconds / 60 % 60, seconds % 60);
    eprintln!("[{h:02}:{m:02}:{s:02} UTC] aria: {message}");
}

/// Set by SIGTERM, SIGINT or SIGHUP; the UI thread's tick sees it. A handler can do little more than this safely.
static QUIT: AtomicBool = AtomicBool::new(false);

/// Quits the dock on its next tick, the way a signal does: background work and Pis are stopped first (Cmd+Q twice in the
/// page, `aria quit`).
pub(crate) fn request_quit() {
    QUIT.store(true, Ordering::SeqCst);
}

/// The systemd unit `aria start` starts, from whether each is installed: the dock's wins when both are (a computer with
/// both is unusual), a hub has only the headless one, and none is no unit at all.
#[cfg(any(target_os = "linux", test))]
fn unit_to_start(dock_installed: bool, serve_installed: bool) -> Option<&'static str> {
    if dock_installed {
        Some("aria.service")
    } else if serve_installed {
        Some("aria-serve.service")
    } else {
        None
    }
}

/// Whether the user's systemd knows `unit` (`systemctl --user cat` reads the unit file and changes nothing).
#[cfg(target_os = "linux")]
fn unit_installed(unit: &str) -> bool {
    std::process::Command::new("systemctl")
        .args(["--user", "cat", unit])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// `aria start`: the service again under the service manager that starts it at login, so it is supervised as usual. On
/// Linux that is the dock's unit or, on a hub, the headless one.
fn start_service() {
    #[cfg(target_os = "macos")]
    let status = std::process::Command::new("launchctl")
        .arg("kickstart")
        .arg(format!("gui/{}/com.aria.dock", unsafe { libc::getuid() }))
        .status();
    #[cfg(target_os = "linux")]
    let status = match unit_to_start(unit_installed("aria.service"), unit_installed("aria-serve.service")) {
        Some(unit) => {
            if unit != "aria.service" {
                println!("Starting {unit} (this computer has no aria.service)");
            }
            std::process::Command::new("systemctl").args(["--user", "start", unit]).status()
        }
        None => Err(std::io::Error::new(std::io::ErrorKind::NotFound, "no Aria unit is installed")),
    };
    match status {
        Ok(status) if status.success() => println!("Aria is running"),
        _ => {
            eprintln!("aria: the service did not start (is Aria installed?); `aria show` starts the dock by itself");
            std::process::exit(1);
        }
    }
}

extern "C" fn on_quit_signal(_: libc::c_int) {
    QUIT.store(true, Ordering::SeqCst);
}

/// The quit signals set [`QUIT`] instead of ending the dock at once, so its tick can stop what it started first.
fn install_quit_handlers() {
    for signal in [libc::SIGTERM, libc::SIGINT, libc::SIGHUP] {
        // SAFETY: a zeroed `sigaction` is a valid empty one; the handler only stores to an atomic.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_quit_signal as extern "C" fn(libc::c_int) as libc::sighandler_t;
            action.sa_flags = libc::SA_RESTART;
            libc::sigemptyset(&mut action.sa_mask);
            if libc::sigaction(signal, &action, std::ptr::null_mut()) != 0 {
                eprintln!("aria: could not handle signal {signal}: {}", std::io::Error::last_os_error());
            }
        }
    }
}

/// Tests that start child processes or hold a listening socket take turns: a fork in one test
/// leaves copies of every open descriptor in the child until it execs, which keeps a dropped socket
/// "listening" and a freshly written script "busy" for a moment.
#[cfg(test)]
pub(crate) static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {

    #[test]
    fn with_a_terminal_plain_aria_is_the_client() {
        assert_eq!(mode(None, true), Mode::Client);
    }

    #[test]
    fn without_one_it_is_the_dock() {
        assert_eq!(mode(None, false), Mode::Dock, "an old unit that still runs plain aria keeps its dock");
        assert_eq!(mode(Some("dock"), true), Mode::Dock);
        assert_eq!(mode(Some("dock"), false), Mode::Dock);
        assert_eq!(mode(Some("toggle"), true), Mode::Command);
    }

    #[test]
    fn serve_is_the_service_without_a_window() {
        assert_eq!(mode(Some("serve"), true), Mode::Serve);
        assert_eq!(mode(Some("serve"), false), Mode::Serve, "a unit runs it with no terminal");
    }
    #[test]
    fn the_unit_to_start_is_the_one_that_is_installed() {
        assert_eq!(unit_to_start(true, false), Some("aria.service"));
        assert_eq!(unit_to_start(false, true), Some("aria-serve.service"), "a hub has only the headless unit");
        assert_eq!(unit_to_start(true, true), Some("aria.service"), "both: the desktop wins");
        assert_eq!(unit_to_start(false, false), None);
    }
    use super::*;
}
