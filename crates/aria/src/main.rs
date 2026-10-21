//! `aria` runs the dock (resident, hidden until you press the hotkey); `aria toggle` shows or hides
//! it, and `aria show` only shows it (a notification's click uses that), starting it first when it
//! is not running. `aria update-ready` (run by the installer after it replaced the files) asks the running dock to
//! restart into the new binary as soon as it is hidden and no session has a run going; it fails when no dock answers,
//! and the installer then restarts the service itself.

mod attachments;
mod chat;
mod control;
mod dock;
mod handlers;
mod jobs;
mod jobs_socket;
mod lines;
mod ui;
mod visibility;
mod watch;
mod worktree;

use aria_core::bridge::{DockMessage, PageMessage};
use aria_core::paths::AriaPaths;
#[cfg(target_os = "linux")]
use aria_hyprland::Registration;
use chat::{ChatCtx, PlatformNotifier, Sessions};
use dock::{Dock, HOTKEY_LABEL};
use jobs::{Jobs, UPDATE_NOTICE_AFTER};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, OnceLock};
use std::time::{Duration, Instant};
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

    // The background work of every session, one for the dock's whole life: its spawner thread is what every job is tied
    // to, so it must outlive them all. The page hears of every change (at most four times a second).
    let jobs = {
        let (slot, sender) = (Arc::new(OnceLock::<Jobs>::new()), reply_sender.clone());
        let seen = slot.clone();
        let jobs = Jobs::new(
            paths.jobs(),
            Arc::new(move || {
                if let Some(jobs) = seen.get() {
                    let _ = sender.send(chat::jobs_message(jobs));
                }
            }),
        );
        let _ = slot.set(jobs.clone());
        jobs
    };

    // The chat owns Pi. Its messages go through one worker thread, in order, so a send is never
    // overtaken by the abort that follows it.
    let dock_visible = Arc::new(AtomicBool::new(false));
    let visible_now = dock_visible.clone();
    let run_dir =
        control::control_socket_path().parent().map(std::path::Path::to_path_buf).unwrap_or_else(std::env::temp_dir);
    // The sandbox hides `$XDG_RUNTIME_DIR` from boxed commands; without it (a run dir under /tmp) the sessions get no
    // jobs socket. macOS has no box (every command asks first), so its run dir is as hidden as it needs to be.
    #[cfg(target_os = "linux")]
    let hidden_dir = std::env::var_os("XDG_RUNTIME_DIR").filter(|dir| !dir.is_empty()).map(std::path::PathBuf::from);
    #[cfg(target_os = "macos")]
    let hidden_dir = Some(run_dir.clone());
    let chat = {
        let sender = reply_sender.clone();
        Sessions::new(
            ChatCtx {
                paths,
                notifier: Arc::new(PlatformNotifier),
                dock_visible: dock_visible.clone(),
                chat_shown: Arc::new(AtomicBool::new(true)),
                run_dir,
                hidden_dir,
                path_var: std::env::var("PATH").unwrap_or_default(),
                quota_fetcher: Arc::new(aria_core::quota::real_fetcher),
                jobs: jobs.clone(),
            },
            Arc::new(move |message| {
                let _ = sender.send(message);
            }),
        )
    };
    chat.forget_gone_sessions();
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

    // Stopping the dock (systemd, launchd, a terminal's Ctrl+C) ends its background work and Pis, never leaves them.
    install_quit_handlers();
    let mut was_visible = false;
    let mut update_ready = false;
    let mut jobs_wait = JobsWait::default();
    let mut ticks: u32 = 0;
    ui::every(16, move || {
        if quit_requested(&QUIT, || {
            eprintln!("aria: stopping");
            jobs.stop_all();
            chat_for_visibility.stop_for_update();
        }) {
            std::process::exit(0);
        }
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
        let visible = visible_now.load(Ordering::SeqCst);
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
        // An update kept waiting by background work alone (a dev server can run all day): its sessions say how to
        // update now, once per wait.
        if update_ready && ticks.is_multiple_of(60) {
            let only_jobs = !chat_for_visibility.busy_without_jobs() && jobs.active() > 0;
            if jobs_wait.due(only_jobs, Instant::now()) {
                chat_for_visibility.tell_update_waits();
            }
        }
        // A new binary is installed: it takes over once nobody is looking and no run or background work would be cut.
        // The tabs and their conversations come back as after any restart.
        if update_ready && ticks.is_multiple_of(60) && !visible && !chat_for_visibility.busy() {
            eprintln!("aria: restarting into the new version");
            chat_for_visibility.stop_for_update();
            // Nothing should be running (`busy` said so), but a job started since must not outlive this dock unseen.
            jobs.stop_all();
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

/// Set by SIGTERM, SIGINT or SIGHUP; the UI thread's tick sees it. A handler can do little more than this safely.
static QUIT: AtomicBool = AtomicBool::new(false);

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

/// Whether the dock should exit now: when a quit was asked for, after `shutdown` (which stops the background work and
/// the sessions' Pis) ran.
fn quit_requested(flag: &AtomicBool, shutdown: impl FnOnce()) -> bool {
    if !flag.load(Ordering::SeqCst) {
        return false;
    }
    shutdown();
    true
}

/// How long an update has waited for background work alone, so the sessions with that work are told once per wait.
#[derive(Default)]
struct JobsWait {
    since: Option<Instant>,
    told: bool,
}

impl JobsWait {
    /// `only_jobs`: the update waits for background work and nothing else right now. True once, when that has lasted
    /// `UPDATE_NOTICE_AFTER`; anything else in the way (a run, a dialog) or no work left starts a new wait.
    fn due(&mut self, only_jobs: bool, now: Instant) -> bool {
        if !only_jobs {
            *self = JobsWait::default();
            return false;
        }
        let since = *self.since.get_or_insert(now);
        if self.told || now.duration_since(since) < UPDATE_NOTICE_AFTER {
            return false;
        }
        self.told = true;
        true
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quit_stops_the_work_before_the_dock_exits() {
        let flag = AtomicBool::new(false);
        let mut stopped = 0;
        assert!(!quit_requested(&flag, || stopped += 1), "no quit asked for: the dock goes on");
        assert_eq!(stopped, 0);
        flag.store(true, Ordering::SeqCst);
        assert!(quit_requested(&flag, || stopped += 1));
        assert_eq!(stopped, 1, "stopped before the exit");
    }

    #[test]
    fn an_update_waiting_for_jobs_alone_is_told_once_per_wait() {
        let start = Instant::now();
        let mut wait = JobsWait::default();
        assert!(!wait.due(true, start));
        assert!(!wait.due(true, start + UPDATE_NOTICE_AFTER - Duration::from_secs(1)));
        assert!(wait.due(true, start + UPDATE_NOTICE_AFTER));
        assert!(!wait.due(true, start + UPDATE_NOTICE_AFTER * 2), "once per wait");
        // A run in between starts a new wait.
        assert!(!wait.due(false, start + UPDATE_NOTICE_AFTER * 2));
        let again = start + UPDATE_NOTICE_AFTER * 3;
        assert!(!wait.due(true, again));
        assert!(wait.due(true, again + UPDATE_NOTICE_AFTER));
    }
}
