//! The service: what Aria does with no window. Sessions and their Pis, the background work, the scheduled tasks, the
//! terminal clients' hub and the keys; the dock (`run_dock`) is a window around it. A [`Service`] touches nothing of the
//! window: the dock hands it whether it is visible and takes back what the page must hear, once per UI tick.

use crate::chat::{self, ChatCtx, Sessions};
use crate::jobs::{Jobs, UPDATE_NOTICE_AFTER};
use crate::{clients, control, handlers, keyring_cli, note, schedule_store, QUIT};
use aria_core::bridge::{DockMessage, PageMessage};
use aria_core::paths::AriaPaths;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, OnceLock};
use std::time::Instant;

/// How often the turn watchdog looks, in UI ticks of 16 ms: every 30 s.
const WATCHDOG_TICKS: u32 = 30_000 / 16;

/// What the service takes from its host.
pub struct Options {
    /// Raises the notification for a prompt nobody sees.
    pub notifier: Arc<dyn chat::Notifier>,
    /// Whether the dock is on screen (a service with no window leaves it false); the host sets it.
    pub dock_visible: Arc<AtomicBool>,
}

/// Where the service lives: its home, and the folders its sockets and its sandbox use.
struct Home {
    paths: AriaPaths,
    vault: keyring_cli::Shared,
    first_run: bool,
    run_dir: PathBuf,
    hidden_dir: Option<PathBuf>,
    /// Where terminals connect (`None`: nobody can, as in a test).
    clients_socket: Option<PathBuf>,
}

/// What one tick of the host's loop found for the host to do.
#[derive(Debug, Default)]
pub struct Tick {
    /// The service has stopped its background work and its Pis: the host exits now.
    pub quit: bool,
    /// `settings.json` changed: the page reads the settings again (and the host's own watchers look at it).
    pub settings_changed: bool,
    /// What the page must hear: the replies that are not for a terminal.
    pub to_page: Vec<DockMessage>,
}

/// How a page's message gets handled; cloned into the window's message callback, which outlives any borrow of the
/// [`Service`]. A service with no window has no page: only a build with one (and the tests that stand in for it) has
/// the inlet.
#[cfg(any(test, feature = "gui"))]
#[derive(Clone)]
pub struct PageInlet {
    ctx: Arc<handlers::Ctx>,
    chat: mpsc::Sender<PageMessage>,
    replies: mpsc::Sender<DockMessage>,
}

#[cfg(any(test, feature = "gui"))]
impl PageInlet {
    /// Chat messages go to the chat's worker thread; the rest are handled here, on a thread of their own when they
    /// block.
    pub fn handle(&self, message: PageMessage) {
        if handlers::is_chat(&message) {
            let _ = self.chat.send(message);
            return;
        }
        let (ctx, sender) = (self.ctx.clone(), self.replies.clone());
        let emit = move |reply: DockMessage| {
            let _ = sender.send(reply);
        };
        if handlers::is_blocking(&message) {
            std::thread::spawn(move || handlers::handle(message, &ctx, &emit));
        } else {
            handlers::handle(message, &ctx, &emit);
        }
    }
}

pub struct Service {
    /// There was no settings file when the service started: the page shows its welcome.
    pub first_run: bool,
    /// Why terminals cannot connect (the clients socket did not bind), when that is so; the dock only logs it, `aria
    /// serve` has nothing else to do and exits.
    pub clients_error: Option<String>,
    #[cfg(any(test, feature = "gui"))]
    inlet: PageInlet,
    hub: clients::Hub,
    jobs: Jobs,
    chat: Sessions,
    vault: keyring_cli::Shared,
    schedules: schedule_store::ScheduleStore,
    scheduling: Arc<AtomicBool>,
    settings_watch: chat::SettingsWatch,
    replies: mpsc::Receiver<DockMessage>,
    was_visible: bool,
    update_ready: bool,
    jobs_wait: JobsWait,
    ticks: u32,
}

/// Which sessions the service has: the ones in `sessions.json`, read when asked (a session a client opens is written there
/// by the chat's worker, and a tab the page saves too).
fn known_sessions(paths: AriaPaths) -> Arc<dyn Fn(&str) -> bool + Send + Sync> {
    Arc::new(move |id| aria_core::sessions::load(&paths).sessions.iter().any(|entry| entry.id == id))
}

impl Service {
    /// Opens Aria's home (the keys move into the keyring and the sealed files are written live before anything below
    /// reads a credential), starts the sessions' worker and listens for terminals.
    pub fn start(options: Options) -> Service {
        let paths = AriaPaths::from_env();
        let first_run = !paths.settings().exists();
        let initial = aria_core::settings::load(&paths).ok().flatten();
        let (paths, vault) = keyring_cli::open_for_dock(paths, initial.as_ref());
        let run_dir = run_dir();
        let hidden_dir = hidden_dir(&run_dir);
        let home = Home {
            paths,
            vault: Arc::new(std::sync::Mutex::new(vault)),
            first_run,
            run_dir,
            hidden_dir,
            clients_socket: Some(clients::socket_path()),
        };
        Service::assemble(home, options)
    }

    fn assemble(home: Home, options: Options) -> Service {
        let Home { paths, vault, first_run, run_dir, hidden_dir, clients_socket } = home;
        // Replies from any thread come back to the host's tick through this channel.
        let (reply_sender, replies) = mpsc::channel::<DockMessage>();
        // The scheduled tasks: changed through every session's jobs socket, run by the scheduler on the tick.
        let schedules = schedule_store::ScheduleStore::open(paths.clone());
        let scheduling = Arc::new(AtomicBool::new(false));
        keyring_cli::spawn_sync(vault.clone());
        let settings_watch = chat::SettingsWatch::new(paths.settings());
        // The chat's messages go through one worker thread (below); the handlers reach it too, to restart every Pi after
        // an MCP import.
        let (chat_sender, chat_receiver) = mpsc::channel::<PageMessage>();
        let ctx = {
            let chat_sender = chat_sender.clone();
            Arc::new(handlers::Ctx {
                restart_engines: Arc::new(move || {
                    let _ = chat_sender.send(PageMessage::EnginesRestart);
                }),
                ..handlers::Ctx::real(paths.clone())
            })
        };

        // The background work of every session, one for the service's whole life: its spawner thread is what every job
        // is tied to, so it must outlive them all. The page hears of every change (at most four times a second).
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

        // Terminal clients (`aria` in a terminal): they speak the page's messages; what they may send goes to the chat's
        // worker or, for a secret, to the handlers, and what the service says reaches them through `hub.route`.
        let hub = {
            let (ctx, sender) = (ctx.clone(), reply_sender.clone());
            let handle: Arc<dyn Fn(PageMessage) + Send + Sync> = Arc::new(move |message| {
                let (ctx, sender) = (ctx.clone(), sender.clone());
                std::thread::spawn(move || {
                    handlers::handle(message, &ctx, &move |reply| {
                        let _ = sender.send(reply);
                    })
                });
            });
            clients::Hub::new(chat_sender.clone(), handle, known_sessions(paths.clone()))
        };
        // The chat owns Pi. Its messages go through one worker thread, in order, so a send is never overtaken by the
        // abort that follows it.
        let chat = {
            let sender = reply_sender.clone();
            Sessions::new(
                ChatCtx {
                    paths,
                    notifier: options.notifier,
                    dock_visible: options.dock_visible,
                    chat_shown: Arc::new(AtomicBool::new(true)),
                    run_dir,
                    hidden_dir,
                    path_var: std::env::var("PATH").unwrap_or_default(),
                    quota_fetcher: Arc::new(aria_core::quota::real_fetcher),
                    jobs: jobs.clone(),
                    schedules: schedules.clone(),
                    watchers: hub.watchers(),
                },
                Arc::new(move |message| {
                    let _ = sender.send(message);
                }),
            )
        };
        chat.forget_gone_sessions();
        let mut clients_error = None;
        if let Some(socket) = clients_socket {
            match clients::bind(&socket) {
                Ok(listener) => hub.serve(listener),
                Err(error) => {
                    let message = format!("terminals cannot connect: {}: {error}", socket.display());
                    note(&message);
                    clients_error = Some(message);
                }
            }
            // The machines linked to this hub come in beside the terminals; a failure here only means none can.
            let machines_socket = socket.with_file_name("machines.sock");
            match clients::bind(&machines_socket) {
                Ok(listener) => hub.serve_machines(listener),
                Err(error) => note(&format!("machines cannot link: {}: {error}", machines_socket.display())),
            }
        }
        let worker = chat.clone();
        std::thread::spawn(move || {
            for message in chat_receiver {
                worker.handle(message);
            }
        });

        Service {
            first_run,
            clients_error,
            #[cfg(any(test, feature = "gui"))]
            inlet: PageInlet { ctx, chat: chat_sender, replies: reply_sender },
            hub,
            jobs,
            chat,
            vault,
            schedules,
            scheduling,
            settings_watch,
            replies,
            was_visible: false,
            update_ready: false,
            jobs_wait: JobsWait::default(),
            ticks: 0,
        }
    }

    /// What the window's message callback does with a page message: it owns the inlet, which outlives any borrow of the
    /// service.
    #[cfg(any(test, feature = "gui"))]
    pub fn page_inlet(&self) -> PageInlet {
        self.inlet.clone()
    }

    /// The installer replaced the files: the service restarts into the new binary once nobody is looking and nothing
    /// would be cut.
    pub fn update_ready(&mut self) {
        eprintln!("aria: a new version is installed; restarting once the dock is hidden and idle");
        self.update_ready = true;
    }

    /// Stops the background work, the sessions' Pis and the vault (a signal, Quit, an update): nothing is left running
    /// unseen.
    pub fn shutdown(&self) {
        stop(&self.jobs, &self.chat, &self.vault);
    }

    /// The same stop, for a callback that must own what it calls (Cmd+Q ends the app without a signal).
    #[cfg(target_os = "macos")]
    pub fn stopper(&self) -> impl Fn() + 'static {
        let (jobs, chat, vault) = (self.jobs.clone(), self.chat.clone(), self.vault.clone());
        move || stop(&jobs, &chat, &vault)
    }

    /// One turn of the host's loop (the dock's 16 ms tick). `visible`: the dock is on screen. The checks run in this
    /// order: a quit asked for, the settings, the turn watchdog, the scheduled tasks, a dock that just hid, an update
    /// kept waiting by background work, the update itself, and last the replies.
    pub fn tick(&mut self, visible: bool) -> Tick {
        if quit_requested(&QUIT, || self.shutdown()) {
            return Tick { quit: true, ..Tick::default() };
        }
        // About once a second: a settings change (the settings screen, `/effort`, `/autocompact`, an editor) reaches
        // every session's Pi when its run settles, and the page reads the settings again.
        self.ticks = self.ticks.wrapping_add(1);
        let settings_changed = self.ticks.is_multiple_of(60) && self.settings_watch.changed();
        if settings_changed {
            // Off the host's thread: preparing the agent directory reads and writes files.
            let sessions = self.chat.clone();
            std::thread::spawn(move || sessions.settings_changed());
        }
        // Every 30 s, the turn watchdog: a run that got nothing from its Pi for `models.turnTimeoutMinutes` is stopped,
        // and the chat says so. Off the host's thread: it reads the settings.
        if self.ticks.is_multiple_of(WATCHDOG_TICKS) {
            let sessions = self.chat.clone();
            std::thread::spawn(move || sessions.check_stalled(Instant::now()));
        }
        // Once a second, the scheduled tasks: each due one runs in its own tab (a time missed while the computer slept or
        // Aria was off runs once, and only today's), and a run that waited for its busy tab starts once it is free. Off
        // the host's thread, one check at a time: starting a tab's Pi takes a moment.
        if self.ticks.is_multiple_of(60) && !self.scheduling.swap(true, Ordering::SeqCst) {
            let (sessions, schedules, scheduling) =
                (self.chat.clone(), self.schedules.clone(), self.scheduling.clone());
            std::thread::spawn(move || {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_millis() as i64);
                for (task, due_at) in schedules.take_due(now, &aria_core::schedules::SystemClock) {
                    let late = if aria_core::schedules::late(due_at, now) { ", late" } else { "" };
                    note(&format!("running the scheduled task {} ({}){late}", task.id, task.title));
                    sessions.start_scheduled(&task, due_at);
                }
                sessions.flush_scheduled();
                scheduling.store(false, Ordering::SeqCst);
            });
        }
        // A dock that just hid with a question still open needs its notification.
        if self.was_visible && !visible {
            self.chat.dock_visibility_changed();
        }
        self.was_visible = visible;
        // An update kept waiting by background work alone (a dev server can run all day): its sessions say how to
        // update now, once per wait.
        if self.update_ready && self.ticks.is_multiple_of(60) {
            let only_jobs = !self.chat.busy_without_jobs() && self.jobs.active() > 0;
            if self.jobs_wait.due(only_jobs, Instant::now()) {
                self.chat.tell_update_waits();
            }
        }
        // A new binary is installed: it takes over once nobody is looking and no run or background work would be cut.
        // The tabs and their conversations come back as after any restart.
        if self.update_ready && self.ticks.is_multiple_of(60) && !visible && !self.chat.busy() {
            eprintln!("aria: restarting into the new version");
            self.chat.stop_for_update();
            // Nothing should be running (`busy` said so), but a job started since must not outlive this service unseen.
            self.jobs.stop_all();
            // The live copies stay for the new process, which seals them when it opens the vault.
            keyring_cli::sync_now(&self.vault);
            let error = control::reexec();
            eprintln!("aria: could not start the new version: {error}");
            std::process::exit(75); // not a clean exit: systemd and launchd start the service again
        }
        let mut to_page = Vec::new();
        while let Ok(reply) = self.replies.try_recv() {
            if self.hub.route(&reply) {
                to_page.push(reply);
            }
        }
        Tick { quit: false, settings_changed, to_page }
    }
}

fn stop(jobs: &Jobs, chat: &Sessions, vault: &keyring_cli::Shared) {
    eprintln!("aria: stopping");
    jobs.stop_all();
    chat.stop_for_update();
    keyring_cli::close(vault);
}

/// Where the sessions' subagent and jobs sockets live: beside the control socket.
fn run_dir() -> PathBuf {
    control::control_socket_path().parent().map(Path::to_path_buf).unwrap_or_else(std::env::temp_dir)
}

/// The folder the sandbox hides from boxed commands: a session's jobs socket is opened only inside it, so a boxed
/// command can never reach it. On Linux it is the runtime directory the run dir is made in (`control::runtime_dir`:
/// `$XDG_RUNTIME_DIR`, else `/run/user/<uid>`); macOS has no box (every command asks first), so its run dir is as
/// hidden as it needs to be.
fn hidden_dir(run_dir: &Path) -> Option<PathBuf> {
    #[cfg(target_os = "linux")]
    let runtime = control::runtime_dir();
    #[cfg(target_os = "macos")]
    let runtime = run_dir.to_path_buf();
    hidden_dir_of(&runtime, run_dir)
}

/// `hidden_dir`'s choice, given where the runtime directory and the run dir are. A runtime directory that fell back to
/// `/tmp` (no `$XDG_RUNTIME_DIR` and no `/run/user/<uid>`) hides nothing: the sessions then get no jobs socket.
fn hidden_dir_of(runtime: &Path, run_dir: &Path) -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        return Some(run_dir.to_path_buf());
    }
    (runtime != Path::new("/tmp")).then(|| runtime.to_path_buf())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;
    use std::time::{Duration, Instant};

    struct Quiet;

    impl chat::Notifier for Quiet {
        fn send(&self, _: &str, _: &str, _: &str) {}
        fn dismiss(&self, _: &str) {}
    }

    fn scratch_service(dir: &std::path::Path) -> Service {
        let paths = AriaPaths::new(dir.join(".aria"));
        paths.ensure_root().unwrap();
        let run_dir = dir.join("run");
        let options = Options { notifier: Arc::new(Quiet), dock_visible: Arc::default() };
        Service::assemble(
            Home { paths, vault: Arc::default(), first_run: true, run_dir, hidden_dir: None, clients_socket: None },
            options,
        )
    }

    #[test]
    fn a_session_is_known_while_sessions_json_has_it() {
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths::new(dir.path().join(".aria"));
        paths.ensure_root().unwrap();
        let known = known_sessions(paths.clone());
        assert!(!known("aa"), "no file, no sessions");
        let entry = aria_core::sessions::add(&paths, "aa", "", None).unwrap();
        assert!(known(&entry.id));
        assert!(!known("zz"));
    }

    #[test]
    fn a_reply_for_a_terminal_does_not_reach_the_page() {
        let dir = tempfile::tempdir().unwrap();
        let mut service = scratch_service(dir.path());
        // A terminal asks for the session list, and the page asks too (its ids are its own).
        let (ours, theirs) = UnixStream::pair().unwrap();
        service.hub.add(theirs);
        ours.set_read_timeout(Some(Duration::from_millis(20))).unwrap();
        writeln!(&ours, r#"{{"type": "sessionsLoad", "id": 1}}"#).unwrap();
        let page_id = 1_792_760_400_000_000;
        service.inlet.replies.send(DockMessage::Sessions { id: page_id, sessions: vec![], active: None }).unwrap();

        let (mut to_page, mut terminal) = (Vec::new(), String::new());
        let start = Instant::now();
        while terminal.is_empty() {
            assert!(start.elapsed() < Duration::from_secs(5), "the terminal never got its reply");
            to_page.extend(service.tick(false).to_page);
            let _ = BufReader::new(&ours).read_line(&mut terminal);
        }
        assert!(terminal.contains(r#""type":"sessions""#) && terminal.contains(r#""id":1"#), "{terminal}");
        // The page's own reply is there, the terminal's is not.
        let ids: Vec<u64> = to_page
            .iter()
            .filter_map(|m| match m {
                DockMessage::Sessions { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec![page_id], "{to_page:?}");
    }

    #[test]
    fn a_page_message_reaches_the_chat_and_its_reply_comes_back_on_the_tick() {
        let dir = tempfile::tempdir().unwrap();
        let mut service = scratch_service(dir.path());
        service.page_inlet().handle(PageMessage::SessionsLoad { id: 7 });
        let start = Instant::now();
        loop {
            assert!(start.elapsed() < Duration::from_secs(5), "the chat never answered");
            let tick = service.tick(false);
            if tick.to_page.iter().any(|m| matches!(m, DockMessage::Sessions { id: 7, .. })) {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

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

    #[test]
    fn the_sandbox_hides_the_runtime_dir_the_run_dir_is_made_in() {
        let (runtime, run_dir) = (Path::new("/run/user/1000"), Path::new("/run/user/1000/aria"));
        let hidden = hidden_dir_of(runtime, run_dir);
        if cfg!(target_os = "macos") {
            assert_eq!(hidden.as_deref(), Some(run_dir), "no box: the run dir is its own hidden folder");
        } else {
            assert_eq!(hidden.as_deref(), Some(runtime));
            assert!(chat::hidden_from_the_box(run_dir, hidden.as_deref()), "the jobs socket may be opened there");
        }
    }

    #[test]
    fn a_tmp_fallback_hides_nothing() {
        if cfg!(target_os = "linux") {
            let hidden = hidden_dir_of(Path::new("/tmp"), Path::new("/tmp/aria"));
            assert_eq!(hidden, None, "a boxed command sees /tmp: no jobs socket there");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_hidden_dir_holds_the_run_dir_whatever_the_environment() {
        // Both follow `control::runtime_dir`, so an unset `$XDG_RUNTIME_DIR` no longer leaves the run dir outside it.
        let run_dir = run_dir();
        if let Some(hidden) = hidden_dir(&run_dir) {
            assert!(run_dir.starts_with(hidden));
        }
    }
}
