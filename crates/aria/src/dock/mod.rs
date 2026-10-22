//! The dock window: the floating surface that hosts the web UI, per platform.
//!
//! Both platforms have the same job: a 700 px surface on the right of the focused screen (1,060 px while the active
//! tab's `/debug` trace panel is shown, `wide`), with the 470 px dock column taking the keyboard when it opens, hosting the embedded `ui/` page and carrying
//! the messages of `aria_core::bridge`.
//!
//! - `gtk` is the Omarchy one: a GTK3 layer-shell surface with WebKitGTK (Linux).
//! - `mac` is the macOS one: a non-activating `NSPanel` with a `WKWebView` (see its module docs).

#[cfg(target_os = "linux")]
mod gtk;
#[cfg(target_os = "linux")]
pub use gtk::Dock;

/// Aria's face beside the notch: its layers and its wing.
#[cfg(target_os = "macos")]
mod face_layers;
#[cfg(target_os = "macos")]
pub mod face_wing;
#[cfg(target_os = "macos")]
mod mac;
#[cfg(target_os = "macos")]
pub use mac::Dock;

use aria_core::bridge::DockMessage;
use include_dir::{include_dir, Dir};
use std::time::{Duration, Instant};

/// The surface's width: the 470 px column and its veil, or wide enough for the trace panel left of the column.
pub(crate) const SURFACE_NARROW: i32 = 700;
pub(crate) const SURFACE_WIDE: i32 = 1060;
/// The dock column, and the trace panel with its gap (ui/dock.css: 560 + 16 px).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))] // only the GTK surface sets an input region
pub(crate) const COLUMN: i32 = 470;
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) const TRACE_COLUMN: i32 = 576;

pub(crate) fn surface_width(wide: bool) -> i32 {
    if wide {
        SURFACE_WIDE
    } else {
        SURFACE_NARROW
    }
}

/// How much of the surface, from its right edge, takes clicks: the column, and the trace panel while wide.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn input_width(wide: bool) -> i32 {
    if wide {
        COLUMN + TRACE_COLUMN
    } else {
        COLUMN
    }
}

/// The page, embedded in the binary and served as `aria://app/<path>`.
pub(crate) static UI: Dir = include_dir!("$CARGO_MANIFEST_DIR/../../ui");

/// The content type of an embedded file, by extension.
pub(crate) fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html",
        "css" => "text/css",
        "js" | "mjs" => "text/javascript",
        "json" => "application/json",
        "svg" => "image/svg+xml",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}

/// May the dock's web view load this address? Only the embedded page (`aria://…`, and the blank page WebKit starts
/// from): a dropped file or a stray link never replaces it.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn stays_in_app(uri: &str) -> bool {
    uri.starts_with("aria://") || uri == "about:blank"
}

/// Deaths closer together than this (measured from the previous one) back off.
const QUIET: Duration = Duration::from_secs(60);
/// The first backed-off delay; each further one doubles it.
const FIRST_BACKOFF: Duration = Duration::from_secs(1);
/// The longest wait before a reload.
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// When to reload the page after its web process dies: at once after a quiet period, and backing off when it keeps
/// dying (1, 2, 4, 8, 16 s, then 30 s) so a page that crashes as it loads does not reload in a tight loop.
#[derive(Default)]
pub(crate) struct Recovery {
    last_death: Option<Instant>,
    delay: Duration,
}

impl Recovery {
    /// Records a death at `now`; returns how long to wait before reloading.
    pub(crate) fn died(&mut self, now: Instant) -> Duration {
        let close = self.last_death.is_some_and(|last| now.saturating_duration_since(last) <= QUIET);
        self.last_death = Some(now);
        self.delay = match (close, self.delay.is_zero()) {
            (false, _) => Duration::ZERO,
            (true, true) => FIRST_BACKOFF,
            (true, false) => (self.delay * 2).min(MAX_BACKOFF),
        };
        self.delay
    }
}

/// What the dock pushes to the page only once (`Platform`, `Version`), kept to send again to a reloaded page.
#[derive(Default)]
pub(crate) struct Replay {
    platform: Option<String>,
    version: Option<String>,
}

impl Replay {
    /// Keeps `json` (the serialised `message`) when it is a `Platform` or a `Version`; ignores any other message.
    pub(crate) fn remember(&mut self, message: &DockMessage, json: &str) {
        match message {
            DockMessage::Platform { .. } => self.platform = Some(json.to_string()),
            DockMessage::Version { .. } => self.version = Some(json.to_string()),
            _ => {}
        }
    }

    /// The kept messages, `Platform` first.
    pub(crate) fn messages(&self) -> Vec<String> {
        self.platform.iter().chain(&self.version).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{input_width, stays_in_app, surface_width, Recovery, Replay};

    #[test]
    fn the_surface_is_700_or_1060() {
        assert_eq!(surface_width(false), 700);
        assert_eq!(surface_width(true), 1060);
    }

    #[test]
    fn input_covers_the_column_or_column_and_trace() {
        assert_eq!(input_width(false), 470);
        assert_eq!(input_width(true), 1046);
        assert!(input_width(true) <= surface_width(true));
    }

    use aria_core::bridge::DockMessage;
    use std::time::{Duration, Instant};

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    #[test]
    fn the_first_death_reloads_at_once() {
        assert_eq!(Recovery::default().died(Instant::now()), Duration::ZERO);
    }

    #[test]
    fn deaths_close_together_back_off_up_to_thirty_seconds() {
        let mut recovery = Recovery::default();
        let start = Instant::now();
        let delays: Vec<Duration> = (0..8).map(|i| recovery.died(start + secs(10 * i))).collect();
        let expected = [0, 1, 2, 4, 8, 16, 30, 30].map(secs);
        assert_eq!(delays, expected);
    }

    #[test]
    fn a_quiet_minute_starts_over() {
        let mut recovery = Recovery::default();
        let start = Instant::now();
        recovery.died(start);
        assert_eq!(recovery.died(start + secs(5)), secs(1));
        assert_eq!(recovery.died(start + secs(5 + 61)), Duration::ZERO);
        assert_eq!(recovery.died(start + secs(5 + 61 + 1)), secs(1));
    }

    #[test]
    fn the_window_is_measured_from_the_previous_death() {
        let mut recovery = Recovery::default();
        let start = Instant::now();
        recovery.died(start);
        assert_eq!(recovery.died(start + secs(59)), secs(1));
        assert_eq!(recovery.died(start + secs(59 + 59)), secs(2));
        assert_eq!(recovery.died(start + secs(59 + 59 + 59)), secs(4));
        // Exactly a minute after the previous death still counts as close.
        assert_eq!(recovery.died(start + secs(59 + 59 + 59 + 60)), secs(8));
    }

    fn platform(hotkey: &str) -> DockMessage {
        DockMessage::Platform {
            hotkey: hotkey.into(),
            taken: Vec::new(),
            os: "linux".into(),
            animation: "compositor".into(),
        }
    }

    fn remember(replay: &mut Replay, message: &DockMessage) -> String {
        let json = serde_json::to_string(message).unwrap();
        replay.remember(message, &json);
        json
    }

    #[test]
    fn nothing_sent_nothing_replayed() {
        assert!(Replay::default().messages().is_empty());
    }

    #[test]
    fn only_the_latest_platform_is_replayed() {
        let mut replay = Replay::default();
        remember(&mut replay, &platform("Super + A"));
        let latest = remember(&mut replay, &platform("Super + Space"));
        assert_eq!(replay.messages(), vec![latest]);
    }

    #[test]
    fn platform_comes_before_version() {
        let mut replay = Replay::default();
        let version = remember(&mut replay, &DockMessage::Version { version: "0.5.0".into() });
        let platform = remember(&mut replay, &platform("Super + A"));
        assert_eq!(replay.messages(), vec![platform, version]);
    }

    #[test]
    fn other_messages_are_not_kept() {
        let mut replay = Replay::default();
        remember(&mut replay, &DockMessage::Open { monitor: "DP-1".into() });
        assert!(replay.messages().is_empty());
    }

    #[test]
    fn only_the_page_itself_loads() {
        assert!(stays_in_app("aria://app/index.html"));
        assert!(stays_in_app("about:blank"));
        for away in ["file:///home/u/photo.png", "https://example.com/", "data:text/html,x", "ARIA://app/x", "aria:/x"]
        {
            assert!(!stays_in_app(away), "{away}");
        }
    }
}
