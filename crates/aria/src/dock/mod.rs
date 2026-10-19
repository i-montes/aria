//! The dock window: the floating surface that hosts the web UI, per platform.
//!
//! Both platforms have the same job: a 700 px surface on the right of the focused screen, with the
//! 470 px dock column taking the keyboard when it opens, hosting the embedded `ui/` page and carrying
//! the messages of `aria_core::bridge`.
//!
//! - `gtk` is the Omarchy one: a GTK3 layer-shell surface with WebKitGTK (Linux).
//! - `mac` is the macOS one: a non-activating `NSPanel` with a `WKWebView` (see its module docs).

#[cfg(target_os = "linux")]
mod gtk;
#[cfg(target_os = "linux")]
pub use gtk::Dock;

#[cfg(target_os = "macos")]
mod mac;
#[cfg(target_os = "macos")]
pub use mac::Dock;

/// The shortcut the interface should show in hints and the first-run notice.
#[cfg(target_os = "linux")]
pub(crate) const HOTKEY_LABEL: &str = "Super + A";
#[cfg(target_os = "macos")]
pub(crate) const HOTKEY_LABEL: &str = "Option + Space";

use include_dir::{include_dir, Dir};

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
