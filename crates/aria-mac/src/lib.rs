//! Everything Aria needs from macOS, the way `aria-hyprland` is everything it needs from Hyprland:
//! a global hotkey, the screen the user is looking at, and nothing to re-register (no config to
//! reload). The panel itself is in the binary (`dock::mac`), because it hosts the page.

#[cfg(target_os = "macos")]
mod hotkey;
#[cfg(target_os = "macos")]
mod screen;

#[cfg(target_os = "macos")]
pub use hotkey::install_hotkey;
#[cfg(target_os = "macos")]
pub use screen::{focused_screen, Screen};
