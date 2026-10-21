//! Everything Aria needs from macOS, the way `aria-hyprland` is everything it needs from Hyprland:
//! a global hotkey (the `shortcut` setting, re-registered when it changes), the screen the user is looking at. The panel
//! itself is in the binary (`dock::mac`), because it hosts the page.

/// Carbon's key codes and the order of registration: platform-free, so its tests run on every platform.
pub mod carbon;

#[cfg(target_os = "macos")]
mod hotkey;
#[cfg(target_os = "macos")]
mod screen;

#[cfg(target_os = "macos")]
pub use hotkey::{change_hotkey, install_hotkey};
#[cfg(target_os = "macos")]
pub use screen::{focused_screen, Screen};
