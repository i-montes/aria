//! Option + Space as a system-wide hotkey, through Carbon's `RegisterEventHotKey`.
//!
//! Carbon is old, but it is still the only way to take a shortcut globally without asking for the
//! Accessibility (or Input Monitoring) permission, which is exactly what a dock needs. The handler is
//! installed on the application event target, so the callback arrives on the main thread inside
//! `NSApplication::run`, next to every other event.

use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

type OSStatus = i32;

#[repr(C)]
struct EventHotKeyID {
    signature: u32,
    id: u32,
}

#[repr(C)]
struct EventTypeSpec {
    event_class: u32,
    event_kind: u32,
}

type EventHandlerProc = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> OSStatus;

/// `kEventClassKeyboard`, the four-character code `'keyb'`.
const EVENT_CLASS_KEYBOARD: u32 = 0x6B65_7962;
/// `kEventHotKeyPressed`.
const EVENT_HOTKEY_PRESSED: u32 = 5;
const HOTKEY_NO_OPTIONS: u32 = 0;
/// `optionKey` as `RegisterEventHotKey` understands modifiers.
const OPTION_MODIFIER: u32 = 0x0800;
/// `kVK_Space`.
const SPACE_KEY_CODE: u32 = 49;
/// `'ARIA'`.
const HOTKEY_SIGNATURE: u32 = 0x4152_4941;

#[link(name = "Carbon", kind = "framework")]
extern "C" {
    fn GetApplicationEventTarget() -> *mut c_void;
    fn InstallEventHandler(
        target: *mut c_void,
        handler: EventHandlerProc,
        num_types: usize,
        types: *const EventTypeSpec,
        user_data: *mut c_void,
        out_ref: *mut *mut c_void,
    ) -> OSStatus;
    fn RegisterEventHotKey(
        key_code: u32,
        modifiers: u32,
        hotkey_id: EventHotKeyID,
        target: *mut c_void,
        options: u32,
        out_ref: *mut *mut c_void,
    ) -> OSStatus;
}

/// One dock at a time, so one callback. Carbon has no user data we can hand a Rust closure, and the
/// shortcut outlives every window: a process-wide slot is the honest shape here.
static ON_PRESSED: OnceLock<Mutex<Option<Box<dyn Fn() + Send>>>> = OnceLock::new();

unsafe extern "C" fn hotkey_pressed(_call: *mut c_void, _event: *mut c_void, _user: *mut c_void) -> OSStatus {
    if let Some(slot) = ON_PRESSED.get() {
        if let Ok(callback) = slot.lock() {
            if let Some(callback) = callback.as_ref() {
                callback();
            }
        }
    }
    0
}

/// Registers Option + Space and calls `on_pressed` every time it is pressed. Must run on the main
/// thread, once the application object exists and before its run loop starts.
pub fn install_hotkey(on_pressed: impl Fn() + Send + 'static) -> Result<(), String> {
    if objc2_foundation::MainThreadMarker::new().is_none() {
        return Err("a global hotkey has to be registered on the main thread".into());
    }

    let slot = ON_PRESSED.get_or_init(|| Mutex::new(None));
    *slot.lock().map_err(|_| "the hotkey callback is poisoned".to_string())? = Some(Box::new(on_pressed));

    let types = EventTypeSpec { event_class: EVENT_CLASS_KEYBOARD, event_kind: EVENT_HOTKEY_PRESSED };
    let mut handler: *mut c_void = std::ptr::null_mut();
    let status = unsafe {
        InstallEventHandler(GetApplicationEventTarget(), hotkey_pressed, 1, &types, std::ptr::null_mut(), &mut handler)
    };
    if status != 0 {
        return Err(format!("InstallEventHandler failed ({status})"));
    }

    let mut hotkey: *mut c_void = std::ptr::null_mut();
    let status = unsafe {
        RegisterEventHotKey(
            SPACE_KEY_CODE,
            OPTION_MODIFIER,
            EventHotKeyID { signature: HOTKEY_SIGNATURE, id: 1 },
            GetApplicationEventTarget(),
            HOTKEY_NO_OPTIONS,
            &mut hotkey,
        )
    };
    if status != 0 {
        return Err(format!("RegisterEventHotKey failed ({status}); Option + Space is probably taken by another app"));
    }

    // No uninstall: both references live until the process ends.
    let _ = (handler, hotkey);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn option_space_is_the_expected_carbon_shortcut() {
        assert_eq!(SPACE_KEY_CODE, 49, "kVK_Space");
        assert_eq!(OPTION_MODIFIER, 1 << 11, "optionKey");
        assert_eq!(EVENT_CLASS_KEYBOARD, u32::from_be_bytes(*b"keyb"));
        assert_eq!(HOTKEY_SIGNATURE, u32::from_be_bytes(*b"ARIA"));
    }
}
