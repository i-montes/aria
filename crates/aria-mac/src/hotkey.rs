//! The dock's shortcut as a system-wide hotkey, through Carbon's `RegisterEventHotKey`: the `shortcut` setting
//! (Option + Space by default), or the first free fallback when another app holds it, registered again when the
//! setting changes.
//!
//! Carbon is old, but it is still the only way to take a shortcut globally without asking for the
//! Accessibility (or Input Monitoring) permission, which is exactly what a dock needs. The handler is
//! installed on the application event target, so the callback arrives on the main thread inside
//! `NSApplication::run`, next to every other event. Which candidate wins and the key codes are in `carbon.rs`, where
//! they are tested on every platform.

use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use aria_core::shortcut::{candidates, InUse, Platform, Shortcut};

use crate::carbon::{self, Outcome};

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
    fn UnregisterEventHotKey(hotkey: *mut c_void) -> OSStatus;
}

type Callback = Box<dyn Fn() + Send>;

/// One dock at a time, so one callback. Carbon has no user data we can hand a Rust closure, and the
/// shortcut outlives every window: a process-wide slot is the honest shape here.
static ON_PRESSED: OnceLock<Mutex<Option<Callback>>> = OnceLock::new();

/// What Carbon holds for the dock. Only touched on the main thread (every entry point checks), so a thread-local is
/// the whole of it: the handler goes in once, and the hotkey reference is what a change unregisters.
#[derive(Default)]
struct Registered {
    handler: bool,
    hotkey: Option<(Shortcut, *mut c_void)>,
}

thread_local! {
    static REGISTERED: RefCell<Registered> = RefCell::new(Registered::default());
}

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

fn on_main_thread() -> Result<(), String> {
    match objc2_foundation::MainThreadMarker::new() {
        Some(_) => Ok(()),
        None => Err("a global hotkey has to be registered on the main thread".into()),
    }
}

/// Registers the `chosen` shortcut (the `shortcut` setting; empty or unreadable means Option + Space), or the first
/// free fallback, and calls `on_pressed` every time it is pressed. Must run on the main thread, once the application
/// object exists and before its run loop starts. `Err` when not one candidate could be registered.
pub fn install_hotkey(chosen: &str, on_pressed: impl Fn() + Send + 'static) -> Result<InUse, String> {
    on_main_thread()?;
    let slot = ON_PRESSED.get_or_init(|| Mutex::new(None));
    *slot.lock().map_err(|_| "the hotkey callback is poisoned".to_string())? = Some(Box::new(on_pressed));

    if !REGISTERED.with_borrow(|r| r.handler) {
        let types = EventTypeSpec { event_class: EVENT_CLASS_KEYBOARD, event_kind: EVENT_HOTKEY_PRESSED };
        let mut handler: *mut c_void = std::ptr::null_mut();
        // SAFETY: on the main thread (checked above); `types` and `handler` outlive the call, which copies the spec,
        // and `hotkey_pressed` is a plain `extern "C"` function that lives for the whole process.
        let status = unsafe {
            InstallEventHandler(
                GetApplicationEventTarget(),
                hotkey_pressed,
                1,
                &types,
                std::ptr::null_mut(),
                &mut handler,
            )
        };
        if status != 0 {
            return Err(format!("InstallEventHandler failed ({status})"));
        }
        // No uninstall: the handler lives until the process ends.
        REGISTERED.with_borrow_mut(|r| r.handler = true);
    }
    change_hotkey(chosen)
}

/// The setting changed: registers the new choice (or the first free fallback) and only then unregisters the old one,
/// so a choice with nothing free leaves the old shortcut working. The same handler fires for whichever is registered.
/// Main thread only, after `install_hotkey`.
pub fn change_hotkey(chosen: &str) -> Result<InUse, String> {
    on_main_thread()?;
    let list = candidates(chosen, Platform::Mac);
    let current = REGISTERED.with_borrow(|r| r.hotkey.as_ref().map(|(shortcut, _)| shortcut.clone()));
    let mut new_ref: *mut c_void = std::ptr::null_mut();
    let outcome = carbon::choose(&list, current.as_ref(), |shortcut| match register(shortcut) {
        Some(hotkey) => {
            new_ref = hotkey;
            true
        }
        None => false,
    });
    match outcome {
        Outcome::Registered(in_use) => {
            let old = REGISTERED.with_borrow_mut(|r| r.hotkey.replace((in_use.shortcut.clone(), new_ref)));
            if let Some((_, old)) = old {
                // SAFETY: `old` came from a successful `RegisterEventHotKey` and was never unregistered (it has just
                // left the slot, so nothing else can unregister it again); main thread, checked above.
                let status = unsafe { UnregisterEventHotKey(old) };
                if status != 0 {
                    eprintln!("aria: UnregisterEventHotKey failed ({status})");
                }
            }
            Ok(in_use)
        }
        Outcome::Kept(in_use) => Ok(in_use),
        Outcome::Nothing { taken } => {
            let tried: Vec<String> = taken.iter().map(Shortcut::label).collect();
            Err(format!("RegisterEventHotKey refused {}; other apps probably hold them", tried.join(", ")))
        }
    }
}

/// Registers one shortcut with Carbon; its reference, or `None` when Carbon refuses it (another app holds it).
fn register(shortcut: &Shortcut) -> Option<*mut c_void> {
    let (key_code, modifiers) = carbon::carbon_codes(shortcut)?;
    let mut hotkey: *mut c_void = std::ptr::null_mut();
    // SAFETY: only called from `change_hotkey`, on the main thread; `hotkey` outlives the call, which writes the
    // reference into it, and the event target is the application's, alive for the whole process.
    let status = unsafe {
        RegisterEventHotKey(
            key_code,
            modifiers,
            EventHotKeyID { signature: HOTKEY_SIGNATURE, id: 1 },
            GetApplicationEventTarget(),
            HOTKEY_NO_OPTIONS,
            &mut hotkey,
        )
    };
    (status == 0).then_some(hotkey)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_carbon_constants_are_the_expected_ones() {
        assert_eq!(carbon::carbon_codes(&aria_core::shortcut::default_shortcut(Platform::Mac)), Some((49, 1 << 11)));
        assert_eq!(EVENT_CLASS_KEYBOARD, u32::from_be_bytes(*b"keyb"));
        assert_eq!(HOTKEY_SIGNATURE, u32::from_be_bytes(*b"ARIA"));
    }
}
