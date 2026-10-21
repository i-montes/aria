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
/// `kEventHotKeyExclusive`: a combination another app registered exclusively is refused (`eventHotKeyExistsErr`),
/// and once Aria holds one, nobody else can take it. Without it Carbon accepts a combination someone else holds.
const HOTKEY_EXCLUSIVE: u32 = 1;
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
    fn CopySymbolicHotKeys(out_array: *mut CFTypeRef) -> OSStatus;
}

type CFTypeRef = *const c_void;
/// `kCFNumberSInt64Type`.
const CF_NUMBER_SINT64: isize = 4;
/// `kCFStringEncodingUTF8`.
const CF_STRING_UTF8: u32 = 0x0800_0100;

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFArrayGetCount(array: CFTypeRef) -> isize;
    fn CFArrayGetValueAtIndex(array: CFTypeRef, index: isize) -> CFTypeRef;
    fn CFDictionaryGetValue(dictionary: CFTypeRef, key: CFTypeRef) -> CFTypeRef;
    fn CFStringCreateWithBytes(
        allocator: CFTypeRef,
        bytes: *const u8,
        length: isize,
        encoding: u32,
        external: u8,
    ) -> CFTypeRef;
    fn CFNumberGetValue(number: CFTypeRef, number_type: isize, out: *mut c_void) -> u8;
    fn CFBooleanGetValue(boolean: CFTypeRef) -> u8;
    fn CFGetTypeID(value: CFTypeRef) -> usize;
    fn CFNumberGetTypeID() -> usize;
    fn CFBooleanGetTypeID() -> usize;
    fn CFDictionaryGetTypeID() -> usize;
    fn CFRelease(value: CFTypeRef);
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
    if !REGISTERED.with_borrow(|r| r.handler) {
        return Err("the hotkey handler is not installed".into());
    }
    let list = candidates(chosen, Platform::Mac);
    // The system's own shortcuts (Spotlight, input sources…): Carbon would accept them, the system would keep them.
    let system = system_hotkeys();
    let current = REGISTERED.with_borrow(|r| r.hotkey.as_ref().map(|(shortcut, _)| shortcut.clone()));
    let mut new_ref: *mut c_void = std::ptr::null_mut();
    let outcome = carbon::choose(&list, current.as_ref(), |shortcut| {
        if carbon::held_by_system(&system, shortcut) {
            return false;
        }
        match register(shortcut) {
            Some(hotkey) => {
                new_ref = hotkey;
                true
            }
            None => false,
        }
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
            Err(format!("{} are taken (by the system or other apps)", tried.join(", ")))
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
            HOTKEY_EXCLUSIVE,
            &mut hotkey,
        )
    };
    (status == 0).then_some(hotkey)
}

/// The system's shortcuts, from `CopySymbolicHotKeys`; empty when Carbon gives none.
fn system_hotkeys() -> Vec<carbon::SymbolicHotKey> {
    let mut array: CFTypeRef = std::ptr::null();
    // SAFETY: main thread (callers check); `array` outlives the call, which stores a +1 reference (Copy rule) in it.
    let status = unsafe { CopySymbolicHotKeys(&mut array) };
    if status != 0 || array.is_null() {
        return Vec::new();
    }
    let keys = ["kHISymbolicHotKeyCode", "kHISymbolicHotKeyModifiers", "kHISymbolicHotKeyEnabled"].map(|name| {
        // SAFETY: `name` is valid UTF-8 for the given length and outlives the call, which copies it; the result is a
        // +1 string (Create rule) released below.
        unsafe { CFStringCreateWithBytes(std::ptr::null(), name.as_ptr(), name.len() as isize, CF_STRING_UTF8, 0) }
    });
    let mut list = Vec::new();
    if keys.iter().all(|key| !key.is_null()) {
        // SAFETY: `array` is the CFArray Carbon returned and stays alive until released below; each value is checked
        // to be a dictionary before it is read; the values read from it are borrowed (Get rule), used at once, and
        // checked for their type before `CFNumberGetValue`/`CFBooleanGetValue`.
        unsafe {
            for index in 0..CFArrayGetCount(array) {
                let entry = CFArrayGetValueAtIndex(array, index);
                if entry.is_null() || CFGetTypeID(entry) != CFDictionaryGetTypeID() {
                    continue;
                }
                let number = |key: CFTypeRef| {
                    let value = CFDictionaryGetValue(entry, key);
                    let mut out: i64 = 0;
                    (!value.is_null()
                        && CFGetTypeID(value) == CFNumberGetTypeID()
                        && CFNumberGetValue(value, CF_NUMBER_SINT64, (&mut out as *mut i64).cast()) != 0)
                        .then_some(out)
                };
                let enabled = {
                    let value = CFDictionaryGetValue(entry, keys[2]);
                    !value.is_null() && CFGetTypeID(value) == CFBooleanGetTypeID() && CFBooleanGetValue(value) != 0
                };
                list.push(carbon::SymbolicHotKey { code: number(keys[0]), modifiers: number(keys[1]), enabled });
            }
        }
    }
    // SAFETY: each non-null key and the array are +1 references this function owns, released exactly once.
    unsafe {
        for key in keys.into_iter().filter(|key| !key.is_null()) {
            CFRelease(key);
        }
        CFRelease(array);
    }
    list
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
