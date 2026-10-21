//! What `RegisterEventHotKey` needs for a shortcut, and which of the candidates the dock keeps. Nothing here calls
//! Carbon, so it builds and is tested on every platform; `hotkey.rs` (macOS only) does the calls.

use aria_core::shortcut::{InUse, Shortcut};

/// Carbon's modifier flags (`Events.h`, `EventModifiers`), as `RegisterEventHotKey` takes them.
pub const CMD_KEY: u32 = 1 << 8;
pub const SHIFT_KEY: u32 = 1 << 9;
pub const OPTION_KEY: u32 = 1 << 11;
pub const CONTROL_KEY: u32 = 1 << 12;

/// The virtual key codes of `Events.h` (HIToolbox), `kVK_ANSI_*` for the US layout's positions. They name physical
/// keys, so on another layout a letter is the key in that position (as every macOS hotkey works).
const KEY_CODES: &[(&str, u32)] = &[
    ("A", 0x00),     // kVK_ANSI_A
    ("S", 0x01),     // kVK_ANSI_S
    ("D", 0x02),     // kVK_ANSI_D
    ("F", 0x03),     // kVK_ANSI_F
    ("H", 0x04),     // kVK_ANSI_H
    ("G", 0x05),     // kVK_ANSI_G
    ("Z", 0x06),     // kVK_ANSI_Z
    ("X", 0x07),     // kVK_ANSI_X
    ("C", 0x08),     // kVK_ANSI_C
    ("V", 0x09),     // kVK_ANSI_V
    ("B", 0x0B),     // kVK_ANSI_B
    ("Q", 0x0C),     // kVK_ANSI_Q
    ("W", 0x0D),     // kVK_ANSI_W
    ("E", 0x0E),     // kVK_ANSI_E
    ("R", 0x0F),     // kVK_ANSI_R
    ("Y", 0x10),     // kVK_ANSI_Y
    ("T", 0x11),     // kVK_ANSI_T
    ("1", 0x12),     // kVK_ANSI_1
    ("2", 0x13),     // kVK_ANSI_2
    ("3", 0x14),     // kVK_ANSI_3
    ("4", 0x15),     // kVK_ANSI_4
    ("6", 0x16),     // kVK_ANSI_6
    ("5", 0x17),     // kVK_ANSI_5
    ("9", 0x19),     // kVK_ANSI_9
    ("7", 0x1A),     // kVK_ANSI_7
    ("8", 0x1C),     // kVK_ANSI_8
    ("0", 0x1D),     // kVK_ANSI_0
    ("O", 0x1F),     // kVK_ANSI_O
    ("U", 0x20),     // kVK_ANSI_U
    ("I", 0x22),     // kVK_ANSI_I
    ("P", 0x23),     // kVK_ANSI_P
    ("L", 0x25),     // kVK_ANSI_L
    ("J", 0x26),     // kVK_ANSI_J
    ("K", 0x28),     // kVK_ANSI_K
    ("N", 0x2D),     // kVK_ANSI_N
    ("M", 0x2E),     // kVK_ANSI_M
    ("Space", 0x31), // kVK_Space
    ("F5", 0x60),    // kVK_F5
    ("F6", 0x61),    // kVK_F6
    ("F7", 0x62),    // kVK_F7
    ("F3", 0x63),    // kVK_F3
    ("F8", 0x64),    // kVK_F8
    ("F9", 0x65),    // kVK_F9
    ("F11", 0x67),   // kVK_F11
    ("F10", 0x6D),   // kVK_F10
    ("F12", 0x6F),   // kVK_F12
    ("F4", 0x76),    // kVK_F4
    ("F2", 0x78),    // kVK_F2
    ("F1", 0x7A),    // kVK_F1
];

/// The key code and modifier flags of a macOS shortcut (canonical, as `aria_core::shortcut::parse` gives it), or
/// `None` for one Carbon cannot take (a Linux modifier, a key outside the table, no modifier).
pub fn carbon_codes(shortcut: &Shortcut) -> Option<(u32, u32)> {
    let key = KEY_CODES.iter().find(|(name, _)| *name == shortcut.key)?.1;
    let mut modifiers = 0;
    for name in &shortcut.mods {
        modifiers |= match name.as_str() {
            "Command" => CMD_KEY,
            "Shift" => SHIFT_KEY,
            "Option" => OPTION_KEY,
            "Control" => CONTROL_KEY,
            _ => return None,
        };
    }
    (modifiers != 0).then_some((key, modifiers))
}

/// The four modifier flags; a symbolic hotkey's modifiers may carry other bits (the function key, caps lock).
const MODIFIER_MASK: u32 = CMD_KEY | SHIFT_KEY | OPTION_KEY | CONTROL_KEY;

/// One entry of Carbon's `CopySymbolicHotKeys` (the system's own shortcuts: Spotlight's Command + Space, the input
/// sources' Control + Space, Mission Control…), as read from its dictionary: `kHISymbolicHotKeyCode`,
/// `kHISymbolicHotKeyModifiers` (Carbon modifier flags) and `kHISymbolicHotKeyEnabled`. A value missing from the
/// dictionary is `None` (an entry without a key, like a mouse button, never matches).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolicHotKey {
    pub code: Option<i64>,
    pub modifiers: Option<i64>,
    pub enabled: bool,
}

/// Whether one of the system's enabled shortcuts is `shortcut`. `RegisterEventHotKey` accepts those, and then the
/// system keeps the keys (Command + Space opens Spotlight, not Aria), so the dock counts them as taken.
pub fn held_by_system(system: &[SymbolicHotKey], shortcut: &Shortcut) -> bool {
    let Some((code, modifiers)) = carbon_codes(shortcut) else { return false };
    system.iter().any(|hotkey| {
        hotkey.enabled
            && hotkey.code == Some(i64::from(code))
            && hotkey.modifiers.is_some_and(|m| (m as u32) & MODIFIER_MASK == modifiers)
    })
}

/// What a registration pass ends with.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// A new shortcut was registered: the previous one (if any) is to be unregistered.
    Registered(InUse),
    /// The one already registered stays: it comes first among the free candidates, or nothing else could be taken.
    Kept(InUse),
    /// Nothing is registered and none of the candidates could be.
    Nothing { taken: Vec<Shortcut> },
}

/// Goes through `candidates` in order (the chosen one, the default, the fallbacks) and keeps the first that works:
/// `current` when it comes up (already registered, nothing to do), else the first `register` accepts. A shortcut
/// Carbon has no codes for counts as taken. The new one is registered before the old one goes, so a change that finds
/// nothing free leaves the old one working.
pub fn choose(
    candidates: &[Shortcut],
    current: Option<&Shortcut>,
    mut register: impl FnMut(&Shortcut) -> bool,
) -> Outcome {
    let mut taken = Vec::new();
    for candidate in candidates {
        if Some(candidate) == current {
            return Outcome::Kept(InUse { shortcut: candidate.clone(), taken });
        }
        if carbon_codes(candidate).is_some() && register(candidate) {
            return Outcome::Registered(InUse { shortcut: candidate.clone(), taken });
        }
        taken.push(candidate.clone());
    }
    match current {
        Some(current) => Outcome::Kept(InUse { shortcut: current.clone(), taken }),
        None => Outcome::Nothing { taken },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_core::shortcut::{candidates, parse, Platform};

    fn mac(text: &str) -> Shortcut {
        parse(text, Platform::Mac).expect("a macOS shortcut")
    }

    #[test]
    fn carbon_codes_for_option_space_and_letters() {
        assert_eq!(carbon_codes(&mac("Option + Space")), Some((49, 1 << 11)), "kVK_Space, optionKey");
        assert_eq!(carbon_codes(&mac("Command + Shift + A")), Some((0x00, (1 << 8) | (1 << 9))));
        assert_eq!(carbon_codes(&mac("Control + Option + K")), Some((0x28, (1 << 12) | (1 << 11))));
        assert_eq!(carbon_codes(&mac("Command + Control + Option + Shift + Z")), Some((0x06, 0x1B00)));
        // A sample of Events.h's ANSI table, which does not follow the alphabet.
        for (key, code) in [("B", 0x0B), ("Q", 0x0C), ("T", 0x11), ("Y", 0x10), ("O", 0x1F), ("N", 0x2D), ("M", 0x2E)] {
            assert_eq!(carbon_codes(&mac(&format!("Command + {key}"))).map(|c| c.0), Some(code), "{key}");
        }
        for (key, code) in [("1", 0x12), ("5", 0x17), ("6", 0x16), ("7", 0x1A), ("8", 0x1C), ("9", 0x19), ("0", 0x1D)] {
            assert_eq!(carbon_codes(&mac(&format!("Option + {key}"))).map(|c| c.0), Some(code), "{key}");
        }
        let f_keys = [122, 120, 99, 118, 96, 97, 98, 100, 101, 109, 103, 111];
        for (n, code) in (1..=12).zip(f_keys) {
            assert_eq!(carbon_codes(&mac(&format!("Control + F{n}"))).map(|c| c.0), Some(code), "F{n}");
        }
    }

    #[test]
    fn every_key_the_setting_accepts_has_a_code() {
        let keys = ('A'..='Z')
            .chain('0'..='9')
            .map(String::from)
            .chain(["Space".into()])
            .chain((1..=12).map(|n| format!("F{n}")));
        let mut codes = Vec::new();
        for key in keys {
            let (code, _) =
                carbon_codes(&mac(&format!("Command + {key}"))).unwrap_or_else(|| panic!("{key} has no code"));
            codes.push(code);
        }
        let count = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), count, "no two keys share a code");
    }

    #[test]
    fn the_system_s_enabled_shortcuts_are_taken() {
        let entry = |code, modifiers, enabled| SymbolicHotKey { code: Some(code), modifiers: Some(modifiers), enabled };
        // The shape `CopySymbolicHotKeys` gives: Spotlight, the input sources, a disabled one, extra flag bits, an
        // entry with no key code.
        let system = [
            entry(49, 1 << 8, true),                 // Command + Space (Spotlight)
            entry(49, 1 << 12, true),                // Control + Space (previous input source)
            entry(49, (1 << 12) | (1 << 11), false), // Control + Option + Space, turned off
            entry(0x11, (1 << 8) | (1 << 17), true), // Command + T with the function-key bit
            SymbolicHotKey { code: None, modifiers: Some(1 << 8), enabled: true },
            entry(65535, 0, true), // no key
        ];
        assert!(held_by_system(&system, &mac("Command + Space")));
        assert!(held_by_system(&system, &mac("Control + Space")));
        assert!(!held_by_system(&system, &mac("Control + Option + Space")), "a disabled one is free");
        assert!(held_by_system(&system, &mac("Command + T")), "other flag bits do not matter");
        assert!(!held_by_system(&system, &mac("Command + Shift + T")), "the modifiers must match exactly");
        assert!(!held_by_system(&system, &mac("Option + Space")));
        assert!(!held_by_system(&[], &mac("Command + Space")));
    }

    #[test]
    fn linux_shortcuts_have_no_carbon_codes() {
        let linux = parse("SUPER + SHIFT + A", Platform::Linux).unwrap();
        assert_eq!(carbon_codes(&linux), None);
        assert_eq!(carbon_codes(&Shortcut { mods: vec![], key: "A".into() }), None, "no modifier");
    }

    fn labels(list: &[Shortcut]) -> Vec<String> {
        list.iter().map(Shortcut::label).collect()
    }

    #[test]
    fn the_first_candidate_carbon_accepts_is_registered() {
        let list = candidates("Command + Shift + K", Platform::Mac);
        let mut tried = Vec::new();
        let outcome = choose(&list, None, |s| {
            tried.push(s.label());
            s.label() == "Control + Option + Space"
        });
        assert_eq!(tried, ["Command + Shift + K", "Option + Space", "Control + Option + Space"]);
        let Outcome::Registered(in_use) = outcome else { panic!("{outcome:?}") };
        assert_eq!(in_use.shortcut.label(), "Control + Option + Space");
        assert_eq!(labels(&in_use.taken), ["Command + Shift + K", "Option + Space"]);
    }

    #[test]
    fn the_registered_one_is_kept_when_it_comes_first() {
        let list = candidates("", Platform::Mac);
        let outcome = choose(&list, Some(&mac("Option + Space")), |_| panic!("nothing new is registered"));
        assert_eq!(outcome, Outcome::Kept(InUse { shortcut: mac("Option + Space"), taken: vec![] }));

        // The person picks a free shortcut while a fallback is in use: the new one wins.
        let list = candidates("Command + K", Platform::Mac);
        let outcome = choose(&list, Some(&mac("Control + Option + Space")), |_| true);
        assert_eq!(outcome, Outcome::Registered(InUse { shortcut: mac("Command + K"), taken: vec![] }));
    }

    #[test]
    fn nothing_free_keeps_the_old_one_or_reports_it() {
        let list = candidates("Command + K", Platform::Mac);
        let outcome = choose(&list, Some(&mac("Command + J")), |_| false);
        let Outcome::Kept(in_use) = outcome else { panic!("{outcome:?}") };
        assert_eq!(in_use.shortcut.label(), "Command + J", "the old one still works");
        assert_eq!(labels(&in_use.taken), labels(&list));

        let outcome = choose(&list, None, |_| false);
        assert_eq!(outcome, Outcome::Nothing { taken: list.clone() });
    }
}
