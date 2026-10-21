//! The dock's shortcut (the `shortcut` setting): parsing, its canonical form and the order in which the dock tries
//! shortcuts when one is taken. Platform-free; Hyprland (`aria-hyprland`) and macOS (Carbon, `aria-mac`) register what
//! this module picks.
//!
//! The format is modifiers then one key, joined by ` + ` (`SUPER + SHIFT + A`). Modifiers are `SUPER`, `CTRL`, `ALT`,
//! `SHIFT` on Linux and `Command`, `Control`, `Option`, `Shift` on macOS; the key is a letter, a digit, `Space` or
//! `F1`–`F12`. Parsing ignores case and spaces and accepts a few common aliases (`ctrl`, `cmd`, `opt`…).

/// Which set of modifier names (and which default) applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Linux,
    Mac,
}

impl Platform {
    /// The platform this build runs on.
    pub fn current() -> Platform {
        if cfg!(target_os = "macos") {
            Platform::Mac
        } else {
            Platform::Linux
        }
    }

    /// The modifiers in their canonical spelling and order.
    pub fn modifiers(self) -> &'static [&'static str] {
        match self {
            Platform::Linux => &["SUPER", "CTRL", "ALT", "SHIFT"],
            Platform::Mac => &["Command", "Control", "Option", "Shift"],
        }
    }

    /// The canonical modifier for a name typed in any case, or one of its aliases.
    fn modifier(self, name: &str) -> Option<&'static str> {
        let name = name.to_ascii_lowercase();
        let canonical = match (self, name.as_str()) {
            (Platform::Linux, "super" | "win" | "meta" | "logo") => "SUPER",
            (Platform::Linux, "ctrl" | "control") => "CTRL",
            (Platform::Linux, "alt") => "ALT",
            (Platform::Linux, "shift") => "SHIFT",
            (Platform::Mac, "command" | "cmd") => "Command",
            (Platform::Mac, "control" | "ctrl") => "Control",
            (Platform::Mac, "option" | "opt" | "alt") => "Option",
            (Platform::Mac, "shift") => "Shift",
            _ => return None,
        };
        Some(canonical)
    }
}

/// A shortcut in canonical form: its modifiers (canonical spelling, canonical order, at least one) and its key (`A`,
/// `7`, `Space`, `F5`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shortcut {
    pub mods: Vec<String>,
    pub key: String,
}

impl Shortcut {
    /// The canonical form, the one the setting holds: `SUPER + SHIFT + A`, `Option + Space`.
    pub fn label(&self) -> String {
        self.mods.iter().map(String::as_str).chain([self.key.as_str()]).collect::<Vec<_>>().join(" + ")
    }

    /// The form the page shows (one `<kbd>` per part): title case, `Super + Shift + A`, `Option + Space`.
    pub fn display(&self) -> String {
        self.mods.iter().map(|m| title_case(m)).chain([self.key.clone()]).collect::<Vec<_>>().join(" + ")
    }
}

fn title_case(word: &str) -> String {
    let lower = word.to_ascii_lowercase();
    let mut chars = lower.chars();
    match chars.next() {
        Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
        None => String::new(),
    }
}

/// The canonical key: `A`–`Z`, `0`–`9`, `Space`, `F1`–`F12`.
fn key(name: &str) -> Option<String> {
    let upper = name.to_ascii_uppercase();
    let mut chars = upper.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) if c.is_ascii_alphanumeric() => return Some(upper),
        _ => {}
    }
    if upper == "SPACE" {
        return Some("Space".into());
    }
    let number: u8 = upper.strip_prefix('F')?.parse().ok()?;
    let canonical = format!("F{number}");
    ((1..=12).contains(&number) && canonical == upper).then_some(canonical)
}

/// The shortcut `text` names, in canonical form, or `None` when it is not one: an unknown modifier or key, no modifier,
/// an empty part.
pub fn parse(text: &str, platform: Platform) -> Option<Shortcut> {
    let parts: Vec<&str> = text.split('+').map(str::trim).collect();
    let (last, mods) = parts.split_last()?;
    let key = key(last)?;
    let mut named = Vec::new();
    for part in mods {
        named.push(platform.modifier(part)?);
    }
    let mods: Vec<String> = platform.modifiers().iter().filter(|m| named.contains(m)).map(|m| m.to_string()).collect();
    if mods.is_empty() {
        return None;
    }
    Some(Shortcut { mods, key })
}

/// The platform's default: `SUPER + A` on Omarchy, `Option + Space` on macOS (what an empty setting means).
pub fn default_shortcut(platform: Platform) -> Shortcut {
    let text = match platform {
        Platform::Linux => "SUPER + A",
        Platform::Mac => "Option + Space",
    };
    parse(text, platform).expect("the default parses")
}

/// The shortcuts the dock tries, in order, the first free one winning: the chosen one (when it parses), the default,
/// then the platform's fallbacks. No shortcut appears twice.
pub fn candidates(chosen: &str, platform: Platform) -> Vec<Shortcut> {
    let fallbacks: &[&str] = match platform {
        Platform::Linux => &["SUPER + SHIFT + A", "SUPER + ALT + A"],
        Platform::Mac => &["Control + Option + Space", "Command + Option + Space"],
    };
    let mut list: Vec<Shortcut> = Vec::new();
    let all = parse(chosen, platform)
        .into_iter()
        .chain([default_shortcut(platform)])
        .chain(fallbacks.iter().filter_map(|text| parse(text, platform)));
    for shortcut in all {
        if !list.contains(&shortcut) {
            list.push(shortcut);
        }
    }
    list
}

/// The first of `candidates` that `taken` does not refuse, and the ones before it (taken). When every one is taken, the
/// first is used anyway (as the dock did before shortcuts could be chosen) and they are all reported taken.
pub fn first_free(candidates: &[Shortcut], taken: impl Fn(&Shortcut) -> bool) -> (Shortcut, Vec<Shortcut>) {
    let skipped: Vec<Shortcut> = candidates.iter().take_while(|s| taken(s)).cloned().collect();
    match candidates.get(skipped.len()) {
        Some(free) => (free.clone(), skipped),
        None => (candidates.first().cloned().expect("there is always a default"), skipped),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label(text: &str, platform: Platform) -> Option<String> {
        parse(text, platform).map(|s| s.label())
    }

    #[test]
    fn parsing_is_forgiving_and_canonical() {
        use Platform::*;
        assert_eq!(label("super+shift+a", Linux).as_deref(), Some("SUPER + SHIFT + A"));
        assert_eq!(label("  Shift +  SUPER+ a ", Linux).as_deref(), Some("SUPER + SHIFT + A"), "canonical order");
        assert_eq!(label("ctrl + alt + space", Linux).as_deref(), Some("CTRL + ALT + Space"));
        assert_eq!(label("Control+Super+f5", Linux).as_deref(), Some("SUPER + CTRL + F5"));
        assert_eq!(label("win + 7", Linux).as_deref(), Some("SUPER + 7"));
        assert_eq!(label("SUPER + SUPER + A", Linux).as_deref(), Some("SUPER + A"), "a repeated modifier counts once");
        assert_eq!(label("option+space", Mac).as_deref(), Some("Option + Space"));
        assert_eq!(label("shift + cmd + opt + k", Mac).as_deref(), Some("Command + Option + Shift + K"));
        assert_eq!(label("ctrl + F12", Mac).as_deref(), Some("Control + F12"));
        let s = parse("super + shift + a", Linux).unwrap();
        assert_eq!(s, Shortcut { mods: vec!["SUPER".into(), "SHIFT".into()], key: "A".into() });
        assert_eq!(s.display(), "Super + Shift + A", "the page's form");
        assert_eq!(parse("ctrl+alt+space", Linux).unwrap().display(), "Ctrl + Alt + Space");
        assert_eq!(parse("option + space", Mac).unwrap().display(), "Option + Space");
        assert_eq!(default_shortcut(Linux).display(), "Super + A");
        assert_eq!(default_shortcut(Mac).label(), "Option + Space");
    }

    #[test]
    fn invalid_shortcuts_are_refused() {
        use Platform::*;
        for text in [
            "",
            "A",
            "SUPER",
            "SUPER +",
            "+ A",
            "SUPER + + A",
            "SUPER + AB",
            "SUPER + F13",
            "SUPER + F0",
            "SUPER + F05",
            "SUPER + Return",
            "SUPER + é",
            "HYPER + A",
            "SUPER + A + B",
            "Command + A",
            r#"SUPER + A"); os.execute("x"#,
        ] {
            assert_eq!(parse(text, Linux), None, "{text:?}");
        }
        assert_eq!(parse("SUPER + A", Mac), None, "Linux names are not macOS ones");
        assert_eq!(parse("Option", Mac), None);
    }

    #[test]
    fn the_candidates_come_in_order_without_repeats() {
        use Platform::*;
        let labels = |chosen: &str, p| candidates(chosen, p).iter().map(Shortcut::label).collect::<Vec<_>>();
        assert_eq!(
            labels("ctrl + alt + k", Linux),
            ["CTRL + ALT + K", "SUPER + A", "SUPER + SHIFT + A", "SUPER + ALT + A"]
        );
        assert_eq!(labels("", Linux), ["SUPER + A", "SUPER + SHIFT + A", "SUPER + ALT + A"], "empty: the default");
        assert_eq!(labels("nonsense", Linux), ["SUPER + A", "SUPER + SHIFT + A", "SUPER + ALT + A"]);
        assert_eq!(labels("super + alt + a", Linux), ["SUPER + ALT + A", "SUPER + A", "SUPER + SHIFT + A"]);
        assert_eq!(labels("", Mac), ["Option + Space", "Control + Option + Space", "Command + Option + Space"]);
    }

    #[test]
    fn the_first_free_fallback_wins() {
        let list = candidates("ctrl + alt + k", Platform::Linux);
        let taken = ["CTRL + ALT + K", "SUPER + A"];
        let (used, skipped) = first_free(&list, |s| taken.contains(&s.label().as_str()));
        assert_eq!(used.label(), "SUPER + SHIFT + A", "the chosen one and the default are taken");
        assert_eq!(skipped.iter().map(Shortcut::label).collect::<Vec<_>>(), taken);

        let (used, skipped) = first_free(&list, |_| false);
        assert_eq!((used.label().as_str(), skipped.len()), ("CTRL + ALT + K", 0), "free: the chosen one");

        let (used, skipped) = first_free(&list, |_| true);
        assert_eq!(used.label(), "CTRL + ALT + K", "all taken: the first one anyway");
        assert_eq!(skipped.len(), 4);
    }
}
