//! The dock's palette in the terminal: 24-bit colour when the terminal says it has it (`COLORTERM` truecolor or 24bit),
//! the nearest of the 256 colours otherwise. One accent, used for the cursor and what is chosen; green, amber and coral
//! only for states.

use ratatui::style::Color;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Palette {
    pub ground: Color,
    pub text: Color,
    pub grey: Color,
    pub faint: Color,
    pub rule: Color,
    pub accent: Color,
    pub green: Color,
    pub amber: Color,
    pub coral: Color,
}

impl Palette {
    pub fn truecolor() -> Palette {
        Palette {
            ground: Color::Rgb(0x0A, 0x0A, 0x0C),
            text: Color::Rgb(0xD6, 0xD6, 0xDC),
            grey: Color::Rgb(0x8A, 0x8A, 0x94),
            faint: Color::Rgb(0x55, 0x55, 0x5E),
            rule: Color::Rgb(0x26, 0x26, 0x2B),
            accent: Color::Rgb(0xB9, 0xA8, 0xFF),
            green: Color::Rgb(0x8B, 0xE0, 0xB0),
            amber: Color::Rgb(0xF0, 0xB4, 0x4C),
            coral: Color::Rgb(0xFF, 0x9A, 0x8B),
        }
    }

    /// The nearest xterm-256 colours.
    pub fn indexed() -> Palette {
        Palette {
            ground: Color::Indexed(232),
            text: Color::Indexed(253),
            grey: Color::Indexed(245),
            faint: Color::Indexed(240),
            rule: Color::Indexed(235),
            accent: Color::Indexed(147),
            green: Color::Indexed(115),
            amber: Color::Indexed(215),
            coral: Color::Indexed(210),
        }
    }

    /// From `COLORTERM`'s value.
    pub fn from_env(colorterm: Option<&str>) -> Palette {
        match colorterm {
            Some("truecolor" | "24bit") => Palette::truecolor(),
            _ => Palette::indexed(),
        }
    }

    pub fn detect() -> Palette {
        Palette::from_env(std::env::var("COLORTERM").ok().as_deref())
    }

    /// A quota window: normal below 80 %, amber from 80 %, coral from 95 %.
    pub fn for_quota(&self, percent: f64) -> Color {
        if percent >= 95.0 {
            self.coral
        } else if percent >= 80.0 {
            self.amber
        } else {
            self.grey
        }
    }

    /// The context: amber from 5 points before it compacts.
    pub fn for_context(&self, percent: f64, compact_at: u8) -> Color {
        if percent >= f64::from(compact_at) - 5.0 {
            self.amber
        } else {
            self.grey
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::Color;

    #[test]
    fn truecolor_uses_the_palette_and_256_the_nearest() {
        assert_eq!(Palette::truecolor().accent, Color::Rgb(0xB9, 0xA8, 0xFF));
        assert!(matches!(Palette::indexed().accent, Color::Indexed(_)));
        assert_eq!(Palette::from_env(Some("truecolor")), Palette::truecolor());
        assert_eq!(Palette::from_env(Some("24bit")), Palette::truecolor());
        assert_eq!(Palette::from_env(None), Palette::indexed());
    }

    #[test]
    fn quota_and_context_colours_follow_their_thresholds() {
        let p = Palette::truecolor();
        assert_eq!(p.for_quota(79.0), p.grey);
        assert_eq!(p.for_quota(80.0), p.amber);
        assert_eq!(p.for_quota(95.0), p.coral);
        assert_eq!(p.for_context(34.0, 40), p.grey);
        assert_eq!(p.for_context(35.0, 40), p.amber);
    }
}
