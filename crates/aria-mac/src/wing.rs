//! Where Aria's face goes on a Mac: in a small wing beside the notch, as if the notch grew a little, or, with no notch on
//! any screen, in a small pill at the top centre of the main screen. Pure (AppKit's coordinates in, a frame out), so its
//! tests run on every platform; `screen::screens` reads the real screens.

/// A rectangle in AppKit's global coordinates (origin at the bottom left of the main screen).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Area {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// What the wing needs to know of a screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenInfo {
    /// The whole screen.
    pub frame: Area,
    /// Below the menu bar and beside the Dock.
    pub visible: Area,
    /// How tall the notch is (`safeAreaInsets.top`); 0 without one.
    pub notch_top: f64,
    /// Where the notch ends on the right (`auxiliaryTopRightArea`'s left edge), when there is one.
    pub notch_right: Option<f64>,
    /// The main screen (the one with the menu bar).
    pub main: bool,
}

/// How wide the wing is.
pub const WING_WIDTH: f64 = 38.0;

/// The wing's frame and whether it sits beside a notch (else it is the pill): the first notched screen wins; without
/// one, the main screen's top centre, as tall as its menu bar.
pub fn wing_frame(screens: &[ScreenInfo]) -> Option<(Area, bool)> {
    let top = |s: &ScreenInfo| s.frame.y + s.frame.height;
    if let Some((s, right)) = screens.iter().find_map(|s| s.notch_right.filter(|_| s.notch_top > 0.0).map(|r| (s, r))) {
        return Some((Area { x: right, y: top(s) - s.notch_top, width: WING_WIDTH, height: s.notch_top }, true));
    }
    let s = screens.iter().find(|s| s.main).or(screens.first())?;
    // The menu bar's height, or the usual one when it hides itself (the visible area then reaches the top).
    let bar = top(s) - (s.visible.y + s.visible.height);
    let height = if bar >= 1.0 { bar } else { MENU_BAR };
    let x = s.frame.x + (s.frame.width - WING_WIDTH) / 2.0;
    Some((Area { x, y: top(s) - height, width: WING_WIDTH, height }, false))
}

/// How far the wing reaches under the notch: the notch's own bottom corner is rounded, and the wing covers that curve
/// (hidden behind the notch) so the two read as one shape with one straight bottom edge.
pub const OVERLAP: f64 = 12.0;
/// The concave curve at the wing's top outer corner(s), where it meets the screen's top edge, as the notch's own.
pub const FLARE: f64 = 6.0;
/// The convex radius of the wing's bottom outer corner(s), about the notch's own.
pub const BOTTOM: f64 = 8.0;

/// The panel that holds the wing: the wing's area, plus the part under the notch and the flare beside a notch, or a
/// flare on each side for the pill.
pub fn panel_frame(wing: Area, notch: bool) -> Area {
    if notch {
        Area { x: wing.x - OVERLAP, width: wing.width + OVERLAP + FLARE, ..wing }
    } else {
        Area { x: wing.x - FLARE, width: wing.width + 2.0 * FLARE, ..wing }
    }
}

/// Where the wing's own area (what the face is centred in) starts inside its panel.
pub fn wing_inset(notch: bool) -> f64 {
    if notch {
        OVERLAP
    } else {
        FLARE
    }
}

/// The menu bar's height when it hides itself and the screen cannot say.
const MENU_BAR: f64 = 24.0;

#[cfg(test)]
mod tests {
    use super::*;

    fn area(x: f64, y: f64, width: f64, height: f64) -> Area {
        Area { x, y, width, height }
    }

    fn laptop(x: f64, main: bool) -> ScreenInfo {
        ScreenInfo {
            frame: area(x, 0.0, 1512.0, 982.0),
            visible: area(x, 0.0, 1512.0, 950.0),
            notch_top: 32.0,
            notch_right: Some(x + 851.0),
            main,
        }
    }

    #[test]
    fn beside_the_notch() {
        assert_eq!(wing_frame(&[laptop(0.0, true)]), Some((area(851.0, 950.0, 38.0, 32.0), true)));
    }

    #[test]
    fn the_notched_screen_wins_over_an_external_main() {
        let external = ScreenInfo {
            frame: area(0.0, 0.0, 2560.0, 1440.0),
            visible: area(0.0, 0.0, 2560.0, 1415.0),
            notch_top: 0.0,
            notch_right: None,
            main: true,
        };
        assert_eq!(wing_frame(&[external, laptop(-1512.0, false)]), Some((area(-661.0, 950.0, 38.0, 32.0), true)));
    }

    #[test]
    fn no_notch_is_a_pill_on_the_main_screen() {
        let other = ScreenInfo {
            frame: area(1920.0, 0.0, 1920.0, 1080.0),
            visible: area(1920.0, 0.0, 1920.0, 1055.0),
            notch_top: 0.0,
            notch_right: None,
            main: false,
        };
        let main = ScreenInfo {
            frame: area(0.0, 0.0, 1920.0, 1080.0),
            visible: area(0.0, 0.0, 1920.0, 1055.0),
            main: true,
            ..other
        };
        assert_eq!(wing_frame(&[other, main]), Some((area(941.0, 1055.0, 38.0, 25.0), false)));
    }

    #[test]
    fn a_hidden_menu_bar_still_gets_a_pill() {
        let main = ScreenInfo {
            frame: area(0.0, 0.0, 1920.0, 1080.0),
            visible: area(0.0, 0.0, 1920.0, 1080.0),
            notch_top: 0.0,
            notch_right: None,
            main: true,
        };
        assert_eq!(wing_frame(&[main]), Some((area(941.0, 1056.0, 38.0, 24.0), false)));
    }

    #[test]
    fn beside_a_notch_the_panel_reaches_under_it_and_flares_out() {
        let wing = area(851.0, 950.0, 38.0, 32.0);
        assert_eq!(panel_frame(wing, true), area(839.0, 950.0, 56.0, 32.0));
        assert_eq!(wing_inset(true), 12.0);
    }

    #[test]
    fn the_pill_flares_on_both_sides() {
        let wing = area(941.0, 1055.0, 38.0, 25.0);
        assert_eq!(panel_frame(wing, false), area(935.0, 1055.0, 50.0, 25.0));
        assert_eq!(wing_inset(false), 6.0);
    }

    #[test]
    fn no_screen_no_wing() {
        assert_eq!(wing_frame(&[]), None);
    }
}
