//! The screen the user is looking at: the one under the mouse. macOS has no notion of a focused
//! monitor that survives the keyboard moving between apps, and a dock opened by a shortcut is always
//! opened where the pointer is.

use objc2_app_kit::{NSEvent, NSScreen};
use objc2_foundation::{MainThreadMarker, NSPoint, NSRect, NSSize};

/// A rectangle in AppKit's global coordinates (origin at the bottom left of the main screen).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    pub fn ns_rect(&self) -> NSRect {
        NSRect::new(NSPoint::new(self.x, self.y), NSSize::new(self.width, self.height))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Screen {
    pub name: String,
    /// The whole screen, menu bar and Dock included.
    pub frame: Rect,
    /// What an ordinary window can use: below the menu bar and beside the Dock.
    pub visible: Rect,
}

fn rect(frame: NSRect) -> Rect {
    Rect { x: frame.origin.x, y: frame.origin.y, width: frame.size.width, height: frame.size.height }
}

fn contains(frame: NSRect, point: NSPoint) -> bool {
    point.x >= frame.origin.x
        && point.x < frame.origin.x + frame.size.width
        && point.y >= frame.origin.y
        && point.y < frame.origin.y + frame.size.height
}

pub fn focused_screen() -> Option<Screen> {
    let mtm = MainThreadMarker::new()?;
    let mouse = NSEvent::mouseLocation();
    let screens = NSScreen::screens(mtm);
    let screen = screens.iter().find(|screen| contains(screen.frame(), mouse)).or_else(|| NSScreen::mainScreen(mtm))?;
    Some(Screen {
        name: screen.localizedName().to_string(),
        frame: rect(screen.frame()),
        visible: rect(screen.visibleFrame()),
    })
}

/// Every screen as the face's wing needs it (`wing::wing_frame` decides where the wing goes).
pub fn screens() -> Vec<crate::wing::ScreenInfo> {
    let Some(mtm) = MainThreadMarker::new() else { return Vec::new() };
    let main = NSScreen::mainScreen(mtm).map(|m| m.frame());
    let area =
        |r: NSRect| crate::wing::Area { x: r.origin.x, y: r.origin.y, width: r.size.width, height: r.size.height };
    NSScreen::screens(mtm)
        .iter()
        .map(|screen| {
            let frame = screen.frame();
            let notch_top = screen.safeAreaInsets().top;
            // Where the notch ends: the screen's right edge less the area right of the notch (its width, so it holds
            // whether AppKit gives that area in the screen's coordinates or the global ones).
            let notch_right = (notch_top > 0.0)
                .then(|| frame.origin.x + frame.size.width - screen.auxiliaryTopRightArea().size.width);
            crate::wing::ScreenInfo {
                frame: area(frame),
                visible: area(screen.visibleFrame()),
                notch_top,
                notch_right,
                main: main == Some(frame),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect_at(x: f64, y: f64, width: f64, height: f64) -> NSRect {
        NSRect::new(NSPoint::new(x, y), NSSize::new(width, height))
    }

    #[test]
    fn a_point_belongs_to_the_screen_that_contains_it() {
        let left = rect_at(0.0, 0.0, 1920.0, 1080.0);
        let right = rect_at(1920.0, 0.0, 2560.0, 1440.0);
        assert!(contains(left, NSPoint::new(10.0, 10.0)));
        assert!(!contains(left, NSPoint::new(1920.0, 10.0)));
        assert!(contains(right, NSPoint::new(1920.0, 10.0)));
        assert!(!contains(right, NSPoint::new(4480.0, 10.0)));
    }

    #[test]
    fn a_screen_above_the_main_one_uses_its_global_origin() {
        let above = rect_at(0.0, 1080.0, 1920.0, 1080.0);
        assert!(contains(above, NSPoint::new(100.0, 1500.0)));
        assert!(!contains(above, NSPoint::new(100.0, 500.0)));
    }

    #[test]
    fn a_screen_carries_both_its_frame_and_its_visible_area() {
        let frame = rect_at(0.0, 0.0, 1920.0, 1080.0);
        let visible = rect_at(0.0, 0.0, 1920.0, 1055.0);
        let screen = Screen { name: "Built-in".into(), frame: rect(frame), visible: rect(visible) };
        assert_eq!(screen.frame.width, 1920.0);
        assert_eq!(screen.visible.height, 1055.0);
        assert_eq!(screen.visible.ns_rect(), NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(1920.0, 1055.0)));
    }
}
