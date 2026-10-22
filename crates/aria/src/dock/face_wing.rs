//! Aria's face beside the notch: a small black panel at the notch's right edge, as tall as the notch, as if the notch
//! grew a little (`aria_mac::wing` decides where; with no notch, a pill at the top centre of the main screen). It plays
//! what `crate::face` says Aria is doing, with `face_motion` for the motion and `face_layers` for the drawing, and a
//! click on it opens or closes the dock. The dock's main-thread tick drives it (`tick`, every 16 ms).

use super::face_layers::FaceLayers;
use crate::face_motion::Motion;
use aria_mac::wing::{panel_frame, wing_frame, wing_inset, Area, BOTTOM, FLARE};
use objc2::rc::Retained;
use objc2::{define_class, msg_send, DefinedClass, MainThreadOnly};
use objc2_app_kit::{
    NSBackingStoreType, NSColor, NSEvent, NSPanel, NSStatusWindowLevel, NSView, NSWindowCollectionBehavior,
    NSWindowOcclusionState, NSWindowStyleMask,
};
use objc2_core_foundation::{CGAffineTransform, CGPoint};
use objc2_core_graphics::{CGColor, CGMutablePath};
use objc2_foundation::{MainThreadMarker, NSPoint, NSRect, NSSize};
use objc2_quartz_core::CAShapeLayer;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How big the face is in the wing (as in the Omarchy bar).
const FACE: f64 = 22.0;
/// How often the screens are looked at again (a display plugged in, the lid closed).
const PLACE_EVERY: Duration = Duration::from_secs(2);

pub struct ClickIvars {
    on_click: RefCell<Option<Box<dyn Fn()>>>,
}

define_class!(
    // SAFETY: NSView has no subclassing requirements, and FaceView does not implement Drop.
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[name = "AriaFaceView"]
    #[ivars = ClickIvars]
    struct FaceView;

    impl FaceView {
        #[unsafe(method(mouseDown:))]
        fn mouse_down(&self, _event: &NSEvent) {
            if let Some(on_click) = self.ivars().on_click.borrow().as_ref() {
                on_click();
            }
        }

        // The first click counts even though Aria is not the active app.
        #[unsafe(method(acceptsFirstMouse:))]
        fn accepts_first_mouse(&self, _event: Option<&NSEvent>) -> bool {
            true
        }
    }
);

impl FaceView {
    fn new(mtm: MainThreadMarker, frame: NSRect, on_click: Box<dyn Fn()>) -> Retained<FaceView> {
        let this = Self::alloc(mtm).set_ivars(ClickIvars { on_click: RefCell::new(Some(on_click)) });
        unsafe { msg_send![super(this), initWithFrame: frame] }
    }
}

pub struct FaceWing {
    panel: Retained<NSPanel>,
    view: Retained<FaceView>,
    face: FaceLayers,
    motion: RefCell<Motion>,
    /// The face `crate::face` last told (any thread writes it; the tick reads it).
    wanted: Arc<Mutex<&'static str>>,
    /// `ARIA_FACE=<word>`: a face forced for a look at it.
    forced: Option<String>,
    last_tick: Cell<Option<Instant>>,
    placed_at: Cell<Instant>,
    placed: Cell<Option<(Area, bool)>>,
}

fn ns_rect(a: Area) -> NSRect {
    NSRect::new(NSPoint::new(a.x, a.y), NSSize::new(a.width, a.height))
}

impl FaceWing {
    /// The wing, on screen; `None` when there is no screen to put it on. `on_click` toggles the dock.
    pub fn new(mtm: MainThreadMarker, on_click: Box<dyn Fn()>) -> Option<Rc<FaceWing>> {
        let (area, notch) = wing_frame(&aria_mac::screens())?;
        let rect = ns_rect(panel_frame(area, notch));
        let panel: Retained<NSPanel> = unsafe {
            msg_send![
                NSPanel::alloc(mtm),
                initWithContentRect: rect,
                styleMask: NSWindowStyleMask::Borderless | NSWindowStyleMask::NonactivatingPanel,
                backing: NSBackingStoreType::Buffered,
                defer: false
            ]
        };
        unsafe { panel.setReleasedWhenClosed(false) };
        panel.setOpaque(false);
        panel.setBackgroundColor(Some(&NSColor::clearColor()));
        panel.setHasShadow(false);
        // Above the menu bar and its icons, so it sits in the notch's row.
        panel.setLevel(NSStatusWindowLevel + 1);
        panel.setCollectionBehavior(
            NSWindowCollectionBehavior::CanJoinAllSpaces
                | NSWindowCollectionBehavior::FullScreenAuxiliary
                | NSWindowCollectionBehavior::Stationary
                | NSWindowCollectionBehavior::IgnoresCycle,
        );
        panel.setHidesOnDeactivate(false);
        panel.setMovable(false);
        panel.setBecomesKeyOnlyIfNeeded(true);

        let view = FaceView::new(mtm, NSRect::new(NSPoint::new(0.0, 0.0), rect.size), on_click);
        view.setWantsLayer(true);
        panel.setContentView(Some(&view));
        let face = FaceLayers::new();
        let host = view.layer()?;
        host.setBackgroundColor(Some(&CGColor::new_srgb(0.0, 0.0, 0.0, 1.0)));
        host.addSublayer(face.root());

        let wanted = Arc::new(Mutex::new("idle"));
        let into = wanted.clone();
        crate::face::on_change(Box::new(move |word| *into.lock().unwrap_or_else(|e| e.into_inner()) = word));
        let wing = Rc::new(FaceWing {
            panel,
            view,
            face,
            motion: RefCell::new(Motion::new(seed())),
            wanted,
            forced: std::env::var("ARIA_FACE").ok().filter(|w| !w.is_empty()),
            last_tick: Cell::new(None),
            placed_at: Cell::new(Instant::now()),
            placed: Cell::new(None),
        });
        wing.put(area, notch);
        wing.panel.orderFrontRegardless();
        Some(wing)
    }

    /// Moves the wing to `area` and shapes it as the notch is shaped: beside a notch it reaches under it (so the
    /// bottom edge runs on from the notch's), curves out into the screen's top edge and rounds its bottom outer corner;
    /// the pill does both on each side.
    fn put(&self, area: Area, notch: bool) {
        if self.placed.get() == Some((area, notch)) {
            return;
        }
        self.placed.set(Some((area, notch)));
        let panel = panel_frame(area, notch);
        self.panel.setFrame_display(ns_rect(panel), true);
        let Some(host) = self.view.layer() else { return };
        let mask = CAShapeLayer::new();
        mask.setPath(Some(&outline(panel.width, panel.height, notch)));
        // SAFETY: the mask is a fresh layer with no superlayer, as `mask` requires.
        unsafe { host.setMask(Some(&mask)) };
        let root = self.face.root();
        root.setPosition(CGPoint::new(wing_inset(notch) + area.width / 2.0, area.height / 2.0));
        let k = FACE / 96.0;
        root.setAffineTransform(CGAffineTransform { a: k, b: 0.0, c: 0.0, d: k, tx: 0.0, ty: 0.0 });
    }

    /// One tick of the dock's clock: the screens now and then, and a frame of the face while it can be seen.
    pub fn tick(&self) {
        let now = Instant::now();
        if now.duration_since(self.placed_at.get()) >= PLACE_EVERY {
            self.placed_at.set(now);
            if let Some((area, notch)) = wing_frame(&aria_mac::screens()) {
                self.put(area, notch);
            }
        }
        let dt = self.last_tick.replace(Some(now)).map_or(0.0, |at| now.duration_since(at).as_secs_f64());
        // A sleeping display, a locked screen: no frames; the face takes the gap as one capped step after.
        if !self.panel.occlusionState().contains(NSWindowOcclusionState::Visible) {
            return;
        }
        let word = match &self.forced {
            Some(word) => word.clone(),
            None => self.wanted.lock().unwrap_or_else(|e| e.into_inner()).to_string(),
        };
        let mut motion = self.motion.borrow_mut();
        motion.want(&word);
        let frame = motion.step(dt);
        self.face.paint(&frame);
    }
}

/// A seed for the face's randomness: the clock, so two docks do not glance alike.
fn seed() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64)
}

/// The wing's outline in its panel (the view's layer: y up, origin at the bottom left). On the outer side: the screen's
/// top edge curves down into the side (concave, as the notch meets the menu bar), and the side rounds into the bottom
/// edge (convex). Beside a notch the inner side is straight (it is hidden under the notch); the pill curves on both.
fn outline(w: f64, h: f64, notch: bool) -> objc2_core_foundation::CFRetained<CGMutablePath> {
    let (f, b) = (FLARE, BOTTOM.min(h - FLARE));
    let p = CGMutablePath::new();
    let none = std::ptr::null();
    // SAFETY (every call): a null transform is allowed and means none.
    unsafe {
        CGMutablePath::move_to_point(Some(&p), none, 0.0, h);
        CGMutablePath::add_line_to_point(Some(&p), none, w, h);
        CGMutablePath::add_quad_curve_to_point(Some(&p), none, w - f, h, w - f, h - f);
        CGMutablePath::add_line_to_point(Some(&p), none, w - f, b);
        CGMutablePath::add_quad_curve_to_point(Some(&p), none, w - f, 0.0, w - f - b, 0.0);
        if notch {
            CGMutablePath::add_line_to_point(Some(&p), none, 0.0, 0.0);
        } else {
            CGMutablePath::add_line_to_point(Some(&p), none, f + b, 0.0);
            CGMutablePath::add_quad_curve_to_point(Some(&p), none, f, 0.0, f, b);
            CGMutablePath::add_line_to_point(Some(&p), none, f, h - f);
            CGMutablePath::add_quad_curve_to_point(Some(&p), none, f, h, 0.0, h);
        }
    }
    CGMutablePath::close_subpath(Some(&p));
    p
}
