//! Aria's face as Core Animation layers: the Omarchy widget's drawing (`install/omarchy/aria.face/Face.qml`) rebuilt
//! as a `CALayer` tree in its 96 x 96 space (y down: the stage is geometry-flipped), painted from a
//! `face_motion::Frame` each tick. Shapes are made once; a frame only moves, scales, shows and hides them, with
//! implicit animations off, except the accent's colour, which fades in 300 ms as in the widget.

use crate::face_motion::{EyeKind, Frame, MouthKind, Rgb};
use objc2::rc::Retained;
use objc2_core_foundation::{CFRetained, CGAffineTransform, CGFloat, CGPoint, CGRect, CGSize};
use objc2_core_graphics::{CGColor, CGMutablePath, CGPath};
use objc2_foundation::NSString;
use objc2_quartz_core::{kCALineCapRound, kCALineJoinRound, CALayer, CAShapeLayer, CATextLayer, CATransaction};
use std::cell::RefCell;
use std::collections::HashMap;

const INK: Rgb = Rgb(0x0A, 0x0A, 0x0C);
const PAPER: Rgb = Rgb(0xF4, 0xF4, 0xF5);
const PAGE: Rgb = Rgb(0xE4, 0xE4, 0xE7);
const MINI: Rgb = Rgb(0x8B, 0x7B, 0xE8);
const MINI_EDGE: Rgb = Rgb(0x12, 0x12, 0x12);
const HAND: Rgb = Rgb(0x7D, 0xD3, 0xFC);
const SMOKE: Rgb = Rgb(0x8A, 0x8A, 0x94);
const CONFETTI: [Rgb; 5] =
    [Rgb(0x9B, 0xE7, 0xC4), Rgb(0xB9, 0xA8, 0xFF), Rgb(0xF0, 0xB4, 0x4C), Rgb(0x7D, 0xD3, 0xFC), PAPER];

/// The clock's drawing is the widget's 40-unit one scaled into 34.6.
const CLOCK: f64 = 34.6 / 40.0;

// ---------------------------------------------------------------------------------------------- 2-D transforms

/// An affine transform in the row-vector form Core Animation uses: `then` applies `self` first.
#[derive(Clone, Copy)]
struct Affine(CGAffineTransform);

impl Affine {
    fn identity() -> Affine {
        Affine(CGAffineTransform { a: 1.0, b: 0.0, c: 0.0, d: 1.0, tx: 0.0, ty: 0.0 })
    }
    fn then(self, o: Affine) -> Affine {
        let (m, n) = (self.0, o.0);
        Affine(CGAffineTransform {
            a: m.a * n.a + m.b * n.c,
            b: m.a * n.b + m.b * n.d,
            c: m.c * n.a + m.d * n.c,
            d: m.c * n.b + m.d * n.d,
            tx: m.tx * n.a + m.ty * n.c + n.tx,
            ty: m.tx * n.b + m.ty * n.d + n.ty,
        })
    }
    fn scale(self, sx: f64, sy: f64) -> Affine {
        self.then(Affine(CGAffineTransform { a: sx, b: 0.0, c: 0.0, d: sy, tx: 0.0, ty: 0.0 }))
    }
    /// Degrees, clockwise on screen (the stage's y points down), as QML's `Rotation`.
    fn rotate(self, degrees: f64) -> Affine {
        let (s, c) = degrees.to_radians().sin_cos();
        self.then(Affine(CGAffineTransform { a: c, b: s, c: -s, d: c, tx: 0.0, ty: 0.0 }))
    }
    fn translate(self, x: f64, y: f64) -> Affine {
        self.then(Affine(CGAffineTransform { a: 1.0, b: 0.0, c: 0.0, d: 1.0, tx: x, ty: y }))
    }
}

// ---------------------------------------------------------------------------------------------- SVG paths

/// The SVG path data the widget uses (`M L H V Q T A Z`, absolute), as a Core Graphics path scaled by `k`.
fn svg_path(d: &str, k: f64) -> CFRetained<CGMutablePath> {
    let path = CGMutablePath::new();
    let m = CGAffineTransform { a: k, b: 0.0, c: 0.0, d: k, tx: 0.0, ty: 0.0 };
    let tokens = svg_tokens(d);
    let (mut i, mut cmd) = (0, 'M');
    let (mut x, mut y) = (0.0, 0.0);
    let mut last_q: Option<(f64, f64)> = None;
    let num = |i: &mut usize| -> f64 {
        let v = match tokens.get(*i) {
            Some(Token::Num(v)) => *v,
            _ => 0.0,
        };
        *i += 1;
        v
    };
    while i < tokens.len() {
        if let Token::Cmd(c) = tokens[i] {
            cmd = c;
            i += 1;
            if c == 'Z' {
                CGMutablePath::close_subpath(Some(&path));
                continue;
            }
        }
        // SAFETY (every call below): `&m` points to a transform that lives through the call.
        match cmd {
            'M' => {
                (x, y) = (num(&mut i), num(&mut i));
                unsafe { CGMutablePath::move_to_point(Some(&path), &m, x, y) };
                cmd = 'L'; // more pairs after M are lines
                last_q = None;
            }
            'L' => {
                (x, y) = (num(&mut i), num(&mut i));
                unsafe { CGMutablePath::add_line_to_point(Some(&path), &m, x, y) };
                last_q = None;
            }
            'H' => {
                x = num(&mut i);
                unsafe { CGMutablePath::add_line_to_point(Some(&path), &m, x, y) };
                last_q = None;
            }
            'V' => {
                y = num(&mut i);
                unsafe { CGMutablePath::add_line_to_point(Some(&path), &m, x, y) };
                last_q = None;
            }
            'Q' => {
                let (cx, cy, nx, ny) = (num(&mut i), num(&mut i), num(&mut i), num(&mut i));
                unsafe { CGMutablePath::add_quad_curve_to_point(Some(&path), &m, cx, cy, nx, ny) };
                (x, y, last_q) = (nx, ny, Some((cx, cy)));
            }
            'T' => {
                let (cx, cy) = last_q.map_or((x, y), |(qx, qy)| (2.0 * x - qx, 2.0 * y - qy));
                let (nx, ny) = (num(&mut i), num(&mut i));
                unsafe { CGMutablePath::add_quad_curve_to_point(Some(&path), &m, cx, cy, nx, ny) };
                (x, y, last_q) = (nx, ny, Some((cx, cy)));
            }
            'A' => {
                let (r, _, _, large, sweep, nx, ny) =
                    (num(&mut i), num(&mut i), num(&mut i), num(&mut i), num(&mut i), num(&mut i), num(&mut i));
                let (cx, cy, r, a0, a1) = arc_center(x, y, nx, ny, r, large != 0.0, sweep != 0.0);
                unsafe { CGMutablePath::add_arc(Some(&path), &m, cx, cy, r, a0, a1, sweep == 0.0) };
                (x, y, last_q) = (nx, ny, None);
            }
            _ => i += 1,
        }
    }
    path
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Token {
    Cmd(char),
    Num(f64),
}

fn svg_tokens(d: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut number = String::new();
    let flush = |number: &mut String, out: &mut Vec<Token>| {
        if let Ok(v) = number.parse() {
            out.push(Token::Num(v));
        }
        number.clear();
    };
    for c in d.chars() {
        if c.is_ascii_alphabetic() {
            flush(&mut number, &mut out);
            out.push(Token::Cmd(c));
        } else if (c == '-' && !number.is_empty() && !number.ends_with('e')) || (c == '.' && number.contains('.')) {
            // a sign or a second point starts the next number ("1-2", "1.5.5")
            flush(&mut number, &mut out);
            number.push(c);
        } else if c.is_ascii_digit() || c == '.' || c == '-' {
            number.push(c);
        } else {
            flush(&mut number, &mut out);
        }
    }
    flush(&mut number, &mut out);
    out
}

/// A circular SVG arc from (x1, y1) to (x2, y2) as a centre, radius and angles (the SVG spec's conversion, rx = ry).
fn arc_center(x1: f64, y1: f64, x2: f64, y2: f64, r: f64, large: bool, sweep: bool) -> (f64, f64, f64, f64, f64) {
    let (xp, yp) = ((x1 - x2) / 2.0, (y1 - y2) / 2.0);
    let mut r = r.abs().max(1e-9);
    let lambda = (xp * xp + yp * yp) / (r * r);
    if lambda > 1.0 {
        r *= lambda.sqrt();
    }
    let num = (r * r * r * r - r * r * yp * yp - r * r * xp * xp).max(0.0);
    let den = r * r * yp * yp + r * r * xp * xp;
    let coef = if large == sweep { -1.0 } else { 1.0 } * if den > 0.0 { (num / den).sqrt() } else { 0.0 };
    let (cxp, cyp) = (coef * yp, -coef * xp);
    let (cx, cy) = (cxp + (x1 + x2) / 2.0, cyp + (y1 + y2) / 2.0);
    let a0 = (y1 - cy).atan2(x1 - cx);
    let a1 = (y2 - cy).atan2(x2 - cx);
    (cx, cy, r, a0, a1)
}

// ---------------------------------------------------------------------------------------------- layers

fn layer() -> Retained<CALayer> {
    CALayer::new()
}

/// Puts `l` at the stage rectangle (x, y, w, h), turning about (ox, oy) in its own coordinates.
fn place(l: &CALayer, x: f64, y: f64, w: f64, h: f64, ox: f64, oy: f64) {
    let anchor = CGPoint::new(if w > 0.0 { ox / w } else { 0.0 }, if h > 0.0 { oy / h } else { 0.0 });
    l.setAnchorPoint(anchor);
    l.setBounds(CGRect::new(CGPoint::new(0.0, 0.0), CGSize::new(w, h)));
    l.setPosition(CGPoint::new(x + ox, y + oy));
}

/// A rounded rectangle (or circle) filled with `color`, turning about its centre.
fn blob(parent: &CALayer, x: f64, y: f64, w: f64, h: f64, radius: f64) -> Retained<CALayer> {
    let l = layer();
    place(&l, x, y, w, h, w / 2.0, h / 2.0);
    l.setCornerRadius(radius);
    parent.addSublayer(&l);
    l
}

fn shape(parent: &CALayer, x: f64, y: f64, w: f64, h: f64) -> Retained<CAShapeLayer> {
    let l = CAShapeLayer::new();
    place(&l, x, y, w, h, 0.0, 0.0);
    // SAFETY: the constants are Core Animation's own, valid for the program's life.
    unsafe {
        l.setLineCap(kCALineCapRound);
        l.setLineJoin(kCALineJoinRound);
    }
    l.setFillColor(None);
    parent.addSublayer(&l);
    l
}

fn group(parent: &CALayer, x: f64, y: f64, w: f64, h: f64, ox: f64, oy: f64) -> Retained<CALayer> {
    let l = layer();
    place(&l, x, y, w, h, ox, oy);
    parent.addSublayer(&l);
    l
}

struct Eye {
    root: Retained<CALayer>,
    pill: Retained<CALayer>,
    open: Retained<CALayer>,
    stroke: Retained<CAShapeLayer>,
}

struct Mini {
    root: Retained<CALayer>,
    eyes: Retained<CALayer>,
}

/// The face's layers. `root()` is the 96 x 96 stage; the wing scales and centres it.
pub struct FaceLayers {
    stage: Retained<CALayer>,
    rings: Vec<Retained<CALayer>>,
    body: Retained<CALayer>,
    roll: Retained<CALayer>,
    circle: Retained<CALayer>,
    gaze: Retained<CALayer>,
    lids: Retained<CALayer>,
    eyes: [Eye; 2],
    mouth: Retained<CALayer>,
    mouth_o: Retained<CALayer>,
    mouth_line: Retained<CAShapeLayer>,
    yawn: Retained<CALayer>,
    book: Retained<CALayer>,
    page: Retained<CALayer>,
    laptop: Retained<CALayer>,
    lines: Vec<Retained<CALayer>>,
    cursor: Retained<CALayer>,
    minis: [Mini; 2],
    clock: Retained<CALayer>,
    marks: Retained<CAShapeLayer>,
    hand: Retained<CALayer>,
    confetti: Vec<Retained<CALayer>>,
    puffs: Vec<Retained<CALayer>>,
    z: Retained<CATextLayer>,
    paths: HashMap<&'static str, CFRetained<CGMutablePath>>,
    colors: RefCell<HashMap<Rgb, CFRetained<CGColor>>>,
    accent: RefCell<Option<Rgb>>,
}

const EYE_HAPPY: &str = "M3.4 11.05Q8.5 3.4 13.6 11.05";
const EYE_CLOSED: &str = "M3.4 7.65Q8.5 12.75 13.6 7.65";
const EYE_X: &str = "M5.1 5.1L11.9 11.9M11.9 5.1L5.1 11.9";
const MOUTH_HMM: &str = "M3.8 5.4L6.2 4.9";
const MOUTH_SMILE: &str = "M2.5 4Q5 7.5 7.5 4";
const MOUTH_WAVE: &str = "M2 5.5Q3.5 3.5 5 5.5T8 5.5";

impl FaceLayers {
    pub fn new() -> FaceLayers {
        let colors = RefCell::new(HashMap::new());
        let color = |c: Rgb| -> CFRetained<CGColor> {
            colors
                .borrow_mut()
                .entry(c)
                .or_insert_with(|| CGColor::new_srgb(c.0 as f64 / 255.0, c.1 as f64 / 255.0, c.2 as f64 / 255.0, 1.0))
                .clone()
        };
        let stage = layer();
        place(&stage, 0.0, 0.0, 96.0, 96.0, 48.0, 48.0);
        stage.setGeometryFlipped(true);

        let rings = (0..3)
            .map(|_| {
                let r = blob(&stage, 11.5, 11.5, 73.0, 73.0, 36.5);
                r.setBorderWidth(3.0);
                r.setHidden(true);
                r
            })
            .collect();

        let body = group(&stage, 0.0, 0.0, 96.0, 96.0, 48.0, 84.5);
        let roll = group(&body, 0.0, 0.0, 96.0, 96.0, 48.0, 48.0);
        let circle = blob(&roll, 11.5, 11.5, 73.0, 73.0, 36.5);

        let gaze = group(&roll, 48.0, 48.0, 0.0, 0.0, 0.0, 0.0);
        let lids = group(&gaze, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let eye = |x: f64| {
            let root = group(&lids, x, 0.0, 0.0, 0.0, 0.0, 0.0);
            let pill = blob(&root, -3.9, -8.65, 7.8, 17.3, 3.9);
            pill.setBackgroundColor(Some(&color(INK)));
            let open = blob(&root, -5.1, -5.1, 10.2, 10.2, 5.1);
            open.setBackgroundColor(Some(&color(INK)));
            let stroke = shape(&root, -8.5, -8.5, 17.0, 17.0);
            stroke.setStrokeColor(Some(&color(INK)));
            Eye { root, pill, open, stroke }
        };
        let eyes = [eye(-14.0), eye(14.0)];
        let mouth = group(&gaze, 0.0, 16.0, 0.0, 0.0, 0.0, 0.0);
        let mouth_o = blob(&mouth, -1.7, -1.7, 3.4, 3.4, 1.7);
        mouth_o.setBackgroundColor(Some(&color(INK)));
        let mouth_line = shape(&mouth, -5.0, -5.0, 10.0, 10.0);
        mouth_line.setStrokeColor(Some(&color(INK)));
        mouth_line.setLineWidth(1.8);
        let yawn = group(&gaze, -6.5, 9.5, 13.0, 16.0, 6.5, 3.2);
        yawn.setCornerRadius(6.5);
        yawn.setBackgroundColor(Some(&color(INK)));

        // the book (reading)
        let book = group(&roll, 20.0, 63.0, 56.0, 27.0, 0.0, 0.0);
        for (d, c) in [
            ("M1 6L28 10L55 6V25L28 27L1 25Z", INK),
            ("M3 5Q15 1.5 28 8V24Q15 18.5 3 22Z", PAPER),
            ("M53 5Q41 1.5 28 8V24Q41 18.5 53 22Z", PAPER),
        ] {
            let s = shape(&book, 0.0, 0.0, 56.0, 27.0);
            s.setPath(Some(&svg_path(d, 1.0)));
            s.setFillColor(Some(&color(c)));
        }
        let page = group(&book, 28.0, 0.0, 28.0, 27.0, 0.0, 13.5);
        let leaf = shape(&page, 0.0, 0.0, 28.0, 27.0);
        leaf.setPath(Some(&svg_path("M25 5Q13 1.5 0 8V24Q13 18.5 25 22Z", 1.0)));
        leaf.setFillColor(Some(&color(PAGE)));

        // the laptop's screen (editing, working)
        let laptop = group(&roll, 18.0, 61.0, 60.0, 31.0, 0.0, 0.0);
        let frame = blob(&laptop, 7.0, 0.8, 46.0, 24.5, 3.0);
        frame.setBackgroundColor(Some(&color(INK)));
        frame.setBorderColor(Some(&color(PAPER)));
        frame.setBorderWidth(1.6);
        let glass = group(&laptop, 9.5, 3.0, 41.0, 20.0, 0.0, 0.0);
        glass.setMasksToBounds(true);
        let lines = (0..12).map(|_| blob(&glass, 0.0, 0.0, 0.0, 1.8, 0.9)).collect();
        let cursor = blob(&glass, 0.0, 0.0, 1.6, 2.6, 0.0);
        cursor.setBackgroundColor(Some(&color(PAPER)));

        // the subagents (delegating)
        let mini = |x: f64| {
            let root = group(&stage, x, 67.2, 25.0, 25.0, 12.5, 12.5);
            let disc = blob(&root, 0.0, 0.0, 25.0, 25.0, 12.5);
            disc.setBackgroundColor(Some(&color(MINI)));
            disc.setBorderColor(Some(&color(MINI_EDGE)));
            disc.setBorderWidth(2.0);
            let eyes = group(&root, 0.0, 0.0, 25.0, 25.0, 0.0, 0.0);
            for dx in [7.0, 14.5] {
                let dot = blob(&eyes, dx, 9.25, 3.5, 3.5, 1.75);
                dot.setBackgroundColor(Some(&color(INK)));
            }
            Mini { root, eyes }
        };
        let minis = [mini(-1.9), mini(73.0)];

        // the alarm clock (scheduled)
        let clock = group(&stage, 64.0, -1.0, 34.6, 34.6, 17.3, 17.3);
        for (d, width) in
            [("M8 10.5A6 6 0 0 1 15.5 5.5M32 10.5A6 6 0 0 0 24.5 5.5", 4.5), ("M12 35L9.5 38.5M28 35L30.5 38.5", 3.0)]
        {
            let s = shape(&clock, 0.0, 0.0, 34.6, 34.6);
            s.setPath(Some(&svg_path(d, CLOCK)));
            s.setStrokeColor(Some(&color(INK)));
            s.setLineWidth(width * CLOCK);
        }
        let marks = shape(&clock, 0.0, 0.0, 34.6, 34.6);
        marks.setPath(Some(&svg_path("M5 9L1.5 6M35 9L38.5 6M20 2.5V-1", CLOCK)));
        marks.setStrokeColor(Some(&color(PAPER)));
        marks.setLineWidth(2.0 * CLOCK);
        let dial = blob(&clock, 6.0 * CLOCK, 8.0 * CLOCK, 28.0 * CLOCK, 28.0 * CLOCK, 14.0 * CLOCK);
        dial.setBackgroundColor(Some(&color(PAPER)));
        dial.setBorderColor(Some(&color(INK)));
        dial.setBorderWidth(2.6);
        let hand = group(&clock, 20.0 * CLOCK - 1.0, 11.5 * CLOCK, 2.0, 10.5 * CLOCK, 1.0, 10.5 * CLOCK);
        hand.setCornerRadius(1.0);
        hand.setBackgroundColor(Some(&color(HAND)));

        // the confetti (done), the smoke (error), the z (sleeping)
        let confetti = (0..8)
            .map(|i| {
                let c = blob(&stage, 48.0 - 3.1, 36.5 - 3.1, 6.2, 6.2, if i % 2 == 1 { 3.1 } else { 1.0 });
                c.setBackgroundColor(Some(&color(CONFETTI[i % 5])));
                c
            })
            .collect();
        let puffs = (0..3)
            .map(|_| {
                let p = blob(&stage, 48.0 - 5.75, 13.4 - 5.75, 11.5, 11.5, 5.75);
                p.setBackgroundColor(Some(&color(SMOKE)));
                p
            })
            .collect();
        let z = CATextLayer::new();
        place(&z, 74.0, -2.0, 16.0, 28.0, 0.0, 0.0);
        z.setFontSize(21.0);
        // SAFETY: an NSString is what `string` takes.
        unsafe { z.setString(Some(&NSString::from_str("z"))) };
        z.setContentsScale(2.0);
        stage.addSublayer(&z);

        let paths = [
            ("happy", svg_path(EYE_HAPPY, 1.0)),
            ("closed", svg_path(EYE_CLOSED, 1.0)),
            ("x", svg_path(EYE_X, 1.0)),
            ("hmm", svg_path(MOUTH_HMM, 1.0)),
            ("smile", svg_path(MOUTH_SMILE, 1.0)),
            ("wave", svg_path(MOUTH_WAVE, 1.0)),
        ]
        .into_iter()
        .collect();

        FaceLayers {
            stage,
            rings,
            body,
            roll,
            circle,
            gaze,
            lids,
            eyes,
            mouth,
            mouth_o,
            mouth_line,
            yawn,
            book,
            page,
            laptop,
            lines,
            cursor,
            minis,
            clock,
            marks,
            hand,
            confetti,
            puffs,
            z,
            paths,
            colors,
            accent: RefCell::new(None),
        }
    }

    /// The 96 x 96 stage, turning about its centre.
    pub fn root(&self) -> &CALayer {
        &self.stage
    }

    fn color(&self, c: Rgb) -> CFRetained<CGColor> {
        self.colors
            .borrow_mut()
            .entry(c)
            .or_insert_with(|| CGColor::new_srgb(c.0 as f64 / 255.0, c.1 as f64 / 255.0, c.2 as f64 / 255.0, 1.0))
            .clone()
    }

    /// Shows one frame.
    pub fn paint(&self, f: &Frame) {
        // The accent fades as in the widget (300 ms); everything else follows the frame exactly.
        if *self.accent.borrow() != Some(f.accent) {
            let first = self.accent.borrow().is_none();
            *self.accent.borrow_mut() = Some(f.accent);
            CATransaction::begin();
            CATransaction::setDisableActions(first);
            CATransaction::setAnimationDuration(0.3);
            let accent = self.color(f.accent);
            self.circle.setBackgroundColor(Some(&accent));
            CATransaction::commit();
        }
        CATransaction::begin();
        CATransaction::setDisableActions(true);
        self.paint_frame(f);
        CATransaction::commit();
    }

    fn paint_frame(&self, f: &Frame) {
        let accent = self.color(f.accent);
        let set = |l: &CALayer, t: Affine| l.setAffineTransform(t.0);
        let show = |l: &CALayer, on: bool| l.setHidden(!on);

        for (ring, p) in self.rings.iter().zip(f.ring_p) {
            let on = (0.0..1.0).contains(&p);
            show(ring, on);
            if on {
                ring.setBorderColor(Some(&accent));
                ring.setOpacity((0.6 * (1.0 - p)) as f32);
                let s = 1.0 + 0.45 * (1.0 - (1.0 - p).powi(3));
                set(ring, Affine::identity().scale(s, s));
            }
        }

        self.body.setOpacity(f.body_op as f32);
        set(
            &self.body,
            Affine::identity().scale(f.body_sx, f.body_sy).rotate(f.body_rot).translate(f.body_tx, f.body_ty),
        );
        set(&self.roll, Affine::identity().rotate(f.roll_deg).translate(f.roll_x, 0.0));

        self.gaze.setPosition(CGPoint::new(48.0 + f.eye_x, 48.0 + f.eye_y));
        set(&self.lids, Affine::identity().scale(1.0, f.lid_scale));
        for (eye, (kind, squeeze)) in self.eyes.iter().zip([(f.look.l, f.eye_l_scale), (f.look.r, f.eye_r_scale)]) {
            set(&eye.root, Affine::identity().scale(1.0, squeeze));
            show(&eye.pill, kind == EyeKind::Line);
            show(&eye.open, kind == EyeKind::Open);
            let stroke = match kind {
                EyeKind::Happy => Some("happy"),
                EyeKind::Closed => Some("closed"),
                EyeKind::X => Some("x"),
                _ => None,
            };
            show(&eye.stroke, stroke.is_some());
            if let Some(name) = stroke {
                eye.stroke.setPath(self.paths.get(name).map(|p| -> &CGPath { p }));
                eye.stroke.setLineWidth(if kind == EyeKind::X { 4.4 } else { 3.4 });
            }
        }
        self.mouth.setPosition(CGPoint::new(f.mouth_x, 16.0));
        show(&self.mouth_o, f.look.m == MouthKind::O);
        let line = match f.look.m {
            MouthKind::Hmm => Some("hmm"),
            MouthKind::Smile => Some("smile"),
            MouthKind::Wave => Some("wave"),
            _ => None,
        };
        show(&self.mouth_line, line.is_some());
        if let Some(name) = line {
            self.mouth_line.setPath(self.paths.get(name).map(|p| -> &CGPath { p }));
        }
        show(&self.yawn, f.gape > 0.02);
        set(&self.yawn, Affine::identity().scale(f.gape, f.gape));

        show(&self.book, f.book_op > 0.01);
        self.book.setOpacity(f.book_op as f32);
        set(&self.book, Affine::identity().translate(0.0, (1.0 - f.book_op) * 14.0));
        set(&self.page, Affine::identity().scale(f.page_scale, 1.0));

        show(&self.laptop, f.lap_op > 0.01);
        if f.lap_op > 0.01 {
            self.laptop.setOpacity(f.lap_op as f32);
            set(&self.laptop, Affine::identity().translate(0.0, (1.0 - f.lap_op) * 14.0));
            for (i, line) in self.lines.iter().enumerate() {
                let y = 2.0 + f.scroll_y + i as f64 * 3.6;
                place(line, 3.0 + f.line_ind[i], y, f.line_w[i], 1.8, 0.0, 0.0);
                line.setBackgroundColor(Some(&self.color(f.line_c[i])));
            }
            place(&self.cursor, f.cursor_x, f.cursor_y, 1.6, 2.6, 0.0, 0.0);
            self.cursor.setOpacity(f.cursor_op as f32);
        }

        for (mini, (ty, ex, ey)) in
            self.minis.iter().zip([(f.mini_ly, f.mini_lex, f.mini_ley), (f.mini_ry, f.mini_rex, f.mini_rey)])
        {
            show(&mini.root, f.dl_op > 0.01);
            mini.root.setOpacity(f.dl_op as f32);
            let s = 0.6 + 0.4 * f.dl_op;
            set(&mini.root, Affine::identity().scale(s, s).translate(0.0, ty));
            set(&mini.eyes, Affine::identity().translate(ex, ey));
        }

        show(&self.clock, f.clock_op > 0.01);
        self.clock.setOpacity(f.clock_op as f32);
        set(&self.clock, Affine::identity().scale(0.85, 0.85).rotate(f.clock_shake));
        show(&self.marks, f.bells_on);
        set(&self.hand, Affine::identity().rotate(f.hand_deg));

        for (piece, p) in self.confetti.iter().zip(f.parts) {
            let alive = p.life > 0.0;
            piece.setOpacity(if alive { (p.life * 2.5).min(1.0) as f32 } else { 0.0 });
            set(piece, Affine::identity().rotate(p.rot).translate(p.x * 96.0, p.y * 96.0));
        }
        for (j, (puff, p)) in self.puffs.iter().zip(f.puff_p).enumerate() {
            let on = (0.0..1.0).contains(&p);
            puff.setOpacity(if on { (0.7 * (1.0 - p)) as f32 } else { 0.0 });
            let s = if on { 0.6 + p } else { 1.0 };
            let dx = if on { [-0.08, 0.06, -0.01][j] * 96.0 * p } else { 0.0 };
            let dy = if on { -0.4 * 96.0 * p } else { 0.0 };
            set(puff, Affine::identity().scale(s, s).translate(dx, dy));
        }

        let sleeping = f.shown == "sleeping";
        show(&self.z, sleeping);
        if sleeping {
            self.z.setForegroundColor(Some(&accent));
            let zp = f.z_p;
            self.z.setOpacity(if zp < 0.4 { zp / 0.4 } else { 1.0 - (zp - 0.4) / 0.6 } as f32);
            place(&self.z, 74.0 + 8.0 * zp, -2.0 - 20.0 * zp, 16.0, 28.0, 0.0, 0.0);
        }
    }
}

/// `CGFloat` is `f64` on every Mac Aria runs on; this keeps the arithmetic above honest about it.
const _: () = assert!(std::mem::size_of::<CGFloat>() == 8);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::face_motion::{Motion, WORDS};

    fn settled(word: &str) -> (Motion, Frame) {
        let mut m = Motion::new(11);
        m.step(1.0 / 60.0);
        m.want(word);
        let mut f = m.step(1.0 / 60.0);
        for _ in 0..600 {
            f = m.step(1.0 / 60.0);
        }
        (m, f)
    }

    #[test]
    fn the_layers_take_a_frame() {
        let face = FaceLayers::new();
        for word in WORDS {
            let (mut m, _) = settled(word);
            let mut last = m.step(1.0 / 60.0);
            for _ in 0..60 {
                last = m.step(1.0 / 60.0);
                face.paint(&last);
            }
            assert_eq!(face.body.opacity(), last.body_op as f32, "{word}");
            assert_eq!(face.book.isHidden(), word != "reading", "{word}: the book is the reader's");
            assert_eq!(face.laptop.isHidden(), !matches!(word, "editing" | "working"), "{word}: the laptop");
            assert_eq!(face.z.isHidden(), word != "sleeping", "{word}: the z");
            assert_eq!(face.eyes[0].open.isHidden(), word != "asking", "{word}: round eyes");
        }
    }

    #[test]
    fn svg_paths_read_as_the_widget_writes_them() {
        assert_eq!(
            svg_tokens("M2 5.5Q3.5 3.5 5 5.5T8 5.5"),
            [
                Token::Cmd('M'),
                Token::Num(2.0),
                Token::Num(5.5),
                Token::Cmd('Q'),
                Token::Num(3.5),
                Token::Num(3.5),
                Token::Num(5.0),
                Token::Num(5.5),
                Token::Cmd('T'),
                Token::Num(8.0),
                Token::Num(5.5)
            ]
        );
        assert_eq!(svg_tokens("V-1")[1], Token::Num(-1.0));
        assert_eq!(svg_tokens("L1.5 6M35")[2..4], [Token::Num(6.0), Token::Cmd('M')]);
    }

    #[test]
    fn a_quarter_arc_finds_its_centre() {
        // from (10, 0) to (0, 10), radius 10, small arc, sweep 1: the centre is the origin
        let (cx, cy, r, a0, a1) = arc_center(10.0, 0.0, 0.0, 10.0, 10.0, false, true);
        assert!(cx.abs() < 1e-9 && cy.abs() < 1e-9 && (r - 10.0).abs() < 1e-9, "{cx} {cy} {r}");
        assert!(a0.abs() < 1e-9 && (a1 - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }

    /// Renders every face to `$ARIA_FACE_RENDER/<word>.rgba` (96 x 96 at 4x, RGBA) for a look by eye:
    /// `ARIA_FACE_RENDER=/some/dir cargo test -p aria render_every_face -- --ignored`.
    #[test]
    #[ignore]
    fn render_every_face() {
        use objc2_core_graphics::{CGBitmapContextCreate, CGColorSpace, CGContext, CGImageAlphaInfo};
        let Some(dir) = std::env::var_os("ARIA_FACE_RENDER") else { return };
        let face = FaceLayers::new();
        let scale = 4.0;
        let size = (96.0 * scale) as usize;
        for word in WORDS {
            let (mut m, _) = settled(word);
            let at = match word {
                "done" => 50,
                "error" => 30,
                _ => 120,
            };
            let mut f = m.step(1.0 / 60.0);
            for _ in 0..at {
                f = m.step(1.0 / 60.0);
            }
            if word == "done" {
                // the confetti flies just after the face changes: show it
                let mut fresh = Motion::new(11);
                fresh.step(1.0 / 60.0);
                fresh.want("done");
                for _ in 0..50 {
                    f = fresh.step(1.0 / 60.0);
                }
            }
            face.paint(&f);
            let mut pixels = vec![0u8; size * size * 4];
            let space = CGColorSpace::new_device_rgb().unwrap();
            // SAFETY: the buffer is size * size * 4 bytes and lives until the context is dropped below.
            let ctx = unsafe {
                CGBitmapContextCreate(
                    pixels.as_mut_ptr().cast(),
                    size,
                    size,
                    8,
                    size * 4,
                    Some(&space),
                    CGImageAlphaInfo::PremultipliedLast.0,
                )
            }
            .unwrap();
            // a bitmap's y points up; the stage's points down (as on screen, where its host is not flipped)
            CGContext::translate_ctm(Some(&ctx), 0.0, size as f64);
            CGContext::scale_ctm(Some(&ctx), scale, -scale);
            // the stage turns about its centre: draw it from its own origin
            face.root().renderInContext(&ctx);
            drop(ctx);
            std::fs::write(std::path::Path::new(&dir).join(format!("{word}.rgba")), &pixels).unwrap();
        }
    }
}
