//! Aria's face in motion, without drawing: a port of the Omarchy bar widget's logic
//! (`install/omarchy/aria.face/Face.qml`) — its looks, springs, timers and sequences — that turns a word ("reading",
//! "asking"…) and the time that passes into a [`Frame`]: every value the widget's `paint()` sets. The macOS wing paints
//! frames with Core Animation (`dock/face_layers.rs`). Everything happens in the QML's 96 x 96 space, with its constants
//! as they are; a seeded generator makes a face's motion repeatable.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))] // only the macOS wing draws it; Omarchy's bar keeps its QML

/// A colour, as the widget writes them (`#B9A8FF`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Rgb(pub u8, pub u8, pub u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EyeKind {
    #[default]
    Line,
    Open,
    Happy,
    Closed,
    X,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MouthKind {
    #[default]
    None,
    Hmm,
    Smile,
    O,
    Wave,
}

/// The eyes and mouth a face wears.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Look {
    pub l: EyeKind,
    pub r: EyeKind,
    pub m: MouthKind,
}

/// One confetti piece: where it is (in units of the 96 px stage), its turn and how much life it has left.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Part {
    pub x: f64,
    pub y: f64,
    pub rot: f64,
    pub life: f64,
}

/// What one frame paints (the QML's `paint()` properties).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Frame {
    pub shown: &'static str,
    pub accent: Rgb,
    pub look: Look,
    pub body_tx: f64,
    pub body_ty: f64,
    pub body_rot: f64,
    pub body_sx: f64,
    pub body_sy: f64,
    pub body_op: f64,
    pub roll_x: f64,
    pub roll_deg: f64,
    pub eye_x: f64,
    pub eye_y: f64,
    pub lid_scale: f64,
    pub eye_l_scale: f64,
    pub eye_r_scale: f64,
    pub mouth_x: f64,
    pub gape: f64,
    pub book_op: f64,
    pub lap_op: f64,
    pub page_scale: f64,
    pub line_w: [f64; 12],
    pub line_c: [Rgb; 12],
    pub line_ind: [f64; 12],
    pub scroll_y: f64,
    pub cursor_x: f64,
    pub cursor_y: f64,
    pub cursor_op: f64,
    pub dl_op: f64,
    pub mini_ly: f64,
    pub mini_ry: f64,
    pub mini_lex: f64,
    pub mini_ley: f64,
    pub mini_rex: f64,
    pub mini_rey: f64,
    pub ring_p: [f64; 3],
    pub clock_op: f64,
    pub clock_shake: f64,
    pub hand_deg: f64,
    pub bells_on: bool,
    pub parts: [Part; 8],
    pub puff_p: [f64; 3],
    pub z_p: f64,
}

/// The words the face knows, in the widget's order.
pub const WORDS: [&str; 11] = [
    "idle",
    "thinking",
    "reading",
    "editing",
    "working",
    "delegating",
    "asking",
    "scheduled",
    "done",
    "error",
    "sleeping",
];

const LAVENDER: Rgb = Rgb(0xB9, 0xA8, 0xFF);
const GREY: Rgb = Rgb(0xA1, 0xA1, 0xAA);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Glance,
    Think,
    Read,
    Type,
    Terminal,
    Delegate,
    Center,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Body {
    Still,
    Float,
    Shake,
    Breathe,
}

/// A word's eyes, mouth, how the eyes move, the body's own motion and its colour (the widget's `looks` and `accents`).
fn style(word: &str) -> (Look, Mode, Body, Rgb) {
    use EyeKind::*;
    let eyes = |e: EyeKind, m: MouthKind| Look { l: e, r: e, m };
    match word {
        "thinking" => (eyes(Line, MouthKind::Hmm), Mode::Think, Body::Still, LAVENDER),
        "reading" => (eyes(Line, MouthKind::None), Mode::Read, Body::Still, LAVENDER),
        "editing" => (eyes(Line, MouthKind::None), Mode::Type, Body::Still, LAVENDER),
        "working" => (eyes(Line, MouthKind::None), Mode::Terminal, Body::Still, LAVENDER),
        "delegating" => (eyes(Line, MouthKind::None), Mode::Delegate, Body::Still, LAVENDER),
        "asking" => (eyes(Open, MouthKind::O), Mode::Center, Body::Float, Rgb(0xF0, 0xB4, 0x4C)),
        "scheduled" => (eyes(Line, MouthKind::Smile), Mode::Glance, Body::Still, Rgb(0x7D, 0xD3, 0xFC)),
        "done" => (eyes(Happy, MouthKind::Smile), Mode::Center, Body::Still, Rgb(0x9B, 0xE7, 0xC4)),
        "error" => (eyes(X, MouthKind::Wave), Mode::Center, Body::Shake, Rgb(0xFF, 0x8F, 0x87)),
        "sleeping" => (eyes(Closed, MouthKind::None), Mode::Down, Body::Breathe, Rgb(0x8A, 0x8A, 0x94)),
        _ => (eyes(Line, MouthKind::None), Mode::Glance, Body::Still, LAVENDER),
    }
}

/// A spring's position and speed.
#[derive(Debug, Clone, Copy, Default)]
struct Sp {
    x: f64,
    v: f64,
}

fn sp(x: f64) -> Sp {
    Sp { x, v: 0.0 }
}

fn spring(o: &mut Sp, target: f64, k: f64, c: f64, dt: f64) {
    o.v += (-k * (o.x - target) - c * o.v) * dt;
    o.x += o.v * dt;
}

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

#[derive(Debug, Clone, Copy, Default)]
struct Pose {
    tx: f64,
    ty: f64,
    rot: f64,
    sx: f64,
    sy: f64,
    op: f64,
}

fn pose(body: Body, t: f64) -> Pose {
    use std::f64::consts::PI;
    let mut p = Pose { sx: 1.0, sy: 1.0, op: 1.0, ..Pose::default() };
    match body {
        Body::Float => p.ty = -0.04 * (0.5 - 0.5 * (2.0 * PI * t / 2.6).cos()),
        Body::Shake => {
            let q = ((t % 2.8) / 2.8 - 0.7) / 0.3;
            if q > 0.0 {
                p.tx = 0.03 * (2.0 * PI * 2.0 * q).sin() * (PI * q).sin();
            }
        }
        Body::Breathe => {
            let b = (2.0 * PI * t / 4.0).sin();
            p.sx = 0.99 + 0.03 * b;
            p.sy = p.sx;
            p.op = 0.875 + 0.125 * b;
        }
        Body::Still => {}
    }
    p
}

/// The jump of `done`: squash, a full turn in the air, land, then a content sway (`u`: seconds since it began).
fn celebration(u: f64) -> (f64, f64, f64, f64) {
    use std::f64::consts::PI;
    let (mut sx, mut sy, mut ty, mut spin) = (1.0, 1.0, 0.0, 0.0);
    if u < 0.25 {
        let a = (PI * u / 0.25).sin();
        sy = 1.0 - 0.12 * a;
        sx = 1.0 + 0.1 * a;
    } else if u < 0.85 {
        let v = (u - 0.25) / 0.6;
        let s1 = (PI * v).sin();
        ty = -0.24 * s1;
        spin = 360.0 * if v < 0.5 { 2.0 * v * v } else { 1.0 - (-2.0 * v + 2.0).powi(2) / 2.0 };
        sy = 1.0 + 0.06 * s1;
        sx = 2.0 - sy;
    } else if u < 1.15 {
        let l = (PI * (u - 0.85) / 0.3).sin();
        sy = 1.0 - 0.1 * l;
        sx = 1.0 + 0.08 * l;
    } else {
        ty = -0.02 * (0.5 - 0.5 * (2.0 * PI * (u - 1.15) / 1.4).cos());
    }
    (sx, sy, ty, spin)
}

/// xorshift64*: enough randomness for glances and confetti, repeatable for a seed.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn between(&mut self, lo: f64, hi: f64) -> f64 {
        self.unit() * (hi - lo) + lo
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScreenKind {
    Edit,
    Command,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Type,
    Prompt,
    Output,
    Done,
}

#[derive(Debug, Clone, Copy, Default)]
struct Line {
    w: f64,
    ind: f64,
    c: Rgb,
    start: f64,
}

/// The laptop's screen: code being typed (editing) or a command and its output (working).
struct Screen {
    kind: ScreenKind,
    lines: [Line; 12],
    total: f64,
    typed: f64,
    shown: usize,
    up_to: usize,
    phase: Phase,
    at: f64,
    next: f64,
    row: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct Particle {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    life: f64,
    rot: f64,
    vr: f64,
}

/// One of the widget's `Timer`s: when it next fires.
#[derive(Debug, Clone, Copy)]
struct Clock {
    at: f64,
}

impl Clock {
    fn after(t: f64, ms: f64) -> Clock {
        Clock { at: t + ms / 1000.0 }
    }
}

pub struct Motion {
    rng: Rng,
    wanted: &'static str,
    shown: &'static str,
    swap_at: Option<f64>,
    t: f64,
    // springs
    gx: Sp,
    gy: Sp,
    lid: Sp,
    yawn: Sp,
    roll: Sp,
    mix: Sp,
    tilt: Sp,
    nod: Sp,
    squint: Sp,
    ponder: Sp,
    bowed: Sp,
    read_amt: Sp,
    lap_amt: Sp,
    flip: Sp,
    tap: Sp,
    lean: Sp,
    scroll: Sp,
    dl: Sp,
    hop_l: Sp,
    hop_r: Sp,
    ml: (Sp, Sp),
    mr: (Sp, Sp),
    knock: Sp,
    sched: Sp,
    wake: Sp,
    jolt: Sp,
    tock: Sp,
    hand: Sp,
    err: Sp,
    // targets and sequence state
    tgx: f64,
    tgy: f64,
    lid_target: f64,
    yawn_target: f64,
    roll_target: f64,
    rolling: bool,
    tilt_target: f64,
    squint_target: f64,
    think_side: f64,
    read_x: f64,
    read_row: u32,
    returning: bool,
    flip_target: f64,
    typing: bool,
    lean_on: bool,
    scr: Option<Screen>,
    focus_side: f64,
    prev_body: Body,
    cur_body: Body,
    since: f64,
    rings: [f64; 3],
    ring_idx: usize,
    parts: Vec<Particle>,
    burst: bool,
    ticks: u32,
    woke: bool,
    ask_since: f64,
    // the timers
    glance: Clock,
    unblink: Option<f64>,
    blink: Clock,
    blink_closing: bool,
    yawn_clock: Clock,
    yawn_open: bool,
    roll_clock: Clock,
    roll_leg: u8,
    roll_dir: f64,
    think: Clock,
    type_clock: Clock,
    type_end: f64,
    delegate: Clock,
    ask: Clock,
    ask_second: bool,
}

impl Motion {
    pub fn new(seed: u64) -> Motion {
        Motion {
            rng: Rng(seed.max(1)),
            wanted: "idle",
            shown: "idle",
            swap_at: None,
            t: 0.0,
            gx: sp(0.0),
            gy: sp(0.0),
            lid: sp(1.0),
            yawn: sp(0.0),
            roll: sp(0.0),
            mix: sp(1.0),
            tilt: sp(0.0),
            nod: sp(0.0),
            squint: sp(0.0),
            ponder: sp(0.0),
            bowed: sp(0.0),
            read_amt: sp(0.0),
            lap_amt: sp(0.0),
            flip: sp(0.0),
            tap: sp(0.0),
            lean: sp(0.0),
            scroll: sp(0.0),
            dl: sp(0.0),
            hop_l: sp(0.0),
            hop_r: sp(0.0),
            ml: (sp(0.0), sp(0.0)),
            mr: (sp(0.0), sp(0.0)),
            knock: sp(0.0),
            sched: sp(0.0),
            wake: sp(0.0),
            jolt: sp(0.0),
            tock: sp(0.0),
            hand: sp(0.0),
            err: sp(0.0),
            tgx: 0.0,
            tgy: 0.0,
            lid_target: 1.0,
            yawn_target: 0.0,
            roll_target: 0.0,
            rolling: false,
            tilt_target: 0.0,
            squint_target: 0.0,
            think_side: 1.0,
            read_x: -0.9,
            read_row: 0,
            returning: false,
            flip_target: 0.0,
            typing: false,
            lean_on: false,
            scr: None,
            focus_side: -1.0,
            prev_body: Body::Still,
            cur_body: Body::Still,
            since: 0.0,
            rings: [-9.0; 3],
            ring_idx: 0,
            parts: Vec::new(),
            burst: false,
            ticks: 0,
            woke: false,
            ask_since: 0.0,
            glance: Clock::after(0.0, 800.0),
            unblink: None,
            blink: Clock::after(0.0, 2500.0),
            blink_closing: true,
            yawn_clock: Clock::after(0.0, 4500.0),
            yawn_open: false,
            roll_clock: Clock::after(0.0, 8000.0),
            roll_leg: 0,
            roll_dir: 1.0,
            think: Clock::after(0.0, 300.0),
            type_clock: Clock::after(0.0, 400.0),
            type_end: 0.0,
            delegate: Clock::after(0.0, 600.0),
            ask: Clock::after(0.0, 900.0),
            ask_second: false,
        }
    }

    /// What Aria is doing now; the face changes behind a blink (an unknown word is idle).
    pub fn want(&mut self, word: &str) {
        let next = WORDS.iter().find(|w| **w == word).copied().unwrap_or("idle");
        if next == self.wanted {
            return;
        }
        self.wanted = next;
        if next == self.shown {
            self.swap_at = None;
            return;
        }
        self.lid_target = 0.05;
        self.yawn_target = 0.0;
        self.swap_at = Some(self.t + 0.13);
    }

    /// The face on screen (it follows `want` once the eyes have closed).
    #[cfg(test)]
    pub fn shown(&self) -> &'static str {
        self.shown
    }

    fn style(&self) -> (Look, Mode, Body, Rgb) {
        style(self.shown)
    }

    /// A new face: the body blends into its motion and its sequence starts.
    fn enter(&mut self, next: &'static str) {
        self.prev_body = self.cur_body;
        self.cur_body = style(next).2;
        self.mix = sp(0.0);
        self.since = self.t;
        self.scr = None;
        self.read_x = -0.9;
        self.read_row = 0;
        if next == "asking" {
            self.ask_since = self.t;
        }
        if next == "scheduled" {
            self.woke = false;
            self.ticks = 0;
        }
        if next == "done" {
            self.burst = false;
        }
        self.shown = next;
        self.lid_target = 1.0;
    }

    fn new_screen(&mut self, kind: ScreenKind) {
        let accent = self.style().3;
        let mut lines = [Line::default(); 12];
        for (i, line) in lines.iter_mut().enumerate() {
            *line = if kind == ScreenKind::Command {
                if i == 0 {
                    Line { w: 14.0, ind: 0.0, c: accent, start: 0.0 }
                } else {
                    Line { w: self.rng.between(10.0, 34.0), ind: 0.0, c: GREY, start: 0.0 }
                }
            } else {
                let ind = [0.0, 4.0, 4.0, 8.0][(self.rng.unit() * 4.0) as usize % 4];
                Line { w: self.rng.between(8.0, 26.0), ind, c: if i % 3 == 2 { GREY } else { accent }, start: 0.0 }
            };
        }
        let mut start = 0.0;
        for line in lines.iter_mut() {
            line.start = start;
            start += line.w;
        }
        let up_to = self.rng.between(6.0, 12.0).floor() as usize;
        self.scr = Some(Screen {
            kind,
            lines,
            total: start,
            typed: 0.0,
            shown: 0,
            up_to,
            phase: if kind == ScreenKind::Command { Phase::Prompt } else { Phase::Type },
            at: self.t,
            next: 0.0,
            row: 0,
        });
        self.lean_on = false;
    }

    fn scroll_rows(&self) -> f64 {
        self.scr.as_ref().map_or(0.0, |s| (s.row as f64 - 4.0).max(0.0))
    }

    /// Advances the laptop's screen; returns where the eyes look.
    fn run_screen(&mut self, dt: f64) -> (f64, f64) {
        let t = self.t;
        let typing = self.typing;
        let rows = self.scroll_rows();
        let Some(sc) = self.scr.as_mut() else { return (0.0, 0.0) };
        if sc.kind == ScreenKind::Edit {
            if typing {
                sc.typed += 34.0 * dt;
            }
            if sc.typed >= sc.total {
                self.new_screen(ScreenKind::Edit);
                return (-0.7, 0.55);
            }
            let mut row = 0;
            while row < 11 && sc.typed >= sc.lines[row].start + sc.lines[row].w {
                row += 1;
            }
            sc.row = row;
            let ln = sc.lines[row];
            let cx = ln.ind + clamp(sc.typed - ln.start, 0.0, ln.w);
            return (cx / 36.0 * 1.4 - 0.7, 0.5 + (row as f64 - rows) * 0.07);
        }
        match sc.phase {
            Phase::Prompt => {
                sc.typed = sc.lines[0].w.min(sc.typed + 30.0 * dt);
                if sc.typed >= sc.lines[0].w && t - sc.at > 1.0 {
                    sc.phase = Phase::Output;
                    sc.at = t;
                    sc.next = t + 0.25;
                    let lean = self.rng.unit() < 0.6;
                    self.lean_on = lean;
                    let sc = self.scr.as_ref().unwrap();
                    return (sc.typed / 36.0 * 1.4 - 0.7, 0.5);
                }
                (sc.typed / 36.0 * 1.4 - 0.7, 0.5)
            }
            Phase::Output => {
                if t >= sc.next {
                    sc.shown += 1;
                    let pause = self.rng.unit() < 0.1;
                    let gap = if pause { 0.5 } else { self.rng.between(0.07, 0.18) };
                    let sc = self.scr.as_mut().unwrap();
                    sc.next = t + gap;
                    if sc.shown >= sc.up_to {
                        sc.phase = Phase::Done;
                        sc.at = t;
                        self.lean_on = false;
                    }
                }
                let sc = self.scr.as_mut().unwrap();
                sc.row = sc.shown;
                (-0.45 + 0.15 * (t * 3.0).sin(), 0.5 + (sc.row as f64 - rows) * 0.07)
            }
            _ => {
                let since = t - sc.at;
                if since > 1.5 {
                    self.new_screen(ScreenKind::Command);
                    return (-0.7, 0.5);
                }
                (if since < 1.1 { 0.6 * (since * 6.0).sin() } else { 0.0 }, 0.45)
            }
        }
    }

    /// The widget's timers: each fires when its time comes and sets the next one, as the QML's `Timer`s do.
    fn timers(&mut self) {
        let t = self.t;
        let (look, mode, _, _) = self.style();
        if let Some(at) = self.swap_at {
            if t >= at {
                self.swap_at = None;
                let next = self.wanted;
                self.enter(next);
            }
        }
        if let Some(at) = self.unblink {
            if t >= at {
                self.unblink = None;
                if self.lid_target == 0.1 {
                    self.lid_target = 1.0;
                }
            }
        }
        // glances (idle, scheduled), and the reader's saccades
        if t >= self.glance.at {
            let ms = match mode {
                Mode::Glance => {
                    let center = self.rng.unit() < 0.25;
                    self.tgx = if center { 0.0 } else { self.rng.between(-1.0, 1.0) };
                    self.tgy = if center { 0.0 } else { self.rng.between(-0.7, 0.7) };
                    self.rng.between(1200.0, 2800.0)
                }
                Mode::Read => {
                    self.read_x += 0.45;
                    self.returning = self.read_x > 0.9;
                    if self.returning {
                        self.read_x = -0.9;
                        self.read_row = (self.read_row + 1) % 3;
                        if self.read_row == 0 {
                            self.flip_target = 1.0;
                        }
                        if self.lid_target == 1.0 {
                            self.lid_target = 0.1;
                            self.unblink = Some(t + 0.12);
                        }
                    }
                    self.tgx = self.read_x;
                    self.tgy = 0.5 + self.read_row as f64 * 0.18;
                    if self.returning {
                        750.0
                    } else {
                        self.rng.between(360.0, 520.0)
                    }
                }
                _ => 400.0,
            };
            self.glance = Clock::after(t, ms);
        }
        // blinks
        if t >= self.blink.at {
            let opens = matches!(look.l, EyeKind::Line | EyeKind::Open);
            let ms = if self.blink_closing && opens && self.lid_target == 1.0 {
                self.lid_target = 0.1;
                self.blink_closing = false;
                120.0
            } else {
                if self.lid_target == 0.1 {
                    self.lid_target = 1.0;
                }
                self.blink_closing = true;
                self.rng.between(2000.0, 5000.0)
            };
            self.blink = Clock::after(t, ms);
        }
        // an idle yawn every 9 to 16 s, never while rolling
        if t >= self.yawn_clock.at {
            let ms = if !self.yawn_open {
                if self.shown != "idle" || self.rolling {
                    self.rng.between(9000.0, 16000.0)
                } else {
                    self.yawn_target = 1.0;
                    self.yawn_open = true;
                    1500.0
                }
            } else {
                self.yawn_target = 0.0;
                self.yawn_open = false;
                self.rng.between(9000.0, 16000.0)
            };
            self.yawn_clock = Clock::after(t, ms);
        }
        // bored: a roll one way, then the other, then back
        if t >= self.roll_clock.at {
            let ms = match self.roll_leg {
                0 => {
                    if self.shown != "idle" || self.yawn_target > 0.0 {
                        self.rng.between(8000.0, 15000.0)
                    } else {
                        self.roll_dir = if self.rng.unit() < 0.5 { 1.0 } else { -1.0 };
                        self.rolling = true;
                        self.roll_target = 0.15 * self.roll_dir;
                        self.roll_leg = 1;
                        1700.0
                    }
                }
                1 => {
                    self.roll_target = -0.15 * self.roll_dir;
                    self.roll_leg = 2;
                    1800.0
                }
                2 => {
                    self.roll_target = 0.0;
                    self.roll_leg = 3;
                    1800.0
                }
                _ => {
                    self.rolling = false;
                    self.roll_leg = 0;
                    self.rng.between(8000.0, 15000.0)
                }
            };
            self.roll_clock = Clock::after(t, ms);
        }
        // thinking: a corner, the far eye narrowed, head tilted; then the other corner, with a nod
        if t >= self.think.at {
            let ms = if mode == Mode::Think {
                let side = self.think_side;
                self.tgx = 0.8 * side;
                self.tgy = -0.85;
                self.tilt_target = 8.0 * side;
                self.squint_target = side;
                self.nod.v += 0.2;
                self.think_side = -side;
                self.rng.between(3000.0, 4200.0)
            } else {
                400.0
            };
            self.think = Clock::after(t, ms);
        }
        // editing types in bursts
        if t >= self.type_clock.at {
            let ms = if mode != Mode::Type {
                self.typing = false;
                300.0
            } else if !self.typing {
                self.typing = true;
                self.type_end = t + self.rng.between(0.7, 1.6);
                10.0
            } else if t > self.type_end {
                self.typing = false;
                self.rng.between(400.0, 1100.0)
            } else {
                self.tap.v += 0.45;
                self.rng.between(85.0, 170.0)
            };
            self.type_clock = Clock::after(t, ms);
        }
        // delegating: checks on one subagent, then the other; the one it looks at hops
        if t >= self.delegate.at {
            if mode == Mode::Delegate {
                self.focus_side = -self.focus_side;
                self.nod.v += 0.2;
                if self.focus_side < 0.0 {
                    self.hop_l.v -= 1.4;
                } else {
                    self.hop_r.v -= 1.4;
                }
            }
            let ms = self.rng.between(1600.0, 2600.0);
            self.delegate = Clock::after(t, ms);
        }
        // asking: two taps on the glass, sooner the longer nobody answers (4.2 s down to 1.6 s)
        if t >= self.ask.at {
            let ms = if self.shown != "asking" {
                self.ask_second = false;
                500.0
            } else {
                self.knock.v += 7.0;
                self.rings[self.ring_idx % 3] = t;
                self.ring_idx += 1;
                if !self.ask_second {
                    self.ask_second = true;
                    190.0
                } else {
                    self.ask_second = false;
                    (4200.0 - (t - self.ask_since) * 120.0).max(1600.0)
                }
            };
            self.ask = Clock::after(t, ms);
        }
    }

    /// One frame: `dt` seconds later (at most 0.05).
    pub fn step(&mut self, dt: f64) -> Frame {
        use std::f64::consts::PI;
        let dt = dt.clamp(0.0, 0.05);
        self.t += dt;
        self.timers();
        let (_, mode, _, _) = self.style();
        let yawning = self.yawn_target > 0.0;
        if self.shown != "idle" {
            self.roll_target = 0.0;
        }
        let thinking = mode == Mode::Think && !yawning;
        let reading = mode == Mode::Read;
        let desk = match mode {
            Mode::Type => Some(ScreenKind::Edit),
            Mode::Terminal => Some(ScreenKind::Command),
            _ => None,
        };
        let delegating = mode == Mode::Delegate;
        let mut tx = if yawning {
            0.0
        } else if self.rolling {
            clamp(self.roll.v * 6.0, -1.0, 1.0)
        } else {
            self.tgx
        };
        let mut ty = if yawning { -0.25 } else { self.tgy };
        if thinking {
            tx += 0.12 * (self.t * 0.8).sin();
            ty += 0.06 * (self.t * 1.1 + 1.0).sin();
        }
        if let Some(kind) = desk {
            if self.scr.as_ref().is_none_or(|s| s.kind != kind) {
                self.new_screen(kind);
            }
            (tx, ty) = self.run_screen(dt);
        }
        if delegating {
            tx = 0.85 * self.focus_side;
            ty = 0.55;
        }
        if mode == Mode::Center {
            tx = 0.0;
            ty = if self.shown == "asking" { 0.1 } else { 0.0 };
        }
        if mode == Mode::Down {
            tx = 0.0;
            ty = 0.55;
        }
        let (mut gk, mut gc) = if thinking { (45.0, 12.0) } else { (220.0, 18.0) };
        if reading {
            (gk, gc) = if self.returning { (60.0, 15.5) } else { (420.0, 41.0) };
        }
        if desk.is_some() {
            (gk, gc) = (300.0, 34.0);
        }

        // scheduled: the alarm for a second, a start awake, then a tick a second
        let mut ringing = false;
        if self.shown == "scheduled" {
            let since = self.t - self.since;
            if since < 1.1 {
                ringing = true;
            } else {
                if !self.woke {
                    self.woke = true;
                    self.jolt.v -= 1.3;
                    self.lid.v += 6.0;
                }
                let tick = (since - 1.4).floor();
                if tick >= 0.0 && tick as u32 + 1 > self.ticks {
                    self.ticks = tick as u32 + 1;
                }
            }
        }
        // done: confetti at the top of the jump
        if self.shown == "done" && !self.burst && self.t - self.since >= 0.55 {
            self.burst = true;
            self.parts = (0..8)
                .map(|i| {
                    let angle = -PI / 2.0 + (i as f64 - 3.5) * 0.34 + self.rng.between(-0.1, 0.1);
                    let speed = self.rng.between(0.9, 1.45);
                    let vr = self.rng.between(-420.0, 420.0);
                    Particle {
                        x: 0.0,
                        y: -0.12,
                        vx: angle.cos() * speed,
                        vy: angle.sin() * speed,
                        life: 1.0,
                        rot: 0.0,
                        vr,
                    }
                })
                .collect();
        }
        for p in self.parts.iter_mut().filter(|p| p.life > 0.0) {
            p.vy += 2.6 * dt;
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.rot += p.vr * dt;
            p.life -= dt / 1.4;
        }

        let n = (dt / (1.0 / 240.0)).ceil().max(1.0) as usize;
        let h = dt / n as f64;
        let scheduled = self.shown == "scheduled";
        let error = self.shown == "error";
        let rows = self.scroll_rows();
        for _ in 0..n {
            spring(&mut self.gx, tx, gk, gc, h);
            spring(&mut self.gy, ty, gk, gc, h);
            spring(&mut self.lid, self.lid_target, 1100.0, 56.0, h);
            spring(&mut self.yawn, self.yawn_target, 38.0, 11.0, h);
            spring(&mut self.mix, 1.0, 90.0, 19.0, h);
            spring(&mut self.roll, self.roll_target, 14.0, 7.2, h);
            spring(&mut self.tilt, if thinking { self.tilt_target } else { 0.0 }, 28.0, 9.0, h);
            spring(&mut self.ponder, if thinking { 1.0 } else { 0.0 }, 40.0, 12.0, h);
            spring(&mut self.squint, if thinking { self.squint_target } else { 0.0 }, 32.0, 10.0, h);
            spring(&mut self.nod, 0.0, 30.0, 9.0, h);
            spring(&mut self.bowed, if reading || desk.is_some() { 1.0 } else { 0.0 }, 36.0, 12.0, h);
            spring(&mut self.read_amt, if reading { 1.0 } else { 0.0 }, 36.0, 12.0, h);
            spring(&mut self.lap_amt, if desk.is_some() { 1.0 } else { 0.0 }, 36.0, 12.0, h);
            spring(&mut self.flip, self.flip_target, 70.0, 17.0, h);
            spring(&mut self.tap, 0.0, 600.0, 34.0, h);
            let lean = if self.lean_on && desk == Some(ScreenKind::Command) { 1.0 } else { 0.0 };
            spring(&mut self.lean, lean, 20.0, 9.0, h);
            spring(&mut self.scroll, rows, 120.0, 22.0, h);
            spring(&mut self.dl, if delegating { 1.0 } else { 0.0 }, 40.0, 12.0, h);
            spring(&mut self.hop_l, 0.0, 200.0, 12.0, h);
            spring(&mut self.hop_r, 0.0, 200.0, 12.0, h);
            let left = delegating && self.focus_side < 0.0;
            let right = delegating && self.focus_side > 0.0;
            spring(&mut self.ml.0, if left { 0.8 } else { 0.0 }, 200.0, 20.0, h);
            spring(&mut self.ml.1, if left { -0.8 } else { 0.6 }, 200.0, 20.0, h);
            spring(&mut self.mr.0, if right { -0.8 } else { 0.0 }, 200.0, 20.0, h);
            spring(&mut self.mr.1, if right { -0.8 } else { 0.6 }, 200.0, 20.0, h);
            spring(&mut self.knock, 0.0, 520.0, 26.0, h);
            spring(&mut self.sched, if scheduled { 1.0 } else { 0.0 }, 40.0, 12.0, h);
            spring(&mut self.wake, if ringing { 1.0 } else { 0.0 }, 160.0, 22.0, h);
            spring(&mut self.jolt, 0.0, 160.0, 10.0, h);
            let tock = if scheduled && self.ticks > 0 {
                if self.ticks % 2 == 1 {
                    3.0
                } else {
                    -3.0
                }
            } else {
                0.0
            };
            spring(&mut self.tock, tock, 300.0, 22.0, h);
            spring(&mut self.hand, self.ticks as f64 * 30.0, 400.0, 28.0, h);
            spring(&mut self.err, if error { 1.0 } else { 0.0 }, 25.0, 9.0, h);
        }
        if self.flip_target == 1.0 && self.flip.x > 0.985 {
            self.flip_target = 0.0;
            self.flip = sp(0.0);
        }
        self.paint(ringing)
    }

    fn paint(&self, ringing: bool) -> Frame {
        use std::f64::consts::PI;
        let (look, _, _, accent) = self.style();
        let a = clamp(self.yawn.x, 0.0, 1.15);
        let w = clamp(self.mix.x, 0.0, 1.0);
        let from = pose(self.prev_body, self.t);
        let to = pose(self.cur_body, self.t);
        let m = |f: fn(&Pose) -> f64| f(&from) + (f(&to) - f(&from)) * w;
        let (cel_sx, cel_sy, cel_ty, spin) =
            if self.shown == "done" { celebration(self.t - self.since) } else { (1.0, 1.0, 0.0, 0.0) };
        let er = clamp(self.err.x, 0.0, 1.0);
        let bw = clamp(self.bowed.x, 0.0, 1.0);
        let ln = clamp(self.lean.x, 0.0, 1.0);
        let kn = self.knock.x;
        let mut f = Frame {
            shown: self.shown,
            accent,
            look,
            body_sx: cel_sx
                * (1.0 + 0.07 * er)
                * m(|p| p.sx)
                * (1.0 - 0.04 * a)
                * (1.0 + 0.025 * ln)
                * (1.0 + 0.25 * kn),
            body_sy: cel_sy
                * (1.0 - 0.08 * er)
                * m(|p| p.sy)
                * (1.0 + 0.08 * a)
                * (1.0 + 0.025 * ln)
                * (1.0 + 0.25 * kn),
            body_ty: (m(|p| p.ty) + cel_ty + 0.03 * er - 0.03 * a
                + self.nod.x
                + 0.015 * bw
                + self.tap.x
                + 0.02 * ln
                + self.jolt.x * 0.1)
                * 96.0,
            body_tx: m(|p| p.tx) * 96.0,
            body_rot: m(|p| p.rot) + self.tilt.x + 3.0 * bw + self.tock.x,
            body_op: m(|p| p.op),
            roll_x: self.roll.x * 96.0,
            roll_deg: self.roll.x / 0.38 * 57.2958 + spin,
            ..Frame::default()
        };
        let pd = clamp(self.ponder.x, 0.0, 1.0);
        f.eye_x = self.gx.x * 6.0 * (1.0 + 0.5 * pd);
        f.eye_y = self.gy.x * 5.0 * (1.0 + 0.6 * pd) + 4.0 * bw;
        let q = self.squint.x;
        f.eye_l_scale = 1.0 - 0.5 * q.max(0.0);
        f.eye_r_scale = 1.0 - 0.5 * (-q).max(0.0);
        f.mouth_x = -1.6 * q + 0.7 * pd * (self.t * 1.7).sin();
        f.lid_scale = clamp(
            self.lid
                .x
                .min(1.0 - 0.9 * clamp(a, 0.0, 1.0))
                .min(1.0 - 0.15 * bw)
                .min(1.0 - 0.9 * clamp(self.wake.x, 0.0, 1.0)),
            0.05,
            1.05,
        );
        f.gape = a.max(0.0);
        f.book_op = clamp(self.read_amt.x, 0.0, 1.0);
        f.lap_op = clamp(self.lap_amt.x, 0.0, 1.0);
        f.page_scale = 1.0 - 2.0 * clamp(self.flip.x, 0.0, 1.0);
        f.line_c = [accent; 12];
        if let Some(sc) = self.scr.as_ref() {
            for i in 0..12 {
                let ln = sc.lines[i];
                f.line_c[i] = ln.c;
                f.line_ind[i] = ln.ind;
                f.line_w[i] = if sc.kind == ScreenKind::Edit {
                    clamp(sc.typed - ln.start, 0.0, ln.w)
                } else if i == 0 {
                    sc.typed
                } else if i <= sc.shown {
                    ln.w
                } else {
                    0.0
                };
            }
            f.scroll_y = -self.scroll.x * 3.6;
            let row = if sc.kind == ScreenKind::Edit { sc.row } else { 0 };
            let cl = sc.lines[row];
            let typed = if sc.kind == ScreenKind::Edit { clamp(sc.typed - cl.start, 0.0, cl.w) } else { sc.typed };
            f.cursor_x = 3.0 + cl.ind + typed + 0.8;
            f.cursor_y = 2.0 + row as f64 * 3.6 - self.scroll.x * 3.6 - 0.4;
            let solid = sc.kind == ScreenKind::Edit && self.typing;
            f.cursor_op = if sc.kind == ScreenKind::Command && sc.phase != Phase::Prompt {
                0.0
            } else if solid || (self.t * 9.0).sin() > 0.0 {
                1.0
            } else {
                0.0
            };
        }
        f.dl_op = clamp(self.dl.x, 0.0, 1.0);
        let mini = 25.0;
        let busy_l = if self.focus_side < 0.0 { 0.0 } else { -0.07 * (self.t * 2.0 * PI * 1.7).sin().abs() };
        let busy_r = if self.focus_side > 0.0 { 0.0 } else { -0.07 * (self.t * 2.0 * PI * 1.7 + 1.3).sin().abs() };
        f.mini_ly = (busy_l + self.hop_l.x * 0.1) * mini;
        f.mini_ry = (busy_r + self.hop_r.x * 0.1) * mini;
        f.mini_lex = self.ml.0.x * mini * 0.12;
        f.mini_ley = self.ml.1.x * mini * 0.1;
        f.mini_rex = self.mr.0.x * mini * 0.12;
        f.mini_rey = self.mr.1.x * mini * 0.1;
        for i in 0..3 {
            f.ring_p[i] = (self.t - self.rings[i]) / 0.75;
        }
        f.clock_op = clamp(self.sched.x, 0.0, 1.0);
        f.clock_shake = if ringing { 14.0 * (self.t * 2.0 * PI * 9.0).sin() } else { 0.0 };
        f.bells_on = ringing;
        f.hand_deg = self.hand.x;
        for (slot, p) in f.parts.iter_mut().zip(&self.parts) {
            *slot = Part { x: p.x, y: p.y, rot: p.rot, life: p.life };
        }
        for j in 0..3 {
            f.puff_p[j] = if self.shown == "error" { (self.t - self.since - j as f64 * 0.14) / 1.1 } else { -1.0 };
        }
        f.z_p = (self.t % 3.0) / 3.0;
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FPS: f64 = 60.0;

    /// A face showing `word`, settled past its blink.
    fn showing(word: &str) -> Motion {
        let mut m = Motion::new(7);
        m.step(1.0 / FPS);
        m.want(word);
        run(&mut m, 0.3);
        m
    }

    /// Runs `secs` at 60 Hz and returns every frame.
    fn run(m: &mut Motion, secs: f64) -> Vec<Frame> {
        (0..(secs * FPS).round() as usize).map(|_| m.step(1.0 / FPS)).collect()
    }

    #[test]
    fn every_word_has_its_look_and_accent() {
        let cases = [
            ("idle", Look { l: EyeKind::Line, r: EyeKind::Line, m: MouthKind::None }, Rgb(0xB9, 0xA8, 0xFF)),
            ("asking", Look { l: EyeKind::Open, r: EyeKind::Open, m: MouthKind::O }, Rgb(0xF0, 0xB4, 0x4C)),
            ("error", Look { l: EyeKind::X, r: EyeKind::X, m: MouthKind::Wave }, Rgb(0xFF, 0x8F, 0x87)),
            ("sleeping", Look { l: EyeKind::Closed, r: EyeKind::Closed, m: MouthKind::None }, Rgb(0x8A, 0x8A, 0x94)),
            ("done", Look { l: EyeKind::Happy, r: EyeKind::Happy, m: MouthKind::Smile }, Rgb(0x9B, 0xE7, 0xC4)),
            ("thinking", Look { l: EyeKind::Line, r: EyeKind::Line, m: MouthKind::Hmm }, Rgb(0xB9, 0xA8, 0xFF)),
            ("scheduled", Look { l: EyeKind::Line, r: EyeKind::Line, m: MouthKind::Smile }, Rgb(0x7D, 0xD3, 0xFC)),
        ];
        for (word, look, accent) in cases {
            let mut m = showing(word);
            let f = m.step(1.0 / FPS);
            assert_eq!((f.shown, f.look, f.accent), (word, look, accent), "{word}");
        }
        let mut m = showing("nonsense");
        assert_eq!(m.step(1.0 / FPS).shown, "idle");
    }

    #[test]
    fn a_change_waits_for_the_blink() {
        let mut m = Motion::new(1);
        m.step(1.0 / FPS);
        m.want("done");
        run(&mut m, 0.1);
        assert_eq!(m.shown(), "idle", "the eyes close first");
        run(&mut m, 0.1);
        assert_eq!(m.shown(), "done");
    }

    #[test]
    fn the_last_word_wins() {
        let mut m = Motion::new(1);
        m.step(1.0 / FPS);
        m.want("done");
        run(&mut m, 0.05);
        m.want("error");
        run(&mut m, 0.3);
        assert_eq!(m.shown(), "error");
    }

    #[test]
    fn reading_opens_the_book_and_turns_pages() {
        let mut m = showing("reading");
        let frames = run(&mut m, 10.0);
        assert!(frames[120].book_op > 0.9, "{}", frames[120].book_op);
        assert!(frames.iter().any(|f| f.page_scale < 0.9), "a page turns");
    }

    #[test]
    fn editing_types_lines() {
        let mut m = showing("editing");
        let frames = run(&mut m, 4.0);
        let typed = |f: &Frame| f.line_w.iter().sum::<f64>();
        assert!(frames.last().unwrap().lap_op > 0.9);
        assert!(typed(&frames[239]) > typed(&frames[30]), "{} → {}", typed(&frames[30]), typed(&frames[239]));
    }

    #[test]
    fn working_scrolls_a_command() {
        let mut m = showing("working");
        let frames = run(&mut m, 4.0);
        let shown = |f: &Frame| f.line_w[1..].iter().filter(|w| **w > 0.0).count();
        assert!(frames.iter().any(|f| shown(f) >= 3), "the output comes");
        assert!(frames.iter().filter(|f| shown(f) >= 1).all(|f| f.cursor_op == 0.0), "no cursor during the output");
    }

    #[test]
    fn delegating_hops_subagents() {
        let mut m = showing("delegating");
        let frames = run(&mut m, 3.0);
        assert!(frames.last().unwrap().dl_op > 0.9);
        assert!(frames.iter().any(|f| f.mini_ly.abs() > 0.5 || f.mini_ry.abs() > 0.5));
    }

    #[test]
    fn asking_taps_sooner_over_time() {
        let mut m = showing("asking");
        // a ring starts when its progress is near 0: collect the times of each pair's first tap
        let frames = run(&mut m, 40.0);
        let mut starts = Vec::new();
        for (i, f) in frames.iter().enumerate() {
            let prev = if i > 0 { &frames[i - 1] } else { f };
            for k in 0..3 {
                let started =
                    f.ring_p[k] >= 0.0 && f.ring_p[k] < 0.05 && !(prev.ring_p[k] >= 0.0 && prev.ring_p[k] < 0.05);
                if started {
                    starts.push(i as f64 / FPS);
                }
            }
        }
        let pairs: Vec<f64> = starts.iter().step_by(2).cloned().collect();
        assert!(pairs.len() > 6, "{starts:?}");
        let early = pairs[1] - pairs[0];
        let late = pairs[pairs.len() - 1] - pairs[pairs.len() - 2];
        assert!(late < early, "early {early}, late {late}");
    }

    #[test]
    fn scheduled_rings_then_ticks() {
        let mut m = Motion::new(3);
        m.step(1.0 / FPS);
        m.want("scheduled");
        let frames = run(&mut m, 5.0);
        assert!(frames[20..60].iter().any(|f| f.bells_on), "it rings first");
        assert!(!frames[200].bells_on);
        assert!(frames.last().unwrap().hand_deg > 30.0, "{}", frames.last().unwrap().hand_deg);
    }

    #[test]
    fn done_throws_confetti_once() {
        let mut m = Motion::new(3);
        m.step(1.0 / FPS);
        m.want("done");
        let frames = run(&mut m, 6.0);
        assert!(frames[60].parts.iter().any(|p| p.life > 0.0), "confetti flies");
        assert!(frames.last().unwrap().parts.iter().all(|p| p.life <= 0.0), "and lands, once");
    }

    #[test]
    fn error_puffs_and_sags() {
        let mut m = Motion::new(3);
        m.step(1.0 / FPS);
        m.want("error");
        let frames = run(&mut m, 1.0);
        let f = &frames[45];
        assert!(f.puff_p[0] > 0.0 && f.puff_p[0] < 1.0, "{:?}", f.puff_p);
        assert!(f.body_sy < 1.0);
    }

    #[test]
    fn sleeping_floats_a_z() {
        let mut m = showing("sleeping");
        let frames = run(&mut m, 4.0);
        assert!(frames.iter().any(|f| f.z_p > 0.8) && frames.iter().any(|f| f.z_p < 0.2));
    }

    #[test]
    fn idle_blinks_and_glances() {
        let mut m = Motion::new(9);
        let frames = run(&mut m, 8.0);
        assert!(frames.iter().any(|f| f.lid_scale < 0.5), "it blinks");
        assert!(frames.iter().any(|f| f.eye_x.abs() > 1.0), "it glances");
    }

    #[test]
    fn every_look_stays_finite() {
        for word in [
            "idle",
            "thinking",
            "reading",
            "editing",
            "working",
            "delegating",
            "asking",
            "scheduled",
            "done",
            "error",
            "sleeping",
        ] {
            let mut m = showing(word);
            for f in run(&mut m, 600.0) {
                let values = [
                    f.body_tx,
                    f.body_ty,
                    f.body_rot,
                    f.body_sx,
                    f.body_sy,
                    f.body_op,
                    f.roll_x,
                    f.roll_deg,
                    f.eye_x,
                    f.eye_y,
                    f.lid_scale,
                    f.mouth_x,
                    f.gape,
                    f.scroll_y,
                    f.cursor_x,
                    f.cursor_y,
                    f.hand_deg,
                ];
                assert!(values.iter().all(|v| v.is_finite()), "{word}: {f:?}");
                assert!(f.body_tx.abs() < 200.0 && f.body_ty.abs() < 200.0, "{word}: {f:?}");
            }
        }
    }

    #[test]
    fn a_long_gap_is_one_capped_step() {
        let mut a = Motion::new(5);
        let mut b = Motion::new(5);
        assert_eq!(a.step(30.0), b.step(0.05));
    }

    #[test]
    fn the_same_seed_gives_the_same_frames() {
        let mut a = showing("editing");
        let mut b = showing("editing");
        assert_eq!(run(&mut a, 3.0), run(&mut b, 3.0));
    }
}
