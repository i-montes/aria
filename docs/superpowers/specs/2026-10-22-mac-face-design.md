# Aria's face beside the notch (macOS) — design

On Omarchy, Aria's face lives in the bar (`install/omarchy/aria.face/Face.qml`, a Quickshell widget) and plays what
`crates/aria/src/face.rs` says Aria is doing. On macOS nothing draws it yet: the dock writes `~/.aria/run/face` and no one
reads it. This brings the same face to the Mac, drawn natively.

Decided with the owner (2026-10-22): the face sits in a small black **wing beside the notch**, on its right, as if the
notch grew a little; on a screen without a notch it is a small black pill at the top centre. It is drawn with **Core
Animation from Rust**, not a web view. A click opens or closes the dock, as on Omarchy.

## Three parts

### 1. The face's motion — `crates/aria/src/face_motion.rs` (pure, every platform)

A port of `Face.qml`'s logic, with no drawing: the same eleven looks (`idle thinking reading editing working delegating
asking scheduled done error sleeping`), the same accents, springs (`spring(o, target, k, c, dt)` integrated in steps of at
most 1/240 s), timers (glances and the reader's saccades, blinks, the idle yawn, the bored roll, the thinking corners,
typing bursts, the delegating hops, the taps on the glass), sequences (the alarm clock's ring then tick-tock, the jump with
confetti, the puff of smoke, the floating z) and the laptop's screen (typed code, a command's scrolling output).

```rust
pub struct Motion { /* the QML's `s` and the timers' state, a seeded RNG */ }
impl Motion {
    pub fn new(seed: u64) -> Motion;
    pub fn want(&mut self, word: &str);          // the change waits behind a blink (130 ms), as in the QML; unknown → idle
    pub fn step(&mut self, dt: f64) -> Frame;     // dt capped at 0.05 s
}
pub struct Frame { /* every value the QML's paint() sets */ }
```

`Frame` holds exactly what `paint()` computes (body transform and opacity, roll, gaze, lids, squeeze, mouth offset, gape,
book and page, laptop lines, scroll and cursor, subagents, rings, clock and hand, bells, confetti, puffs, z) plus `shown`
and `accent`. Timers are clocks inside `step` (a deadline per timer, rearmed as each QML `Timer` does), so the whole face
is deterministic for a seed and testable without a display. The constants are the QML's, copied as they are.

### 2. The drawing — `crates/aria/src/dock/face_layers.rs` (macOS)

A `CALayer` tree in the QML's 96 × 96 space, scaled to the wing: the rings, the body (circle in the accent), the gaze
group (two eyes, mouth, yawn), the book, the laptop (frame, 12 lines, cursor), the two subagents, the alarm clock, the 8
confetti pieces, the 3 puffs, the z (`CATextLayer`). Eye and mouth kinds are `CAShapeLayer` paths built once from the
QML's SVG paths; each frame sets transforms, opacities, widths and colours inside a `CATransaction` with actions
disabled. The accent's colour change keeps the QML's 300 ms fade (the one implicit animation). Frames come from a 60 Hz
timer on the main run loop, which stops while the screen sleeps or the session is locked and starts again after.

Dependencies: `objc2-quartz-core` and `objc2-core-graphics` (the objc2 0.3 family already used).

### 3. The wing — `crates/aria/src/dock/face_wing.rs` (macOS) and pure geometry in `aria-mac/src/screen.rs`

- A borderless, non-activating `NSPanel` that never takes the keyboard, above the menu bar (`NSStatusWindowLevel` + 1),
  on every Space and beside full-screen apps (`canJoinAllSpaces | fullScreenAuxiliary | stationary | ignoresCycle`).
- **Where.** The screen with a notch (`safeAreaInsets.top > 0`; `auxiliaryTopLeftArea` / `auxiliaryTopRightArea` say
  where it ends): the wing starts at the notch's right edge, as tall as the notch, 38 pt wide, black, its bottom-right
  corner rounded (10 pt) so it reads as part of the notch. Without a notch anywhere: the main screen, centred at the top,
  as tall as the menu bar, 38 pt wide, both bottom corners rounded. Placed again when screens change
  (`NSApplicationDidChangeScreenParametersNotification`).
- `wing_frame(screens) -> Rect` is pure (inputs: each screen's frame, visible frame, safe-area top and auxiliary areas),
  tested for a notched laptop alone, a notched laptop with an external main screen, and no notch.
- The face is 22 pt (as in the Omarchy bar), centred in the wing.
- **A click** toggles the dock, as `aria toggle` does.

### The state

`face.rs` gains `on_change(callback)`: the dock registers one at start, called with the face whenever it changes (the
file is still written, as on Omarchy). The callback hops to the main thread and calls `Motion::want`.

## Errors and edge cases

- No main thread marker or no screen: the wing is not shown; the dock works as before and logs it once.
- Display asleep, screen locked: the frame timer stops; the face resumes where it is (the motion takes the gap as one
  capped step).
- A screen change while a face plays: the wing moves; the motion goes on.
- Linux is unchanged: `face_motion.rs` compiles there and is tested, but nothing uses it yet (the bar keeps its QML).

## Testing

- `face_motion.rs` (both platforms): every word gives its look and accent, an unknown one idle; the change waits for the
  blink; springs settle; each look's sequence moves what it moves (the book shows when reading, the laptop types and its
  lines grow when editing, the command's output scrolls when working, the subagents hop when delegating, rings appear
  when asking and come sooner the longer it waits, the clock rings then ticks when scheduled, confetti flies once when
  done, smoke when error, the z rises when sleeping); idle blinks, glances, yawns and rolls over a long run; a frame's
  values stay finite and in range over ten simulated minutes of every look; the same seed gives the same frames.
- `wing_frame`: the three screen layouts.
- `face.rs`: `on_change` is called once per change.
- Live on the owner's Mac: the wing beside the notch, each face played (`aria face <word>` is not added; a debug
  environment variable `ARIA_FACE=<word>` forces one), a click opens the dock, CPU while idle under 1 %.

## Out of scope

Hiding the wing from the settings, a right-click menu, hover details, using `face_motion.rs` for the Omarchy bar.
