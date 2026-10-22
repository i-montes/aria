import QtQuick
import QtQuick.Shapes
import Quickshell
import Quickshell.Io
import qs.Commons
import qs.Ui

// Aria's face in the Omarchy bar, installed by `aria bar install`. The dock writes what Aria is doing, one word, to
// $XDG_RUNTIME_DIR/aria/face (`omarchy-shell aria.face set <word>` does too); the face plays it:
//   idle        glances, blinks, yawns, rolls side to side when bored
//   thinking    looks up into a corner, one eye narrowed, head tilted, "hmm"
//   reading     head bowed over an open book, eyes word by word, a page turns every three lines
//   editing     types on its laptop in bursts; code fills the screen
//   working     a command: its output scrolls up, it leans in, checks it when it stops
//   delegating  two little subagents work beside it; it checks on each in turn
//   asking      round eyes, taps on the glass (rings), sooner the longer it waits
//   scheduled   its alarm clock rings, it starts awake, then tick-tocks with the clock
//   done        a jump with a full turn, confetti, then a content sway
//   error       a puff of smoke, it sags, x x eyes, a soft shake now and then
//   sleeping    eyes shut, it breathes, a z floats up
// Everything is drawn in a 96 x 96 space (the design's) scaled to the bar, and moved frame by frame with springs.
// A click runs `aria toggle`.
BarWidget {
  id: root
  moduleName: "aria.face"

  readonly property real size: 22
  readonly property real k: size / 96
  readonly property color ink: "#0A0A0C"
  readonly property var accents: ({
    idle: "#B9A8FF", thinking: "#B9A8FF", reading: "#B9A8FF", editing: "#B9A8FF", working: "#B9A8FF",
    delegating: "#B9A8FF", asking: "#F0B44C", scheduled: "#7DD3FC", done: "#9BE7C4", error: "#FF8F87",
    sleeping: "#8A8A94"
  })
  // eyes, mouth, how the eyes move, and the body's own motion
  readonly property var looks: ({
    idle: { l: "line", r: "line", m: "", mode: "glance", body: "still" },
    thinking: { l: "line", r: "line", m: "hmm", mode: "think", body: "still" },
    reading: { l: "line", r: "line", m: "", mode: "read", body: "still" },
    editing: { l: "line", r: "line", m: "", mode: "type", body: "still" },
    working: { l: "line", r: "line", m: "", mode: "terminal", body: "still" },
    delegating: { l: "line", r: "line", m: "", mode: "delegate", body: "still" },
    asking: { l: "open", r: "open", m: "o", mode: "center", body: "float" },
    scheduled: { l: "line", r: "line", m: "smile", mode: "glance", body: "still" },
    done: { l: "happy", r: "happy", m: "smile", mode: "center", body: "still" },
    error: { l: "x", r: "x", m: "wave", mode: "center", body: "shake" },
    sleeping: { l: "closed", r: "closed", m: "", mode: "down", body: "breathe" }
  })

  // What the file says, and what is on screen (the change waits behind a blink).
  property string wanted: "idle"
  property string shown: "idle"
  readonly property var look: looks[shown] || looks.idle
  readonly property color accent: accents[shown] || "#B9A8FF"

  implicitWidth: 26
  implicitHeight: barSize

  // ------------------------------------------------------------------ the state file
  FileView {
    id: stateFile
    path: Quickshell.env("XDG_RUNTIME_DIR") + "/aria/face"
    watchChanges: true
    printErrors: false
    onFileChanged: reload()
    onLoaded: {
      var next = text().trim()
      if (next && root.looks[next]) root.wanted = next
    }
  }

  IpcHandler {
    target: "aria.face"
    // Written by another process: the shell does not see its own writes, so the other bars would miss it.
    function set(state: string): void {
      Quickshell.execDetached(["sh", "-c", "printf '%s\\n' \"$1\" > \"$2\"", "sh", state, stateFile.path])
    }
  }

  onWantedChanged: {
    if (wanted === shown) return
    s.lidTarget = 0.05
    s.yawnTarget = 0
    swap.restart()
  }
  Timer {
    id: swap
    interval: 130
    onTriggered: root.enter(root.wanted)
  }

  // ------------------------------------------------------------------ motion state
  property var s: ({
    t: 0, gx: sp(0), gy: sp(0), lid: sp(1), yawn: sp(0), roll: sp(0), mix: sp(1), tilt: sp(0), nod: sp(0),
    squint: sp(0), ponder: sp(0), bowed: sp(0), readAmt: sp(0), lapAmt: sp(0), flip: sp(0), tap: sp(0), lean: sp(0),
    scroll: sp(0), dl: sp(0), hopL: sp(0), hopR: sp(0), mL: { x: sp(0), y: sp(0) }, mR: { x: sp(0), y: sp(0) },
    knock: sp(0), sched: sp(0), wake: sp(0), jolt: sp(0), tock: sp(0), hand: sp(0), err: sp(0),
    tgx: 0, tgy: 0, lidTarget: 1, yawnTarget: 0, rollTarget: 0, rolling: false, tiltTarget: 0, squintTarget: 0,
    thinkSide: 1, readX: -0.9, readRow: 0, returning: false, flipTarget: 0, typing: false, leanOn: false,
    scr: null, focusSide: -1, prevBody: "still", curBody: "still", since: 0,
    rings: [-9, -9, -9], ringIdx: 0, parts: [], burst: false, ticks: 0, woke: false, askSince: 0
  })
  function sp(x) { return { x: x, v: 0 } }

  // what the frame paints
  property real bodyTx: 0
  property real bodyTy: 0
  property real bodyRot: 0
  property real bodySx: 1
  property real bodySy: 1
  property real bodyOp: 1
  property real rollX: 0
  property real rollDeg: 0
  property real eyeX: 0
  property real eyeY: 0
  property real lidScale: 1
  property real eyeLScale: 1
  property real eyeRScale: 1
  property real mouthX: 0
  property real gape: 0
  property real bookOp: 0
  property real lapOp: 0
  property real pageScale: 1
  property var lineW: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  property var lineC: ["#B9A8FF", "#B9A8FF", "#B9A8FF", "#B9A8FF", "#B9A8FF", "#B9A8FF", "#B9A8FF", "#B9A8FF", "#B9A8FF", "#B9A8FF", "#B9A8FF", "#B9A8FF"]
  property var lineInd: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  property real scrollY: 0
  property real cursorX: 0
  property real cursorY: 0
  property real cursorOp: 0
  property real dlOp: 0
  property real miniLy: 0
  property real miniRy: 0
  property real miniLex: 0
  property real miniLey: 0
  property real miniRex: 0
  property real miniRey: 0
  property var ringP: [-1, -1, -1]
  property real clockOp: 0
  property real clockShake: 0
  property real handDeg: 0
  property bool bellsOn: false
  property var partState: []
  property var puffP: [-1, -1, -1]
  property real zP: 0

  function between(lo, hi) { return Math.random() * (hi - lo) + lo }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)) }
  function spring(o, target, kk, c, dt) {
    o.v += (-kk * (o.x - target) - c * o.v) * dt
    o.x += o.v * dt
  }

  // A new state: the eyes change behind the blink, the body blends into its motion, and its sequence starts.
  function enter(next) {
    var st = s
    st.prevBody = st.curBody
    st.curBody = (looks[next] || looks.idle).body
    st.mix = sp(0)
    st.since = st.t
    st.scr = null
    st.readX = -0.9
    st.readRow = 0
    if (next === "asking") st.askSince = st.t
    if (next === "scheduled") { st.woke = false; st.ticks = 0 }
    if (next === "done") st.burst = false
    shown = next
    st.lidTarget = 1
  }

  // ------------------------------------------------------------------ poses
  function pose(kind, t) {
    var p = { tx: 0, ty: 0, rot: 0, sx: 1, sy: 1, op: 1 }
    if (kind === "float") {
      p.ty = -0.04 * (0.5 - 0.5 * Math.cos(2 * Math.PI * t / 2.6))
    } else if (kind === "shake") {
      var q = ((t % 2.8) / 2.8 - 0.7) / 0.3
      if (q > 0) p.tx = 0.03 * Math.sin(2 * Math.PI * 2 * q) * Math.sin(Math.PI * q)
    } else if (kind === "breathe") {
      var b = Math.sin(2 * Math.PI * t / 4)
      p.sx = p.sy = 0.99 + 0.03 * b
      p.op = 0.875 + 0.125 * b
    }
    return p
  }

  function celebration(u) {
    var p = { sx: 1, sy: 1, ty: 0, spin: 0 }
    if (u < 0.25) {
      var a = Math.sin(Math.PI * u / 0.25)
      p.sy = 1 - 0.12 * a; p.sx = 1 + 0.1 * a
    } else if (u < 0.85) {
      var v = (u - 0.25) / 0.6
      var s1 = Math.sin(Math.PI * v)
      p.ty = -0.24 * s1
      p.spin = 360 * (v < 0.5 ? 2 * v * v : 1 - Math.pow(-2 * v + 2, 2) / 2)
      p.sy = 1 + 0.06 * s1; p.sx = 2 - p.sy
    } else if (u < 1.15) {
      var l = Math.sin(Math.PI * (u - 0.85) / 0.3)
      p.sy = 1 - 0.1 * l; p.sx = 1 + 0.08 * l
    } else {
      p.ty = -0.02 * (0.5 - 0.5 * Math.cos(2 * Math.PI * (u - 1.15) / 1.4))
    }
    return p
  }

  // ------------------------------------------------------------------ the laptop's screen
  function newScreen(kind) {
    var lines = []
    for (var i = 0; i < 12; i++) {
      if (kind === "command") lines.push(i === 0 ? { w: 14, ind: 0, c: accent } : { w: between(10, 34), ind: 0, c: "#A1A1AA" })
      else lines.push({ w: between(8, 26), ind: [0, 4, 4, 8][Math.floor(Math.random() * 4)], c: i % 3 === 2 ? "#A1A1AA" : accent })
    }
    var start = 0
    for (var j = 0; j < lines.length; j++) { lines[j].start = start; start += lines[j].w }
    s.scr = { kind: kind, lines: lines, total: start, typed: 0, shown: 0, upTo: Math.floor(between(6, 12)),
              phase: kind === "command" ? "prompt" : "type", at: s.t, next: 0, row: 0 }
    s.leanOn = false
    lineC = lines.map(function (ln) { return ln.c })
    lineInd = lines.map(function (ln) { return ln.ind })
  }
  function scrollRows() { return s.scr ? Math.max(0, s.scr.row - 4) : 0 }
  function runScreen(dt) {
    var sc = s.scr
    if (sc.kind === "edit") {
      if (s.typing) sc.typed += 34 * dt
      if (sc.typed >= sc.total) { newScreen("edit"); return { x: -0.7, y: 0.55 } }
      var row = 0
      while (row < 11 && sc.typed >= sc.lines[row].start + sc.lines[row].w) row++
      sc.row = row
      var ln = sc.lines[row]
      var cx = ln.ind + clamp(sc.typed - ln.start, 0, ln.w)
      return { x: cx / 36 * 1.4 - 0.7, y: 0.5 + (row - scrollRows()) * 0.07 }
    }
    if (sc.phase === "prompt") {
      sc.typed = Math.min(sc.lines[0].w, sc.typed + 30 * dt)
      if (sc.typed >= sc.lines[0].w && s.t - sc.at > 1) { sc.phase = "output"; sc.at = s.t; sc.next = s.t + 0.25; s.leanOn = Math.random() < 0.6 }
      return { x: sc.typed / 36 * 1.4 - 0.7, y: 0.5 }
    }
    if (sc.phase === "output") {
      if (s.t >= sc.next) {
        sc.shown++
        sc.next = s.t + (Math.random() < 0.1 ? 0.5 : between(0.07, 0.18))
        if (sc.shown >= sc.upTo) { sc.phase = "done"; sc.at = s.t; s.leanOn = false }
      }
      sc.row = sc.shown
      return { x: -0.45 + 0.15 * Math.sin(s.t * 3), y: 0.5 + (sc.row - scrollRows()) * 0.07 }
    }
    var since = s.t - sc.at
    if (since > 1.5) { newScreen("command"); return { x: -0.7, y: 0.5 } }
    return { x: since < 1.1 ? 0.6 * Math.sin(since * 6) : 0, y: 0.45 }
  }
  function paintScreen() {
    var sc = s.scr
    if (!sc) return
    var w = []
    for (var i = 0; i < 12; i++) {
      var ln = sc.lines[i]
      if (sc.kind === "edit") w.push(clamp(sc.typed - ln.start, 0, ln.w))
      else w.push(i === 0 ? sc.typed : (i <= sc.shown ? ln.w : 0))
    }
    lineW = w
    scrollY = -s.scroll.x * 3.6
    var row = sc.kind === "edit" ? sc.row : 0
    var cl = sc.lines[row]
    cursorX = 3 + cl.ind + (sc.kind === "edit" ? clamp(sc.typed - cl.start, 0, cl.w) : sc.typed) + 0.8
    cursorY = 2 + row * 3.6 - s.scroll.x * 3.6 - 0.4
    var solid = sc.kind === "edit" ? s.typing : false
    cursorOp = sc.kind === "command" && sc.phase !== "prompt" ? 0 : (solid || Math.sin(s.t * 9) > 0 ? 1 : 0)
  }

  // ------------------------------------------------------------------ one frame
  function step(dt) {
    var st = s
    st.t += dt
    var mode = look.mode
    var yawning = st.yawnTarget > 0
    if (shown !== "idle") st.rollTarget = 0
    var thinking = mode === "think" && !yawning
    var reading = mode === "read"
    var desk = mode === "type" ? "edit" : (mode === "terminal" ? "command" : "")
    var delegating = mode === "delegate"
    var tx = yawning ? 0 : (st.rolling ? clamp(st.roll.v * 6, -1, 1) : st.tgx)
    var ty = yawning ? -0.25 : st.tgy
    if (thinking) { tx += 0.12 * Math.sin(st.t * 0.8); ty += 0.06 * Math.sin(st.t * 1.1 + 1) }
    if (desk) {
      if (!st.scr || st.scr.kind !== desk) newScreen(desk)
      var at = runScreen(dt)
      tx = at.x; ty = at.y
    }
    if (delegating) { tx = 0.85 * st.focusSide; ty = 0.55 }
    if (mode === "center") { tx = 0; ty = shown === "asking" ? 0.1 : 0 }
    if (mode === "down") { tx = 0; ty = 0.55 }
    var gk = thinking ? 45 : 220
    var gc = thinking ? 12 : 18
    if (reading) { gk = st.returning ? 60 : 420; gc = st.returning ? 15.5 : 41 }
    if (desk) { gk = 300; gc = 34 }

    // scheduled: the alarm for a second, a start awake, then a tick a second
    var ringing = false
    if (shown === "scheduled") {
      var since = st.t - st.since
      if (since < 1.1) ringing = true
      else {
        if (!st.woke) { st.woke = true; st.jolt.v -= 1.3; st.lid.v += 6 }
        var tick = Math.floor(since - 1.4)
        if (tick >= 0 && tick + 1 > st.ticks) st.ticks = tick + 1
      }
    }
    // done: confetti at the top of the jump
    if (shown === "done" && !st.burst && st.t - st.since >= 0.55) {
      st.burst = true
      var parts = []
      for (var i = 0; i < 8; i++) {
        var angle = -Math.PI / 2 + (i - 3.5) * 0.34 + between(-0.1, 0.1)
        var speed = between(0.9, 1.45)
        parts.push({ x: 0, y: -0.12, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed, life: 1, rot: 0, vr: between(-420, 420) })
      }
      st.parts = parts
    }
    for (var p = 0; p < st.parts.length; p++) {
      var pt = st.parts[p]
      if (pt.life <= 0) continue
      pt.vy += 2.6 * dt; pt.x += pt.vx * dt; pt.y += pt.vy * dt; pt.rot += pt.vr * dt; pt.life -= dt / 1.4
    }

    var n = Math.max(1, Math.ceil(dt / (1 / 240)))
    var h = dt / n
    for (var j = 0; j < n; j++) {
      spring(st.gx, tx, gk, gc, h)
      spring(st.gy, ty, gk, gc, h)
      spring(st.lid, st.lidTarget, 1100, 56, h)
      spring(st.yawn, st.yawnTarget, 38, 11, h)
      spring(st.mix, 1, 90, 19, h)
      spring(st.roll, st.rollTarget, 14, 7.2, h)
      spring(st.tilt, thinking ? st.tiltTarget : 0, 28, 9, h)
      spring(st.ponder, thinking ? 1 : 0, 40, 12, h)
      spring(st.squint, thinking ? st.squintTarget : 0, 32, 10, h)
      spring(st.nod, 0, 30, 9, h)
      spring(st.bowed, reading || desk ? 1 : 0, 36, 12, h)
      spring(st.readAmt, reading ? 1 : 0, 36, 12, h)
      spring(st.lapAmt, desk ? 1 : 0, 36, 12, h)
      spring(st.flip, st.flipTarget, 70, 17, h)
      spring(st.tap, 0, 600, 34, h)
      spring(st.lean, st.leanOn && desk === "command" ? 1 : 0, 20, 9, h)
      spring(st.scroll, scrollRows(), 120, 22, h)
      spring(st.dl, delegating ? 1 : 0, 40, 12, h)
      spring(st.hopL, 0, 200, 12, h)
      spring(st.hopR, 0, 200, 12, h)
      spring(st.mL.x, delegating && st.focusSide < 0 ? 0.8 : 0, 200, 20, h)
      spring(st.mL.y, delegating && st.focusSide < 0 ? -0.8 : 0.6, 200, 20, h)
      spring(st.mR.x, delegating && st.focusSide > 0 ? -0.8 : 0, 200, 20, h)
      spring(st.mR.y, delegating && st.focusSide > 0 ? -0.8 : 0.6, 200, 20, h)
      spring(st.knock, 0, 520, 26, h)
      spring(st.sched, shown === "scheduled" ? 1 : 0, 40, 12, h)
      spring(st.wake, ringing ? 1 : 0, 160, 22, h)
      spring(st.jolt, 0, 160, 10, h)
      spring(st.tock, shown === "scheduled" && st.ticks ? (st.ticks % 2 ? 3 : -3) : 0, 300, 22, h)
      spring(st.hand, st.ticks * 30, 400, 28, h)
      spring(st.err, shown === "error" ? 1 : 0, 25, 9, h)
    }
    if (st.flipTarget === 1 && st.flip.x > 0.985) { st.flipTarget = 0; st.flip = sp(0) }
    paint(ringing)
  }

  function paint(ringing) {
    var st = s
    var a = clamp(st.yawn.x, 0, 1.15)
    var w = clamp(st.mix.x, 0, 1)
    var from = pose(st.prevBody, st.t)
    var to = pose(st.curBody, st.t)
    function m(key) { return from[key] + (to[key] - from[key]) * w }
    var cel = shown === "done" ? celebration(st.t - st.since) : { sx: 1, sy: 1, ty: 0, spin: 0 }
    var er = clamp(st.err.x, 0, 1)
    var bw = clamp(st.bowed.x, 0, 1)
    var ln = clamp(st.lean.x, 0, 1)
    var kn = st.knock.x
    bodySx = cel.sx * (1 + 0.07 * er) * m("sx") * (1 - 0.04 * a) * (1 + 0.025 * ln) * (1 + 0.25 * kn)
    bodySy = cel.sy * (1 - 0.08 * er) * m("sy") * (1 + 0.08 * a) * (1 + 0.025 * ln) * (1 + 0.25 * kn)
    bodyTy = (m("ty") + cel.ty + 0.03 * er - 0.03 * a + st.nod.x + 0.015 * bw + st.tap.x + 0.02 * ln + st.jolt.x * 0.1) * 96
    bodyTx = m("tx") * 96
    bodyRot = m("rot") + st.tilt.x + 3 * bw + st.tock.x
    bodyOp = m("op")
    rollX = st.roll.x * 96
    rollDeg = st.roll.x / 0.38 * 57.2958 + cel.spin
    var pd = clamp(st.ponder.x, 0, 1)
    eyeX = st.gx.x * 6 * (1 + 0.5 * pd)
    eyeY = st.gy.x * 5 * (1 + 0.6 * pd) + 4 * bw
    var q = st.squint.x
    eyeLScale = 1 - 0.5 * Math.max(0, q)
    eyeRScale = 1 - 0.5 * Math.max(0, -q)
    mouthX = -1.6 * q + 0.7 * pd * Math.sin(st.t * 1.7)
    lidScale = clamp(Math.min(st.lid.x, 1 - 0.9 * clamp(a, 0, 1), 1 - 0.15 * bw, 1 - 0.9 * clamp(st.wake.x, 0, 1)), 0.05, 1.05)
    gape = Math.max(0, a)
    bookOp = clamp(st.readAmt.x, 0, 1)
    lapOp = clamp(st.lapAmt.x, 0, 1)
    pageScale = 1 - 2 * clamp(st.flip.x, 0, 1)
    if (lapOp > 0.01) paintScreen()
    dlOp = clamp(st.dl.x, 0, 1)
    var mini = 25
    var busyL = st.focusSide < 0 ? 0 : -0.07 * Math.abs(Math.sin(st.t * 2 * Math.PI * 1.7))
    var busyR = st.focusSide > 0 ? 0 : -0.07 * Math.abs(Math.sin(st.t * 2 * Math.PI * 1.7 + 1.3))
    miniLy = (busyL + st.hopL.x * 0.1) * mini
    miniRy = (busyR + st.hopR.x * 0.1) * mini
    miniLex = st.mL.x.x * mini * 0.12; miniLey = st.mL.y.x * mini * 0.1
    miniRex = st.mR.x.x * mini * 0.12; miniRey = st.mR.y.x * mini * 0.1
    var rp = []
    for (var i = 0; i < 3; i++) rp.push((st.t - st.rings[i]) / 0.75)
    ringP = rp
    clockOp = clamp(st.sched.x, 0, 1)
    clockShake = ringing ? 14 * Math.sin(st.t * 2 * Math.PI * 9) : 0
    bellsOn = ringing
    handDeg = st.hand.x
    // New objects each frame: the confetti's bindings only look again when what they read is new.
    partState = st.parts.map(function (q) { return { x: q.x, y: q.y, rot: q.rot, life: q.life } })
    var pp = []
    for (var j = 0; j < 3; j++) pp.push(shown === "error" ? (st.t - st.since - j * 0.14) / 1.1 : -1)
    puffP = pp
    zP = (st.t % 3) / 3
  }

  FrameAnimation {
    running: root.visible
    onTriggered: root.step(Math.min(0.05, frameTime))
  }

  // ------------------------------------------------------------------ the timed behaviours
  Timer { // glances (idle, scheduled), and the reader's saccades
    interval: 800; running: true
    onTriggered: {
      var st = root.s
      var mode = root.look.mode
      if (mode === "glance") {
        var center = Math.random() < 0.25
        st.tgx = center ? 0 : root.between(-1, 1)
        st.tgy = center ? 0 : root.between(-0.7, 0.7)
        interval = root.between(1200, 2800)
      } else if (mode === "read") {
        st.readX += 0.45
        st.returning = st.readX > 0.9
        if (st.returning) {
          st.readX = -0.9
          st.readRow = (st.readRow + 1) % 3
          if (st.readRow === 0) st.flipTarget = 1
          if (st.lidTarget === 1) { st.lidTarget = 0.1; unblink.restart() }
        }
        st.tgx = st.readX
        st.tgy = 0.5 + st.readRow * 0.18
        interval = st.returning ? 750 : root.between(360, 520)
      } else {
        interval = 400
      }
      restart()
    }
  }
  Timer { id: unblink; interval: 120; onTriggered: if (root.s.lidTarget === 0.1) root.s.lidTarget = 1 }

  Timer { // blinks
    interval: 2500; running: true
    property bool closing: true
    onTriggered: {
      var e = root.look
      var opens = e.l === "line" || e.l === "open"
      if (closing && opens && root.s.lidTarget === 1) { root.s.lidTarget = 0.1; interval = 120; closing = false }
      else { if (root.s.lidTarget === 0.1) root.s.lidTarget = 1; interval = root.between(2000, 5000); closing = true }
      restart()
    }
  }

  Timer { // an idle yawn every 9 to 16 s, never while rolling
    interval: 4500; running: true
    property bool open: false
    onTriggered: {
      if (!open) {
        if (root.shown !== "idle" || root.s.rolling) { interval = root.between(9000, 16000); restart(); return }
        root.s.yawnTarget = 1; open = true; interval = 1500
      } else {
        root.s.yawnTarget = 0; open = false; interval = root.between(9000, 16000)
      }
      restart()
    }
  }

  Timer { // bored: a roll one way, then the other, then back, every 8 to 15 s
    interval: 8000; running: true
    property int leg: 0
    property int dir: 1
    onTriggered: {
      var st = root.s
      if (leg === 0) {
        if (root.shown !== "idle" || st.yawnTarget > 0) { interval = root.between(8000, 15000); restart(); return }
        dir = Math.random() < 0.5 ? 1 : -1
        st.rolling = true; st.rollTarget = 0.15 * dir; leg = 1; interval = 1700
      } else if (leg === 1) { st.rollTarget = -0.15 * dir; leg = 2; interval = 1800 }
      else if (leg === 2) { st.rollTarget = 0; leg = 3; interval = 1800 }
      else { st.rolling = false; leg = 0; interval = root.between(8000, 15000) }
      restart()
    }
  }

  Timer { // thinking: a corner, the far eye narrowed, head tilted; every 3 to 4 s the other corner, with a nod
    interval: 300; running: true
    onTriggered: {
      var st = root.s
      if (root.look.mode === "think") {
        var side = st.thinkSide
        st.tgx = 0.8 * side; st.tgy = -0.85
        st.tiltTarget = 8 * side; st.squintTarget = side
        st.nod.v += 0.2
        st.thinkSide = -side
        interval = root.between(3000, 4200)
      } else interval = 400
      restart()
    }
  }

  Timer { // editing types in bursts: taps 85 to 170 ms apart, then a short pause
    id: typing
    interval: 400; running: true
    property real end: 0
    onTriggered: {
      var st = root.s
      if (root.look.mode !== "type") { st.typing = false; interval = 300 }
      else if (!st.typing) { st.typing = true; end = st.t + root.between(0.7, 1.6); interval = 10 }
      else if (st.t > end) { st.typing = false; interval = root.between(400, 1100) }
      else { st.tap.v += 0.45; interval = root.between(85, 170) }
      restart()
    }
  }

  Timer { // delegating: checks on one subagent, then the other; the one it looks at hops to report
    interval: 600; running: true
    onTriggered: {
      var st = root.s
      if (root.look.mode === "delegate") {
        st.focusSide = -st.focusSide
        st.nod.v += 0.2
        if (st.focusSide < 0) st.hopL.v -= 1.4; else st.hopR.v -= 1.4
      }
      interval = root.between(1600, 2600)
      restart()
    }
  }

  Timer { // asking: two taps on the glass, sooner the longer nobody answers (4.2 s down to 1.6 s)
    interval: 900; running: true
    property bool second: false
    onTriggered: {
      var st = root.s
      if (root.shown !== "asking") { second = false; interval = 500; restart(); return }
      st.knock.v += 7
      st.rings[st.ringIdx % 3] = st.t
      st.ringIdx++
      if (!second) { second = true; interval = 190 }
      else { second = false; interval = Math.max(1600, 4200 - (st.t - st.askSince) * 120) }
      restart()
    }
  }

  // ------------------------------------------------------------------ drawing (96 x 96, scaled to the bar)
  Item {
    id: stage
    width: 96
    height: 96
    anchors.centerIn: parent
    scale: root.k

    // the rings a tap on the glass leaves
    Repeater {
      model: 3
      Rectangle {
        required property int index
        readonly property real p: root.ringP[index]
        x: 11.5; y: 11.5; width: 73; height: 73; radius: 36.5
        color: "transparent"
        border.color: root.accent
        border.width: 3
        visible: p >= 0 && p < 1
        opacity: visible ? 0.6 * (1 - p) : 0
        scale: visible ? 1 + 0.45 * (1 - Math.pow(1 - p, 3)) : 1
      }
    }

    Item {
      id: body
      width: 96; height: 96
      opacity: root.bodyOp
      transform: [
        Scale { origin.x: 48; origin.y: 84.5; xScale: root.bodySx; yScale: root.bodySy },
        Rotation { origin.x: 48; origin.y: 84.5; angle: root.bodyRot },
        Translate { x: root.bodyTx; y: root.bodyTy }
      ]

      Item {
        width: 96; height: 96
        transform: [
          Rotation { origin.x: 48; origin.y: 48; angle: root.rollDeg },
          Translate { x: root.rollX }
        ]

        Rectangle {
          x: 11.5; y: 11.5; width: 73; height: 73; radius: 36.5
          color: root.accent
          Behavior on color { ColorAnimation { duration: 300 } }
        }

        // eyes and mouth, moved together by the gaze
        Item {
          id: gaze
          x: 48 + root.eyeX
          y: 48 + root.eyeY

          Item {
            id: lids
            transform: Scale { yScale: root.lidScale }
            Eye { x: -14; kind: root.look.l; squeeze: root.eyeLScale }
            Eye { x: 14; kind: root.look.r; squeeze: root.eyeRScale }
          }

          Mouth { x: root.mouthX; y: 16; kind: root.look.m }

          Rectangle { // the yawn
            x: -6.5; y: 9.5; width: 13; height: 16; radius: 6.5
            color: root.ink
            visible: root.gape > 0.02
            transform: Scale { origin.x: 6.5; origin.y: 3.2; xScale: root.gape; yScale: root.gape }
          }
        }

        // the book (reading)
        Item {
          x: 20; y: 63; width: 56; height: 27
          opacity: root.bookOp
          visible: opacity > 0.01
          transform: Translate { y: (1 - root.bookOp) * 14 }
          Shape {
            anchors.fill: parent
            antialiasing: true
            ShapePath { fillColor: root.ink; strokeColor: "transparent"; PathSvg { path: "M1 6L28 10L55 6V25L28 27L1 25Z" } }
            ShapePath { fillColor: "#F4F4F5"; strokeColor: "transparent"; PathSvg { path: "M3 5Q15 1.5 28 8V24Q15 18.5 3 22Z" } }
            ShapePath { fillColor: "#F4F4F5"; strokeColor: "transparent"; PathSvg { path: "M53 5Q41 1.5 28 8V24Q41 18.5 53 22Z" } }
          }
          Item {
            x: 28; width: 28; height: 27
            transform: Scale { origin.x: 0; xScale: root.pageScale }
            Shape {
              anchors.fill: parent
              antialiasing: true
              ShapePath { fillColor: "#E4E4E7"; strokeColor: "transparent"; PathSvg { path: "M25 5Q13 1.5 0 8V24Q13 18.5 25 22Z" } }
            }
          }
        }

        // the laptop's screen (editing, working)
        Item {
          x: 18; y: 61; width: 60; height: 31
          opacity: root.lapOp
          visible: opacity > 0.01
          transform: Translate { y: (1 - root.lapOp) * 14 }
          Rectangle {
            x: 7; y: 0.8; width: 46; height: 24.5; radius: 3
            color: root.ink
            border.color: "#F4F4F5"
            border.width: 1.6
          }
          Item {
            x: 9.5; y: 3; width: 41; height: 20
            clip: true
            Column {
              x: 3; y: 2 + root.scrollY
              spacing: 1.8
              Repeater {
                model: 12
                Rectangle {
                  required property int index
                  x: root.lineInd[index]
                  width: root.lineW[index]
                  height: 1.8
                  radius: 0.9
                  color: root.lineC[index]
                }
              }
            }
            Rectangle {
              x: root.cursorX; y: root.cursorY; width: 1.6; height: 2.6
              color: "#F4F4F5"
              opacity: root.cursorOp
            }
          }
        }
      }
    }

    // the subagents (delegating)
    Repeater {
      model: 2
      Item {
        required property int index
        readonly property bool isLeft: index === 0
        x: isLeft ? -1.9 : 73; y: 67.2; width: 25; height: 25
        opacity: root.dlOp
        visible: opacity > 0.01
        scale: 0.6 + 0.4 * root.dlOp
        transform: Translate { y: isLeft ? root.miniLy : root.miniRy }
        Rectangle { anchors.fill: parent; radius: 12.5; color: "#8B7BE8"; border.color: "#121212"; border.width: 2 }
        Row {
          x: (25 - 2 * 3.5 - 4) / 2 + (isLeft ? root.miniLex : root.miniRex)
          y: 11 - 1.75 + (isLeft ? root.miniLey : root.miniRey)
          spacing: 4
          Rectangle { width: 3.5; height: 3.5; radius: 1.75; color: root.ink }
          Rectangle { width: 3.5; height: 3.5; radius: 1.75; color: root.ink }
        }
      }
    }

    // the alarm clock (scheduled)
    Item {
      x: 64; y: -1; width: 34.6; height: 34.6; scale: 0.85
      opacity: root.clockOp
      visible: opacity > 0.01
      transformOrigin: Item.Center
      rotation: root.clockShake
      Shape {
        anchors.fill: parent
        antialiasing: true
        transform: Scale { xScale: 34.6 / 40; yScale: 34.6 / 40 }
        ShapePath { strokeColor: root.ink; strokeWidth: 4.5; fillColor: "transparent"; capStyle: ShapePath.RoundCap; PathSvg { path: "M8 10.5A6 6 0 0 1 15.5 5.5M32 10.5A6 6 0 0 0 24.5 5.5" } }
        ShapePath { strokeColor: root.ink; strokeWidth: 3; fillColor: "transparent"; capStyle: ShapePath.RoundCap; PathSvg { path: "M12 35L9.5 38.5M28 35L30.5 38.5" } }
        ShapePath { strokeColor: root.bellsOn ? "#F4F4F5" : "transparent"; strokeWidth: 2; fillColor: "transparent"; capStyle: ShapePath.RoundCap; PathSvg { path: "M5 9L1.5 6M35 9L38.5 6M20 2.5V-1" } }
      }
      Rectangle { x: 6 * 34.6 / 40; y: 8 * 34.6 / 40; width: 28 * 34.6 / 40; height: width; radius: width / 2; color: "#F4F4F5"; border.color: root.ink; border.width: 2.6 }
      Rectangle { // the hand
        x: 20 * 34.6 / 40 - 1; y: 11.5 * 34.6 / 40; width: 2; height: 10.5 * 34.6 / 40; radius: 1
        color: "#7DD3FC"
        transform: Rotation { origin.x: 1; origin.y: 10.5 * 34.6 / 40; angle: root.handDeg }
      }
    }

    // the confetti (done)
    Repeater {
      model: 8
      Rectangle {
        required property int index
        readonly property var pt: root.partState.length > index ? root.partState[index] : null
        x: 48 - 3.1 + (pt ? pt.x * 96 : 0)
        y: 36.5 - 3.1 + (pt ? pt.y * 96 : 0)
        width: 6.2; height: 6.2
        radius: index % 2 ? 3.1 : 1
        rotation: pt ? pt.rot : 0
        color: ["#9BE7C4", "#B9A8FF", "#F0B44C", "#7DD3FC", "#F4F4F5"][index % 5]
        opacity: pt && pt.life > 0 ? Math.min(1, pt.life * 2.5) : 0
      }
    }

    // the smoke (error)
    Repeater {
      model: 3
      Rectangle {
        required property int index
        readonly property real p: root.puffP[index]
        readonly property bool on: p >= 0 && p < 1
        x: 48 - 5.75 + (on ? [-0.08, 0.06, -0.01][index] * 96 * p : 0)
        y: 13.4 - 5.75 - (on ? 0.4 * 96 * p : 0)
        width: 11.5; height: 11.5; radius: 5.75
        color: "#8A8A94"
        opacity: on ? 0.7 * (1 - p) : 0
        scale: on ? 0.6 + p : 1
      }
    }

    // the z (sleeping)
    Text {
      x: 74 + 8 * root.zP; y: -2 - 20 * root.zP
      text: "z"
      font.pixelSize: 21
      font.weight: Font.DemiBold
      color: root.accent
      visible: root.shown === "sleeping"
      opacity: root.zP < 0.4 ? root.zP / 0.4 : 1 - (root.zP - 0.4) / 0.6
    }
  }

  // ------------------------------------------------------------------ eye and mouth shapes
  // Inline components do not see the file's ids, so the colour comes in as a property.
  component Eye: Item {
    property string kind: "line"
    property color ink: "#0A0A0C"
    property real squeeze: 1
    transform: Scale { yScale: squeeze }
    Rectangle { // an upright pill
      visible: kind === "line"
      x: -3.9; y: -8.65; width: 7.8; height: 17.3; radius: 3.9
      color: ink
    }
    Rectangle { // a round open eye
      visible: kind === "open"
      x: -5.1; y: -5.1; width: 10.2; height: 10.2; radius: 5.1
      color: ink
    }
    Shape {
      visible: kind === "happy" || kind === "closed" || kind === "x"
      x: -8.5; y: -8.5; width: 17; height: 17
      antialiasing: true
      ShapePath {
        strokeColor: ink; strokeWidth: kind === "x" ? 4.4 : 3.4; fillColor: "transparent"
        capStyle: ShapePath.RoundCap; joinStyle: ShapePath.RoundJoin
        PathSvg {
          path: kind === "happy" ? "M3.4 11.05Q8.5 3.4 13.6 11.05"
              : kind === "closed" ? "M3.4 7.65Q8.5 12.75 13.6 7.65"
              : "M5.1 5.1L11.9 11.9M11.9 5.1L5.1 11.9"
        }
      }
    }
  }

  component Mouth: Item {
    property string kind: ""
    property color ink: "#0A0A0C"
    Rectangle { // "o"
      visible: kind === "o"
      x: -1.7; y: -1.7; width: 3.4; height: 3.4; radius: 1.7
      color: ink
    }
    Shape {
      visible: kind === "hmm" || kind === "smile" || kind === "wave"
      x: -5; y: -5; width: 10; height: 10
      antialiasing: true
      ShapePath {
        strokeColor: ink; strokeWidth: 1.8; fillColor: "transparent"
        capStyle: ShapePath.RoundCap; joinStyle: ShapePath.RoundJoin
        PathSvg {
          path: kind === "hmm" ? "M3.8 5.4L6.2 4.9"
              : kind === "smile" ? "M2.5 4Q5 7.5 7.5 4"
              : "M2 5.5Q3.5 3.5 5 5.5T8 5.5"
        }
      }
    }
  }

  MouseArea {
    anchors.fill: parent
    cursorShape: Qt.PointingHandCursor
    onClicked: {
      if (root.bar) root.bar.run("aria toggle")
      else Quickshell.execDetached(["aria", "toggle"])
    }
  }
}
