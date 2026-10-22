import QtQuick
import Quickshell
import Quickshell.Io
import qs.Commons
import qs.Ui

// Aria's face in the Omarchy bar, installed by `aria bar install`: a coloured dot with two eyes. So far
// the idle face (glances, blinks, yawns, rolls side to side when bored), driven frame by frame with
// springs at the display's rate.
// Its state is the word in $XDG_RUNTIME_DIR/aria/face (Aria writes it; `omarchy-shell aria.face set
// <state>` does too); for now the state only changes its colour. A click runs `aria toggle`.
BarWidget {
  id: root
  moduleName: "aria.face"

  readonly property real size: 22
  readonly property real k: size / 96
  property string expr: "idle"
  readonly property var accents: ({
    idle: "#B9A8FF", thinking: "#B9A8FF", reading: "#B9A8FF", editing: "#B9A8FF", working: "#B9A8FF",
    delegating: "#B9A8FF", asking: "#F0B44C", scheduled: "#7DD3FC", done: "#9BE7C4", error: "#FF8F87",
    sleeping: "#8A8A94"
  })
  readonly property color accent: accents[expr] || "#B9A8FF"

  implicitWidth: 26
  implicitHeight: barSize

  // Spring state: position x, velocity v, and where it is heading.
  property var s: ({
    gx: { x: 0, v: 0 }, gy: { x: 0, v: 0 }, lid: { x: 1, v: 0 }, yawn: { x: 0, v: 0 }, roll: { x: 0, v: 0 },
    tgx: 0, tgy: 0, lidTarget: 1, yawnTarget: 0, rollTarget: 0, rolling: false
  })

  // What the frame loop paints.
  property real eyeX: 0
  property real eyeY: 0
  property real lidScale: 1
  property real gape: 0
  property real rollX: 0
  property real rollDeg: 0
  property real stretch: 0

  function between(lo, hi) { return Math.random() * (hi - lo) + lo }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)) }
  function spring(o, target, kk, c, dt) {
    o.v += (-kk * (o.x - target) - c * o.v) * dt
    o.x += o.v * dt
  }

  function step(dt) {
    var st = root.s
    var yawning = st.yawnTarget > 0
    var tx = yawning ? 0 : (st.rolling ? clamp(st.roll.v * 6, -1, 1) : st.tgx)
    var ty = yawning ? -0.25 : st.tgy
    var n = Math.max(1, Math.ceil(dt / (1 / 240)))
    var h = dt / n
    for (var i = 0; i < n; i++) {
      spring(st.gx, tx, 220, 18, h)
      spring(st.gy, ty, 220, 18, h)
      spring(st.lid, st.lidTarget, 1100, 56, h)
      spring(st.yawn, st.yawnTarget, 38, 11, h)
      spring(st.roll, st.rollTarget, 14, 7.2, h)
    }
    var a = clamp(st.yawn.x, 0, 1.15)
    eyeX = st.gx.x * 6 * k
    eyeY = st.gy.x * 5 * k
    lidScale = clamp(Math.min(st.lid.x, 1 - 0.9 * clamp(a, 0, 1)), 0.05, 1.05)
    gape = Math.max(0, a)
    stretch = a
    rollX = st.roll.x * size
    rollDeg = (st.roll.x / 0.38) * 57.2958
  }

  FrameAnimation {
    running: root.visible
    onTriggered: root.step(Math.min(0.05, frameTime))
  }

  Timer {
    id: glance
    interval: 800
    running: true
    onTriggered: {
      var center = Math.random() < 0.25
      root.s.tgx = center ? 0 : root.between(-1, 1)
      root.s.tgy = center ? 0 : root.between(-0.7, 0.7)
      interval = root.between(1200, 2800)
      restart()
    }
  }

  Timer {
    id: blink
    interval: 2500
    running: true
    property bool closing: true
    onTriggered: {
      if (closing && root.s.lidTarget === 1) { root.s.lidTarget = 0.1; interval = 120; closing = false }
      else { if (root.s.lidTarget === 0.1) root.s.lidTarget = 1; interval = root.between(2000, 5000); closing = true }
      restart()
    }
  }

  // Yawns every 9 to 16 s, never while rolling.
  Timer {
    id: yawnTimer
    interval: 4500
    running: true
    property bool open: false
    onTriggered: {
      if (!open) {
        if (root.expr !== "idle" || root.s.rolling) { interval = root.between(9000, 16000); restart(); return }
        root.s.yawnTarget = 1; open = true; interval = 1500
      } else {
        root.s.yawnTarget = 0; open = false; interval = root.between(9000, 16000)
      }
      restart()
    }
  }

  // Bored: rolls one way, then the other, then back to the middle, every 8 to 15 s.
  Timer {
    id: rollTimer
    interval: 8000
    running: true
    property int leg: 0
    property int dir: 1
    onTriggered: {
      if (leg === 0) {
        if (root.expr !== "idle" || root.s.yawnTarget > 0) { interval = root.between(8000, 15000); restart(); return }
        dir = Math.random() < 0.5 ? 1 : -1
        root.s.rolling = true
        root.s.rollTarget = 0.15 * dir; leg = 1; interval = 1700
      } else if (leg === 1) {
        root.s.rollTarget = -0.15 * dir; leg = 2; interval = 1800
      } else if (leg === 2) {
        root.s.rollTarget = 0; leg = 3; interval = 1800
      } else {
        root.s.rolling = false; leg = 0; interval = root.between(8000, 15000)
      }
      restart()
    }
  }

  // The state lives in one file that every monitor's bar watches: Aria writes it, and the IPC call
  // (which reaches only one bar) writes it too, so all of them change together.
  FileView {
    id: stateFile
    path: Quickshell.env("XDG_RUNTIME_DIR") + "/aria/face"
    watchChanges: true
    printErrors: false
    onFileChanged: reload()
    onLoaded: {
      var next = text().trim()
      if (next) root.expr = next
    }
  }

  IpcHandler {
    target: "aria.face"
    // Written by another process: the shell does not see its own writes, so the other bars would miss it.
    function set(state: string): void {
      Quickshell.execDetached(["sh", "-c", "printf '%s\\n' \"$1\" > \"$2\"", "sh", state, stateFile.path])
    }
  }

  Item {
    id: face
    width: root.size
    height: root.size
    anchors.centerIn: parent

    Item {
      id: body
      anchors.fill: parent
      transform: [
        Scale { origin.x: face.width / 2; origin.y: face.height * 0.88; xScale: 1 - 0.04 * root.stretch; yScale: 1 + 0.08 * root.stretch },
        Rotation { origin.x: face.width / 2; origin.y: face.height / 2; angle: root.rollDeg },
        Translate { x: root.rollX; y: -0.03 * root.stretch * root.size }
      ]

      Rectangle {
        anchors.centerIn: parent
        width: root.size * 0.76
        height: width
        radius: width / 2
        color: root.accent
        Behavior on color { ColorAnimation { duration: 300 } }
      }

      // Eyes: two upright pills, and the yawn's open mouth below them.
      Item {
        id: gaze
        x: face.width / 2 + root.eyeX
        y: face.height / 2 + root.eyeY

        Rectangle {
          width: 7.8 * root.k
          height: 17.3 * root.k
          radius: width / 2
          color: "#0A0A0C"
          x: -14 * root.k - width / 2
          y: -height / 2
          transform: Scale { origin.y: 17.3 * root.k / 2; yScale: root.lidScale }
        }
        Rectangle {
          width: 7.8 * root.k
          height: 17.3 * root.k
          radius: width / 2
          color: "#0A0A0C"
          x: 14 * root.k - width / 2
          y: -height / 2
          transform: Scale { origin.y: 17.3 * root.k / 2; yScale: root.lidScale }
        }
        Rectangle {
          width: 13 * root.k
          height: 16 * root.k
          radius: width / 2
          color: "#0A0A0C"
          x: -width / 2
          y: 9.5 * root.k
          opacity: root.gape > 0.02 ? 1 : 0
          transform: Scale { origin.x: 6.5 * root.k; origin.y: 3.2 * root.k; xScale: root.gape; yScale: root.gape }
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
