#!/bin/sh
# The terminal client against a real dock, on Linux (Wayland), in a scratch home:
#
#   cargo build -p aria && sh tools/parity/scratch-home.sh sh tools/term/live-check.sh "$PWD/target/debug/aria"
#
# The dock gets its own runtime folder and no Hyprland signature, so it never touches the desktop's binds or the
# running dock; drive.py runs `aria` in a pseudo-terminal: a new session in a folder, !pwd, Ctrl+C, attach again, /exit.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
aria=$1
case "${WAYLAND_DISPLAY:-}" in
  /*) ;;
  "") echo "no WAYLAND_DISPLAY: run it from the desktop session" >&2; exit 2 ;;
  *) WAYLAND_DISPLAY="$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY" ;;
esac
export WAYLAND_DISPLAY
run=$(mktemp -d); chmod 700 "$run"
export XDG_RUNTIME_DIR="$run"
unset HYPRLAND_INSTANCE_SIGNATURE
"$aria" dock > "$run/dock.log" 2>&1 &
dock=$!
trap 'kill $dock 2>/dev/null; sleep 1; rm -rf "$run"' EXIT
for i in $(seq 1 100); do [ -S "$run/aria/clients.sock" ] && break; sleep 0.1; done
[ -S "$run/aria/clients.sock" ] || { echo "FAIL: no clients.sock"; cat "$run/dock.log"; exit 1; }
echo "ok: the dock serves clients.sock"
mkdir -p "$run/x"
python3 -I "$here/drive.py" "$aria" "$run/x" "$ARIA_HOME"
