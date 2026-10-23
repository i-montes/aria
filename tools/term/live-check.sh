#!/bin/sh
# The terminal client against a real dock, on Linux (Wayland), in a scratch home:
#
#   cargo build -p aria && sh tools/parity/scratch-home.sh sh tools/term/live-check.sh "$PWD/target/debug/aria"
#
# The dock gets its own runtime folder and no Hyprland signature, so it never touches the desktop's binds or the
# running dock; drive.py runs `aria` in a pseudo-terminal: it starts the dock, which outlives its terminal; then a new
# session in a folder, !pwd, Ctrl+C, attach again, /exit.
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
# No dock yet: the client starts it, and closing the client's terminal must not take it down.
trap '"$aria" quit >/dev/null 2>&1 || true; sleep 1; rm -rf "$run"' EXIT
mkdir -p "$run/x"
python3 -I "$here/drive.py" "$aria" "$run/x" "$ARIA_HOME" "$run/aria/clients.sock"
