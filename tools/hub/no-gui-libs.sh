#!/bin/sh
# Fails when a binary links a library that needs a desktop (what a computer with no desktop does not have):
#   sh tools/hub/no-gui-libs.sh target/release/aria
set -u
[ $# -eq 1 ] || { echo "usage: sh tools/hub/no-gui-libs.sh <binary>" >&2; exit 2; }
binary=$1
libs=$(ldd "$binary") || { echo "ldd could not read $binary" >&2; exit 2; }
found=$(printf '%s\n' "$libs" | grep -E 'gtk|gdk|webkit|vulkan|asound|javascriptcore' || true)
if [ -n "$found" ]; then
  echo "GUI libraries linked into $binary:" >&2
  printf '%s\n' "$found" >&2
  exit 1
fi
echo "no GUI libraries linked"
