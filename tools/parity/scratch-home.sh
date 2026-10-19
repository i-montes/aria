#!/bin/sh
# Runs a command in Pi's exact session environment for a scratch copy of the installed Aria home, then deletes the
# copy. The real home (~/.aria, or ARIA_REAL_HOME) is only read: the copy gets its settings, credentials and MCP list,
# and links to its pinned Pi and tools; its agent directory, sessions and Engram data are the copy's own.
#
#   sh tools/parity/scratch-home.sh <command> [args...]
#
# The command runs in a fresh git repo with ARIA_HOME (the copy), ARIA_PI (the pinned Pi), ARIA_PI_DIST (its
# pi-coding-agent dist folder) and ARIA_REAL_AUTH (the real auth.json, for checks that a secret stays hidden).
set -eu
[ $# -ge 1 ] || { echo "usage: scratch-home.sh <command> [args...]" >&2; exit 2; }
repo=$(cd "$(dirname "$0")/../.." && pwd)
real=${ARIA_REAL_HOME:-$HOME/.aria}
[ -d "$real/pi" ] || { echo "no pinned Pi in $real: install Aria first" >&2; exit 2; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM
home="$tmp/home"
work="$tmp/work"
mkdir -p "$home/bin" "$work"
chmod 700 "$tmp"
for file in settings.json auth.json mcp.json mcp-auth.json; do
  [ -f "$real/$file" ] && cp "$real/$file" "$home/$file" && chmod 600 "$home/$file"
done
ln -s "$real/pi" "$home/pi"
for tool in "$real"/bin/*; do
  [ -x "$tool" ] && [ "$(basename "$tool")" != pi ] && ln -s "$tool" "$home/bin/$(basename "$tool")"
done
git -C "$work" init -q

env=$(cd "$repo" && ARIA_HOME="$home" ARIA_PARITY_WORKDIR="$work" cargo run -q -p aria-core --example parity_env)
eval "$env"
export ARIA_HOME="$home"
export ARIA_REAL_AUTH="$real/auth.json"
cd "$work"
set +e
"$@"
code=$?
exit "$code"
