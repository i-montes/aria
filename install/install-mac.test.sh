#!/bin/sh
# Run: sh install/install-mac.test.sh
# Checks the macOS branch of the installer without touching the system: dry run, LaunchAgent, guards.
# The Omarchy branch has its own test (install.test.sh).
set -eu
if [ "$(uname -s)" != Darwin ]; then
  echo "skip: install-mac.test.sh checks the macOS installer; run it on a Mac"
  exit 0
fi
here=$(cd "$(dirname "$0")" && pwd)
installer="$here/install.sh"
failures=0
fail() { echo "FAIL: $1"; failures=$((failures + 1)); }
pass() { echo "ok:   $1"; }

line_of() { printf '%s\n' "$1" | grep -nF -- "$2" | head -1 | cut -d: -f1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/node" "$tmp/bin" "$tmp/user"
# A Node that passes the version check, and an npm the installer never has to run in a dry run.
printf '#!/bin/sh\necho v22.23.2\n' > "$tmp/node/node"
printf '#!/bin/sh\n' > "$tmp/node/npm"
chmod +x "$tmp/node/node" "$tmp/node/npm"
printf '#!/bin/sh\n' > "$tmp/fake-aria"
chmod +x "$tmp/fake-aria"
# The installer is a plain sh script; keep the real system tools and fake Node/npm.
export PATH="$tmp/node:/usr/bin:/bin:/usr/sbin:/sbin"

env_run() { HOME="$tmp/user" ARIA_HOME="$tmp/home" ARIA_BIN_DIR="$tmp/bin" "$@"; }

# the dry run lists every step, in order, and changes nothing
out=$(env_run sh "$installer" --dry-run --local "$tmp/fake-aria" 2>&1)
last=0
ordered=1
for step in \
  'would: check macOS 12 or newer' \
  'macOS: no system packages are needed' \
  "would: npm install --prefix $tmp/home/pi/" \
  "would: install -Dm755 $tmp/fake-aria $tmp/bin/aria" \
  "would: mkdir -p $tmp/user/Library/LaunchAgents" \
  "would: write $tmp/user/Library/LaunchAgents/com.aria.dock.plist" \
  "would: launchctl bootstrap gui/$(id -u) $tmp/user/Library/LaunchAgents/com.aria.dock.plist"; do
  at=$(line_of "$out" "$step") || at=""
  if [ -z "$at" ]; then fail "install dry run lacks '$step'"; ordered=0
  elif [ "$at" -le "$last" ]; then fail "install dry run has '$step' out of order"; ordered=0
  else last=$at; fi
done
[ "$ordered" = 1 ] && pass "install dry run lists its steps in order"

if printf '%s\n' "$out" | grep -Eq 'pacman|systemctl|hyprctl'; then
  fail "the macOS dry run still mentions a Linux-only step"
else
  pass "the macOS dry run has no pacman, systemd or Hyprland steps"
fi
printf '%s\n' "$out" | grep -qF 'Aria is installed. Press Option + Space to set it up.' && pass "the install message names Option + Space" || fail "the install message does not name the Mac shortcut"

# the LaunchAgent embedded in the script carries the absolute binary and log paths
plist=$(env_run sh "$installer" --print-unit)
for expected in \
  '<string>com.aria.dock</string>' \
  "<string>$tmp/bin/aria</string>" \
  "<string>$tmp/home/logs/aria.log</string>" \
  '<key>LimitLoadToSessionType</key><string>Aqua</string>'; do
  printf '%s\n' "$plist" | grep -qF "$expected" && pass "LaunchAgent has $expected" || fail "LaunchAgent lacks $expected"
done

# uninstall stops the agent, removes both files, and has no Hyprland step
out=$(env_run sh "$installer" --dry-run --uninstall 2>&1)
printf '%s\n' "$out" | grep -qF "would: launchctl bootout gui/$(id -u)/com.aria.dock" && pass "uninstall stops the LaunchAgent" || fail "uninstall does not stop the LaunchAgent"
printf '%s\n' "$out" | grep -qF "would: rm -f $tmp/user/Library/LaunchAgents/com.aria.dock.plist $tmp/bin/aria" && pass "uninstall removes the agent and the binary" || fail "uninstall does not remove the agent and the binary"
if printf '%s\n' "$out" | grep -Eq 'systemctl|hyprctl'; then fail "the macOS uninstall mentions a Linux-only step"; else pass "the macOS uninstall has no systemd or Hyprland step"; fi

# a bad --local path fails before anything is changed
if env_run sh "$installer" --local /nonexistent/aria >/dev/null 2>&1; then
  fail "a missing --local binary was accepted"
else
  pass "a missing --local binary is refused"
fi
[ ! -e "$tmp/bin/aria" ] && [ ! -e "$tmp/home" ] && pass "nothing was created by the refused install" || fail "the refused install left files behind"

# Node is the same requirement on both platforms
out=$(env_run sh "$installer" --check-node 2>&1) && pass "a modern Node is accepted" || fail "a modern Node was refused: $out"

# the pinned version is the repo's
pinned=$(tr -d '[:space:]' < "$here/../pi-version")
[ "$(env_run sh "$installer" --print-pi-version 2>&1)" = "$pinned" ] && pass "prints the pinned Pi version" || fail "prints a different Pi version"

# syntax, and shellcheck when it is installed
sh -n "$installer" && pass "install.sh parses" || fail "install.sh has a syntax error"
if command -v shellcheck >/dev/null 2>&1; then shellcheck "$installer" && pass "shellcheck is clean" || fail "shellcheck found problems"; else echo "skip: shellcheck is not installed"; fi

[ "$failures" -eq 0 ] && echo "all checks passed" || { echo "$failures check(s) failed"; exit 1; }
