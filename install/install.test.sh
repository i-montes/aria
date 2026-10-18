#!/bin/sh
# Run: sh install/install.test.sh
# Checks the installer without touching the system: its dry run, its embedded unit, its guards.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
installer="$here/install.sh"
failures=0
fail() { echo "FAIL: $1"; failures=$((failures + 1)); }
pass() { echo "ok:   $1"; }

line_of() { printf '%s\n' "$1" | grep -nF -- "$2" | head -1 | cut -d: -f1; }

# the dry run lists every step, in order, and changes nothing
out=$(sh "$installer" --dry-run --local /tmp/fake-aria 2>&1)
last=0
ordered=1
for step in 'sudo pacman -S --needed gtk3 webkit2gtk-4.1 gtk-layer-shell' 'install -Dm755 /tmp/fake-aria' 'install -d -m 700' 'systemctl --user daemon-reload' 'systemctl --user enable --now aria.service'; do
  at=$(line_of "$out" "would: $step") || at=""
  if [ -z "$at" ]; then fail "install dry run lacks 'would: $step'"; ordered=0
  elif [ "$at" -le "$last" ]; then fail "install dry run has '$step' out of order"; ordered=0
  else last=$at; fi
done
[ "$ordered" = 1 ] && pass "install dry run lists its steps in order"

# uninstall dry run
out=$(sh "$installer" --dry-run --uninstall 2>&1)
printf '%s\n' "$out" | grep -qF 'would: systemctl --user disable --now aria.service' && pass "uninstall stops the unit" || fail "uninstall does not stop the unit"
printf '%s\n' "$out" | grep -qF 'hl.unbind' && pass "uninstall removes the runtime hotkey" || fail "uninstall does not unbind SUPER + A"

# the unit embedded in the script is the one in the repo
if sh "$installer" --print-unit | diff -u "$here/aria.service" - >/dev/null; then pass "embedded unit matches install/aria.service"; else fail "embedded unit differs from install/aria.service"; fi

# a bad --local path fails before anything is changed
tmp=$(mktemp -d)
if ARIA_BIN_DIR="$tmp/bin" ARIA_HOME="$tmp/home" sh "$installer" --local /nonexistent/aria >/dev/null 2>&1; then fail "a missing --local binary was accepted"; else pass "a missing --local binary is refused"; fi
[ ! -e "$tmp/bin" ] && [ ! -e "$tmp/home" ] && pass "nothing was created by the refused install" || fail "the refused install left files behind"
rm -r "$tmp"

# unknown flags are refused
if sh "$installer" --frobnicate >/dev/null 2>&1; then fail "an unknown flag was accepted"; else pass "unknown flags are refused"; fi

# syntax, and shellcheck when it is installed
sh -n "$installer" && pass "install.sh parses" || fail "install.sh has a syntax error"
if command -v shellcheck >/dev/null 2>&1; then shellcheck "$installer" && pass "shellcheck is clean" || fail "shellcheck found problems"; else echo "skip: shellcheck is not installed"; fi

[ "$failures" -eq 0 ] && echo "all checks passed" || { echo "$failures check(s) failed"; exit 1; }
