#!/bin/sh
# Run: sh install/install.test.sh
# Checks the Omarchy installer without touching the system: its dry run, its embedded unit, its guards.
# The macOS branch has its own test (install-mac.test.sh).
set -eu
if [ "$(uname -s)" != Linux ]; then
  echo "skip: install.test.sh checks the Linux installer; run it on Omarchy"
  exit 0
fi
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
for step in 'sudo pacman -S --needed gtk3 webkit2gtk-4.1 gtk-layer-shell bubblewrap iproute2 libsecret' 'install -d -m 700' 'install -Dm755 /tmp/fake-aria' 'systemctl --user daemon-reload' 'systemctl --user enable --now aria.service'; do
  at=$(line_of "$out" "would: $step") || at=""
  if [ -z "$at" ]; then fail "install dry run lacks 'would: $step'"; ordered=0
  elif [ "$at" -le "$last" ]; then fail "install dry run has '$step' out of order"; ordered=0
  else last=$at; fi
done
[ "$ordered" = 1 ] && pass "install dry run lists its steps in order"

# the packages the owner's extensions use are installed in Aria's Pi, pinned (pi-packages)
tmp=$(mktemp -d)
out=$(ARIA_HOME="$tmp/home" sh "$installer" --dry-run --local /tmp/fake-aria 2>&1)
while IFS= read -r package; do
  [ -z "$package" ] && continue
  printf '%s\n' "$out" | grep -qF "would: env PI_CODING_AGENT_DIR=$tmp/home/agent PI_SKIP_VERSION_CHECK=1 $tmp/home/pi/" && printf '%s\n' "$out" | grep -qF "install $package" && pass "installs $package" || fail "does not install $package"
done < "$here/../pi-packages"
at_packages=$(line_of "$out" "install git:")
at_binary=$(line_of "$out" "would: install -Dm755")
[ -n "$at_packages" ] && [ -n "$at_binary" ] && [ "$at_packages" -lt "$at_binary" ] && pass "packages are installed before the binary is replaced (a failure leaves the old one running)" || fail "packages come after the binary"
rm -r "$tmp"

# Engram (memory): the pinned release from engram-version, verified, into ~/.aria/bin, before the binary is replaced
engram_version=$(head -1 "$here/../engram-version" | tr -d '[:space:]')
tmp=$(mktemp -d)
out=$(ARIA_HOME="$tmp/home" sh "$installer" --dry-run --local /tmp/fake-aria 2>&1)
at_engram=$(line_of "$out" "would: install engram $engram_version")
at_binary=$(line_of "$out" "would: install -Dm755")
[ -n "$at_engram" ] && [ -n "$at_binary" ] && [ "$at_engram" -lt "$at_binary" ] && pass "Engram $engram_version is installed before the binary is replaced" || fail "the dry run does not install Engram first"
# a download that does not match its checksum is refused and nothing is installed
mkdir -p "$tmp/pkg" && printf '#!/bin/sh\necho engram %s\n' "$engram_version" > "$tmp/pkg/engram" && chmod +x "$tmp/pkg/engram"
tar czf "$tmp/engram.tar.gz" -C "$tmp/pkg" engram
if ARIA_HOME="$tmp/home" ARIA_ENGRAM_URL="file://$tmp/engram.tar.gz" sh "$installer" --install-engram >/dev/null 2>"$tmp/err"; then fail "a download with the wrong checksum was installed"
else grep -qi 'checksum' "$tmp/err" && [ ! -e "$tmp/home/bin/engram" ] && pass "a download with the wrong checksum is refused" || fail "the checksum refusal is unclear or left a file"; fi
# the right one is installed, and a second run keeps it
sum=$(sha256sum "$tmp/engram.tar.gz" | cut -d ' ' -f 1)
if ARIA_HOME="$tmp/home" ARIA_ENGRAM_URL="file://$tmp/engram.tar.gz" ARIA_ENGRAM_SHA256="$sum" sh "$installer" --install-engram >/dev/null 2>&1 && [ -x "$tmp/home/bin/engram" ]; then pass "a verified Engram goes into ~/.aria/bin"; else fail "a verified Engram was not installed"; fi
out=$(ARIA_HOME="$tmp/home" ARIA_ENGRAM_URL="file:///nonexistent" sh "$installer" --install-engram 2>&1) && printf '%s\n' "$out" | grep -qF "Engram $engram_version is already installed" && pass "an installed Engram is not downloaded again" || fail "an installed Engram is downloaded again"
rm -r "$tmp"

# agent-browser (the agents' browser check): the pinned version from agent-browser-version, after the Pi packages and
# before the binary is replaced; a failure only warns
browser_version=$(tr -d '[:space:]' < "$here/../agent-browser-version")
tmp=$(mktemp -d)
out=$(ARIA_HOME="$tmp/home" sh "$installer" --dry-run --local /tmp/fake-aria 2>&1)
last=$(line_of "$out" "install git:") || last=""
[ -n "$last" ] || { fail "the dry run lists no Pi packages"; last=0; }
ordered=1
for step in \
  "would: npm install --prefix $tmp/home/agent-browser/$browser_version --no-audit --no-fund agent-browser@$browser_version" \
  "would: ln -sf $tmp/home/agent-browser/$browser_version/node_modules/.bin/agent-browser $tmp/home/bin/agent-browser" \
  "would: $tmp/home/bin/agent-browser install" \
  "would: install -Dm755"; do
  at=$(line_of "$out" "$step") || at=""
  if [ -z "$at" ]; then fail "install dry run lacks '$step'"; ordered=0
  elif [ "$at" -le "$last" ]; then fail "install dry run has '$step' out of order"; ordered=0
  else last=$at; fi
done
[ "$ordered" = 1 ] && pass "agent-browser $browser_version and its browser are installed after the Pi packages, before the binary"
mkdir -p "$tmp/fail" && printf '#!/bin/sh\nexit 1\n' > "$tmp/fail/npm" && chmod +x "$tmp/fail/npm"
if PATH="$tmp/fail:$PATH" ARIA_HOME="$tmp/home" sh "$installer" --install-agent-browser >/dev/null 2>"$tmp/err"; then
  if grep -qF 'warning:' "$tmp/err"; then pass "a failed agent-browser install warns and the install goes on"; else fail "a failed agent-browser install does not warn"; fi
else fail "a failed agent-browser install stops the installer"; fi
# a bad version: a working fake npm records any call, so only the version check can keep it from running
mkdir -p "$tmp/record" && printf '#!/bin/sh\necho "$*" >> "%s"\n' "$tmp/npm-calls" > "$tmp/record/npm" && chmod +x "$tmp/record/npm"
if PATH="$tmp/record:$PATH" ARIA_HOME="$tmp/home" ARIA_AGENT_BROWSER_VERSION=bad sh "$installer" --install-agent-browser >/dev/null 2>"$tmp/err"; then
  if grep -qF 'not a version number' "$tmp/err" && [ ! -e "$tmp/npm-calls" ] && [ ! -e "$tmp/home/agent-browser" ]; then pass "a bad agent-browser version warns and installs nothing"; else fail "a bad agent-browser version is not refused cleanly"; fi
else fail "a bad agent-browser version stops the installer"; fi
rm -r "$tmp"

# Pi and Node: the pinned version is read from pi-version, Pi is installed under ~/.aria/pi/<version>
pinned=$(tr -d '[:space:]' < "$here/../pi-version")
tmp=$(mktemp -d)
out=$(ARIA_HOME="$tmp/home" sh "$installer" --dry-run --local /tmp/fake-aria 2>&1)
printf '%s\n' "$out" | grep -qF "would: npm install --prefix $tmp/home/pi/$pinned --no-audit --no-fund @earendil-works/pi-coding-agent@$pinned" && pass "install dry run installs the pinned Pi ($pinned)" || fail "install dry run lacks the pinned Pi install"
mkdir -p "$tmp/home/pi/$pinned/node_modules/.bin" && : > "$tmp/home/pi/$pinned/node_modules/.bin/pi" && chmod +x "$tmp/home/pi/$pinned/node_modules/.bin/pi"
out=$(ARIA_HOME="$tmp/home" sh "$installer" --dry-run --local /tmp/fake-aria 2>&1)
printf '%s\n' "$out" | grep -qF "npm install --prefix $tmp/home/pi/" && fail "an installed Pi is installed again" || pass "an installed Pi is not installed again"
fake_node() { mkdir -p "$tmp/$1" && printf '#!/bin/sh\necho %s\n' "$2" > "$tmp/$1/node" && chmod +x "$tmp/$1/node" && printf '#!/bin/sh\n' > "$tmp/$1/npm" && chmod +x "$tmp/$1/npm"; }
fake_node old v20.0.0
fake_node new v22.23.2
if PATH="$tmp/old:/usr/bin:/bin" sh "$installer" --check-node >/dev/null 2>"$tmp/err"; then fail "Node 20 was accepted"; else grep -qF 'Node 22.19 or newer' "$tmp/err" && pass "an old Node is refused with a clear message" || fail "the Node message is missing"; fi
if PATH="$tmp/new:/usr/bin:/bin" sh "$installer" --check-node >/dev/null 2>&1; then pass "Node 22.23 is accepted"; else fail "Node 22.23 was refused"; fi
if PATH="/usr/bin:/bin" sh -c 'command -v node >/dev/null' 2>/dev/null; then :; else
  if PATH="/nonexistent" /bin/sh "$installer" --check-node >/dev/null 2>&1; then fail "a missing Node was accepted"; else pass "a missing Node is refused"; fi
fi
# npm is needed too, and is checked before anything is installed
mkdir -p "$tmp/onlynode" && printf '#!/bin/sh\necho v22.23.2\n' > "$tmp/onlynode/node" && chmod +x "$tmp/onlynode/node" && ln -sf "$(command -v sed)" "$tmp/onlynode/sed"
if PATH="$tmp/onlynode" /bin/sh "$installer" --check-node >/dev/null 2>"$tmp/err"; then fail "a Node without npm was accepted"; else grep -qi 'npm' "$tmp/err" && pass "a missing npm is refused with a clear message" || fail "the npm message is missing"; fi
# the pinned version: next to the script when it is a real file, the published one when piped (curl | sh)
printf '9.9.9\n' > "$tmp/published-pin"
piped=$(cd "$here" && ARIA_PI_VERSION_URL="file://$tmp/published-pin" sh -s -- --print-pi-version < "$installer" 2>&1)
[ "$piped" = "9.9.9" ] && pass "a piped installer reads the published pin, not a pi-version in the current directory's parent" || fail "piped installer used '$piped'"
direct=$(sh "$installer" --print-pi-version 2>&1)
[ "$direct" = "$pinned" ] && pass "a real script file reads the repo's pi-version" || fail "direct run used '$direct'"
# the node link lets the service find Node without mise on its PATH
mkdir -p "$tmp/linkhome"
PATH="$tmp/new:/usr/bin:/bin" ARIA_HOME="$tmp/linkhome" sh "$installer" --link-node >/dev/null 2>&1
[ "$(readlink "$tmp/linkhome/bin/node")" = "$tmp/new/node" ] && pass "the installer links Node into ~/.aria/bin" || fail "the Node link is missing or wrong"
# Claude Code (the Claude provider needs it) is linked too when it is there, and its absence is fine
[ ! -e "$tmp/linkhome/bin/claude" ] && pass "no Claude Code, no link" || fail "a Claude link appeared from nowhere"
printf '#!/bin/sh\n' > "$tmp/new/claude" && chmod +x "$tmp/new/claude"
PATH="$tmp/new:/usr/bin:/bin" ARIA_HOME="$tmp/linkhome" sh "$installer" --link-node >/dev/null 2>&1
[ "$(readlink "$tmp/linkhome/bin/claude")" = "$tmp/new/claude" ] && pass "the installer links Claude Code into ~/.aria/bin" || fail "the Claude Code link is missing or wrong"
rm -r "$tmp"

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

# uninstall only deletes a directory that is really Aria's data
tmp=$(mktemp -d)
mkdir "$tmp/aria-data" "$tmp/empty" "$tmp/other" "$tmp/real"
echo '{}' > "$tmp/aria-data/settings.json"
echo x > "$tmp/other/notes.txt"
echo '{}' > "$tmp/real/auth.json"
ln -s "$tmp/real" "$tmp/link"
can_delete() { sh "$installer" --can-delete "$1" >/dev/null 2>&1; }
expect_yes() { if can_delete "$1"; then pass "may delete $2"; else fail "refused to delete $2"; fi; }
expect_no() { if can_delete "$1"; then fail "would delete $2"; else pass "refuses to delete $2"; fi; }
expect_yes "$tmp/aria-data" "a directory with settings.json"
expect_yes "$tmp/real" "a directory with auth.json"
expect_yes "$tmp/empty" "an empty directory"
expect_no "$tmp/other" "a directory with someone else's files"
expect_no "$tmp/link" "a symlink, even to Aria data"
expect_no "$tmp/missing" "a path that does not exist"
expect_no "/" "the filesystem root"
expect_no "$HOME" "the home directory"
expect_no "$HOME/" "the home directory with a trailing slash"
expect_no "$HOME/." "the home directory spelled with a dot"
rm -r "$tmp/link" "$tmp/aria-data" "$tmp/empty" "$tmp/other" "$tmp/real" "$tmp"

# unknown flags are refused
if sh "$installer" --frobnicate >/dev/null 2>&1; then fail "an unknown flag was accepted"; else pass "unknown flags are refused"; fi

# syntax, and shellcheck when it is installed
sh -n "$installer" && pass "install.sh parses" || fail "install.sh has a syntax error"
if command -v shellcheck >/dev/null 2>&1; then shellcheck "$installer" && pass "shellcheck is clean" || fail "shellcheck found problems"; else echo "skip: shellcheck is not installed"; fi

[ "$failures" -eq 0 ] && echo "all checks passed" || { echo "$failures check(s) failed"; exit 1; }
