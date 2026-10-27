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
for step in 'sudo pacman -S --needed gtk3 webkit2gtk-4.1 gtk-layer-shell bubblewrap iproute2 libsecret alsa-lib vulkan-icd-loader' 'install -d -m 700' 'install -Dm755 /tmp/fake-aria' 'systemctl --user daemon-reload' 'systemctl --user enable --now aria.service'; do
  at=$(line_of "$out" "would: $step") || at=""
  if [ -z "$at" ]; then fail "install dry run lacks 'would: $step'"; ordered=0
  elif [ "$at" -le "$last" ]; then fail "install dry run has '$step' out of order"; ordered=0
  else last=$at; fi
done
[ "$ordered" = 1 ] && pass "install dry run lists its steps in order"

# Aria's face goes in the Omarchy bar right after the new binary is in place (the binary carries the widget)
out=$(sh "$installer" --dry-run --local /tmp/fake-aria 2>&1)
at_binary=$(line_of "$out" "would: install -Dm755 /tmp/fake-aria")
at_bar=$(line_of "$out" "would: $HOME/.local/bin/aria bar install")
[ -n "$at_binary" ] && [ -n "$at_bar" ] && [ "$at_bar" -gt "$at_binary" ] && pass "install puts Aria's face in the Omarchy bar after the binary" || fail "install does not run 'aria bar install' after the binary"

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

# --headless: a server with no desktop. No GUI packages, no bar, no Hyprland; the user unit aria-serve.service; a private
# Node when the system's is too old; bubblewrap and iproute2 through the package manager it has.
node_pin=$(head -1 "$here/../node-version" | tr -d '[:space:]')
tmp=$(mktemp -d)
fake_node old v18.19.1
fake_node new v22.23.2
mkdir -p "$tmp/user"
headless() { dirs=$1; shift; PATH="$dirs:/usr/bin:/bin" HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --headless "$@" || true; }
out=$(PATH="$tmp/new:/usr/bin:/bin" HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --headless --dry-run --local /tmp/fake-aria 2>&1) || fail "the headless dry run failed: $out"
if printf '%s\n' "$out" | grep -qF 'aria-serve.service' && printf '%s\n' "$out" | grep -qF 'bubblewrap' && printf '%s\n' "$out" | grep -qF 'iproute2' \
  && ! printf '%s\n' "$out" | grep -qiE 'gtk3|webkit2gtk|omarchy|hyprland|hyprctl|libsecret|alsa|vulkan|aria.service '; then
  pass "headless dry run installs no GUI packages and the serve unit"
else fail "headless dry run is wrong: $out"; fi
printf '%s\n' "$out" | grep -qF 'would: systemctl --user enable --now aria-serve.service' && pass "headless enables aria-serve.service" || fail "headless does not enable aria-serve.service"
printf '%s\n' "$out" | grep -qF 'would: loginctl enable-linger' && pass "headless enables linger" || fail "headless does not enable linger"
printf '%s\n' "$out" | grep -qF 'bar install' && fail "headless puts a face in the Omarchy bar" || pass "headless has no bar widget"
# the package manager it has: apt-get first, else pacman, else only the names
mkdir -p "$tmp/apt" && printf '#!/bin/sh\nexit 0\n' > "$tmp/apt/apt-get" && chmod +x "$tmp/apt/apt-get"
out=$(headless "$tmp/apt:$tmp/new" --dry-run --local /tmp/fake-aria 2>&1)
printf '%s\n' "$out" | grep -qF 'would: sudo apt-get install -y bubblewrap iproute2' && pass "headless installs its packages with apt-get when it exists" || fail "headless does not use apt-get: $out"
mkdir -p "$tmp/bare"
for tool in sh sed cat tr grep head cut awk uname id dirname mktemp rm mkdir install ln date tar curl sha256sum; do
  src=$(command -v "$tool" 2>/dev/null) && ln -sf "$src" "$tmp/bare/$tool" || true
done
ln -sf "$tmp/new/node" "$tmp/bare/node"; ln -sf "$tmp/new/npm" "$tmp/bare/npm"
out=$(PATH="$tmp/bare" HOME="$tmp/user" ARIA_HOME="$tmp/home" /bin/sh "$installer" --headless --dry-run --local /tmp/fake-aria 2>&1) || true
printf '%s\n' "$out" | grep -qF 'bubblewrap iproute2' && ! printf '%s\n' "$out" | grep -qF 'would: sudo' && pass "headless without apt-get or pacman prints the package names and goes on" || fail "headless without a package manager does not go on: $out"
# a refused sudo prints the command for the owner and the install goes on
mkdir -p "$tmp/nosudo" && printf '#!/bin/sh\nexit 1\n' > "$tmp/nosudo/sudo" && chmod +x "$tmp/nosudo/sudo"
if out=$(PATH="$tmp/nosudo:$tmp/apt:$tmp/bare" HOME="$tmp/user" ARIA_HOME="$tmp/home" /bin/sh "$installer" --headless --install-packages 2>&1); then
  printf '%s\n' "$out" | grep -qF 'sudo apt-get install -y bubblewrap iproute2' && pass "a refused sudo prints the command and does not abort" || fail "a refused sudo does not print the command: $out"
else fail "a refused sudo aborted the installer: $out"; fi
# a refused loginctl prints the command and does not abort
mkdir -p "$tmp/nologin" && printf '#!/bin/sh\nexit 1\n' > "$tmp/nologin/loginctl" && chmod +x "$tmp/nologin/loginctl"
if out=$(PATH="$tmp/nologin:/usr/bin:/bin" HOME="$tmp/user" USER=ana ARIA_HOME="$tmp/home" sh "$installer" --headless --enable-linger 2>&1); then
  printf '%s\n' "$out" | grep -qF 'loginctl enable-linger ana' && pass "a refused linger prints the command and does not abort" || fail "a refused linger does not print the command: $out"
else fail "a refused linger aborted the installer: $out"; fi
# the unit
unit=$(sh "$installer" --headless --print-unit 2>&1 || true)
printf '%s\n' "$unit" | grep -qx 'ExecStart=%h/.local/bin/aria serve' && pass "the serve unit runs aria serve" || fail "the serve unit does not run aria serve"
printf '%s\n' "$unit" | grep -qx 'Restart=on-failure' && printf '%s\n' "$unit" | grep -qx 'WantedBy=default.target' && ! printf '%s\n' "$unit" | grep -q 'graphical-session' && pass "the serve unit restarts on failure, starts with the user manager and needs no session" || fail "the serve unit has the wrong lifecycle"
printf '%s\n' "$unit" | grep -q '^Environment=PATH=%h/.aria/node/bin:' && pass "the serve unit's PATH starts with the private Node" || fail "the serve unit's PATH does not start with %h/.aria/node/bin"
if sh "$installer" --headless --print-unit | diff -u "$here/aria-serve.service" - >/dev/null; then pass "embedded serve unit matches install/aria-serve.service"; else fail "embedded serve unit differs from install/aria-serve.service"; fi
# the unit waits a few seconds before it restarts after a failure
printf '%s\n' "$unit" | grep -qx 'RestartSec=5' && pass "the serve unit waits 5 seconds before a restart" || fail "the serve unit has no RestartSec=5"
# a non-login ssh has no XDG_RUNTIME_DIR and no user bus address: a headless install sets them for systemctl --user
out=$(PATH="$tmp/new:/usr/bin:/bin" HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --headless --dry-run --local /tmp/fake-aria 2>&1)
printf '%s\n' "$out" | grep -qF 'XDG_RUNTIME_DIR=/run/user/' && printf '%s\n' "$out" | grep -qF 'DBUS_SESSION_BUS_ADDRESS=unix:path=' && pass "the headless dry run says it sets XDG_RUNTIME_DIR and the user bus address" || fail "the headless dry run does not mention XDG_RUNTIME_DIR and the bus address: $out"
out=$(PATH="$tmp/new:/usr/bin:/bin" HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --dry-run --local /tmp/fake-aria 2>&1)
printf '%s\n' "$out" | grep -qF 'XDG_RUNTIME_DIR' && fail "a desktop install sets XDG_RUNTIME_DIR: $out" || pass "a desktop install does not touch XDG_RUNTIME_DIR"
if [ -d "/run/user/$(id -u)" ]; then
  mkdir -p "$tmp/envctl" && printf '#!/bin/sh\necho "XDG=${XDG_RUNTIME_DIR:-unset} BUS=${DBUS_SESSION_BUS_ADDRESS:-unset}" >> "%s/seen"\nexit 0\n' "$tmp" > "$tmp/envctl/systemctl" && chmod +x "$tmp/envctl/systemctl"
  env -u XDG_RUNTIME_DIR -u DBUS_SESSION_BUS_ADDRESS PATH="$tmp/envctl:/usr/bin:/bin" HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --headless --restart-dock 0 0 >/dev/null 2>&1 || true
  grep -qF "XDG=/run/user/$(id -u) BUS=unix:path=/run/user/$(id -u)/bus" "$tmp/seen" && pass "systemctl --user runs with XDG_RUNTIME_DIR and the bus address a non-login ssh lacks" || fail "systemctl --user did not get them: $(cat "$tmp/seen" 2>/dev/null)"
  rm -f "$tmp/seen"
  XDG_RUNTIME_DIR=/somewhere/else DBUS_SESSION_BUS_ADDRESS=unix:path=/mine PATH="$tmp/envctl:/usr/bin:/bin" HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --headless --restart-dock 0 0 >/dev/null 2>&1 || true
  grep -qF "XDG=/somewhere/else BUS=unix:path=/mine" "$tmp/seen" && pass "values already set are kept" || fail "set values were replaced: $(cat "$tmp/seen" 2>/dev/null)"
else pass "skip: no /run/user/$(id -u) here to check the environment against"; fi
# systemctl --user that fails on a server says what it needs
mkdir -p "$tmp/failctl" && printf '#!/bin/sh\necho "Failed to connect to bus: No medium found" >&2\nexit 1\n' > "$tmp/failctl/systemctl" && chmod +x "$tmp/failctl/systemctl"
if out=$(PATH="$tmp/failctl:/usr/bin:/bin" HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --headless --restart-dock 0 0 2>&1); then fail "a failing systemctl --user did not stop the headless install: $out"
else printf '%s\n' "$out" | grep -qF 'dbus-user-session' && printf '%s\n' "$out" | grep -qi 'linger' && pass "a failing systemctl --user names dbus-user-session and linger" || fail "the systemctl failure does not name dbus-user-session and linger: $out"; fi
# the browser (Chrome is a large download, and not for every arm64 server) only on request on a server
out=$(PATH="$tmp/new:/usr/bin:/bin" HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --headless --dry-run --local /tmp/fake-aria 2>&1)
printf '%s\n' "$out" | grep -qF -- '--with-browser' && ! printf '%s\n' "$out" | grep -qF 'agent-browser@' && pass "headless skips agent-browser and says --with-browser installs it" || fail "headless does not skip agent-browser: $out"
out=$(PATH="$tmp/new:/usr/bin:/bin" HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --headless --with-browser --dry-run --local /tmp/fake-aria 2>&1)
printf '%s\n' "$out" | grep -qF 'agent-browser@' && pass "headless --with-browser installs agent-browser" || fail "headless --with-browser does not install agent-browser: $out"
out=$(PATH="$tmp/new:/usr/bin:/bin" HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --dry-run --local /tmp/fake-aria 2>&1)
printf '%s\n' "$out" | grep -qF 'agent-browser@' && pass "a desktop install still installs agent-browser" || fail "a desktop install lost agent-browser: $out"
# a Mac has no systemd user units: --headless is Linux only
mkdir -p "$tmp/darwin" && printf '#!/bin/sh\ncase "$1" in -s) echo Darwin ;; -m) echo arm64 ;; *) echo Darwin ;; esac\n' > "$tmp/darwin/uname" && chmod +x "$tmp/darwin/uname"
if PATH="$tmp/darwin:$tmp/new:/usr/bin:/bin" HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --headless --dry-run --local /tmp/fake-aria >"$tmp/out" 2>"$tmp/err"; then fail "--headless was accepted on macOS"
else grep -qi 'headless' "$tmp/err" && grep -qi 'linux' "$tmp/err" && pass "--headless on macOS dies with a clear message" || fail "the macOS refusal is unclear: $(cat "$tmp/err")"; fi
# Node: an old one is left alone and a private pinned one goes under ~/.aria/node; a new enough one is kept
out=$(headless "$tmp/old" --dry-run --local /tmp/fake-aria 2>&1)
printf '%s\n' "$out" | grep -qF "Node $node_pin into $tmp/home/node" && pass "an old Node is replaced by a private one ($node_pin)" || fail "an old Node does not get a private one: $out"
out=$(headless "$tmp/new" --dry-run --local /tmp/fake-aria 2>&1)
printf '%s\n' "$out" | grep -qF "into $tmp/home/node" && fail "a new enough Node was replaced: $out" || pass "a new enough Node is kept"
rm -f "$tmp/bare/node" "$tmp/bare/npm"
out=$(PATH="$tmp/bare" HOME="$tmp/user" ARIA_HOME="$tmp/home" /bin/sh "$installer" --headless --dry-run --local /tmp/fake-aria 2>&1) || true
printf '%s\n' "$out" | grep -qF "Node $node_pin into $tmp/home/node" && pass "a missing Node gets a private one" || fail "a missing Node does not get a private one: $out"
out=$(headless "$tmp/old" --dry-run --local /tmp/fake-aria 2>&1)
printf '%s\n' "$out" | grep -qF 'pinned for linux-' && ! printf '%s\n' "$out" | grep -qF 'SHASUMS256.txt' && pass "the Node download is checked against the checksum pinned in node-version" || fail "the Node download is not checked against the pinned checksum"
# the private Node is planned but not on disk yet (a fresh install): the link still points at it
printf '%s\n' "$out" | grep -qF "would: link $tmp/home/bin/node to $tmp/home/node/bin/node" && pass "a planned private Node is what link_node links on a fresh install" || fail "a planned private Node is not linked: $out"
printf '%s\n' "$out" | grep -qF "https://nodejs.org/dist/v$node_pin/node-v$node_pin-linux-" && pass "the private Node comes from nodejs.org" || fail "the private Node's address is wrong"
case "$(uname -m)" in aarch64 | arm64) node_arch=arm64 ;; x86_64) node_arch=x64 ;; *) node_arch="" ;; esac
if [ -n "$node_arch" ]; then
  printf '%s\n' "$out" | grep -qF "node-v$node_pin-linux-$node_arch.tar.xz" && pass "the private Node is the build for this machine ($node_arch)" || fail "the private Node is not the $node_arch build"
  # a real download from a local mirror: a wrong checksum deletes it and dies, a right one is unpacked
  dist="$tmp/dist/v$node_pin"; tarball="node-v$node_pin-linux-$node_arch.tar.xz"
  mkdir -p "$dist" "$tmp/nodepkg/node-v$node_pin-linux-$node_arch/bin"
  printf '#!/bin/sh\necho v%s\n' "$node_pin" > "$tmp/nodepkg/node-v$node_pin-linux-$node_arch/bin/node"
  chmod +x "$tmp/nodepkg/node-v$node_pin-linux-$node_arch/bin/node"
  tar cJf "$dist/$tarball" -C "$tmp/nodepkg" "node-v$node_pin-linux-$node_arch" 2>/dev/null || tar cf - -C "$tmp/nodepkg" "node-v$node_pin-linux-$node_arch" | xz > "$dist/$tarball"
  # the pin is the trust root: a piped installer reads it from ARIA_NODE_VERSION_URL, and the mirror serves only the tarball
  piped_node() { (cd "$here" && HOME="$tmp/user" ARIA_HOME="$tmp/nodehome" ARIA_NODE_VERSION_URL="file://$tmp/pin" ARIA_NODE_DIST="$1" sh -s -- --install-node < "$installer"); }
  printf '%s\nlinux-%s 0000000000000000000000000000000000000000000000000000000000000000\n' "$node_pin" "$node_arch" > "$tmp/pin"
  if piped_node "file://$tmp/dist" >/dev/null 2>"$tmp/err"; then fail "a Node whose hash is not the pinned one was installed"
  else grep -qi 'checksum' "$tmp/err" && [ ! -e "$tmp/nodehome/node" ] && [ -z "$(ls -A "$tmp/nodehome")" ] && pass "a Node download that does not match the pin is refused and leaves nothing" || fail "the Node pin refusal is unclear or left files: $(cat "$tmp/err")"; fi
  printf '%s\nlinux-%s %s\n' "$node_pin" "$node_arch" "$(sha256sum "$dist/$tarball" | cut -d ' ' -f 1)" > "$tmp/pin"
  if piped_node "file://$tmp/dist" >/dev/null 2>&1 && [ "$("$tmp/nodehome/node/bin/node" --version)" = "v$node_pin" ] && [ "$(ls -A "$tmp/nodehome")" = node ]; then pass "a Node that matches the pin is unpacked under ~/.aria/node"; else fail "a Node that matches the pin was not installed cleanly"; fi
  # a pin with no line for this CPU dies
  printf '%s\nlinux-other %s\n' "$node_pin" "$(sha256sum "$dist/$tarball" | cut -d ' ' -f 1)" > "$tmp/pin"
  rm -rf "$tmp/nodehome"
  if piped_node "file://$tmp/dist" >/dev/null 2>"$tmp/err"; then fail "a pin with no line for this CPU was accepted"
  else grep -qF "linux-$node_arch" "$tmp/err" && [ ! -e "$tmp/nodehome/node" ] && pass "a pin with no checksum for this CPU dies" || fail "the missing-pin message is unclear: $(cat "$tmp/err")"; fi
  # a malformed hash is refused when the pin is read
  printf '%s\nlinux-%s nothex\n' "$node_pin" "$node_arch" > "$tmp/pin"
  if piped_node "file://$tmp/dist" >/dev/null 2>"$tmp/err"; then fail "a malformed pinned hash was accepted"
  else grep -qF 'not' "$tmp/err" && pass "a malformed pinned hash is refused" || fail "the malformed-hash message is unclear"; fi
  # http:// (or anything but https:// and a local file://) is refused: the download would be open to tampering
  if piped_node "http://nodejs.org/dist" >/dev/null 2>"$tmp/err"; then fail "an http:// Node mirror was accepted"
  else grep -qF 'https://' "$tmp/err" && pass "an http:// Node mirror is refused" || fail "the http refusal is unclear"; fi
  # the node link goes to it
  mkdir -p "$tmp/nodehome/node/bin" && printf '#!/bin/sh\n' > "$tmp/nodehome/node/bin/node" && chmod +x "$tmp/nodehome/node/bin/node"
  PATH="$tmp/new:/usr/bin:/bin" ARIA_HOME="$tmp/nodehome" sh "$installer" --link-node >/dev/null 2>&1
  [ "$(readlink "$tmp/nodehome/bin/node")" = "$tmp/nodehome/node/bin/node" ] && pass "link_node prefers the private Node" || fail "link_node ignored the private Node"
else echo "skip: no Node build for $(uname -m)"; fi
# any other CPU dies with a clear message
mkdir -p "$tmp/riscv" && printf '#!/bin/sh\ncase "$1" in -s) echo Linux ;; -m) echo riscv64 ;; *) echo Linux ;; esac\n' > "$tmp/riscv/uname" && chmod +x "$tmp/riscv/uname"
if PATH="$tmp/riscv:/usr/bin:/bin" HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --install-node >/dev/null 2>"$tmp/err"; then fail "an unsupported CPU was accepted"
else grep -qF 'riscv64' "$tmp/err" && pass "an unsupported CPU dies with a clear message" || fail "the unsupported CPU message is unclear"; fi
# the pin is next to the script, else published beside pi-version
printf '9.9.9\nlinux-x64 %s\n' 0000000000000000000000000000000000000000000000000000000000000000 > "$tmp/published-node"
piped=$(cd "$here" && ARIA_NODE_VERSION_URL="file://$tmp/published-node" sh -s -- --print-node-version < "$installer" 2>&1)
[ "$piped" = "9.9.9" ] && pass "a piped installer reads the published Node pin" || fail "piped installer used '$piped' for the Node pin"
direct=$(sh "$installer" --print-node-version 2>&1)
[ "$direct" = "$node_pin" ] && pass "a real script file reads the repo's node-version" || fail "direct run used '$direct' for the Node pin"
# uninstall stops the serve unit
out=$(HOME="$tmp/user" sh "$installer" --headless --dry-run --uninstall 2>&1)
printf '%s\n' "$out" | grep -qF 'would: systemctl --user disable --now aria-serve.service' && ! printf '%s\n' "$out" | grep -q 'hyprctl\|bar remove' && pass "headless uninstall stops the serve unit and touches no desktop" || fail "headless uninstall is wrong: $out"
rm -r "$tmp"

# uninstall dry run
out=$(sh "$installer" --dry-run --uninstall 2>&1)
printf '%s\n' "$out" | grep -qF 'would: systemctl --user disable --now aria.service' && pass "uninstall stops the unit" || fail "uninstall does not stop the unit"
printf '%s\n' "$out" | grep -qF 'hl.unbind' && pass "uninstall removes the runtime hotkey" || fail "uninstall does not unbind SUPER + A"
at_bar=$(line_of "$out" "would: $HOME/.local/bin/aria bar remove")
at_rm=$(line_of "$out" "would: rm -f")
[ -n "$at_bar" ] && [ -n "$at_rm" ] && [ "$at_bar" -lt "$at_rm" ] && pass "uninstall takes Aria's face out of the Omarchy bar before deleting the binary" || fail "uninstall does not run 'aria bar remove' before deleting the binary"

# the service manager runs the dock as `aria dock`: plain `aria` in a terminal is the client
sh "$installer" --print-unit | grep -qx 'ExecStart=%h/.local/bin/aria dock' && pass "the unit runs aria dock" || fail "the unit does not run aria dock"
grep -qF '<key>ProgramArguments</key><array><string>$BIN_DIR/aria</string><string>dock</string></array>' "$installer" && pass "the LaunchAgent runs aria dock" || fail "the LaunchAgent does not run aria dock"

# the unit embedded in the script is the one in the repo
if sh "$installer" --print-unit | diff -u "$here/aria.service" - >/dev/null; then pass "embedded unit matches install/aria.service"; else fail "embedded unit differs from install/aria.service"; fi

# a bad --local path fails before anything is changed
tmp=$(mktemp -d)
if ARIA_BIN_DIR="$tmp/bin" ARIA_HOME="$tmp/home" sh "$installer" --local /nonexistent/aria >/dev/null 2>&1; then fail "a missing --local binary was accepted"; else pass "a missing --local binary is refused"; fi
[ ! -e "$tmp/bin" ] && [ ! -e "$tmp/home" ] && pass "nothing was created by the refused install" || fail "the refused install left files behind"
rm -r "$tmp"

# the download address: GitHub Releases of i-montes/aria, a pinned release with ARIA_VERSION, the environment wins
repo_url=https://github.com/i-montes/aria/releases
tmp=$(mktemp -d)
mkdir -p "$tmp/user" "$tmp/old" "$tmp/x86"
# the download is refused on a Linux that is not x86_64, so this block is an x86_64 machine whatever runs it
printf '#!/bin/sh\ncase "$1" in -s) echo Linux ;; -m) echo x86_64 ;; *) echo Linux ;; esac\n' > "$tmp/x86/uname" && chmod +x "$tmp/x86/uname"
saved_path=$PATH
PATH="$tmp/x86:$PATH"
printf '#!/bin/sh\necho v18.19.1\n' > "$tmp/old/node" && printf '#!/bin/sh\n' > "$tmp/old/npm" && chmod +x "$tmp/old/node" "$tmp/old/npm"
# a piped installer (curl | sh) has no sibling files, so it reads every pin from beside the binary
dl() { (cd "$here" && HOME="$tmp/user" ARIA_HOME="$tmp/home" "$@" sh -s -- --dry-run --local /tmp/fake-aria < "$installer" 2>&1); }
out=$(env -u ARIA_DOWNLOAD_URL -u ARIA_VERSION -u ARIA_REPO sh "$installer" --dry-run 2>&1)
printf '%s\n' "$out" | grep -qF "would: download $repo_url/latest/download/aria-x86_64-linux and verify it against $repo_url/latest/download/aria-x86_64-linux.sha256" && pass "the default download is the latest GitHub release of i-montes/aria" || fail "the default download is not the GitHub latest release: $out"
printf '%s\n' "$out" | grep -qF 'aria.example' && fail "the placeholder address is still there" || pass "no placeholder address is left"
out=$(dl env -u ARIA_DOWNLOAD_URL -u ARIA_VERSION -u ARIA_REPO)
ok=1
for sibling in pi-version pi-packages engram-version agent-browser-version; do
  printf '%s\n' "$out" | grep -qF "would: download $repo_url/latest/download/$sibling" || { ok=0; fail "$sibling is not read from beside the latest binary: $out"; }
done
[ "$ok" = 1 ] && pass "a piped installer reads the pins from beside the latest binary"
out=$(cd "$here" && PATH="$tmp/old:/usr/bin:/bin" HOME="$tmp/user" ARIA_HOME="$tmp/home" env -u ARIA_DOWNLOAD_URL -u ARIA_VERSION -u ARIA_REPO sh -s -- --headless --dry-run --local /tmp/fake-aria < "$installer" 2>&1)
printf '%s\n' "$out" | grep -qF "would: download $repo_url/latest/download/node-version" && pass "the Node pin is read from beside the latest binary too" || fail "node-version is not read from beside the binary: $out"
out=$(dl env -u ARIA_DOWNLOAD_URL -u ARIA_REPO ARIA_VERSION=0.20.2)
ok=1
for sibling in pi-version pi-packages engram-version agent-browser-version; do
  printf '%s\n' "$out" | grep -qF "would: download $repo_url/download/v0.20.2/$sibling" || { ok=0; fail "$sibling of a pinned release is not read from that release: $out"; }
done
[ "$ok" = 1 ] && pass "a pinned release reads its pins from the same release"
out=$(env -u ARIA_DOWNLOAD_URL -u ARIA_REPO ARIA_VERSION=0.20.2 sh "$installer" --dry-run 2>&1)
printf '%s\n' "$out" | grep -qF "would: download $repo_url/download/v0.20.2/aria-x86_64-linux and verify it against $repo_url/download/v0.20.2/aria-x86_64-linux.sha256" && pass "ARIA_VERSION=0.20.2 pins v0.20.2" || fail "ARIA_VERSION=0.20.2 does not pin v0.20.2: $out"
out=$(env -u ARIA_DOWNLOAD_URL -u ARIA_REPO ARIA_VERSION=v0.20.2 sh "$installer" --dry-run 2>&1)
printf '%s\n' "$out" | grep -qF "would: download $repo_url/download/v0.20.2/aria-x86_64-linux " && pass "ARIA_VERSION=v0.20.2 is the same release (no doubled v)" || fail "ARIA_VERSION=v0.20.2 is not normalised: $out"
out=$(env -u ARIA_DOWNLOAD_URL ARIA_REPO=ana/aria-fork sh "$installer" --dry-run 2>&1)
printf '%s\n' "$out" | grep -qF "would: download https://github.com/ana/aria-fork/releases/latest/download/aria-x86_64-linux " && pass "ARIA_REPO points the download at another repository" || fail "ARIA_REPO is ignored: $out"
out=$(env -u ARIA_DOWNLOAD_URL ARIA_REPO=ana/aria-fork ARIA_VERSION=1.2.3 sh "$installer" --dry-run 2>&1)
printf '%s\n' "$out" | grep -qF "would: download https://github.com/ana/aria-fork/releases/download/v1.2.3/aria-x86_64-linux " && pass "ARIA_REPO and ARIA_VERSION combine" || fail "ARIA_REPO with ARIA_VERSION is wrong: $out"
out=$(ARIA_DOWNLOAD_URL=https://mirror.test/aria/aria-x86_64-linux ARIA_VERSION=9.9.9 ARIA_REPO=ana/aria-fork sh "$installer" --dry-run 2>&1)
printf '%s\n' "$out" | grep -qF "would: download https://mirror.test/aria/aria-x86_64-linux and verify it against https://mirror.test/aria/aria-x86_64-linux.sha256" && pass "ARIA_DOWNLOAD_URL wins over ARIA_VERSION and ARIA_REPO" || fail "ARIA_DOWNLOAD_URL does not win: $out"
out=$(dl env ARIA_DOWNLOAD_URL=https://mirror.test/aria/aria-x86_64-linux ARIA_VERSION=9.9.9)
printf '%s\n' "$out" | grep -qF 'would: download https://mirror.test/aria/pi-packages' && pass "the pins follow ARIA_DOWNLOAD_URL" || fail "the pins do not follow ARIA_DOWNLOAD_URL: $out"
out=$(dl env -u ARIA_DOWNLOAD_URL -u ARIA_VERSION -u ARIA_REPO ARIA_PI_VERSION_URL=https://mirror.test/pi)
printf '%s\n' "$out" | grep -qF 'would: download https://mirror.test/pi' && pass "ARIA_PI_VERSION_URL still wins" || fail "ARIA_PI_VERSION_URL no longer wins: $out"
for bad in 'latest' '0.20' 'v1.2.3;rm' '../x' '1.2.3 --local x'; do
  if env -u ARIA_DOWNLOAD_URL "ARIA_VERSION=$bad" sh "$installer" --dry-run >/dev/null 2>"$tmp/err"; then fail "ARIA_VERSION='$bad' was accepted"
  else grep -qF 'ARIA_VERSION' "$tmp/err" && pass "ARIA_VERSION='$bad' is refused" || fail "ARIA_VERSION='$bad' is refused without naming it"; fi
done
for bad in 'nobody' 'a/b/c' 'a b/c' '/aria' 'ana/..' 'ana/aria?x=1' '.x/y' 'a./b'; do
  if env -u ARIA_DOWNLOAD_URL "ARIA_REPO=$bad" sh "$installer" --dry-run >/dev/null 2>"$tmp/err"; then fail "ARIA_REPO='$bad' was accepted"
  else grep -qF 'ARIA_REPO' "$tmp/err" && pass "ARIA_REPO='$bad' is refused" || fail "ARIA_REPO='$bad' is refused without naming it"; fi
done
# a stray value in the environment must not stop a --local install, which downloads no binary
ARIA_VERSION=latest ARIA_REPO=nobody sh "$installer" --dry-run --local /tmp/fake-aria >/dev/null 2>&1 && pass "a bad ARIA_VERSION or ARIA_REPO does not stop a --local install" || fail "a bad ARIA_VERSION or ARIA_REPO stopped a --local install"
# an ARIA_VERSION that is not used (ARIA_DOWNLOAD_URL decides the address) is not checked
ARIA_DOWNLOAD_URL=https://mirror.test/aria-x86_64-linux ARIA_VERSION=latest sh "$installer" --dry-run >/dev/null 2>&1 && pass "an unused ARIA_VERSION is not checked" || fail "an unused ARIA_VERSION stopped the installer"
# --help documents the variables and the one-line install, and stops at the end of the header
help=$(sh "$installer" --help)
printf '%s\n' "$help" | grep -qF 'ARIA_REPO' && printf '%s\n' "$help" | grep -qF 'ARIA_VERSION' && printf '%s\n' "$help" | grep -qF 'curl -fsSL https://github.com/i-montes/aria/releases/latest/download/install.sh | sh' && pass "--help documents ARIA_REPO, ARIA_VERSION and the one-line install" || fail "--help lacks ARIA_REPO, ARIA_VERSION or the one-line install"
printf '%s\n' "$help" | grep -qF 'aria.example' && fail "--help has the placeholder" || pass "--help has no placeholder"
printf '%s\n' "$help" | grep -qF 'set -eu' && fail "--help runs past the header" || pass "--help stops at the end of the header"

# the real download (the --fetch-binary hook): a verified binary, and every failure says what to do and leaves nothing
rel="$tmp/rel"; mkdir -p "$rel" "$tmp/scratch"
printf '#!/bin/sh\necho aria\n' > "$rel/aria-x86_64-linux"
(cd "$rel" && sha256sum aria-x86_64-linux > aria-x86_64-linux.sha256)
want=$(sha256sum "$rel/aria-x86_64-linux" | cut -d ' ' -f 1)
fetch() { TMPDIR="$tmp/scratch" HOME="$tmp/user" ARIA_HOME="$tmp/home" ARIA_DOWNLOAD_URL="$1" sh "$installer" --fetch-binary 2>&1; }
if out=$(fetch "file://$rel/aria-x86_64-linux") && printf '%s\n' "$out" | grep -qF "fetched $want" && [ -z "$(ls -A "$tmp/scratch")" ]; then pass "a binary whose checksum file is sha256sum's output is verified and the scratch directory is removed"; else fail "a good download failed or left files: $out"; fi
printf '%s\n' "$want" > "$rel/aria-x86_64-linux.sha256"
out=$(fetch "file://$rel/aria-x86_64-linux") && pass "a checksum file with only the digest is accepted too" || fail "a bare digest was refused: $out"
(cd "$rel" && sha256sum aria-x86_64-linux > aria-x86_64-linux.sha256)
check_failure() { # label, output of the failed fetch, URL expected in it
  if printf '%s\n' "$2" | grep -qF "$3" && printf '%s\n' "$2" | grep -qi 'macOS' && printf '%s\n' "$2" | grep -qF 'cargo build --release' && printf '%s\n' "$2" | grep -qF 'sh install/install.sh --local target/release/aria' && [ -z "$(ls -A "$tmp/scratch")" ]; then
    pass "$1: the message names the URL, macOS and the build from a clone, and nothing is left"
  else fail "$1: unclear message or files left: $2 / $(ls -A "$tmp/scratch")"; fi
}
url="file://$tmp/none/aria-x86_64-linux"
if out=$(fetch "$url"); then fail "a missing binary was accepted"; else check_failure "a missing binary" "$out" "$url"; fi
mkdir -p "$tmp/stub"
printf '#!/bin/sh\necho "curl: (22) The requested URL returned error: 404" >&2\nexit 22\n' > "$tmp/stub/curl" && chmod +x "$tmp/stub/curl"
url=https://github.com/i-montes/aria/releases/latest/download/aria-x86_64-linux
if out=$(PATH="$tmp/stub:$PATH" fetch "$url"); then fail "an HTTP 404 was accepted"; else check_failure "an HTTP 404" "$out" "$url"; fi
# the binary is there but its checksum file is not
mkdir -p "$tmp/nosum" && cp "$rel/aria-x86_64-linux" "$tmp/nosum/"
if out=$(fetch "file://$tmp/nosum/aria-x86_64-linux"); then fail "a download without a checksum file was accepted"
else printf '%s\n' "$out" | grep -qF "file://$tmp/nosum/aria-x86_64-linux.sha256" && [ -z "$(ls -A "$tmp/scratch")" ] && pass "a missing checksum file names its URL and leaves nothing" || fail "the missing-checksum message is unclear: $out"; fi
# a checksum that does not match
mkdir -p "$tmp/badsum" && cp "$rel/aria-x86_64-linux" "$tmp/badsum/" && printf '%064d  aria-x86_64-linux\n' 0 > "$tmp/badsum/aria-x86_64-linux.sha256"
if out=$(fetch "file://$tmp/badsum/aria-x86_64-linux"); then fail "a wrong checksum was accepted"
else printf '%s\n' "$out" | grep -qF 'does not match its checksum' && [ -z "$(ls -A "$tmp/scratch")" ] && pass "a wrong checksum is refused and leaves nothing" || fail "the checksum refusal is unclear: $out"; fi
[ ! -e "$tmp/user/.local/bin/aria" ] && pass "no failed download installed a binary" || fail "a binary was installed by a failed download"
printf '%s\n' "$out" | grep -qF 'run the command again' && pass "a checksum mismatch says a release may have been published meanwhile" || fail "the mismatch has no hint to run it again: $out"
# a mismatch against a real (stub) download from the default address: both curls are https only, also across redirects
mkdir -p "$tmp/rec" && printf '#!/bin/sh\necho "curl $*" >> "%s/curl-calls"\nout=""; for a in "$@"; do if [ "$out" = next ]; then : > "$a"; printf x > "$a"; out=""; fi; [ "$a" = -o ] && out=next; done\nif [ -z "$out" ] && ! printf "%%s" "$*" | grep -q -- " -o "; then echo 0.1.2; fi\nexit 0\n' "$tmp" > "$tmp/rec/curl" && chmod +x "$tmp/rec/curl"
rm -f "$tmp/curl-calls"
PATH="$tmp/rec:$PATH" TMPDIR="$tmp/scratch" HOME="$tmp/user" env -u ARIA_DOWNLOAD_URL -u ARIA_VERSION -u ARIA_REPO sh "$installer" --fetch-binary >/dev/null 2>&1 || true
[ "$(grep -c -- "--proto =https --proto-redir =https" "$tmp/curl-calls")" = 2 ] && pass "the default address is fetched with --proto =https --proto-redir =https (binary and checksum)" || fail "the default download is not https only: $(cat "$tmp/curl-calls" 2>/dev/null)"
rm -f "$tmp/curl-calls"
PATH="$tmp/rec:$PATH" TMPDIR="$tmp/scratch" HOME="$tmp/user" ARIA_DOWNLOAD_URL=https://mirror.test/aria-x86_64-linux sh "$installer" --fetch-binary >/dev/null 2>&1 || true
grep -q -- '--proto' "$tmp/curl-calls" && fail "an overridden address got --proto: $(cat "$tmp/curl-calls")" || pass "an ARIA_DOWNLOAD_URL override is not limited to https"
rm -f "$tmp/curl-calls"
(cd "$here" && PATH="$tmp/rec:$PATH" HOME="$tmp/user" env -u ARIA_DOWNLOAD_URL -u ARIA_PI_VERSION_URL sh -s -- --print-pi-version < "$installer" >/dev/null 2>&1) || true
grep -q -- '--proto =https --proto-redir =https' "$tmp/curl-calls" && pass "the pins of the default address are https only too" || fail "the pin download is not https only: $(cat "$tmp/curl-calls" 2>/dev/null)"
rm -f "$tmp/curl-calls"
(cd "$here" && PATH="$tmp/rec:$PATH" HOME="$tmp/user" ARIA_PI_VERSION_URL="file://$tmp/rel/none" sh -s -- --print-pi-version < "$installer" >/dev/null 2>&1) || true
grep -q -- '--proto' "$tmp/curl-calls" && fail "an overridden pin address got --proto" || pass "an ARIA_PI_VERSION_URL override is not limited to https"
# the download is for x86_64 Linux: any other CPU is refused before anything is fetched, with the way out
for cpu in aarch64 armv7l riscv64; do
  mkdir -p "$tmp/cpu-$cpu" && printf '#!/bin/sh\ncase "$1" in -s) echo Linux ;; -m) echo %s ;; *) echo Linux ;; esac\n' "$cpu" > "$tmp/cpu-$cpu/uname" && chmod +x "$tmp/cpu-$cpu/uname"
  rm -f "$tmp/curl-calls"
  # the way out: an aarch64 server installs as a hub (it has a build); the others build their own
  if [ "$cpu" = aarch64 ]; then way_out='sh install.sh --mode hub'; else way_out='sh tools/build-headless.sh'; fi
  if out=$(PATH="$tmp/cpu-$cpu:$tmp/rec:$PATH" TMPDIR="$tmp/scratch" HOME="$tmp/user" ARIA_DOWNLOAD_URL="file://$rel/aria-x86_64-linux" sh "$installer" --fetch-binary 2>&1); then fail "a $cpu Linux was given the x86_64 binary"
  elif printf '%s\n' "$out" | grep -qF "$cpu" && printf '%s\n' "$out" | grep -qF 'cargo build --release' && printf '%s\n' "$out" | grep -qF 'sh install/install.sh --local target/release/aria' && printf '%s\n' "$out" | grep -qF "$way_out" && [ ! -e "$tmp/curl-calls" ] && [ -z "$(ls -A "$tmp/scratch")" ]; then pass "a $cpu Linux is refused before any download, saying how to build and what to install as a server"
  else fail "the $cpu refusal is unclear or downloaded first: $out"; fi
done
if PATH="$tmp/cpu-aarch64:$PATH" sh "$installer" --dry-run >/dev/null 2>&1; then fail "the dry run of an aarch64 Linux promises a download"; else pass "the dry run says no too on an aarch64 Linux"; fi
PATH="$tmp/cpu-aarch64:$PATH" sh "$installer" --dry-run --local /tmp/fake-aria >/dev/null 2>&1 && pass "--local still works on an aarch64 Linux" || fail "--local was refused on an aarch64 Linux"
mkdir -p "$tmp/cpu-amd64" && printf '#!/bin/sh\ncase "$1" in -s) echo Linux ;; -m) echo amd64 ;; *) echo Linux ;; esac\n' > "$tmp/cpu-amd64/uname" && chmod +x "$tmp/cpu-amd64/uname"
PATH="$tmp/cpu-amd64:$PATH" TMPDIR="$tmp/scratch" ARIA_DOWNLOAD_URL="file://$rel/aria-x86_64-linux" sh "$installer" --fetch-binary >/dev/null 2>&1 && pass "amd64 is x86_64" || fail "amd64 was refused"
# one EXIT trap for every scratch directory: a later trap must not drop an earlier cleanup
[ "$(grep -c '^ *trap ' "$installer")" = 1 ] && pass "the installer has one EXIT trap (it removes the binary's and Node's scratch directories)" || fail "the installer sets more than one EXIT trap"
PATH=$saved_path
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

# --- the use: this computer's dock, or a hub ---
tmp=$(mktemp -d)
mkdir -p "$tmp/user"
mode() { env -u DISPLAY -u WAYLAND_DISPLAY -u HYPRLAND_INSTANCE_SIGNATURE -u ARIA_MODE -u ARIA_SERVER -u XDG_CONFIG_HOME ${DESK:+DISPLAY=:0} ${AMODE:+ARIA_MODE=$AMODE} ${ASERVER:+ARIA_SERVER=$ASERVER} HOME="$tmp/user" ARIA_TTY_IN="${TTY_IN:-/nonexistent}" ARIA_TTY_OUT="$tmp/tty-out" sh "$installer" --resolve-mode "$@" 2>&1; }
# nothing to ask on: the default follows the desktop, and a hub on this server is the default for a server
[ "$(mode | tail -1)" = "use=hub server=here headless=1" ] && pass "with no desktop and nothing to ask, the use is a hub on this server" || fail "no desktop does not default to a hub: $(mode)"
[ "$(DESK=1 mode | tail -1)" = "use=desktop server=none headless=0" ] && pass "with a desktop and nothing to ask, the use is the dock" || fail "a desktop does not default to the dock: $(DESK=1 mode)"
[ "$(mode --headless | tail -1)" = "use=hub server=here headless=1" ] && pass "--headless is a hub on this server" || fail "--headless changed: $(mode --headless)"
[ "$(mode --mode hub | tail -1)" = "use=hub server=here headless=1" ] && pass "--mode hub with nothing to ask is a hub on this server" || fail "--mode hub changed: $(mode --mode hub)"
[ "$(DESK=1 mode --mode desktop | tail -1)" = "use=desktop server=none headless=0" ] && pass "--mode desktop is the dock" || fail "--mode desktop changed"
[ "$(DESK=1 mode --server ana@box | tail -1)" = "use=hub server=ana@box headless=0" ] && pass "--server user@host is a hub on another computer, and this one gets the dock" || fail "--server user@host changed: $(DESK=1 mode --server ana@box)"
[ "$(DESK=1 ASERVER=here mode | tail -1)" = "use=hub server=here headless=1" ] && pass "ARIA_SERVER=here is a hub on this server" || fail "ARIA_SERVER changed"
[ "$(AMODE=hub DESK=1 mode | tail -1)" = "use=hub server=here headless=1" ] && pass "ARIA_MODE=hub is read" || fail "ARIA_MODE is not read"
for bad in '--mode bogus' '--headless --mode desktop' '--mode desktop --server ana@box' '--server nobody' '--server -oProxyCommand=x@y' '--server a@b@c' '--server ana@box;ls' '--server @box'; do
  # shellcheck disable=SC2086
  if mode $bad >/dev/null; then fail "'$bad' was accepted"; else pass "'$bad' is refused"; fi
done
# an Aria that is already installed keeps its use, and nobody is asked
mkdir -p "$tmp/user/.config/systemd/user"
: > "$tmp/user/.config/systemd/user/aria-serve.service"
[ "$(DESK=1 mode | tail -1)" = "use=hub server=here headless=1" ] && pass "an installed hub stays a hub on an update" || fail "an installed hub was changed: $(DESK=1 mode)"
rm "$tmp/user/.config/systemd/user/aria-serve.service"
: > "$tmp/user/.config/systemd/user/aria.service"
[ "$(mode | tail -1)" = "use=desktop server=none headless=0" ] && pass "an installed dock stays the dock on an update" || fail "an installed dock was changed: $(mode)"
rm "$tmp/user/.config/systemd/user/aria.service"
# the questions, answered from a file standing in for the terminal
answers() { printf '%b' "$1" > "$tmp/answers"; TTY_IN="$tmp/answers" mode; }
out=$(answers '\n'); [ "$(printf '%s\n' "$out" | tail -1)" = "use=hub server=here headless=1" ] && pass "Enter takes the default use (a hub when there is no desktop), then the default server (this one)" || fail "Enter did not take the defaults: $out"
out=$(DESK=1 answers '\n\n'); [ "$(printf '%s\n' "$out" | tail -1)" = "use=desktop server=none headless=0" ] && pass "Enter takes the default use (the dock when there is a desktop)" || fail "Enter on a desktop: $out"
out=$(DESK=1 answers '2\n2\n'); [ "$(printf '%s\n' "$out" | tail -1)" = "use=hub server=here headless=1" ] && pass "2 then 2 is a hub on this computer" || fail "2 then 2: $out"
out=$(DESK=1 answers '2\n1\nana@box\n'); [ "$(printf '%s\n' "$out" | tail -1)" = "use=hub server=ana@box headless=0" ] && pass "2, 1 and an address is a hub on another computer" || fail "2, 1, address: $out"
out=$(DESK=1 answers '2\n1\nnonsense\nana@box\n'); [ "$(printf '%s\n' "$out" | tail -1)" = "use=hub server=ana@box headless=0" ] && pass "a bad address is asked for again" || fail "a bad address: $out"
out=$(DESK=1 answers 'x\nx\nx\n') && fail "three wrong answers were accepted: $out" || pass "three wrong answers stop the installer"
out=$(DESK=1 answers 'hub\nhere\n'); [ "$(printf '%s\n' "$out" | tail -1)" = "use=hub server=here headless=1" ] && pass "the words hub and here are accepted as answers" || fail "words as answers: $out"
grep -q 'Which computer is the server' "$tmp/tty-out" && grep -q 'How will you use Aria' "$tmp/tty-out" && pass "the questions say what each answer means" || fail "the questions are not on the terminal"
grep -q 'a server that keeps Aria running' "$tmp/tty-out" && pass "the hub is explained in plain words" || fail "the hub is not explained"
# --yes asks nothing, even where there is a terminal
out=$(printf '2\n2\n' > "$tmp/answers"; TTY_IN="$tmp/answers" DESK=1 mode --yes); [ "$(printf '%s\n' "$out" | tail -1)" = "use=desktop server=none headless=0" ] && pass "--yes asks nothing and takes the default" || fail "--yes asked: $out"
rm -r "$tmp"

# a hub on another computer: the dock goes in here, then this Aria sets the server up
tmp=$(mktemp -d)
out=$(HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --dry-run --local /tmp/fake-aria --server ana@box 2>&1)
printf '%s\n' "$out" | grep -qF 'would: systemctl --user enable --now aria.service' && pass "a hub elsewhere still puts the dock on this computer" || fail "no dock for a hub elsewhere: $out"
printf '%s\n' "$out" | grep -qF "would: $tmp/user/.local/bin/aria hub init ana@box --yes" && pass "a hub elsewhere is set up with aria hub init after the dock" || fail "no hub init for a hub elsewhere: $out"
printf '%s\n' "$out" | grep -qF '[7/7]' && pass "a hub elsewhere has a seventh step" || fail "no seventh step: $out"
printf '%s\n' "$out" | grep -qF '[6/6]' && fail "a hub elsewhere counts six steps" || pass "the steps are counted to seven"
rm -r "$tmp"

# the screen: no colour in a pipe or with NO_COLOR, and the marks are plain where the locale is not UTF-8
tmp=$(mktemp -d)
out=$(LC_ALL=C HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --dry-run --local /tmp/fake-aria --mode desktop 2>&1)
printf '%s\n' "$out" | grep -q "$(printf '\033')" && fail "a pipe got colour codes" || pass "a pipe gets no colour codes"
printf '%s\n' "$out" | grep -qE '^\[1/6\] Checking this computer' && pass "the steps are numbered" || fail "the steps are not numbered: $out"
printf '%s\n' "$out" | grep -qE '^  ok ' && pass "the marks are plain where the locale is not UTF-8" || fail "the marks are not plain in the C locale"
out=$(LC_ALL=en_US.UTF-8 HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --dry-run --local /tmp/fake-aria --mode desktop 2>&1)
printf '%s\n' "$out" | grep -q '^  ✓ ' && pass "the marks are check marks in a UTF-8 locale" || fail "no check marks in a UTF-8 locale"
rm -r "$tmp"

# every line of the screen fits a narrow terminal (a remote login can be 80 columns wide), whatever the answers
tmp=$(mktemp -d)
mkdir -p "$tmp/user"
printf '2\n1\nana@my-server\n' > "$tmp/answers"
: > "$tmp/tty-out"
env -u XDG_CONFIG_HOME -u HYPRLAND_INSTANCE_SIGNATURE -u WAYLAND_DISPLAY DISPLAY=:0 HOME="$tmp/user" ARIA_TTY_IN="$tmp/answers" ARIA_TTY_OUT="$tmp/tty-out" sh "$installer" --resolve-mode >/dev/null 2>&1
wide=$(awk 'length($0) > 78' "$tmp/tty-out")
[ -z "$wide" ] && pass "the questions fit 78 columns" || fail "a question line is wider than 78 columns: $wide"
for args in '--mode desktop' '--mode hub --server here' '--server ana@my-server'; do
  # shellcheck disable=SC2086
  out=$(env -u XDG_CONFIG_HOME HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --dry-run --local /tmp/fake-aria $args 2>&1)
  wide=$(printf '%s\n' "$out" | grep -v '^would:' | awk 'length($0) > 78')
  [ -z "$wide" ] && pass "the screen of '$args' fits 78 columns" || fail "a line of '$args' is wider than 78 columns: $wide"
done
rm -r "$tmp"

# run by aria hub init on the server (ARIA_NESTED=1) the installer is one step of a longer screen: no banner, plan, numbers or question
tmp=$(mktemp -d)
mkdir -p "$tmp/user"
out=$(LC_ALL=C ARIA_NESTED=1 HOME="$tmp/user" ARIA_HOME="$tmp/home" sh "$installer" --dry-run --local /tmp/fake-aria --headless 2>&1)
printf '%s\n' "$out" | grep -qE '^\[[0-9]/[0-9]\]' && fail "a nested installer numbers its steps" || pass "a nested installer does not number its steps"
printf '%s\n' "$out" | grep -qF 'Press Enter' && fail "a nested installer waits for an Enter" || pass "a nested installer asks nothing to start"
printf '%s\n' "$out" | grep -q 'steps\. Nothing changes\|setting up' && fail "a nested installer prints its banner" || pass "a nested installer prints no banner"
printf '%s\n' "$out" | grep -qE '^  - Checking this server' && printf '%s\n' "$out" | grep -qE '^  - Starting Aria' && pass "a nested installer lists its steps, one line each" || fail "the nested steps are missing: $out"
printf '%s\n' "$out" | grep -qF 'would: systemctl --user enable --now aria-serve.service' && pass "a nested installer does everything the same" || fail "a nested installer skipped work"
rm -r "$tmp"

# the address a computer links to a hub by: Tailscale's, else the one this login came to, else the name
tmp=$(mktemp -d)
mkdir -p "$tmp/ts" "$tmp/none"
printf '#!/bin/sh\necho 100.64.0.9\n' > "$tmp/ts/tailscale"; chmod +x "$tmp/ts/tailscale"
for tool in head hostname uname dirname; do ln -s "$(command -v $tool)" "$tmp/none/$tool"; done
[ "$(PATH="$tmp/ts:$tmp/none" SSH_CONNECTION='1.2.3.4 5 9.9.9.9 22' /bin/sh "$installer" --print-address)" = 100.64.0.9 ] && pass "the hub is reached by its Tailscale address when it has one" || fail "the Tailscale address was not chosen"
[ "$(PATH="$tmp/none" SSH_CONNECTION='1.2.3.4 5 9.9.9.9 22' /bin/sh "$installer" --print-address)" = 9.9.9.9 ] && pass "without Tailscale, the address this login came to" || fail "the ssh address was not chosen"
[ -n "$(PATH="$tmp/none" /bin/sh -c 'unset SSH_CONNECTION; exec /bin/sh "$0" --print-address' "$installer")" ] && pass "without either, its name" || fail "no address at all"
rm -r "$tmp"

# the headless build has its own artifact, by CPU
tmp=$(mktemp -d)
for cpu in aarch64 x86_64; do
  mkdir -p "$tmp/cpu-$cpu" && printf '#!/bin/sh\ncase "$1" in -s) echo Linux ;; -m) echo %s ;; *) echo Linux ;; esac\n' "$cpu" > "$tmp/cpu-$cpu/uname" && chmod +x "$tmp/cpu-$cpu/uname"
  out=$(PATH="$tmp/cpu-$cpu:$PATH" HOME="$tmp/user" ARIA_HOME="$tmp/home" env -u ARIA_DOWNLOAD_URL -u ARIA_VERSION -u ARIA_REPO sh "$installer" --headless --dry-run --fetch-binary 2>&1)
  printf '%s\n' "$out" | grep -qF "releases/latest/download/aria-$cpu-linux-headless " && pass "a hub on $cpu downloads aria-$cpu-linux-headless" || fail "a hub on $cpu asks for the wrong file: $out"
done
out=$(env -u ARIA_DOWNLOAD_URL -u ARIA_VERSION -u ARIA_REPO sh "$installer" --dry-run --fetch-binary 2>&1)
printf '%s\n' "$out" | grep -qF "releases/latest/download/aria-x86_64-linux " && pass "the dock still downloads aria-x86_64-linux" || fail "the dock's artifact changed: $out"
rm -r "$tmp"

# the sandbox on a server: a profile for bwrap alone, written only with consent
tmp=$(mktemp -d)
mkdir -p "$tmp/bin" "$tmp/etc"
printf '1\n' > "$tmp/restrict"; : > "$tmp/abi"
printf '#!/bin/sh\n[ -f "%s" ] && exit 0\necho "bwrap: setting up uid map: Permission denied" >&2\nexit 1\n' "$tmp/etc/aria-bwrap" > "$tmp/bin/bwrap"
printf '#!/bin/sh\nexec "$@"\n' > "$tmp/bin/sudo"
printf '#!/bin/sh\necho "$@" >> "%s/parser-calls"\n' "$tmp" > "$tmp/bin/apparmor_parser"
chmod +x "$tmp/bin/"*
sandbox() { PATH="$tmp/bin:/usr/bin:/bin" HOME="$tmp/user" ARIA_BWRAP_PROFILE="$tmp/etc/aria-bwrap" ARIA_USERNS_SYSCTL="${RESTRICT:-$tmp/restrict}" ARIA_APPARMOR_ABI="$tmp/abi" ARIA_TTY_IN="${TTY_IN:-/nonexistent}" ARIA_TTY_OUT="$tmp/tty-out" sh "$installer" --ensure-sandbox "$@" 2>&1; }
out=$(sandbox); [ ! -e "$tmp/etc/aria-bwrap" ] && printf '%s\n' "$out" | grep -qF 'every command asks' && pass "with no terminal and no --yes the sandbox profile is not written, and the installer says what happens" || fail "the profile was written without consent: $out"
printf 'n\n' > "$tmp/answers"; out=$(TTY_IN="$tmp/answers" sandbox); [ ! -e "$tmp/etc/aria-bwrap" ] && pass "answering no leaves the sandbox rules alone" || fail "the profile was written on a no: $out"
printf '\n' > "$tmp/answers"; out=$(TTY_IN="$tmp/answers" sandbox); [ -f "$tmp/etc/aria-bwrap" ] && grep -q 'profile aria-bwrap .* flags=(unconfined)' "$tmp/etc/aria-bwrap" && grep -q 'userns,' "$tmp/etc/aria-bwrap" && grep -qF -- '-r' "$tmp/parser-calls" && pass "Enter (the default yes) writes the bwrap profile and loads it" || fail "the profile was not written on Enter: $out"
printf '%s\n' "$out" | grep -qF 'The sandbox works now' && pass "it says the sandbox works once the profile is in" || fail "no success line: $out"
rm -f "$tmp/etc/aria-bwrap" "$tmp/parser-calls"
out=$(sandbox --yes); [ -f "$tmp/etc/aria-bwrap" ] && pass "--yes writes the profile" || fail "--yes did not write the profile: $out"
rm -f "$tmp/etc/aria-bwrap"
printf '0\n' > "$tmp/restrict0"
out=$(RESTRICT="$tmp/restrict0" sandbox --yes); [ ! -e "$tmp/etc/aria-bwrap" ] && printf '%s\n' "$out" | grep -qF "not Ubuntu's user-namespace restriction" && pass "a sandbox that fails for another reason is not given a profile" || fail "a profile was written for another cause: $out"
if grep -rqF 'sysctl' "$installer" && grep -n 'sysctl -w\|sysctl --' "$installer" | grep -v '^.*#' | grep -q .; then fail "the installer changes a sysctl"; else pass "the installer never changes the sysctl"; fi
rm -r "$tmp"
sh "$installer" --print-bwrap-profile | grep -qF 'profile aria-bwrap /usr/bin/bwrap flags=(unconfined)' && pass "the profile is for bwrap alone" || fail "the profile is not for bwrap alone"

# a failing noisy step shows the end of its log and keeps it
tmp=$(mktemp -d)
mkdir -p "$tmp/bin" "$tmp/scratch"
printf '#!/bin/sh\necho "E: Unable to locate package bubblewrap" >&2\nexit 100\n' > "$tmp/bin/apt-get"
printf '#!/bin/sh\nexec "$@"\n' > "$tmp/bin/sudo"
chmod +x "$tmp/bin/"*
# a PATH with the tools the step needs and no bwrap or ss, so it has something to install
for tool in mktemp tail sed uname cat rm; do ln -s "$(command -v $tool)" "$tmp/bin/$tool"; done
out=$(PATH="$tmp/bin" TMPDIR="$tmp/scratch" HOME="$tmp/user" ARIA_HOME="$tmp/home" /bin/sh "$installer" --headless --install-packages 2>&1) || true
printf '%s\n' "$out" | grep -qF 'Unable to locate package bubblewrap' && pass "a failed step shows what it printed" || fail "a failed step hides its output: $out"
rm -r "$tmp"

# unknown flags are refused
if sh "$installer" --frobnicate >/dev/null 2>&1; then fail "an unknown flag was accepted"; else pass "unknown flags are refused"; fi

# syntax, and shellcheck when it is installed
sh -n "$installer" && pass "install.sh parses" || fail "install.sh has a syntax error"
if command -v shellcheck >/dev/null 2>&1; then shellcheck "$installer" && pass "shellcheck is clean" || fail "shellcheck found problems"; else echo "skip: shellcheck is not installed"; fi

[ "$failures" -eq 0 ] && echo "all checks passed" || { echo "$failures check(s) failed"; exit 1; }
