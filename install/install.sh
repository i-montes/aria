#!/bin/sh
# Aria installer for Omarchy (Hyprland with the Lua config) and macOS.
#
#   curl -fsSL https://<your site>/install.sh | sh     install, or update in place
#   sh install.sh --local target/release/aria          install a binary you built yourself
#   sh install.sh --uninstall                          remove Aria (asks before deleting ~/.aria)
#   sh install.sh --dry-run [...]                      print what would happen and change nothing
#   sh install.sh --restart-now [...]                  restart a running dock at once, even mid-run
#   sh install.sh --headless [...]                     a server with no desktop (Linux): `aria serve` as a user unit
#
# An update does not cut a run: the running dock is told a new version is installed and restarts into it once it
# is hidden and no session has a run going. Only a changed service file, --restart-now, or a dock too old to
# understand that restarts it at once.
#
# It detects the platform. On Omarchy it installs Aria's system libraries and a systemd user unit that
# starts the dock at login; on macOS it installs a LaunchAgent and needs no system packages. Both put
# the `aria` binary in ~/.local/bin, create ~/.aria, and install the pinned Pi under
# ~/.aria/pi/<version> (Node 22.19 or newer is required). On Omarchy it never writes to
# ~/.config/hypr/: the dock registers SUPER + A with Hyprland itself, every time it starts. With the Omarchy shell it puts
# Aria's face in the bar (~/.config/omarchy/plugins/aria.face, placed with `omarchy plugin enable`). On macOS
# the shortcut is Option + Space, registered by the dock with Carbon (no Accessibility permission).
set -eu

BIN_DIR=${ARIA_BIN_DIR:-$HOME/.local/bin}
DATA_DIR=${ARIA_HOME:-$HOME/.aria}
UNIT_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
UNIT=aria.service
SERVE_UNIT=aria-serve.service
AGENT_DIR=$HOME/Library/LaunchAgents
AGENT=com.aria.dock.plist
LABEL=com.aria.dock
# The dock's toolkit, plus bubblewrap (the sandbox runs every command in bwrap), iproute2 (the process guard tells the
# model to find a port's process with `ss`), libsecret (`secret-tool`, how Aria keeps keys in the keyring), and for
# dictation alsa-lib (the microphone) and vulkan-icd-loader (whisper.cpp on the GPU; without a driver it runs on the CPU).
# bun is not needed: only the checks use it.
PACKAGES="gtk3 webkit2gtk-4.1 gtk-layer-shell bubblewrap iproute2 libsecret alsa-lib vulkan-icd-loader"

# Only a real file has siblings: under `curl | sh` $0 is just "sh" and the current directory is not ours.
SCRIPT_DIR=""
if [ -f "$0" ]; then SCRIPT_DIR=$(cd "$(dirname "$0")" 2>/dev/null && pwd) || SCRIPT_DIR=""; fi
PI_PACKAGE=@earendil-works/pi-coding-agent
# A server has no toolkit: the sandbox (bwrap) and `ss` for the process guard are all it needs from the system.
HEADLESS_PACKAGES="bubblewrap iproute2"

DRY_RUN=0
HEADLESS=0
NODE_PRIVATE=0
NODE_PINS=""
RESTART_NOW=0
SERVICE_CHANGED=0
LOCAL=""
MODE=install
TMP=""

say() { printf '%s\n' "$*"; }
die() { printf 'aria installer: %s\n' "$*" >&2; exit 1; }
warn() { printf 'aria installer: warning: %s\n' "$*" >&2; }

# Runs a command, or only prints it in a dry run.
run() {
  if [ "$DRY_RUN" = 1 ]; then
    printf 'would: %s\n' "$*"
  else
    "$@"
  fi
}

# The platform picks the artifact and the service manager. It is detected lazily, when an install (or a
# platform-specific hook) actually needs it, so the test hooks (--check-node, --link-node) work with a
# minimal PATH and a missing uname only matters then.
PLATFORM=""
# Placeholders until releases are published; override with the environment.
DOWNLOAD_URL=${ARIA_DOWNLOAD_URL:-}
PI_VERSION_URL=${ARIA_PI_VERSION_URL:-}
NODE_VERSION_URL=${ARIA_NODE_VERSION_URL:-}
NODE_DIST=${ARIA_NODE_DIST:-https://nodejs.org/dist}

detect_platform() {
  [ -n "$PLATFORM" ] && return
  platform=$(uname -s 2>/dev/null) || die "could not detect the system: uname is missing"
  case "$platform" in
    Linux) artifact=aria-x86_64-linux ;;
    Darwin)
      machine=$(uname -m 2>/dev/null) || die "could not detect the Mac architecture"
      case "$machine" in
        arm64) artifact=aria-aarch64-apple-darwin ;;
        x86_64) artifact=aria-x86_64-apple-darwin ;;
        *) die "unsupported Mac architecture: $machine" ;;
      esac
      ;;
    *) die "unsupported system: $platform (Aria runs on Omarchy/Hyprland and macOS)" ;;
  esac
  PLATFORM=$platform
  if [ -z "$DOWNLOAD_URL" ]; then DOWNLOAD_URL=https://aria.example/releases/latest/$artifact; fi
  if [ -z "$PI_VERSION_URL" ]; then PI_VERSION_URL=${DOWNLOAD_URL%/*}/pi-version; fi
  if [ -z "$NODE_VERSION_URL" ]; then NODE_VERSION_URL=${DOWNLOAD_URL%/*}/node-version; fi
  if [ "$HEADLESS" = 1 ] && [ "$PLATFORM" != Linux ]; then die "--headless is for Linux servers (this is $PLATFORM); a Mac runs the dock"; fi
}

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

# The unit file. `curl | sh` has no sibling files, so the script carries it (install/aria.service
# is the same text; install.test.sh checks they match).
print_unit() {
  if [ "$BIN_DIR" = "$HOME/.local/bin" ]; then exec_path='%h/.local/bin/aria'; else exec_path="$BIN_DIR/aria"; fi
  cat <<UNIT
[Unit]
Description=Aria dock
PartOf=graphical-session.target
After=graphical-session.target

[Service]
ExecStart=$exec_path dock
Restart=on-failure

[Install]
WantedBy=graphical-session.target
UNIT
}

# The server's unit file (install/aria-serve.service is the same text; install.test.sh checks they match). No graphical
# session: it starts with the user manager (loginctl enable-linger keeps that running with nobody logged in). Its PATH
# starts with the private Node, which the service's own PATH would otherwise not have.
print_serve_unit() {
  if [ "$BIN_DIR" = "$HOME/.local/bin" ]; then exec_path='%h/.local/bin/aria'; else exec_path="$BIN_DIR/aria"; fi
  if [ "$DATA_DIR" = "$HOME/.aria" ]; then data_path='%h/.aria'; else data_path="$DATA_DIR"; fi
  cat <<UNIT
[Unit]
Description=Aria service

[Service]
Environment=PATH=$data_path/node/bin:$data_path/bin:%h/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$exec_path serve
Restart=on-failure

[Install]
WantedBy=default.target
UNIT
}

# The LaunchAgent. It starts the dock at login in the user's Aqua session and keeps its output in
# ~/.aria/logs/aria.log, and starts it again if it crashes (systemd's Restart=on-failure). The executable is an absolute
# path: a LaunchAgent has no shell profile.
print_agent() {
  cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>$BIN_DIR/aria</string><string>dock</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ProcessType</key><string>Interactive</string>
  <key>LimitLoadToSessionType</key><string>Aqua</string>
  <key>StandardOutPath</key><string>$DATA_DIR/logs/aria.log</string>
  <key>StandardErrorPath</key><string>$DATA_DIR/logs/aria.log</string>
</dict>
</plist>
PLIST
}

# GNU install creates the folder with -D; the BSD install of macOS has no such flag, and ~/.local/bin often does not exist.
install_binary() {
  if [ "$PLATFORM" = Darwin ]; then
    run mkdir -p "$BIN_DIR"
    run install -m 755 "$SOURCE" "$BIN_DIR/aria"
    # The linker signs it ad hoc under a name that changes with every build and leaves its Info.plist (the microphone's
    # reason) unbound; signed again ad hoc as com.aria.dock, the plist is bound and the name stays. macOS still asks for
    # the microphone once per version (an ad hoc signature is a new app to it on every update).
    run codesign --force --sign - --identifier com.aria.dock "$BIN_DIR/aria" || warn "could not sign $BIN_DIR/aria"
  else
    run install -Dm755 "$SOURCE" "$BIN_DIR/aria"
  fi
}

print_service() {
  detect_platform
  if [ "$PLATFORM" = Darwin ]; then print_agent; elif [ "$HEADLESS" = 1 ]; then print_serve_unit; else print_unit; fi
}

check_session() {
  detect_platform
  if [ "$PLATFORM" = Darwin ]; then
    if [ "$DRY_RUN" = 1 ]; then
      say "would: check macOS 12 or newer"
      return
    fi
    version=$(sw_vers -productVersion 2>/dev/null || echo "")
    major=${version%%.*}
    case "$major" in "" | *[!0-9]*) die "Aria needs macOS 12 or newer (could not read the version)" ;; esac
    [ "$major" -ge 12 ] || die "Aria needs macOS 12 or newer (found $version)"
    return
  fi
  if [ "$HEADLESS" = 1 ]; then return; fi # a server has no session to check
  if [ "$DRY_RUN" = 1 ]; then
    say "would: check this is a Hyprland session with the Lua config (hyprctl eval 'return 1' prints ok)"
    return
  fi
  [ -n "${HYPRLAND_INSTANCE_SIGNATURE:-}" ] || die "run this from inside a Hyprland session (Omarchy)"
  [ "$(hyprctl eval 'return 1' 2>/dev/null)" = "ok" ] || die "Aria needs Hyprland 0.56 or newer with the Lua configuration"
}

install_packages() {
  detect_platform
  if [ "$PLATFORM" = Darwin ]; then
    say "macOS: no system packages are needed"
    return
  fi
  if [ "$DRY_RUN" = 1 ]; then
    # shellcheck disable=SC2086
    run sudo pacman -S --needed $PACKAGES
    return
  fi
  command -v pacman >/dev/null 2>&1 || die "pacman not found: install these packages first: $PACKAGES"
  # shellcheck disable=SC2086
  if ! pacman -Qq $PACKAGES >/dev/null 2>&1; then run sudo pacman -S --needed $PACKAGES; fi
}

# A server's packages: the sandbox and `ss`, with the package manager it has. Never fatal: without them (or without
# sudo) the owner gets the command to run and the sandbox asks every time, as on a Mac.
install_headless_packages() {
  detect_platform
  if command -v apt-get >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    set -- sudo apt-get install -y $HEADLESS_PACKAGES
  elif command -v pacman >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    set -- sudo pacman -S --needed $HEADLESS_PACKAGES
  else
    say "No apt-get or pacman here: install $HEADLESS_PACKAGES with your package manager (until then every command asks before it runs)"
    return 0
  fi
  if [ "$DRY_RUN" = 1 ]; then
    run "$@"
    return 0
  fi
  if command -v bwrap >/dev/null 2>&1 && command -v ss >/dev/null 2>&1; then return 0; fi
  "$@" || {
    warn "could not install $HEADLESS_PACKAGES (no sudo?)"
    say "Run this yourself: $*"
    say "Until then every command asks before it runs."
    return 0
  }
}

# Without linger the user manager (and Aria with it) stops when the last login ends.
enable_linger() {
  who=${USER:-$(id -un)}
  if [ "$DRY_RUN" = 1 ]; then
    run loginctl enable-linger "$who"
    return 0
  fi
  loginctl enable-linger "$who" || {
    warn "could not enable linger, so Aria stops when you log out"
    say "Run this yourself: sudo loginctl enable-linger $who"
  }
}

# True when the Node on PATH (with its npm) is 22.19 or newer. Silent: a server falls back to a private Node.
node_is_current() {
  command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 || return 1
  version=$(node --version 2>/dev/null | sed 's/^v//')
  major=${version%%.*}
  rest=${version#*.}
  minor=${rest%%.*}
  case "$major$minor" in *[!0-9]* | "") return 1 ;; esac
  [ "$major" -gt 22 ] || { [ "$major" -eq 22 ] && [ "$minor" -ge 19 ]; }
}

# Sets NODE_VERSION and NODE_PINS from the pinned Node: the repo's node-version file next to this script, else the
# published one. Its first line is the version, then one "linux-<arch> <sha256>" line per build, like engram-version.
read_node_version() {
  detect_platform
  if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/../node-version" ]; then
    NODE_PINS=$(cat "$SCRIPT_DIR/../node-version")
  elif [ "$DRY_RUN" = 1 ]; then
    say "would: download $NODE_VERSION_URL"
    NODE_VERSION=PINNED
    NODE_PINS=""
    return
  else
    NODE_PINS=$(curl -fsSL "$NODE_VERSION_URL") || die "could not download the pinned Node version from $NODE_VERSION_URL"
  fi
  NODE_VERSION=$(printf '%s\n' "$NODE_PINS" | head -1 | tr -d '[:space:]')
  case "$NODE_VERSION" in *[!0-9.]* | "") die "the pinned Node version is not a version number: '$NODE_VERSION'" ;; esac
  bad=$(printf '%s\n' "$NODE_PINS" | sed 1d | tr -s ' \t' ' ' | sed 's/^ //; s/ $//' | grep -v '^$' | grep -vE '^linux-[a-z0-9]+ [0-9a-f]{64}$' || true)
  [ -z "$bad" ] || die "node-version has a line that is not '<build> <sha256>': '$bad'"
}

# The pinned Node into $DATA_DIR/node, for a server whose own is missing or too old (never replaced: Ubuntu's apt Node is
# what its other software uses). What is checked: the tarball's sha256 equals the one pinned for this CPU in node-version,
# which is in the repo (or published with the installer), so a changed or corrupt download, or a mirror that serves
# something else, is refused. It does not prove nodejs.org's own release is trustworthy: the pin is only as good as the
# moment someone compared it with nodejs.org's SHASUMS256.txt.
install_private_node() {
  detect_platform
  case "$NODE_DIST" in https://* | file://*) ;; *) die "ARIA_NODE_DIST must be an https:// address (or file:// for a local mirror): '$NODE_DIST'" ;; esac
  read_node_version
  case "$(uname -m)" in
    aarch64 | arm64) node_arch=arm64 ;;
    x86_64) node_arch=x64 ;;
    *) die "there is no Node build for this machine ($(uname -m)); install Node 22.19 or newer yourself" ;;
  esac
  NODE_PRIVATE=1
  tarball=node-v$NODE_VERSION-linux-$node_arch.tar.xz
  base=$NODE_DIST/v$NODE_VERSION
  if [ "$NODE_VERSION" = PINNED ]; then
    say "would: install the pinned Node into $DATA_DIR/node from $base/$tarball (verified against the checksum pinned for linux-$node_arch in node-version)"
    return
  fi
  expected=$(printf '%s\n' "$NODE_PINS" | awk -v k="linux-$node_arch" '$1 == k { print $2 }')
  [ -n "$expected" ] || die "node-version pins no checksum for linux-$node_arch; nothing was installed"
  if [ "$DRY_RUN" = 1 ]; then
    say "would: install Node $NODE_VERSION into $DATA_DIR/node from $base/$tarball (verified against the checksum pinned for linux-$node_arch in node-version)"
    return
  fi
  command -v xz >/dev/null 2>&1 || die "xz is needed to unpack Node (on Ubuntu: sudo apt-get install xz-utils)"
  install -d -m 700 "$DATA_DIR"
  # Inside $DATA_DIR, so the final move is a rename on one filesystem; removed on any exit.
  work=$(mktemp -d "$DATA_DIR/.node-install.XXXXXX")
  trap 'rm -rf "$work"' EXIT
  curl -fsSL "$base/$tarball" -o "$work/$tarball" || die "could not download Node from $base/$tarball"
  [ "$(sha256_of "$work/$tarball")" = "$expected" ] || die "the Node download does not match the checksum pinned in node-version; nothing was installed"
  mkdir "$work/stage"
  tar -xJf "$work/$tarball" -C "$work/stage" --strip-components 1 || die "could not unpack the Node download"
  [ ! -e "$DATA_DIR/node" ] || mv "$DATA_DIR/node" "$work/old"
  mv "$work/stage" "$DATA_DIR/node"
  rm -rf "$work"
  trap - EXIT
  say "Installed Node $NODE_VERSION into $DATA_DIR/node"
}

# A server's Node: the private one (when an earlier run put it there) or the system's, kept when it is new enough, else
# the pinned one is installed under $DATA_DIR/node. Pi and npm then run on it.
ensure_server_node() {
  if [ -x "$DATA_DIR/node/bin/node" ]; then PATH="$DATA_DIR/node/bin:$PATH"; export PATH; fi
  if node_is_current; then
    say "Node $(node --version | sed 's/^v//') is new enough; keeping it"
    if [ -x "$DATA_DIR/node/bin/node" ]; then NODE_PRIVATE=1; fi
    return
  fi
  install_private_node
  PATH="$DATA_DIR/node/bin:$PATH"; export PATH
}

# Node 22.19 or newer runs Pi. Prints the reason and fails when it is missing or too old.
check_node() {
  command -v node >/dev/null 2>&1 || die "Node 22.19 or newer is required (install it with your package manager or mise)"
  command -v npm >/dev/null 2>&1 || die "npm is required to install Pi (it usually comes with Node; on Arch install the npm package)"
  version=$(node --version 2>/dev/null | sed 's/^v//')
  major=${version%%.*}
  rest=${version#*.}
  minor=${rest%%.*}
  case "$major$minor" in *[!0-9]* | "") die "could not read the Node version ('$version'); Node 22.19 or newer is required" ;; esac
  if [ "$major" -lt 22 ] || { [ "$major" -eq 22 ] && [ "$minor" -lt 19 ]; }; then
    die "Node 22.19 or newer is required (found $version); install it with your package manager or mise"
  fi
}

# The service has no mise on its PATH: a link to the Node found now lets it, and Pi's children, run.
link_node() {
  run install -d -m 700 "$DATA_DIR/bin"
  if [ -x "$DATA_DIR/node/bin/node" ] || [ "$NODE_PRIVATE" = 1 ]; then
    # a server's private Node: it is the one the service runs, whatever the system has
    node_bin=$DATA_DIR/node/bin/node
  else
    node_bin=""
  fi
  if [ "$DRY_RUN" = 1 ]; then
    if [ -n "$node_bin" ]; then
      say "would: link $DATA_DIR/bin/node to $node_bin (and claude, when Claude Code is installed)"
    else
      say "would: link $DATA_DIR/bin/node to the Node on your PATH (and claude, when Claude Code is installed)"
    fi
  else
    ln -sf "${node_bin:-$(command -v node)}" "$DATA_DIR/bin/node"
    # Claude Code, when installed, is what the Claude provider runs; Pi's processes find it through the same directory.
    if command -v claude >/dev/null 2>&1; then ln -sf "$(command -v claude)" "$DATA_DIR/bin/claude"; fi
  fi
}

# Sets PI_VERSION to the pinned Pi: the repo's pi-version file next to this script, else the published one.
read_pi_version() {
  detect_platform
  if [ -n "${ARIA_PI_VERSION:-}" ]; then
    PI_VERSION=$ARIA_PI_VERSION
  elif [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/../pi-version" ]; then
    PI_VERSION=$(tr -d '[:space:]' < "$SCRIPT_DIR/../pi-version")
  elif [ "$DRY_RUN" = 1 ]; then
    say "would: download $PI_VERSION_URL"
    PI_VERSION=PINNED
    return
  else
    PI_VERSION=$(curl -fsSL "$PI_VERSION_URL") || die "could not download the pinned Pi version from $PI_VERSION_URL"
    PI_VERSION=$(printf '%s' "$PI_VERSION" | tr -d '[:space:]')
  fi
  case "$PI_VERSION" in *[!0-9.]* | "") die "the pinned Pi version is not a version number: '$PI_VERSION'" ;; esac
}

install_pi() {
  [ -n "${PI_VERSION:-}" ] || read_pi_version
  prefix="$DATA_DIR/pi/$PI_VERSION"
  if [ -x "$prefix/node_modules/.bin/pi" ]; then
    say "Pi $PI_VERSION is already installed"
    return
  fi
  run npm install --prefix "$prefix" --no-audit --no-fund "$PI_PACKAGE@$PI_VERSION"
}

# The packages the owner's extensions use (pi-packages, pinned), installed into Aria's own Pi directory.
install_pi_packages() {
  detect_platform
  if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/../pi-packages" ]; then
    packages=$(cat "$SCRIPT_DIR/../pi-packages")
  elif [ "$DRY_RUN" = 1 ]; then
    say "would: download ${DOWNLOAD_URL%/*}/pi-packages"
    return
  else
    packages=$(curl -fsSL "${DOWNLOAD_URL%/*}/pi-packages") || die "could not download the list of Pi packages"
  fi
  run install -d -m 700 "$DATA_DIR/agent"
  for package in $packages; do
    run env PI_CODING_AGENT_DIR="$DATA_DIR/agent" PI_SKIP_VERSION_CHECK=1 "$DATA_DIR/pi/$PI_VERSION/node_modules/.bin/pi" install "$package"
  done
}

# Sets SOURCE to the binary to install: the --local one, or a downloaded and verified one.
fetch_binary() {
  detect_platform
  if [ -n "$LOCAL" ]; then
    SOURCE=$LOCAL
    return
  fi
  if [ "$DRY_RUN" = 1 ]; then
    say "would: download $DOWNLOAD_URL and verify it against $DOWNLOAD_URL.sha256"
    SOURCE=/tmp/aria-download
    return
  fi
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' EXIT
  curl -fsSL "$DOWNLOAD_URL" -o "$TMP/aria"
  curl -fsSL "$DOWNLOAD_URL.sha256" -o "$TMP/aria.sha256"
  expected=$(cut -d ' ' -f 1 "$TMP/aria.sha256")
  actual=$(sha256_of "$TMP/aria")
  [ "$expected" = "$actual" ] || die "the downloaded binary does not match its checksum; nothing was installed"
  chmod +x "$TMP/aria"
  SOURCE=$TMP/aria
}

# Engram, Aria's memory: the release pinned in engram-version (its first line is the version, then one
# "<platform> <sha256>" line per build) into ~/.aria/bin, verified against that checksum. Aria keeps its own copy and
# its own database (~/.aria/engram), apart from an Engram the user may have.
ENGRAM_RELEASES=https://github.com/Gentleman-Programming/engram/releases/download
install_engram() {
  detect_platform
  if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/../engram-version" ]; then
    pins=$(cat "$SCRIPT_DIR/../engram-version")
  elif [ "$DRY_RUN" = 1 ]; then
    say "would: download ${DOWNLOAD_URL%/*}/engram-version"
    say "would: install engram (the pinned version) into $DATA_DIR/bin/engram"
    return
  else
    pins=$(curl -fsSL "${DOWNLOAD_URL%/*}/engram-version") || die "could not download the pinned Engram version"
  fi
  version=$(printf '%s\n' "$pins" | head -1 | tr -d '[:space:]')
  case "$PLATFORM/$(uname -m)" in
    Linux/x86_64) build=linux_amd64 ;;
    Linux/aarch64 | Linux/arm64) build=linux_arm64 ;;
    Darwin/arm64) build=darwin_arm64 ;;
    Darwin/x86_64) build=darwin_amd64 ;;
    *) die "Engram has no build for this machine ($(uname -m))" ;;
  esac
  target="$DATA_DIR/bin/engram"
  if [ -x "$target" ] && "$target" --version 2>/dev/null | grep -qF "$version"; then
    say "Engram $version is already installed"
    return
  fi
  url=${ARIA_ENGRAM_URL:-$ENGRAM_RELEASES/v$version/engram_${version}_$build.tar.gz}
  expected=${ARIA_ENGRAM_SHA256:-$(printf '%s\n' "$pins" | awk -v b="$build" '$1 == b { print $2 }')}
  if [ "$DRY_RUN" = 1 ]; then
    say "would: install engram $version into $target from $url (verified)"
    return
  fi
  work=$(mktemp -d)
  if ! curl -fsSL "$url" -o "$work/engram.tar.gz"; then rm -rf "$work"; die "could not download Engram from $url"; fi
  if [ "$(sha256_of "$work/engram.tar.gz")" != "$expected" ]; then rm -rf "$work"; die "the Engram download does not match its checksum; nothing was installed"; fi
  if ! tar xzf "$work/engram.tar.gz" -C "$work" engram; then rm -rf "$work"; die "the Engram download has no engram binary"; fi
  install -d -m 700 "$DATA_DIR/bin"
  install -m 755 "$work/engram" "$target"
  rm -rf "$work"
  say "Installed Engram $version"
}

# agent-browser, the browser the agents check the pages they build with (the verify-in-browser skill): the version
# pinned in agent-browser-version, installed with npm under ~/.aria/agent-browser/<version> and linked into ~/.aria/bin,
# then its Chrome (in ~/.agent-browser/browsers). Aria works without it, so a failure here only warns.
install_agent_browser() {
  detect_platform
  if [ -n "${ARIA_AGENT_BROWSER_VERSION:-}" ]; then
    version=$ARIA_AGENT_BROWSER_VERSION
  elif [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/../agent-browser-version" ]; then
    version=$(tr -d '[:space:]' < "$SCRIPT_DIR/../agent-browser-version")
  elif [ "$DRY_RUN" = 1 ]; then
    say "would: download ${DOWNLOAD_URL%/*}/agent-browser-version"
    version=PINNED
  else
    version=$(curl -fsSL "${DOWNLOAD_URL%/*}/agent-browser-version") || {
      warn "could not download the pinned agent-browser version; the agents cannot check pages in a browser"
      return 0
    }
    version=$(printf '%s' "$version" | tr -d '[:space:]')
  fi
  if [ "$version" != PINNED ] || [ "$DRY_RUN" != 1 ]; then
    case "$version" in *[!0-9.]* | "")
      warn "the pinned agent-browser version is not a version number: '$version'; agent-browser was not installed"
      return 0
      ;;
    esac
  fi
  prefix="$DATA_DIR/agent-browser/$version"
  if [ -x "$prefix/node_modules/.bin/agent-browser" ]; then
    say "agent-browser $version is already installed"
  else
    run npm install --prefix "$prefix" --no-audit --no-fund "agent-browser@$version" || {
      warn "could not install agent-browser $version; the agents cannot check pages in a browser"
      return 0
    }
  fi
  { run install -d -m 700 "$DATA_DIR/bin" && run ln -sf "$prefix/node_modules/.bin/agent-browser" "$DATA_DIR/bin/agent-browser"; } || {
    warn "could not link agent-browser into $DATA_DIR/bin"
    return 0
  }
  run "$DATA_DIR/bin/agent-browser" install || warn "could not install agent-browser's browser; the browser check will not work until it is installed (run: ~/.aria/bin/agent-browser install)"
}

# sha256sum is Linux's; macOS has shasum.
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

# Writes the service file, and sets SERVICE_CHANGED=1 when it is new or different: a running dock then has to be
# restarted by the service manager to pick it up.
write_unit() {
  detect_platform
  if [ "$PLATFORM" = Darwin ]; then
    run mkdir -p "$AGENT_DIR"
    run install -d -m 700 "$DATA_DIR/logs"
    write_service_file "$AGENT_DIR/$AGENT" "$(print_agent)"
    return
  fi
  run mkdir -p "$UNIT_DIR"
  if [ "$HEADLESS" = 1 ]; then
    write_service_file "$UNIT_DIR/$SERVE_UNIT" "$(print_serve_unit)"
  else
    write_service_file "$UNIT_DIR/$UNIT" "$(print_unit)"
  fi
}

write_service_file() {
  if [ -f "$1" ] && [ "$(cat "$1")" = "$2" ]; then return; fi
  SERVICE_CHANGED=1
  if [ "$DRY_RUN" = 1 ]; then
    say "would: write $1"
  else
    printf '%s\n' "$2" > "$1"
  fi
}

# Asks the running dock to restart into the new binary when it is hidden and idle. Fails (and the caller restarts the
# service) when the service file changed, --restart-now was given, or no dock understands the request.
soft_restart() {
  [ "$SERVICE_CHANGED" = 0 ] && [ "$RESTART_NOW" = 0 ] || return 1
  run "$BIN_DIR/aria" update-ready 2>/dev/null || return 1
  say "Aria is updated. The dock restarts into the new version once it is hidden and nothing is running."
}

# Starts the dock, or brings the running one to the new version. $1 is 1 when the dock was running before the install.
restart_dock() {
  detect_platform
  if [ "$PLATFORM" = Darwin ]; then
    if [ "$1" = 1 ] && soft_restart; then return; fi
    uid=$(id -u)
    # The LaunchAgent may already be loaded from an earlier install: unload it, then load the new file.
    # bootstrap fails with "Input/output error" while the old instance is still exiting, so wait for the
    # label to disappear first. `RunAtLoad` starts the new binary.
    run launchctl bootout "gui/$uid/$LABEL" 2>/dev/null || true
    if [ "$DRY_RUN" != 1 ]; then
      attempts=0
      while launchctl print "gui/$uid/$LABEL" >/dev/null 2>&1 && [ "$attempts" -lt 50 ]; do
        attempts=$((attempts + 1))
        sleep 0.1
      done
    fi
    run launchctl bootstrap "gui/$uid" "$AGENT_DIR/$AGENT"
    say "Aria is installed. Press Option + Space to set it up."
  elif [ "$HEADLESS" = 1 ]; then
    run systemctl --user daemon-reload
    run systemctl --user enable --now "$SERVE_UNIT"
    if [ "$1" = 1 ] && soft_restart; then return; fi
    if [ "$1" = 1 ]; then run systemctl --user restart "$SERVE_UNIT"; fi
    say "Aria is installed and running as $SERVE_UNIT. From your computer: aria connect <user>@<this server>"
  else
    run systemctl --user daemon-reload
    run systemctl --user enable --now "$UNIT"
    if [ "$1" = 1 ] && soft_restart; then return; fi
    # An update replaces the binary under a running dock: restart it so the new one runs.
    if [ "$1" = 1 ]; then run systemctl --user restart "$UNIT"; fi
    say "Aria is installed. Press Super+A to set it up."
  fi
}

# Aria's face in the Omarchy bar, next to the weather: the binary carries the widget and puts it in the shell's plugins
# (`aria bar install`, which backs up shell.json and places it with Omarchy's own command). Skipped where there is no
# Omarchy shell; a failure only warns, since the dock works without it.
install_bar_widget() {
  [ "$PLATFORM" = Darwin ] && return
  [ "$HEADLESS" = 1 ] && return
  if [ "$DRY_RUN" = 1 ]; then
    say "would: $BIN_DIR/aria bar install (Aria's face in the Omarchy bar, when the Omarchy shell is there)"
    return
  fi
  "$BIN_DIR/aria" bar install || warn "Aria's face could not be put in the Omarchy bar; the dock works without it"
}

do_install() {
  detect_platform
  # Refuse a bad --local before anything else, so a mistake changes nothing.
  if [ -n "$LOCAL" ] && [ "$DRY_RUN" != 1 ] && [ ! -x "$LOCAL" ]; then die "$LOCAL is not an executable file"; fi
  check_session
  if [ "$HEADLESS" = 1 ]; then ensure_server_node; else check_node; fi
  read_pi_version # before anything is installed: a version we cannot read must not leave a half install
  if [ "$HEADLESS" = 1 ]; then install_headless_packages; else install_packages; fi
  was_active=0
  if [ "$DRY_RUN" != 1 ]; then
    if [ "$PLATFORM" = Darwin ]; then
      launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1 && was_active=1
    elif [ "$HEADLESS" = 1 ]; then
      systemctl --user is-active --quiet "$SERVE_UNIT" 2>/dev/null && was_active=1
    else
      systemctl --user is-active --quiet "$UNIT" 2>/dev/null && was_active=1
    fi
  fi
  fetch_binary
  # Pi and its packages first: if a download fails, nothing has been replaced and the running dock stays as it was.
  run install -d -m 700 "$DATA_DIR"
  link_node
  install_pi
  install_pi_packages
  install_engram
  install_agent_browser
  install_binary
  install_bar_widget
  write_unit
  if [ "$HEADLESS" = 1 ]; then enable_linger; fi
  restart_dock "$was_active"
}

# True only for a real directory (not a symlink) that is neither / nor the home directory and that
# looks like Aria's data: it holds settings.json or auth.json, or it is empty. Compared by real path,
# so `$HOME/`, `$HOME/.` and friends do not slip past.
safe_to_delete() {
  dir=$1
  [ -n "$dir" ] && [ -d "$dir" ] && [ ! -L "$dir" ] || return 1
  real=$(cd "$dir" 2>/dev/null && pwd -P) || return 1
  home_real=$(cd "$HOME" 2>/dev/null && pwd -P) || return 1
  [ "$real" != "/" ] && [ "$real" != "$home_real" ] || return 1
  [ -e "$real/settings.json" ] || [ -e "$real/auth.json" ] || [ -z "$(ls -A "$real")" ]
}

ask_delete_data() {
  [ -d "$DATA_DIR" ] || return 0
  if [ "$DRY_RUN" = 1 ]; then
    say "would: ask \"Delete $DATA_DIR (your settings and credentials)? [y/N]\" and remove it only on y"
    return
  fi
  answer=n
  if [ -r /dev/tty ]; then
    printf 'Delete %s (your settings and credentials)? [y/N] ' "$DATA_DIR" > /dev/tty
    read -r answer < /dev/tty || answer=n
  fi
  case "$answer" in
    y | Y | yes)
      if safe_to_delete "$DATA_DIR"; then
        rm -rf -- "$DATA_DIR"
        say "Deleted $DATA_DIR"
      else
        say "$DATA_DIR does not look like Aria's data, so it was left alone."
      fi
      ;;
    *) say "Kept $DATA_DIR" ;;
  esac
}

do_uninstall() {
  detect_platform
  if [ "$PLATFORM" = Darwin ]; then
    run launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    run rm -f "$AGENT_DIR/$AGENT" "$BIN_DIR/aria"
  elif [ "$HEADLESS" = 1 ]; then
    run systemctl --user disable --now "$SERVE_UNIT" || true
    run rm -f "$UNIT_DIR/$SERVE_UNIT" "$BIN_DIR/aria"
    run systemctl --user daemon-reload
  else
    run systemctl --user disable --now "$UNIT" || true
    if [ "$DRY_RUN" = 1 ] || [ -x "$BIN_DIR/aria" ]; then run "$BIN_DIR/aria" bar remove || true; fi
    run rm -f "$UNIT_DIR/$UNIT" "$BIN_DIR/aria"
    run systemctl --user daemon-reload
    if [ "$DRY_RUN" = 1 ] || [ -n "${HYPRLAND_INSTANCE_SIGNATURE:-}" ]; then
      run hyprctl eval 'hl.unbind("SUPER + A")' || true
    fi
  fi
  ask_delete_data
  say "Aria is removed."
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --restart-now) RESTART_NOW=1 ;;
    --headless) HEADLESS=1 ;; # give it before --restart-dock, the hook that runs the moment it is read
    --local)
      [ $# -ge 2 ] || die "--local needs the path of a binary"
      LOCAL=$2
      shift
      ;;
    --uninstall) MODE=uninstall ;;
    --print-unit) MODE=print-unit ;;
    --check-node) MODE=check-node ;; # test hooks
    --print-pi-version) MODE=print-pi-version ;;
    --link-node) MODE=link-node ;;
    --install-engram) MODE=install-engram ;; # test hook
    --install-node) MODE=install-node ;; # test hooks for the server's steps
    --install-packages) MODE=install-packages ;;
    --enable-linger) MODE=enable-linger ;;
    --print-node-version) MODE=print-node-version ;;
    --install-agent-browser) MODE=install-agent-browser ;; # test hook
    --restart-dock) # test hook: the last step of an install; $2 is 1 when a dock was running, $3 when the service file changed
      [ $# -ge 3 ] || die "--restart-dock needs two values"
      SERVICE_CHANGED=$3
      restart_dock "$2"
      exit $?
      ;;
    --can-delete) # test hook: exit 0 when the directory is one uninstall would delete
      [ $# -ge 2 ] || die "--can-delete needs a directory"
      safe_to_delete "$2"
      exit $?
      ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
  shift
done

case "$MODE" in
  install) do_install ;;
  uninstall) do_uninstall ;;
  print-unit) print_service ;;
  check-node) check_node ;;
  print-pi-version) read_pi_version; printf '%s\n' "$PI_VERSION" ;;
  link-node) link_node ;;
  install-engram) install_engram ;;
  install-node) install_private_node ;;
  install-packages) install_headless_packages ;;
  enable-linger) enable_linger ;;
  print-node-version) read_node_version; printf '%s\n' "$NODE_VERSION" ;;
  install-agent-browser) install_agent_browser ;;
esac
