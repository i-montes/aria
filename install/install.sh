#!/bin/sh
# Aria installer for Omarchy (Hyprland with the Lua config) and macOS.
#
#   curl -fsSL https://github.com/i-montes/aria/releases/latest/download/install.sh | sh     install, or update in place
#   curl -fsSL https://github.com/i-montes/aria/releases/latest/download/install.sh | ARIA_VERSION=v0.20.2 sh     a chosen release
#   sh install.sh --local target/release/aria          install a binary you built yourself
#   sh install.sh --uninstall                          remove Aria (asks before deleting ~/.aria)
#   sh install.sh --dry-run [...]                      print what would happen and change nothing
#   sh install.sh --restart-now [...]                  restart a running dock at once, even mid-run
#   sh install.sh --mode hub|desktop [...]             skip the question: a hub (a server others connect to) or this computer's dock
#   sh install.sh --server user@host [...]             a hub on another computer: installed from here, and this computer is linked to it
#   sh install.sh --server here [...]                  this computer is the hub (the same as --headless): `aria serve` as a user unit
#   sh install.sh --headless [...]                     a server with no desktop (Linux), `aria serve` as a user unit
#   sh install.sh --yes [...]                          ask nothing: take the answer that is known or the default, and agree to the
#                                                      sandbox's AppArmor profile (below)
#   sh install.sh --headless --with-browser [...]      the same, and agent-browser with its Chrome (a large download, left out on a server)
#
# On Linux it asks one question, "On this computer, or as a hub?", unless the answer is known: --mode or ARIA_MODE, an
# Aria already installed here (an update keeps its use), macOS (always the dock), or nothing to ask on (a dry run, no
# terminal, --yes), which takes the default: the dock on a machine with a desktop, a hub on one without. A hub installs
# the headless build (aria-x86_64-linux-headless or aria-aarch64-linux-headless), Node, Pi, Engram, the user unit and
# linger, and makes the sandbox start: on Ubuntu 24.04, where AppArmor blocks user namespaces, it offers a profile for
# bwrap alone (never the system-wide sysctl) and writes it only on your yes (or --yes).
#
# Where the binary comes from: the GitHub Releases of ARIA_REPO (default i-montes/aria), the latest release, or the one
# named by ARIA_VERSION (v0.20.2 or 0.20.2). The environment variables go to the shell that runs the script
# (`... | ARIA_VERSION=v0.20.2 sh`), not to curl. The binary is <artifact> next to its <artifact>.sha256 and the pins
# (pi-version, pi-packages, engram-version, agent-browser-version, node-version) in the same release; the artifact is
# aria-x86_64-linux, aria-aarch64-apple-darwin or aria-x86_64-apple-darwin. (A script run from a clone reads the pins
# of that clone instead; only a piped one fetches them.) A release has a binary for a platform only if it was built for
# it: when the download fails, build from a clone (`cargo build --release`, then
# `sh install/install.sh --local target/release/aria`). The binary is checked against the .sha256 of the same release,
# and the pins come from it too, so the checks show the download is whole, not that the release is authentic: nothing is
# signed, and this script itself is trusted only through TLS. The dock's Linux binary is for x86_64; a hub's is for x86_64
# or aarch64; any other CPU is refused.
#   ARIA_MODE=desktop|hub      the answer to the first question, as --mode
#   ARIA_SERVER=here|user@host the answer to the second one (which computer is the server), as --server
#   ARIA_REPO=owner/name       the repository whose releases are used (default i-montes/aria)
#   ARIA_VERSION=v0.20.2       a release, instead of the latest
#   ARIA_DOWNLOAD_URL=<url>    the address of the binary itself; beats ARIA_REPO and ARIA_VERSION, and the pins and the
#                              checksum are read from the same directory (<url>.sha256, ${url%/*}/pi-version, ...)
#   ARIA_PI_VERSION_URL, ARIA_NODE_VERSION_URL, ARIA_NODE_DIST, ARIA_ENGRAM_URL    the addresses of the pins and of what the
#                              installer fetches from elsewhere (Node, Engram), each beating what is derived above
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
ASSUME_YES=0
CHOSEN_MODE=${ARIA_MODE:-}
CHOSEN_SERVER=${ARIA_SERVER:-}
SERVER=""
WITH_BROWSER=0
NODE_PRIVATE=0
NODE_PINS=""
RESTART_NOW=0
SERVICE_CHANGED=0
LOCAL=""
MODE=install
TMP=""
NODE_WORK=""

# The one EXIT trap: the scratch directories of the binary's download and of Node's (a second trap would replace the first).
cleanup() {
  if [ -n "$TMP" ]; then rm -rf "$TMP"; fi
  if [ -n "$NODE_WORK" ]; then rm -rf "$NODE_WORK"; fi
  # The log of the noisy steps goes away with a good install and stays after a bad one (die says where it is).
  if [ -n "$LOG_FILE" ] && [ "$KEEP_LOG" != 1 ]; then rm -f "$LOG_FILE"; fi
}
trap cleanup EXIT

# --- the screen ---------------------------------------------------------------------------------------------------------
# A guided screen: numbered steps, what each is for, a mark when it is done, and the noisy output (apt, npm) kept in a log
# that is shown only when a step fails. Colour only on a terminal that takes it (NO_COLOR and TERM=dumb turn it off) and
# plain marks unless the locale is UTF-8. In a pipe or a log the same sentences come out without either.
UI_COLOR=0
MARK_OK=ok
MARK_FAIL=x
RULE=-
STEP_N=0
STEP_TOTAL=6
# Run by `aria hub init` on the server, the installer is one step of a longer screen: ARIA_NESTED=1 leaves out its banner,
# its plan, its question to start and its closing screen, and lists its steps without numbers.
NESTED=${ARIA_NESTED:-0}
MARK_STEP='-'
LOG_FILE=""
KEEP_LOG=0

ui_setup() {
  if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != dumb ]; then UI_COLOR=1; fi
  case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in
    *[Uu][Tt][Ff]-8* | *[Uu][Tt][Ff]8*)
      MARK_OK='✓'
      MARK_FAIL='✗'
      MARK_STEP='·'
      RULE='─'
      ;;
  esac
}
ui_setup

# paint <ansi code> <text>: the text in a colour on a terminal, as it is anywhere else.
paint() {
  code=$1
  shift
  if [ "$UI_COLOR" = 1 ]; then printf '\033[%sm%s\033[0m' "$code" "$*"; else printf '%s' "$*"; fi
}

say() { printf '%s\n' "$*"; }
die() {
  printf 'aria installer: %s\n' "$*" >&2
  if [ -n "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
    KEEP_LOG=1
    printf 'aria installer: what the steps printed is in %s\n' "$LOG_FILE" >&2
  fi
  exit 1
}
warn() { printf '%s %s\n' "$(paint 33 'aria installer: warning:')" "$*" >&2; }

ui_rule() { printf '  '; width=0; while [ "$width" -lt 56 ]; do printf '%s' "$RULE"; width=$((width + 1)); done; printf '\n'; }
ui_title() {
  [ "$NESTED" != 1 ] || return 0
  say ""
  say "  $(paint '1;36' 'Aria') $(paint 2 "· $1")"
  ui_rule
}
# ui_step <title> <what it is for>: the next numbered step.
ui_step() {
  STEP_N=$((STEP_N + 1))
  if [ "$NESTED" = 1 ]; then
    say "  $(paint 2 "$MARK_STEP") $1"
    return
  fi
  say ""
  say "$(paint '1;36' "[$STEP_N/$STEP_TOTAL]") $(paint 1 "$1")"
  if [ -n "${2:-}" ]; then say "      $(paint 2 "$2")"; fi
}
ui_ok() { say "  $(paint 32 "$MARK_OK") $*"; }
ui_info() { say "      $(paint 2 "$*")"; }

# hush <what it does> <command...>: runs the command with its output in the log and says nothing when it worked; a failure
# shows the end of that log. A dry run prints the command like `run` does. quiet is hush and a check mark for what it did.
hush() {
  label=$1
  shift
  if [ "$DRY_RUN" = 1 ]; then
    printf 'would: %s\n' "$*"
    return 0
  fi
  if [ -z "$LOG_FILE" ]; then LOG_FILE=$(mktemp "${TMPDIR:-/tmp}/aria-install.XXXXXX" 2>/dev/null) || LOG_FILE=""; fi
  log=${LOG_FILE:-/dev/null}
  printf '=== %s: %s\n' "$label" "$*" >>"$log"
  if [ "$1" = sudo ]; then say "  $(paint 33 '!') This step uses sudo $(paint 2 '(it may ask for your password)')"; fi
  "$@" >>"$log" 2>&1 && return 0
  status=$?
  say "  $(paint 31 "$MARK_FAIL") $label" >&2
  if [ -n "$LOG_FILE" ]; then
    KEEP_LOG=1
    tail -n 12 "$LOG_FILE" | sed 's/^/        /' >&2
  fi
  return "$status"
}
quiet() {
  label=$1
  hush "$@" || return $?
  if [ "$DRY_RUN" != 1 ]; then ui_ok "$label"; fi
}

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
# Where the binary and its siblings (the checksum, the pins) are published: the GitHub Releases of ARIA_REPO. The
# environment overrides each address.
REPO=${ARIA_REPO:-i-montes/aria}
VERSION=${ARIA_VERSION:-}
DOWNLOAD_URL=${ARIA_DOWNLOAD_URL:-}
PI_VERSION_URL=${ARIA_PI_VERSION_URL:-}
NODE_VERSION_URL=${ARIA_NODE_VERSION_URL:-}
NODE_DIST=${ARIA_NODE_DIST:-https://nodejs.org/dist}

detect_platform() {
  [ -n "$PLATFORM" ] && return
  platform=$(uname -s 2>/dev/null) || die "could not detect the system: uname is missing"
  case "$platform" in
    Linux)
      artifact=aria-x86_64-linux
      if [ "$HEADLESS" = 1 ]; then
        # A hub's binary has no desktop libraries, and exists for the two CPUs servers have.
        cpu=$(uname -m 2>/dev/null) || cpu=""
        case "$cpu" in
          aarch64 | arm64) artifact=aria-aarch64-linux-headless ;;
          x86_64 | amd64) artifact=aria-x86_64-linux-headless ;;
          *) artifact=aria-x86_64-linux-headless ;; # fetch_binary refuses a CPU that has no build
        esac
      fi
      ;;
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
  if [ -z "$DOWNLOAD_URL" ]; then release_url "$artifact"; fi
  if [ -z "$PI_VERSION_URL" ]; then PI_VERSION_URL=${DOWNLOAD_URL%/*}/pi-version; fi
  if [ -z "$NODE_VERSION_URL" ]; then NODE_VERSION_URL=${DOWNLOAD_URL%/*}/node-version; fi
  if [ "$HEADLESS" = 1 ] && [ "$PLATFORM" != Linux ]; then die "--headless is for Linux servers (this is $PLATFORM); a Mac runs the dock"; fi
}

# The binary's address in the GitHub Releases of $REPO: the latest release, or $VERSION (v-prefixed whether or not it was
# given with the v). Both end up in a URL, so only what a repository name and a version are made of is accepted.
release_url() {
  problem=""
  case "$REPO" in
    "" | .* | */ | /* | */*/* | *./* | */.. | */. | *[!A-Za-z0-9._/-]*) problem="ARIA_REPO must look like owner/name (got '$REPO')" ;;
    */*) ;;
    *) problem="ARIA_REPO must look like owner/name (got '$REPO')" ;;
  esac
  tag=""
  if [ -z "$problem" ] && [ -n "$VERSION" ]; then
    tag=v${VERSION#v}
    case "$tag" in
      v[0-9]*.[0-9]*.[0-9]*) ;;
      *) problem="ARIA_VERSION must look like v0.20.2 or 0.20.2 (got '$VERSION')" ;;
    esac
    case "$tag" in *[!A-Za-z0-9.+-]*) problem="ARIA_VERSION must look like v0.20.2 or 0.20.2 (got '$VERSION')" ;; esac
  fi
  if [ -n "$problem" ]; then
    # A --local install downloads no binary: a stray variable in the environment must not stop it (the address only
    # names the pins a piped installer reads, and falls back to the latest release).
    [ -n "$LOCAL" ] || die "$problem"
    REPO=i-montes/aria
    tag=""
  fi
  if [ -z "$tag" ]; then
    DOWNLOAD_URL=https://github.com/$REPO/releases/latest/download/$1
  else
    DOWNLOAD_URL=https://github.com/$REPO/releases/download/$tag/$1
  fi
}

# curl for what the release serves: https only, redirects included (GitHub sends the files to its own storage hosts), but
# only for the default address. $1 is the environment's override of this one address, if any: an override may be
# a file:// path or a mirror, and is the owner's to choose.
download() {
  override=$1
  shift
  if [ -z "${ARIA_DOWNLOAD_URL:-}" ] && [ -z "$override" ]; then
    curl --proto '=https' --proto-redir '=https' -fsSL "$@"
  else
    curl -fsSL "$@"
  fi
}

# The header of this file is the help.
usage() {
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
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
RestartSec=5

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
  if [ "$DRY_RUN" != 1 ]; then ui_ok "Aria is in $BIN_DIR/aria"; fi
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
    ui_ok "macOS: no system packages are needed"
    return
  fi
  if [ "$DRY_RUN" = 1 ]; then
    # shellcheck disable=SC2086
    run sudo pacman -S --needed $PACKAGES
    return
  fi
  command -v pacman >/dev/null 2>&1 || die "pacman not found: install these packages first: $PACKAGES"
  # shellcheck disable=SC2086
  if ! pacman -Qq $PACKAGES >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    quiet "Installed the libraries the dock needs" sudo pacman -S --needed --noconfirm $PACKAGES
  else
    ui_ok "The libraries the dock needs are already installed"
  fi
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
    set -- sudo pacman -S --needed --noconfirm $HEADLESS_PACKAGES
  else
    say "No apt-get or pacman here: install $HEADLESS_PACKAGES with your package manager (until then every command asks before it runs)"
    return 0
  fi
  if [ "$DRY_RUN" = 1 ]; then
    run "$@"
    return 0
  fi
  if command -v bwrap >/dev/null 2>&1 && command -v ss >/dev/null 2>&1; then
    ui_ok "The sandbox tools are already installed"
    return 0
  fi
  quiet "Installed the sandbox tools ($HEADLESS_PACKAGES)" "$@" || {
    warn "could not install $HEADLESS_PACKAGES (no sudo?)"
    say "Run this yourself: $*"
    say "Until then every command asks before it runs."
    return 0
  }
}

# `systemctl --user` talks to the user manager over $XDG_RUNTIME_DIR/bus, and a non-login ssh command (how a hub is set
# up) has neither XDG_RUNTIME_DIR nor DBUS_SESSION_BUS_ADDRESS. Headless mode only: where they are unset and the user's
# runtime directory exists they are set to what a login would have given. A value already there is kept.
user_session_env() {
  if [ "$DRY_RUN" = 1 ]; then
    say "would: use XDG_RUNTIME_DIR=/run/user/$(id -u) and DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus for systemctl --user, when they are not set (a non-login ssh sets neither)"
    return 0
  fi
  if [ -z "${XDG_RUNTIME_DIR:-}" ] && [ -d "/run/user/$(id -u)" ]; then
    XDG_RUNTIME_DIR=/run/user/$(id -u)
    export XDG_RUNTIME_DIR
  fi
  if [ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ] && [ -n "${XDG_RUNTIME_DIR:-}" ]; then
    DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus
    export DBUS_SESSION_BUS_ADDRESS
  fi
}

# `systemctl --user` on a server; when it fails the reason is nearly always the user manager's bus or linger.
server_systemctl() {
  hush "systemctl --user $*" systemctl --user "$@" || die "systemctl --user $* failed. A server needs dbus-user-session (sudo apt-get install -y dbus-user-session) and linger on (sudo loginctl enable-linger ${USER:-$(id -un)}) for its user manager and bus to be there without a login; then run this again"
}

# Without linger the user manager (and Aria with it) stops when the last login ends.
enable_linger() {
  who=${USER:-$(id -un)}
  if [ "$DRY_RUN" = 1 ]; then
    run loginctl enable-linger "$who"
    return 0
  fi
  loginctl enable-linger "$who" && {
    ui_ok "Aria stays on when you log out (linger)"
    return 0
  }
  warn "could not enable linger, so Aria stops when you log out"
  say "  Run this yourself: sudo loginctl enable-linger $who"
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
    NODE_PINS=$(download "${ARIA_NODE_VERSION_URL:-}" "$NODE_VERSION_URL") || die "could not download the pinned Node version from $NODE_VERSION_URL"
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
  NODE_WORK=$(mktemp -d "$DATA_DIR/.node-install.XXXXXX")
  work=$NODE_WORK
  curl -fsSL "$base/$tarball" -o "$work/$tarball" || die "could not download Node from $base/$tarball"
  [ "$(sha256_of "$work/$tarball")" = "$expected" ] || die "the Node download does not match the checksum pinned in node-version; nothing was installed"
  mkdir "$work/stage"
  tar -xJf "$work/$tarball" -C "$work/stage" --strip-components 1 || die "could not unpack the Node download"
  [ ! -e "$DATA_DIR/node" ] || mv "$DATA_DIR/node" "$work/old"
  mv "$work/stage" "$DATA_DIR/node"
  rm -rf "$work"
  NODE_WORK=""
  ui_ok "Installed Node $NODE_VERSION into $DATA_DIR/node"
}

# A server's Node: the private one (when an earlier run put it there) or the system's, kept when it is new enough, else
# the pinned one is installed under $DATA_DIR/node. Pi and npm then run on it.
ensure_server_node() {
  if [ -x "$DATA_DIR/node/bin/node" ]; then PATH="$DATA_DIR/node/bin:$PATH"; export PATH; fi
  if node_is_current; then
    ui_ok "Node $(node --version | sed 's/^v//') is new enough; keeping it"
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
    PI_VERSION=$(download "${ARIA_PI_VERSION_URL:-}" "$PI_VERSION_URL") || die "could not download the pinned Pi version from $PI_VERSION_URL"
    PI_VERSION=$(printf '%s' "$PI_VERSION" | tr -d '[:space:]')
  fi
  case "$PI_VERSION" in *[!0-9.]* | "") die "the pinned Pi version is not a version number: '$PI_VERSION'" ;; esac
}

install_pi() {
  [ -n "${PI_VERSION:-}" ] || read_pi_version
  prefix="$DATA_DIR/pi/$PI_VERSION"
  if [ -x "$prefix/node_modules/.bin/pi" ]; then
    ui_ok "Pi $PI_VERSION is already installed"
    return
  fi
  quiet "Installed the agent (Pi $PI_VERSION)" npm install --prefix "$prefix" --no-audit --no-fund "$PI_PACKAGE@$PI_VERSION"
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
    packages=$(download "" "${DOWNLOAD_URL%/*}/pi-packages") || die "could not download the list of Pi packages"
  fi
  run install -d -m 700 "$DATA_DIR/agent"
  for package in $packages; do
    quiet "Added $(printf '%s' "$package" | sed 's|^git:github.com/||; s|^npm:||; s|@[^@/]*$||')" env PI_CODING_AGENT_DIR="$DATA_DIR/agent" PI_SKIP_VERSION_CHECK=1 "$DATA_DIR/pi/$PI_VERSION/node_modules/.bin/pi" install "$package"
  done
}

# Sets SOURCE to the binary to install: the --local one, or a downloaded and verified one.
fetch_binary() {
  detect_platform
  if [ -n "$LOCAL" ]; then
    SOURCE=$LOCAL
    return
  fi
  # The dock's Linux binary is for x86_64 and a hub's also for aarch64. Another CPU would verify, install and replace its
  # binary with one that cannot run, so it is refused here (--local, above, and Engram's builds do support arm64).
  if [ "$PLATFORM" = Linux ]; then
    cpu=$(uname -m 2>/dev/null) || die "could not detect the CPU"
    case "$cpu" in
      x86_64 | amd64) ;;
      aarch64 | arm64)
        [ "$HEADLESS" = 1 ] || die "Aria publishes the dock's Linux binary only for x86_64 and this machine is $cpu; nothing was installed
For a server, install it as a hub: sh install.sh --mode hub (a hub has an aarch64 binary)
To build the dock yourself, from a clone of the repository: cargo build --release, then: sh install/install.sh --local target/release/aria" ;;
      *) die "Aria publishes a Linux binary only for x86_64 and aarch64 (a hub) and this machine is $cpu; nothing was installed
To build it yourself, from a clone of the repository: cargo build --release, then: sh install/install.sh --local target/release/aria
For a server: sh tools/build-headless.sh, then: sh install/install.sh --headless --local <the binary it prints>" ;;
    esac
  fi
  if [ "$DRY_RUN" = 1 ]; then
    say "would: download $DOWNLOAD_URL and verify it against $DOWNLOAD_URL.sha256"
    SOURCE=/tmp/aria-download
    return
  fi
  TMP=$(mktemp -d)
  # Nothing is installed from here: the download goes to a scratch directory that any exit removes, and install_binary
  # runs only after the checksum matched. The checksum is the release's own file, next to the binary: it shows the
  # download is whole and is the one that was published, not that the release is authentic (nothing is signed).
  if ! download "" "$DOWNLOAD_URL" -o "$TMP/aria"; then
    die "could not download the Aria binary from $DOWNLOAD_URL
There may be no binary for this platform: a release has one only if it was built for it, and macOS binaries are not published for every release (or there is no release yet). Nothing was installed.
To build it yourself, from a clone of the repository: cargo build --release, then: sh install/install.sh --local target/release/aria"
  fi
  download "" "$DOWNLOAD_URL.sha256" -o "$TMP/aria.sha256" || die "could not download the checksum from $DOWNLOAD_URL.sha256; nothing was installed"
  expected=$(cut -d ' ' -f 1 "$TMP/aria.sha256")
  actual=$(sha256_of "$TMP/aria")
  [ "$expected" = "$actual" ] || die "the downloaded binary does not match its checksum; nothing was installed
Hint: a release may have been published while this ran: run the command again"
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
    pins=$(download "" "${DOWNLOAD_URL%/*}/engram-version") || die "could not download the pinned Engram version"
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
    ui_ok "Engram $version is already installed"
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
  ui_ok "Installed Engram $version"
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
    version=$(download "" "${DOWNLOAD_URL%/*}/agent-browser-version") || {
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
    ui_ok "agent-browser $version is already installed"
  else
    quiet "Installed agent-browser $version" npm install --prefix "$prefix" --no-audit --no-fund "agent-browser@$version" || {
      warn "could not install agent-browser $version; the agents cannot check pages in a browser"
      return 0
    }
  fi
  { run install -d -m 700 "$DATA_DIR/bin" && run ln -sf "$prefix/node_modules/.bin/agent-browser" "$DATA_DIR/bin/agent-browser"; } || {
    warn "could not link agent-browser into $DATA_DIR/bin"
    return 0
  }
  quiet "Downloaded the browser agent-browser uses" "$DATA_DIR/bin/agent-browser" install || warn "could not install agent-browser's browser; the browser check will not work until it is installed (run: ~/.aria/bin/agent-browser install)"
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
  if [ "$DRY_RUN" != 1 ]; then ui_ok "The service that starts Aria is set up"; fi
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
    ui_ok "Aria is installed. Press Option + Space to set it up."
  elif [ "$HEADLESS" = 1 ]; then
    user_session_env
    server_systemctl daemon-reload
    server_systemctl enable --now "$SERVE_UNIT"
    if [ "$1" = 1 ] && soft_restart; then return; fi
    if [ "$1" = 1 ]; then server_systemctl restart "$SERVE_UNIT"; fi
    ui_ok "Aria is installed and running as $SERVE_UNIT"
  else
    hush "systemctl --user daemon-reload" systemctl --user daemon-reload
    hush "systemctl --user enable --now $UNIT" systemctl --user enable --now "$UNIT"
    if [ "$1" = 1 ] && soft_restart; then return; fi
    # An update replaces the binary under a running dock: restart it so the new one runs.
    if [ "$1" = 1 ]; then hush "systemctl --user restart $UNIT" systemctl --user restart "$UNIT"; fi
    ui_ok "Aria is installed. Press Super+A to set it up."
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

# --- the use: this computer's dock, or a hub ---------------------------------------------------------------------------

# Which use an existing install is for, from the unit it left: hub, desktop, or nothing.
installed_use() {
  if [ -f "$UNIT_DIR/$SERVE_UNIT" ]; then echo hub
  elif [ -f "$UNIT_DIR/$UNIT" ] || [ -f "$AGENT_DIR/$AGENT" ]; then echo desktop
  fi
}

# True when there is a terminal to ask on: not a dry run, not --yes, and /dev/tty opens (under `curl | sh` stdin is the
# script, so the question goes to the terminal itself).
can_ask() {
  [ "$DRY_RUN" != 1 ] && [ "$ASSUME_YES" != 1 ] && ( : <"$TTY_IN" ) 2>/dev/null
}

# A machine with a desktop session is a computer to use; one without is a server.
default_use() {
  if [ -n "${HYPRLAND_INSTANCE_SIGNATURE:-}${WAYLAND_DISPLAY:-}${DISPLAY:-}" ]; then echo desktop; else echo hub; fi
}

# Reads one line of the answer from the terminal.
# The terminal is opened once and read as a stream, so each question gets the next line (can_ask checked that it opens).
TTY_OPEN=0
tty_read() {
  if [ "$TTY_OPEN" != 1 ]; then
    exec 3<"$TTY_IN"
    TTY_OPEN=1
  fi
  answer=""
  read -r answer <&3 || answer=""
}
tty_say() { printf '%s\n' "$*" >>"$TTY_OUT"; }

# The first question. Sets USE (desktop or hub); Enter takes the default, anything else but a choice asks again.
ask_use() {
  default=$1
  if [ "$default" = hub ]; then number=2; else number=1; fi
  tries=0
  while :; do
    tty_say ""
    tty_say "  $(paint 1 'How will you use Aria?')"
    tty_say ""
    tty_say "    $(paint '1;36' 1)  On this computer"
    tty_say "       the dock: a window that opens over what you are doing,"
    tty_say "       with its own shortcut"
    tty_say ""
    tty_say "    $(paint '1;36' 2)  As a hub"
    tty_say "       a server that keeps Aria running day and night,"
    tty_say "       which you connect to from your computers"
    tty_say ""
    printf '  Type 1 or 2, then Enter [%s]: ' "$number" >>"$TTY_OUT"
    tty_read
    case "$answer" in
      "") USE=$default; return ;;
      1 | desktop | local) USE=desktop; return ;;
      2 | hub) USE=hub; return ;;
    esac
    tries=$((tries + 1))
    [ "$tries" -lt 3 ] || die "answer 1 (this computer) or 2 (a hub), or pass --mode desktop|hub"
  done
}

# True for something that can follow `ssh`: user@host, with nothing a shell or ssh would read as more (no space, no leading dash).
looks_like_server() {
  case "$1" in
    -* | "" | *[!A-Za-z0-9._@:%+-]* | *@*@* | @* | *@) return 1 ;;
    *@*) return 0 ;;
    *) return 1 ;;
  esac
}

# The second question, for a hub: which computer is the server. Sets SERVER (here, or user@host).
ask_server() {
  default=$1 # here or other
  if [ "$default" = here ]; then number=2; else number=1; fi
  tries=0
  while :; do
    tty_say ""
    tty_say "  $(paint 1 'Which computer is the server?')"
    tty_say ""
    tty_say "    $(paint '1;36' 1)  Another one"
    tty_say "       you give me its address (like ana@my-server); I set it up"
    tty_say "       from here and link this computer to it"
    tty_say ""
    tty_say "    $(paint '1;36' 2)  This one"
    tty_say "       this computer is the server; other computers connect to it"
    tty_say ""
    printf '  Type 1 or 2, then Enter [%s]: ' "$number" >>"$TTY_OUT"
    tty_read
    case "$answer" in
      "") if [ "$default" = here ]; then SERVER=here; else SERVER=ask; fi; break ;;
      1) SERVER=ask; break ;;
      2 | here) SERVER=here; break ;;
    esac
    tries=$((tries + 1))
    [ "$tries" -lt 3 ] || die "answer 1 (another computer) or 2 (this one), or pass --server user@host or --server here"
  done
  [ "$SERVER" = ask ] || return 0
  tries=0
  while :; do
    tty_say ""
    printf '  Its ssh address, like ana@my-server (a name or an IP): ' >>"$TTY_OUT"
    tty_read
    if looks_like_server "$answer"; then SERVER=$answer; return 0; fi
    tty_say "  That does not look like user@host."
    tries=$((tries + 1))
    [ "$tries" -lt 3 ] || die "give the server as user@host, for example: --server ana@my-server"
  done
}

use_words() {
  case "$1" in
    hub) echo "a hub on this server" ;;
    here) echo "a hub on this server" ;;
    desktop) echo "this computer's dock" ;;
    *) echo "this computer and the server $1" ;;
  esac
}

# Settles the use before anything is detected or installed: USE (desktop or hub), SERVER (here, or user@host for a hub
# that lives on another computer), and HEADLESS for a hub on this one.
resolve_mode() {
  choice=$CHOSEN_MODE
  SERVER=$CHOSEN_SERVER
  case "$choice" in "" | desktop | hub) ;; *) die "the mode is desktop or hub (got '$choice')" ;; esac
  if [ -n "$SERVER" ] && [ "$SERVER" != here ] && ! looks_like_server "$SERVER"; then die "--server is here, or user@host (got '$SERVER')"; fi
  if [ "$HEADLESS" = 1 ]; then
    [ "$choice" != desktop ] || die "--headless is a hub and --mode desktop is the dock: give one"
    choice=hub
    [ -n "$SERVER" ] || SERVER=here
  fi
  # A server is a hub, whatever else was said.
  if [ -n "$SERVER" ]; then
    [ "$choice" != desktop ] || die "--server is for a hub: --mode hub, or leave --mode out"
    choice=hub
  fi
  if [ -z "$choice" ]; then
    if [ "$(uname -s 2>/dev/null)" = Darwin ]; then
      choice=desktop
    else
      choice=$(installed_use)
      if [ -n "$choice" ]; then
        say "  Aria is already installed here for $(use_words "$choice"): updating it (--mode changes that)"
        [ "$choice" != hub ] || SERVER=here
      else
        default=$(default_use)
        if can_ask; then
          ask_use "$default"
          choice=$USE
        else
          choice=$default
          say "  Installing for $(use_words "$choice") (nothing to ask on; --mode desktop|hub chooses)"
        fi
      fi
    fi
  fi
  if [ "$choice" = hub ] && [ -z "$SERVER" ]; then
    if can_ask; then
      if [ "$(default_use)" = hub ]; then ask_server here; else ask_server other; fi
    else
      SERVER=here
    fi
  fi
  USE=$choice
  if [ "$choice" = hub ] && [ "$SERVER" = here ]; then HEADLESS=1; fi
}

# What is about to happen, in the words of someone who has not read the docs; Enter to go on when there is a terminal.
ui_plan() {
  [ "$NESTED" != 1 ] || return 0
  if [ "$USE" = hub ] && [ "$SERVER" = here ]; then
    ui_title "setting up a hub on this server"
    say "  This server will keep Aria running day and night."
    say "  You use it from your computers."
  elif [ "$USE" = hub ]; then
    ui_title "setting up Aria here and a hub on $SERVER"
    say "  Aria goes on this computer, then on $SERVER, which keeps it"
    say "  running day and night. This computer is linked to it."
  else
    ui_title "setting up Aria on this computer"
    say "  Aria opens over whatever you are doing, and you ask it to work"
    say "  in your language."
  fi
  say ""
  say "  $(paint 2 "$STEP_TOTAL steps. Nothing changes outside your home folder except")"
  say "  $(paint 2 'system packages (it asks for your password for those).')"
  if [ "$USE" = hub ] && [ "$SERVER" = here ]; then
    say "  $(paint 2 'It also stays on after you log out, and asks before it')"
    say "  $(paint 2 'changes the sandbox rules.')"
  fi
  if can_ask; then
    printf '\n  Press Enter to start, or Ctrl+C to cancel. ' >>"$TTY_OUT"
    tty_read
  fi
}

# --- the sandbox on a server ------------------------------------------------------------------------------------------

# Where things are on Ubuntu; the environment overrides them for the tests (and nothing else).
BWRAP_PROFILE=${ARIA_BWRAP_PROFILE:-/etc/apparmor.d/aria-bwrap}
USERNS_SYSCTL=${ARIA_USERNS_SYSCTL:-/proc/sys/kernel/apparmor_restrict_unprivileged_userns}
APPARMOR_ABI=${ARIA_APPARMOR_ABI:-/etc/apparmor.d/abi/4.0}
# The terminal the questions use: /dev/tty, so they work under `curl | sh` (the tests give a file for each direction).
TTY_IN=${ARIA_TTY_IN:-/dev/tty}
TTY_OUT=${ARIA_TTY_OUT:-/dev/tty}

# The probe the hub's own check uses: a namespace, with /dev and /proc mounted like the real call.
sandbox_works() {
  command -v bwrap >/dev/null 2>&1 && bwrap --unshare-user --ro-bind / / --dev /dev --proc /proc true >/dev/null 2>&1
}

# Ubuntu's own shape of a profile that lets one program make user namespaces: bwrap alone, nothing else changes.
print_bwrap_profile() {
  cat <<PROFILE
abi <abi/4.0>,
include <tunables/global>

profile aria-bwrap $1 flags=(unconfined) {
  userns,

  include if exists <local/aria-bwrap>
}
PROFILE
}

# Makes the sandbox start on a server. The usual block is Ubuntu 24.04 restricting user namespaces; the fix is a profile
# for bwrap alone, written with sudo only on a yes (or --yes), and never the system-wide sysctl. Never fatal: without a
# sandbox every command asks first, and the installer says how to fix it.
ensure_sandbox() {
  if [ "$DRY_RUN" = 1 ]; then
    say "would: check that the sandbox starts, and offer an AppArmor profile for bwrap alone ($BWRAP_PROFILE) if Ubuntu blocks it"
    return 0
  fi
  command -v bwrap >/dev/null 2>&1 || return 0 # the packages step already said what to run
  if sandbox_works; then
    ui_ok "The sandbox works"
    return 0
  fi
  restricted=$(cat "$USERNS_SYSCTL" 2>/dev/null || echo 0)
  if [ "$restricted" != 1 ] || [ ! -f "$APPARMOR_ABI" ]; then
    warn "the sandbox does not start, and it is not Ubuntu's user-namespace restriction"
    say "  Aria cannot fix that: every command will ask before it runs."
    return 0
  fi
  path=$(readlink -f "$(command -v bwrap)" 2>/dev/null || command -v bwrap)
  say ""
  say "  The sandbox cannot start: this Ubuntu stops programs from making"
  say "  user namespaces. The fix is a profile in $BWRAP_PROFILE"
  say "  that lets bwrap alone make them. Nothing else changes."
  consent=n
  if [ "$ASSUME_YES" = 1 ]; then
    consent=y
  elif can_ask; then
    printf '  Write that profile (needs sudo)? [Y/n] ' >>"$TTY_OUT"
    tty_read
    consent=${answer:-y}
  fi
  case "$consent" in
    y | Y | yes)
      if print_bwrap_profile "$path" | sudo tee "$BWRAP_PROFILE" >/dev/null && sudo apparmor_parser -r "$BWRAP_PROFILE"; then
        if sandbox_works; then
          ui_ok "The sandbox works now"
        else
          warn "the profile is in place and the sandbox still does not start"
          say "  Every command will ask before it runs."
        fi
      else
        warn "could not write or load $BWRAP_PROFILE (no sudo?)"
        say "  Every command will ask before it runs. To fix it yourself, an"
        say "  administrator writes what 'sh install.sh --print-bwrap-profile'"
        say "  prints to that file and runs: sudo apparmor_parser -r $BWRAP_PROFILE"
      fi
      ;;
    *)
      say "  Left as it is: every command asks before it runs."
      say "  To fix it later, run this installer again and say yes."
      ;;
  esac
}

# True when this computer is linked to a hub (aria hub init remembers it in hubs.json): then an install here also brings
# that server to the version just installed, so the two never drift apart.
has_linked_hubs() {
  [ "$HEADLESS" != 1 ] && [ "$NESTED" != 1 ] && [ -f "$DATA_DIR/hubs.json" ] && grep -q '"key"' "$DATA_DIR/hubs.json" 2>/dev/null
}

update_linked_hubs() {
  has_linked_hubs || return 0
  [ -z "$SERVER" ] || [ "$SERVER" = here ] || return 0 # a server just named is set up by link_to_server
  ui_step "Updating your server" "Brings it to the version you just installed."
  if [ "$DRY_RUN" = 1 ]; then
    if [ "$ASSUME_YES" = 1 ]; then say "would: $BIN_DIR/aria hub update --all --yes"; else say "would: $BIN_DIR/aria hub update --all"; fi
    return 0
  fi
  if [ "$ASSUME_YES" = 1 ]; then set -- --all --yes; else set -- --all; fi
  "$BIN_DIR/aria" hub update "$@" || {
    warn "Aria is installed on this computer, but the server was not updated"
    say "  Run this to finish it (it is safe to run again): aria hub update"
    exit 1
  }
}

# Sets up the server another computer was named for: the hub is installed there and this computer is linked to it, by the
# Aria that was just installed here (`aria hub init`), which also tries the connection.
link_to_server() {
  [ -n "$SERVER" ] && [ "$SERVER" != here ] || return 0
  ui_step "Setting up the server $SERVER" "Installs the hub there, links this computer, tests the connection."
  if [ "$DRY_RUN" = 1 ]; then
    say "would: $BIN_DIR/aria hub init $SERVER --yes"
    return 0
  fi
  "$BIN_DIR/aria" hub init "$SERVER" --yes || {
    warn "Aria is installed on this computer, but the server $SERVER is not ready"
    say "Run this to finish it (it is safe to run again): aria hub init $SERVER"
    exit 1
  }
}

# The address a computer reaches this server by: the Tailscale one when there is one (it works from every device of the
# tailnet), else the address this very ssh login came to (SSH_CONNECTION: client, port, server, port), else its name.
server_address() {
  if command -v tailscale >/dev/null 2>&1; then
    found=$(tailscale ip -4 2>/dev/null | head -1)
    if [ -n "$found" ]; then echo "$found"; return; fi
  fi
  if [ -n "${SSH_CONNECTION:-}" ]; then
    # shellcheck disable=SC2086
    set -- $SSH_CONNECTION
    if [ -n "${3:-}" ]; then echo "$3"; return; fi
  fi
  hostname -f 2>/dev/null || hostname 2>/dev/null || echo this-server
}

# The last screen: what is ready, and the one thing to do to use it. Never a setup step, except the one a server cannot do
# for a computer it has never seen.
ui_done() {
  [ "$DRY_RUN" != 1 ] || return 0
  [ "$NESTED" != 1 ] || return 0
  say ""
  ui_rule
  if [ "$HEADLESS" = 1 ]; then
    host=$(server_address)
    say "  $(paint '1;32' "$MARK_OK Aria is running on this server.")"
    say ""
    say "  One thing is left, and only a computer you will use can do it:"
    say "  this server has never seen it. On that computer, with Aria"
    say "  installed, run:"
    say ""
    say "      $(paint 1 "aria hub init ${USER:-$(id -un)}@$host")"
    say ""
    say "  $(paint 2 "(if you reach this server by another name, use that one)")"
  elif [ -n "$SERVER" ] && [ "$SERVER" != here ]; then
    say "  $(paint '1;32' "$MARK_OK Aria is ready on this computer and on $SERVER.")"
    say ""
    say "  To use the server, open a terminal and type:  $(paint 1 'aria connect')"
    if [ "$PLATFORM" = Darwin ]; then
      say "  To use Aria on this computer: press $(paint 1 'Option + Space')."
    else
      say "  To use Aria on this computer: press $(paint 1 'Super + A')."
    fi
  else
    say "  $(paint '1;32' "$MARK_OK Aria is ready.")"
    say ""
    if has_linked_hubs; then
      say "  Your server is on the same version. To use it, open a terminal"
      say "  and type:  $(paint 1 'aria connect')"
      say ""
    fi
    if [ "$PLATFORM" = Darwin ]; then
      say "  To use it, press $(paint 1 'Option + Space')."
      say "  The first time it walks you through connecting a model."
    else
      say "  To use it, press $(paint 1 'Super + A')."
      say "  The first time it walks you through connecting a model."
    fi
  fi
  say ""
}

do_install() {
  detect_platform
  # Refuse a bad --local before anything else, so a mistake changes nothing.
  if [ -n "$LOCAL" ] && [ "$DRY_RUN" != 1 ] && [ ! -x "$LOCAL" ]; then die "$LOCAL is not an executable file"; fi
  if [ -n "$SERVER" ] && [ "$SERVER" != here ]; then STEP_TOTAL=7; fi
  if has_linked_hubs && { [ -z "$SERVER" ] || [ "$SERVER" = here ]; }; then STEP_TOTAL=7; fi
  ui_plan
  if [ "$HEADLESS" = 1 ]; then where="this server"; else where="this computer"; fi
  ui_step "Checking $where" "That Aria can run here, and which Node and Pi it will use."
  check_session
  if [ "$HEADLESS" = 1 ]; then ensure_server_node; else check_node; fi
  read_pi_version # before anything is installed: a version we cannot read must not leave a half install
  ui_ok "Aria can run on $where"
  if [ "$HEADLESS" = 1 ]; then
    ui_step "Installing what Aria needs" "The sandbox that boxes what the agent runs, and its tools."
    install_headless_packages
    ensure_sandbox
  else
    ui_step "Installing what Aria needs" "The dock's libraries, and the sandbox that boxes what the agent runs."
    install_packages
  fi
  was_active=0
  if [ "$HEADLESS" = 1 ]; then user_session_env; fi
  if [ "$DRY_RUN" != 1 ]; then
    if [ "$PLATFORM" = Darwin ]; then
      launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1 && was_active=1
    elif [ "$HEADLESS" = 1 ]; then
      systemctl --user is-active --quiet "$SERVE_UNIT" 2>/dev/null && was_active=1
    else
      systemctl --user is-active --quiet "$UNIT" 2>/dev/null && was_active=1
    fi
  fi
  ui_step "Downloading Aria" "From the Aria release, checked against its checksum."
  fetch_binary
  if [ "$DRY_RUN" != 1 ]; then
    if [ -n "$LOCAL" ]; then ui_ok "Using the program you gave: $LOCAL"; else ui_ok "Downloaded and checked"; fi
  fi
  # Pi and its packages first: if a download fails, nothing has been replaced and the running dock stays as it was.
  ui_step "Installing the agent and its memory" "Pi (the agent), its extensions, and Engram (long-term memory)."
  run install -d -m 700 "$DATA_DIR"
  link_node
  install_pi
  install_pi_packages
  install_engram
  if [ "$HEADLESS" = 1 ] && [ "$WITH_BROWSER" != 1 ]; then
    ui_info "No browser check on a server (a large download): --with-browser adds it"
  else
    install_agent_browser
  fi
  ui_step "Installing Aria" "The program itself, and the service that starts it for you."
  install_binary
  install_bar_widget
  write_unit
  if [ "$HEADLESS" = 1 ]; then enable_linger; fi
  ui_step "Starting Aria" ""
  restart_dock "$was_active"
  link_to_server
  update_linked_hubs
  ui_done
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
    --with-browser) WITH_BROWSER=1 ;;
    --yes | -y) ASSUME_YES=1 ;;
    --mode)
      [ $# -ge 2 ] || die "--mode needs desktop or hub"
      CHOSEN_MODE=$2
      shift
      ;;
    --server)
      [ $# -ge 2 ] || die "--server needs here, or the server as user@host"
      CHOSEN_SERVER=$2
      shift
      ;;
    --headless) HEADLESS=1 ;; # give it before --restart-dock, the hook that runs the moment it is read
    --local)
      [ $# -ge 2 ] || die "--local needs the path of a binary"
      LOCAL=$2
      shift
      ;;
    --uninstall) MODE=uninstall ;;
    --print-unit) MODE=print-unit ;;
    --print-bwrap-profile) MODE=print-bwrap-profile ;;
    --ensure-sandbox) MODE=ensure-sandbox ;; # test hook
    --resolve-mode) MODE=resolve-mode ;; # test hook: print the use and the server that were settled
    --print-address) MODE=print-address ;; # test hook: the address a computer would link to this server by
    --check-node) MODE=check-node ;; # test hooks
    --print-pi-version) MODE=print-pi-version ;;
    --link-node) MODE=link-node ;;
    --install-engram) MODE=install-engram ;; # test hook
    --install-node) MODE=install-node ;; # test hooks for the server's steps
    --install-packages) MODE=install-packages ;;
    --enable-linger) MODE=enable-linger ;;
    --print-node-version) MODE=print-node-version ;;
    --install-agent-browser) MODE=install-agent-browser ;; # test hook
    --fetch-binary) MODE=fetch-binary ;; # test hook: download and verify the binary, print its checksum
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
  install) resolve_mode; do_install ;;
  uninstall)
    # Remove what is installed: a hub's unit and a dock's differ, and nobody is asked which.
    if [ "$HEADLESS" != 1 ] && [ "$(installed_use)" = hub ]; then HEADLESS=1; fi
    do_uninstall
    ;;
  print-bwrap-profile) print_bwrap_profile /usr/bin/bwrap ;;
  ensure-sandbox) ensure_sandbox ;;
  print-address) server_address ;;
  resolve-mode) resolve_mode; say "use=$USE server=${SERVER:-none} headless=$HEADLESS" ;;
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
  fetch-binary) fetch_binary; if [ "$DRY_RUN" != 1 ]; then say "fetched $(sha256_of "$SOURCE")"; fi ;;
esac
