#!/bin/sh
# Aria installer for Omarchy (Hyprland with the Lua config) and macOS.
#
#   curl -fsSL https://<your site>/install.sh | sh     install, or update in place
#   sh install.sh --local target/release/aria          install a binary you built yourself
#   sh install.sh --uninstall                          remove Aria (asks before deleting ~/.aria)
#   sh install.sh --dry-run [...]                      print what would happen and change nothing
#   sh install.sh --restart-now [...]                  restart a running dock at once, even mid-run
#
# An update does not cut a run: the running dock is told a new version is installed and restarts into it once it
# is hidden and no session has a run going. Only a changed service file, --restart-now, or a dock too old to
# understand that restarts it at once.
#
# It detects the platform. On Omarchy it installs Aria's system libraries and a systemd user unit that
# starts the dock at login; on macOS it installs a LaunchAgent and needs no system packages. Both put
# the `aria` binary in ~/.local/bin, create ~/.aria, and install the pinned Pi under
# ~/.aria/pi/<version> (Node 22.19 or newer is required). On Omarchy it never writes to
# ~/.config/hypr/: the dock registers SUPER + A with Hyprland itself, every time it starts. On macOS
# the shortcut is Option + Space, registered by the dock with Carbon (no Accessibility permission).
set -eu

BIN_DIR=${ARIA_BIN_DIR:-$HOME/.local/bin}
DATA_DIR=${ARIA_HOME:-$HOME/.aria}
UNIT_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
UNIT=aria.service
AGENT_DIR=$HOME/Library/LaunchAgents
AGENT=com.aria.dock.plist
LABEL=com.aria.dock
# The dock's toolkit, plus bubblewrap (the sandbox runs every command in bwrap), iproute2 (the process guard tells the
# model to find a port's process with `ss`) and libsecret (`secret-tool`, how Aria keeps keys in the keyring). bun is not
# needed: only the checks use it.
PACKAGES="gtk3 webkit2gtk-4.1 gtk-layer-shell bubblewrap iproute2 libsecret"

# Only a real file has siblings: under `curl | sh` $0 is just "sh" and the current directory is not ours.
SCRIPT_DIR=""
if [ -f "$0" ]; then SCRIPT_DIR=$(cd "$(dirname "$0")" 2>/dev/null && pwd) || SCRIPT_DIR=""; fi
PI_PACKAGE=@earendil-works/pi-coding-agent

DRY_RUN=0
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
}

usage() {
  sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
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
ExecStart=$exec_path
Restart=on-failure

[Install]
WantedBy=graphical-session.target
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
  <key>ProgramArguments</key><array><string>$BIN_DIR/aria</string></array>
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
  else
    run install -Dm755 "$SOURCE" "$BIN_DIR/aria"
  fi
}

print_service() {
  detect_platform
  if [ "$PLATFORM" = Darwin ]; then print_agent; else print_unit; fi
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
  if [ "$DRY_RUN" = 1 ]; then
    say "would: link $DATA_DIR/bin/node to the Node on your PATH (and claude, when Claude Code is installed)"
  else
    ln -sf "$(command -v node)" "$DATA_DIR/bin/node"
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
  write_service_file "$UNIT_DIR/$UNIT" "$(print_unit)"
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
  else
    run systemctl --user daemon-reload
    run systemctl --user enable --now "$UNIT"
    if [ "$1" = 1 ] && soft_restart; then return; fi
    # An update replaces the binary under a running dock: restart it so the new one runs.
    if [ "$1" = 1 ]; then run systemctl --user restart "$UNIT"; fi
    say "Aria is installed. Press Super+A to set it up."
  fi
}

do_install() {
  detect_platform
  # Refuse a bad --local before anything else, so a mistake changes nothing.
  if [ -n "$LOCAL" ] && [ "$DRY_RUN" != 1 ] && [ ! -x "$LOCAL" ]; then die "$LOCAL is not an executable file"; fi
  check_session
  check_node
  read_pi_version # before anything is installed: a version we cannot read must not leave a half install
  install_packages
  was_active=0
  if [ "$DRY_RUN" != 1 ]; then
    if [ "$PLATFORM" = Darwin ]; then
      launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1 && was_active=1
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
  write_unit
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
  else
    run systemctl --user disable --now "$UNIT" || true
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
  install-agent-browser) install_agent_browser ;;
esac
