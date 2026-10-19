#!/bin/sh
# Aria installer for Omarchy (Hyprland with the Lua config).
#
#   curl -fsSL https://<your site>/install.sh | sh     install, or update in place
#   sh install.sh --local target/release/aria          install a binary you built yourself
#   sh install.sh --uninstall                          remove Aria (asks before deleting ~/.aria)
#   sh install.sh --dry-run [...]                      print what would happen and change nothing
#
# It installs Aria's system libraries, puts the `aria` binary in ~/.local/bin, creates ~/.aria, installs
# the pinned Pi under ~/.aria/pi/<version> (Node 22.19 or newer is required) and a
# systemd user unit that starts the dock at login. It never writes to ~/.config/hypr/: the dock
# registers SUPER + A with Hyprland itself, every time it starts.
set -eu

# Placeholders until releases are published; override with the environment.
DOWNLOAD_URL=${ARIA_DOWNLOAD_URL:-https://aria.example/releases/latest/aria-x86_64-linux}
BIN_DIR=${ARIA_BIN_DIR:-$HOME/.local/bin}
DATA_DIR=${ARIA_HOME:-$HOME/.aria}
UNIT_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
UNIT=aria.service
PACKAGES="gtk3 webkit2gtk-4.1 gtk-layer-shell"

# Only a real file has siblings: under `curl | sh` $0 is just "sh" and the current directory is not ours.
SCRIPT_DIR=""
if [ -f "$0" ]; then SCRIPT_DIR=$(cd "$(dirname "$0")" 2>/dev/null && pwd) || SCRIPT_DIR=""; fi
PI_VERSION_URL=${ARIA_PI_VERSION_URL:-${DOWNLOAD_URL%/*}/pi-version}
PI_PACKAGE=@earendil-works/pi-coding-agent

DRY_RUN=0
LOCAL=""
MODE=install
TMP=""

say() { printf '%s\n' "$*"; }
die() { printf 'aria installer: %s\n' "$*" >&2; exit 1; }

# Runs a command, or only prints it in a dry run.
run() {
  if [ "$DRY_RUN" = 1 ]; then
    printf 'would: %s\n' "$*"
  else
    "$@"
  fi
}

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
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

check_session() {
  if [ "$DRY_RUN" = 1 ]; then
    say "would: check this is a Hyprland session with the Lua config (hyprctl eval 'return 1' prints ok)"
    return
  fi
  [ -n "${HYPRLAND_INSTANCE_SIGNATURE:-}" ] || die "run this from inside a Hyprland session (Omarchy)"
  [ "$(hyprctl eval 'return 1' 2>/dev/null)" = "ok" ] || die "Aria needs Hyprland 0.56 or newer with the Lua configuration"
}

install_packages() {
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

# Sets SOURCE to the binary to install: the --local one, or a downloaded and verified one.
fetch_binary() {
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
  actual=$(sha256sum "$TMP/aria" | cut -d ' ' -f 1)
  [ "$expected" = "$actual" ] || die "the downloaded binary does not match its checksum; nothing was installed"
  chmod +x "$TMP/aria"
  SOURCE=$TMP/aria
}

write_unit() {
  run mkdir -p "$UNIT_DIR"
  if [ "$DRY_RUN" = 1 ]; then
    say "would: write $UNIT_DIR/$UNIT"
  else
    print_unit > "$UNIT_DIR/$UNIT"
  fi
}

do_install() {
  # Refuse a bad --local before anything else, so a mistake changes nothing.
  if [ -n "$LOCAL" ] && [ "$DRY_RUN" != 1 ] && [ ! -x "$LOCAL" ]; then die "$LOCAL is not an executable file"; fi
  check_session
  check_node
  read_pi_version # before anything is installed: a version we cannot read must not leave a half install
  install_packages
  was_active=0
  if [ "$DRY_RUN" != 1 ] && systemctl --user is-active --quiet "$UNIT" 2>/dev/null; then was_active=1; fi
  fetch_binary
  run install -Dm755 "$SOURCE" "$BIN_DIR/aria"
  run install -d -m 700 "$DATA_DIR"
  link_node
  install_pi
  write_unit
  run systemctl --user daemon-reload
  run systemctl --user enable --now "$UNIT"
  # An update replaces the binary under a running dock: restart it so the new one runs.
  if [ "$was_active" = 1 ]; then run systemctl --user restart "$UNIT"; fi
  say "Aria is installed. Press Super+A to set it up."
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
  run systemctl --user disable --now "$UNIT" || true
  run rm -f "$UNIT_DIR/$UNIT" "$BIN_DIR/aria"
  run systemctl --user daemon-reload
  if [ "$DRY_RUN" = 1 ] || [ -n "${HYPRLAND_INSTANCE_SIGNATURE:-}" ]; then
    run hyprctl eval 'hl.unbind("SUPER + A")' || true
  fi
  ask_delete_data
  say "Aria is removed."
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
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
  print-unit) print_unit ;;
  check-node) check_node ;;
  print-pi-version) read_pi_version; printf '%s\n' "$PI_VERSION" ;;
  link-node) link_node ;;
esac
