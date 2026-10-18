#!/bin/sh
# Aria installer for Omarchy (Hyprland with the Lua config).
#
#   curl -fsSL https://<your site>/install.sh | sh     install, or update in place
#   sh install.sh --local target/release/aria          install a binary you built yourself
#   sh install.sh --uninstall                          remove Aria (asks before deleting ~/.aria)
#   sh install.sh --dry-run [...]                      print what would happen and change nothing
#
# It installs Aria's system libraries, puts the `aria` binary in ~/.local/bin, creates ~/.aria and a
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
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
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
  install_packages
  was_active=0
  if [ "$DRY_RUN" != 1 ] && systemctl --user is-active --quiet "$UNIT" 2>/dev/null; then was_active=1; fi
  fetch_binary
  run install -Dm755 "$SOURCE" "$BIN_DIR/aria"
  run install -d -m 700 "$DATA_DIR"
  write_unit
  run systemctl --user daemon-reload
  run systemctl --user enable --now "$UNIT"
  # An update replaces the binary under a running dock: restart it so the new one runs.
  if [ "$was_active" = 1 ]; then run systemctl --user restart "$UNIT"; fi
  say "Aria is installed. Press Super+A to set it up."
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
      if [ -n "$DATA_DIR" ] && [ "$DATA_DIR" != "/" ] && [ "$DATA_DIR" != "$HOME" ]; then rm -rf "$DATA_DIR"; say "Deleted $DATA_DIR"; fi
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
    -h | --help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
  shift
done

case "$MODE" in
  install) do_install ;;
  uninstall) do_uninstall ;;
  print-unit) print_unit ;;
esac
