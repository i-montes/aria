#!/bin/sh
# A stand-in for ssh in the `aria hub init` tests (ARIA_SSH points here). It talks to no network and reads no ssh
# configuration. Every remote command is recorded, one line each, in the file named by ARIA_FAKE_SSH_LOG:
#   <host><TAB><command><TAB>stdin=<bytes>
# The bytes of stdin are never recorded (a bundle holds credentials), only how many there were. `ssh -G <host>` is
# recorded as `-G <host>`. Answers are canned, chosen by the command; env vars select variants:
#   ARIA_FAKE_SSH_ARCH         the answer of `uname -m` (default aarch64)
#   ARIA_FAKE_SSH_PUBLIC       0: sshd listens on a Tailscale address only (default: on every address)
#   ARIA_FAKE_SSH_MISSING      names `command -v` does not find (default none): xz bwrap node npm tailscale
#   ARIA_FAKE_SSH_NODE         the answer of `node --version` (default v18.19.1)
#   ARIA_FAKE_SSH_BWRAP_FAIL   1: the sandbox probe fails
#   ARIA_FAKE_SSH_IMPORT_FAIL  1: `aria hub-import` fails and complains on stderr, quoting the start of what it read
#   ARIA_FAKE_SSH_PASSWORDS    yes: sshd_config allows password logins (default no)
# ARIA_FAKE_SSH_ROOT is the hub's computer: the upload is unpacked, and the key is added to authorized_keys, under
# $ARIA_FAKE_SSH_ROOT/home, the only place a command is really run.
log=${ARIA_FAKE_SSH_LOG:?ARIA_FAKE_SSH_LOG}
root=${ARIA_FAKE_SSH_ROOT:?ARIA_FAKE_SSH_ROOT}

if [ "$1" = "-G" ]; then
  printf '%s\n' "-G $2" >> "$log"
  printf 'user ana\nhostname hub.example\nport %s\n' "${ARIA_FAKE_SSH_SSHPORT:-2222}"
  exit 0
fi

host=""
prev=""
for argument in "$@"; do host=$prev; prev=$argument; done
cmd=$prev
stdin_file="$root/stdin.$$"
mkdir -p "$root/home"
cat > "$stdin_file"
bytes=$(wc -c < "$stdin_file" | tr -d ' ')
printf '%s\t%s\tstdin=%s\n' "$host" "$cmd" "$bytes" >> "$log"
trap 'rm -f "$stdin_file"' EXIT

missing=" ${ARIA_FAKE_SSH_MISSING:-} "
case "$cmd" in
  "uname -m") echo "${ARIA_FAKE_SSH_ARCH:-aarch64}" ;;
  *os-release*) echo "Ubuntu 24.04.2 LTS" ;;
  'echo $HOME') echo /home/ana ;;
  "ss -ltnH -n")
    echo "LISTEN 0 4096 127.0.0.1:631 0.0.0.0:*"
    if [ "${ARIA_FAKE_SSH_PUBLIC:-1}" = 0 ]; then
      echo "LISTEN 0 128 100.64.0.9:2222 0.0.0.0:*"
    else
      echo "LISTEN 0 128 0.0.0.0:2222 0.0.0.0:*"
      echo "LISTEN 0 128 [::]:2222 [::]:*"
    fi ;;
  "command -v "*)
    for name in ${cmd#command -v }; do
      case "$missing" in *" $name "*) ;; *) echo "/usr/bin/$name" ;; esac
    done ;;
  "node --version") echo "${ARIA_FAKE_SSH_NODE:-v18.19.1}" ;;
  *"tar -xf"*) HOME="$root/home" sh -c "$cmd" < "$stdin_file" ;;
  *install.sh*) echo "fake installer: Aria is installed and running as aria-serve.service" ;;
  *hub-import*)
    if [ "${ARIA_FAKE_SSH_IMPORT_FAIL:-0}" = 1 ]; then
      printf 'aria hub-import: the bundle is not one: invalid value: %s\n' "$(cut -c1-60 "$stdin_file")" >&2
      exit 1
    fi
    echo settings.json ;;
  *systemctl*) ;;
  *bwrap*)
    if [ "${ARIA_FAKE_SSH_BWRAP_FAIL:-0}" = 1 ]; then
      echo "bwrap: setting up uid map: Permission denied" >&2
      exit 1
    fi ;;
  *authorized_keys*) HOME="$root/home" sh -c "$cmd" < "$stdin_file" ;;
  *sshd_config*)
    if [ "${ARIA_FAKE_SSH_PASSWORDS:-no}" = yes ]; then echo "PasswordAuthentication yes"; else echo "PasswordAuthentication no"; fi ;;
  *--version*) echo "aria 0.0.0-fake" ;;
  *) echo "fake ssh: unexpected command: $cmd" >&2; exit 97 ;;
esac
