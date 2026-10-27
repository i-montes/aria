#!/bin/sh
# A stand-in for ssh in the `aria hub init` tests (ARIA_SSH points here). It talks to no network and reads no ssh
# configuration. Every remote command is recorded, one line each, in the file named by ARIA_FAKE_SSH_LOG:
#   <host><TAB><command><TAB>stdin=<bytes>
# The bytes of stdin are never recorded (a bundle holds credentials), only how many there were. `ssh -G <host>` is
# recorded as `-G <host>`. Answers are canned, chosen by the command; env vars select variants:
#   ARIA_FAKE_SSH_ARCH         the answer of `uname -m` (default aarch64)
#   ARIA_FAKE_SSH_PUBLIC       0: sshd listens on a Tailscale address only (default: on every address)
#   ARIA_FAKE_SSH_SS           fail: `ss` fails (not installed); empty: it answers nothing; fail-until-install: it fails
#                              until the installer has run
#   ARIA_FAKE_SSH_MISSING      names `command -v` does not find (default none): xz bwrap node npm tailscale
#   ARIA_FAKE_SSH_NODE         the answer of `node --version` (default v18.19.1)
#   ARIA_FAKE_SSH_BWRAP_NOPROC 1: the probe fails only when it mounts /proc (a container-style server)
#   ARIA_FAKE_SSH_BWRAP_FAIL   1: the sandbox probe fails (bwrap cannot make a user namespace); missing: there is no bwrap
#   ARIA_FAKE_SSH_IMPORT_FAIL  1: `aria hub-import` fails and complains on stderr, quoting the start of what it read
#   ARIA_FAKE_SSH_PASSWORDS    yes: sshd_config allows password logins (default no)
#   ARIA_FAKE_SSH_LINGER       no: the user does not linger on the hub; fail: `loginctl` fails (default: yes)
#   ARIA_FAKE_SSH_ACTIVE       1: aria-serve.service is already active before the run (default: it is not)
#   ARIA_FAKE_SSH_VERSION      the version `aria --version` says on the hub (default 0.0.0-fake); ARIA_FAKE_SSH_VERSION_AFTER, when
#                              set, is what it says once the installer has run
#   ARIA_FAKE_SSH_DOWN         hosts (space separated) that cannot be reached: ssh exits 255 for every command to them
#   ARIA_FAKE_SSH_HELLO        fail: `aria client-stdio` (the connection `hub init` tries at the end) exits at once; otherwise it
#                              answers the hello with protocol ARIA_FAKE_SSH_PROTOCOL (default 1) and the version
#                              ARIA_FAKE_SSH_HELLO_VERSION (default x); once a hub has "updated itself" (below) it says
#                              ARIA_FAKE_SSH_HELLO_VERSION_AFTER instead, when that is set
#   ARIA_FAKE_SSH_SELFUPDATE   what the terminal bridge does with a `hubUpdate` request (`aria hub update` on a hub linked with
#                              its key): ok (two progress lines, then a good result for the version asked, and the hub counts
#                              as updated), fail (a progress line and a failed result: a 404), refuse (a failed result:
#                              a downgrade), wrong (a good result for another version), unknown (an old hub: an error,
#                              `not-for-clients`), silent (no answer at all), eof (a progress line and the connection ends,
#                              the hub counting as updated). The request is logged as `<host><TAB>hubUpdate <version>`; the
#                              arguments of the bridge's ssh are kept in $ARIA_FAKE_SSH_ROOT/ssh-args.
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
# The terminal's own connection (`<host> ~/.local/bin/aria client-stdio`, the last two words as arguments of their own): no
# stdin to slurp, a hello to answer, and it stays open until its client closes it.
a3=""; a2=""; a1=""
for argument in "$@"; do a3=$a2; a2=$a1; a1=$argument; done
if [ "$a1" = client-stdio ] && [ "$a2" = "~/.local/bin/aria" ]; then
  host=$a3
  cmd="$a2 $a1"
  printf '%s\t%s\tstdin=0\n' "$host" "$cmd" >> "$log"
  if [ "${ARIA_FAKE_SSH_HELLO:-ok}" = fail ]; then echo "fake ssh: the connection is refused" >&2; exit 255; fi
  printf '%s\n' "$*" > "$root/ssh-args"
  read -r hello
  hello_version=${ARIA_FAKE_SSH_HELLO_VERSION:-x}
  if [ -f "$root/self-updated" ] && [ -n "${ARIA_FAKE_SSH_HELLO_VERSION_AFTER:-}" ]; then hello_version=$ARIA_FAKE_SSH_HELLO_VERSION_AFTER; fi
  printf '{"type":"hello","protocol":%s,"version":"%s"}\n' "${ARIA_FAKE_SSH_PROTOCOL:-1}" "$hello_version"
  if [ -n "${ARIA_FAKE_SSH_SELFUPDATE:-}" ]; then
    # A connection that only says hello (the checks of `hub update`) is closed by its client: no request, nothing to log.
    read -r request || exit 0
    asked=$(printf '%s' "$request" | sed -n 's/.*"version":"\([^"]*\)".*/\1/p')
    id=$(printf '%s' "$request" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
    printf '%s\thubUpdate %s\tstdin=0\n' "$host" "$asked" >> "$log"
    case "$ARIA_FAKE_SSH_SELFUPDATE" in
      ok)
        printf '{"type":"hubUpdateProgress","id":%s,"line":"- Checking this server"}\n' "$id"
        printf '{"type":"hubUpdateProgress","id":%s,"line":"- Installing Aria"}\n' "$id"
        touch "$root/self-updated"
        printf '{"type":"hubUpdateResult","id":%s,"ok":true,"version":"%s"}\n' "$id" "$asked" ;;
      wrong)
        printf '{"type":"hubUpdateResult","id":%s,"ok":true,"version":"9.9.9"}\n' "$id" ;;
      fail)
        printf '{"type":"hubUpdateProgress","id":%s,"line":"curl: (22) The requested URL returned error: 404"}\n' "$id"
        printf '{"type":"hubUpdateResult","id":%s,"ok":false,"reason":"curl: (22) The requested URL returned error: 404"}\n' "$id" ;;
      refuse)
        printf '{"type":"hubUpdateResult","id":%s,"ok":false,"reason":"a downgrade is refused"}\n' "$id" ;;
      unknown)
        printf '{"type":"error","id":%s,"code":"not-for-clients"}\n' "$id" ;;
      eof)
        printf '{"type":"hubUpdateProgress","id":%s,"line":"- Installing Aria"}\n' "$id"
        touch "$root/self-updated"
        exit 0 ;;
      silent) ;;
    esac
  fi
  exec sleep 30
fi
stdin_file="$root/stdin.$$"
mkdir -p "$root/home"
cat > "$stdin_file"
bytes=$(wc -c < "$stdin_file" | tr -d ' ')
printf '%s\t%s\tstdin=%s\n' "$host" "$cmd" "$bytes" >> "$log"
trap 'rm -f "$stdin_file"' EXIT
case " ${ARIA_FAKE_SSH_DOWN:-} " in
  *" $host "*) echo "ssh: connect to host $host port 22: Connection refused" >&2; exit 255 ;;
esac

missing=" ${ARIA_FAKE_SSH_MISSING:-} "
case "$cmd" in
  "uname -m") echo "${ARIA_FAKE_SSH_ARCH:-aarch64}" ;;
  *os-release*) echo "Ubuntu 24.04.2 LTS" ;;
  'echo $HOME') echo /home/ana ;;
  "ss -ltnH -n")
    case "${ARIA_FAKE_SSH_SS:-ok}" in
      fail) echo "sh: ss: command not found" >&2; exit 127 ;;
      empty) exit 0 ;;
      fail-until-install)
        if [ ! -f "$root/installed" ]; then echo "sh: ss: command not found" >&2; exit 127; fi ;;
    esac
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
  *"test -f"*) HOME="$root/home" sh -c "$cmd" ;;
  *loginctl*)
    case "${ARIA_FAKE_SSH_LINGER:-yes}" in
      fail) echo "Failed to connect to bus: No medium found" >&2; exit 1 ;;
      *) echo "Linger=${ARIA_FAKE_SSH_LINGER:-yes}" ;;
    esac ;;
  *is-active*)
    if [ "${ARIA_FAKE_SSH_ACTIVE:-0}" = 1 ]; then echo active; else echo inactive; exit 3; fi ;;
  *"tar -xf"*) HOME="$root/home" sh -c "$cmd" < "$stdin_file" ;;
  *install.sh*) touch "$root/installed"; echo "fake installer: Aria is installed and running as aria-serve.service" ;;
  *hub-import*)
    if [ "${ARIA_FAKE_SSH_IMPORT_FAIL:-0}" = 1 ]; then
      printf 'aria hub-import: the bundle is not one: invalid value: %s\n' "$(cut -c1-60 "$stdin_file")" >&2
      exit 1
    fi
    # what a real hub-import would have written: the names the bundle holds (the bundle is kept for the tests)
    cp "$stdin_file" "$root/last-bundle"
    for name in settings.json auth.json mcp.json mcp-auth.json; do
      if grep -q "\"$name\":" "$stdin_file"; then echo "$name"; fi
    done ;;
  *systemctl*) ;;
  *bwrap*)
    case "${ARIA_FAKE_SSH_BWRAP_FAIL:-0}" in
      1) echo "bwrap: setting up uid map: Permission denied" >&2; exit 1 ;;
      missing) echo "sh: 1: bwrap: not found" >&2; exit 127 ;;
    esac
    # A container-style server: a user namespace works, mounting /proc does not (only the real call's `--proc` fails).
    if [ "${ARIA_FAKE_SSH_BWRAP_NOPROC:-0}" = 1 ]; then
      case "$cmd" in
        *"--proc /proc"*) echo "bwrap: Can't mount proc on /newroot/proc: Operation not permitted" >&2; exit 1 ;;
      esac
    fi ;;
  *authorized_keys*) HOME="$root/home" sh -c "$cmd" < "$stdin_file" ;;
  *sshd_config*)
    if [ "${ARIA_FAKE_SSH_PASSWORDS:-no}" = yes ]; then echo "PasswordAuthentication yes"; else echo "PasswordAuthentication no"; fi ;;
  *--version*)
    if [ -f "$root/installed" ] && [ -n "${ARIA_FAKE_SSH_VERSION_AFTER:-}" ]; then echo "aria $ARIA_FAKE_SSH_VERSION_AFTER"
    else echo "aria ${ARIA_FAKE_SSH_VERSION:-0.0.0-fake}"; fi ;;
  *) echo "fake ssh: unexpected command: $cmd" >&2; exit 97 ;;
esac
