#!/bin/sh
# Acceptance of the hub (milestone M1) with no display, no ssh and no model provider: a headless `aria serve` in a scratch
# home, and the terminal client reaching it the way `aria connect` does, with ARIA_SSH pointing at tools/hub/ssh-shim.sh,
# which runs the hub's `aria client-stdio` on this computer instead of ssh.
#
#   cargo build -p aria --no-default-features && sh tools/hub/accept.sh target/debug/aria
#
# Everything lives in temporary folders (HOME, ARIA_HOME and XDG_RUNTIME_DIR of the hub and of the client), and the hub's
# PATH holds nothing, so it never reaches the owner's keyring or home. Checks, one `ok:` or `FAIL:` line each:
#   (a) the list over connect shows the hub's sessions    (b) a session opened over connect has no folder from the client
#   (c) killing the client leaves the hub and the session, and a second connect lists it
#   (d) /schedules over connect says there are no scheduled tasks
#   (e) a client with another protocol is refused (exit 3)   (f) a secretSave through the bridge is `not-for-remote`
#   (g) `aria quit` stops the hub (exit 0, nothing answers on the socket: the file of a stopped service stays, as with a dock)
# (c) covers the session and the service surviving the client. A RUN started over connect that goes on after the client
# dies needs a model provider, which a scratch home does not have: the owner checks that by hand (the spec's M1 "done
# when"), and it is not faked here.
set -u
here=$(cd "$(dirname "$0")" && pwd)
[ $# -eq 1 ] && [ -x "$1" ] || { echo "usage: sh tools/hub/accept.sh <path of the headless aria binary>" >&2; exit 2; }
aria=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
failures=0
pass() { echo "ok: $1"; }
fail() { echo "FAIL: $1"; failures=$((failures + 1)); }

tmp=$(mktemp -d); chmod 700 "$tmp"
hub_home="$tmp/hub/home"; hub_run="$tmp/hub/run"
mkdir -p "$hub_home/.aria" "$hub_run" "$tmp/hub/bin" "$tmp/client/home/.aria" "$tmp/client/run" "$tmp/client/cwd"
chmod 700 "$hub_run" "$tmp/client/run"
serve_pid=
# The hub stops with this script, whatever way it ends; then the scratch folders go.
cleanup() {
  [ -z "$serve_pid" ] || kill "$serve_pid" 2>/dev/null
  [ -z "$serve_pid" ] || wait "$serve_pid" 2>/dev/null
  rm -rf "$tmp"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

# The hub's sessions, as an earlier use would have left them.
cat > "$hub_home/.aria/sessions.json" <<'JSON'
{"sessions": [{"id": "seed-1", "title": "Fix the checkout widget"}, {"id": "seed-2", "title": "Review the release notes"}], "active": null}
JSON

hub_env() { env -i HOME="$hub_home" ARIA_HOME="$hub_home/.aria" XDG_RUNTIME_DIR="$hub_run" PATH="$tmp/hub/bin" "$@"; }
socket="$hub_run/aria/clients.sock"

hub_env "$aria" serve >"$tmp/serve.log" 2>&1 &
serve_pid=$!
waited=0
while ! grep -q "serving with no window" "$tmp/serve.log" 2>/dev/null || [ ! -S "$socket" ]; do
  kill -0 "$serve_pid" 2>/dev/null || { echo "FAIL: the hub did not start:"; cat "$tmp/serve.log"; exit 1; }
  waited=$((waited + 1))
  [ "$waited" -lt 400 ] || { echo "FAIL: the hub never came up:"; cat "$tmp/serve.log"; exit 1; }
  sleep 0.05
done

# (a)-(d): the terminal client over the shim, in a pseudo-terminal. It is never given the hub's folders: only the shim is.
export ARIA_SSH="$here/ssh-shim.sh" ARIA_SHIM_ARIA="$aria" ARIA_SHIM_HOME="$hub_home" ARIA_SHIM_RUN="$hub_run" \
  ARIA_SHIM_PIDFILE="$tmp/shim.pid"
if env HOME="$tmp/client/home" ARIA_HOME="$tmp/client/home/.aria" XDG_RUNTIME_DIR="$tmp/client/run" PATH="$tmp/hub/bin:/usr/bin:/bin" \
  python3 -I "$here/drive_connect.py" "$aria" "$tmp/client/cwd" "$hub_home/.aria" "$socket" "$tmp/shim.pid"; then :; else
  fail "the terminal over connect (see the last line above)"
fi

# (e) another protocol: the bridge answers with the mismatch and exits 3
out=$(printf '{"type":"hello","protocol":99,"version":"0.0.0"}\n' | ARIA_SHIM_PIDFILE= "$here/ssh-shim.sh" host aria client-stdio 2>&1)
code=$?
if [ "$code" -eq 3 ] && printf '%s' "$out" | grep -q protocol-mismatch; then
  pass "a client with another protocol is refused (exit 3, protocol-mismatch)"
else
  fail "a client with another protocol is refused (exit $code: $out)"
fi

# (f) a secret does not go in from another computer: the bridge's hello marks the connection remote
out=$( (printf '{"type":"hello","protocol":1,"version":"0.0.0"}\n{"type":"secretSave","id":7,"name":"k","value":"v"}\n'; sleep 2) |
  ARIA_SHIM_PIDFILE= "$here/ssh-shim.sh" host aria client-stdio 2>&1)
if printf '%s' "$out" | grep -q '"code":"not-for-remote"' && printf '%s' "$out" | grep -q '"id":7'; then
  pass "a secretSave through client-stdio is answered not-for-remote"
else
  fail "a secretSave through client-stdio is answered not-for-remote ($out)"
fi
if [ -e "$hub_home/.aria/secrets.json" ]; then fail "the refused secret left a file"; fi

# (g) aria quit
out=$(hub_env "$aria" quit 2>&1)
code=$?
waited=0
while kill -0 "$serve_pid" 2>/dev/null && [ "$waited" -lt 200 ]; do waited=$((waited + 1)); sleep 0.05; done
# A stopped service leaves its socket file behind (the next start replaces it): what counts is that nothing answers.
answers() { python3 -I -c 'import socket, sys; s = socket.socket(socket.AF_UNIX); s.connect(sys.argv[1])' "$socket" 2>/dev/null; }
if [ "$code" -eq 0 ] && ! kill -0 "$serve_pid" 2>/dev/null && ! answers; then
  wait "$serve_pid" 2>/dev/null; serve_status=$?
  [ "$serve_status" -eq 0 ] && pass "aria quit stops the hub (exit 0, nothing answers on the socket)" || fail "the hub exited $serve_status after quit"
  serve_pid=
else
  fail "aria quit stops the hub (quit exit $code, $(answers && echo "the socket still answers"): $out)"
fi

[ "$failures" -eq 0 ] || { echo "$failures failed"; exit 1; }
echo "hub acceptance passed"
