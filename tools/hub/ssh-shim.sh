#!/bin/sh
# A stand-in for ssh in tools/hub/accept.sh (ARIA_SSH points here): it ignores every argument ssh would get (the host and
# the remote command) and runs the hub's own `aria client-stdio` on this computer, the way `ssh <host> aria client-stdio`
# would on the hub. No network, no ssh configuration. accept.sh sets:
#   ARIA_SHIM_ARIA  the binary to run         ARIA_SHIM_HOME  the hub's ARIA_HOME and HOME's folder
#   ARIA_SHIM_RUN   the hub's XDG_RUNTIME_DIR ARIA_SHIM_PIDFILE  where to write the bridge's process id (it is `exec`ed, so
#                   it is this script's), so a check can kill it as a dropped connection would
: "${ARIA_SHIM_ARIA:?}" "${ARIA_SHIM_HOME:?}" "${ARIA_SHIM_RUN:?}"
[ -z "${ARIA_SHIM_PIDFILE:-}" ] || echo $$ > "$ARIA_SHIM_PIDFILE"
exec env HOME="$ARIA_SHIM_HOME" ARIA_HOME="$ARIA_SHIM_HOME/.aria" XDG_RUNTIME_DIR="$ARIA_SHIM_RUN" "$ARIA_SHIM_ARIA" client-stdio
