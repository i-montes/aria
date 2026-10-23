#!/bin/sh
# Runs a command in Pi's session environment for a scratch copy of the installed Aria home, then deletes the copy. The
# real home (~/.aria, or ARIA_REAL_HOME) is only read: the copy gets its settings, API keys, MCP list and a copy of Pi's
# packages, and links to its pinned Pi and tools; its agent directory, sessions and Engram data are the copy's own.
#
#   sh tools/parity/scratch-home.sh <command> [args...]
#
# Sign-ins (OAuth in auth.json and mcp-auth.json) are not copied: a scratch Pi that refreshes one would leave the
# dock's copy stale if the provider rotates its tokens. ARIA_PARITY_OAUTH=1 copies them, for a check that needs Claude
# Max or Notion.
#
# The command runs in a fresh git repo with ARIA_HOME (the copy), ARIA_PI (the pinned Pi), ARIA_PI_DIST (its
# pi-coding-agent dist folder) and ARIA_REAL_AUTH (the real auth.json, for checks that a secret stays hidden).
set -eu
[ $# -ge 1 ] || { echo "usage: scratch-home.sh <command> [args...]" >&2; exit 2; }
repo=$(cd "$(dirname "$0")/../.." && pwd)
real=${ARIA_REAL_HOME:-$HOME/.aria}
[ -d "$real/pi" ] || { echo "no pinned Pi in $real: install Aria first" >&2; exit 2; }

# Under the cache folder (the same file system as ~/.aria on most machines), so Pi's packages copy fast.
cache=${XDG_CACHE_HOME:-$HOME/.cache}
mkdir -p "$cache"
tmp=$(mktemp -d "$cache/aria-parity.XXXXXX")
trap 'rm -rf "$tmp"' EXIT INT TERM HUP
home="$tmp/home"
work="$tmp/work"
mkdir -p "$home/bin" "$home/agent" "$work"
chmod 700 "$tmp"
[ -f "$real/settings.json" ] && cp "$real/settings.json" "$home/settings.json"
# The API keys only, unless sign-ins are asked for. With the keyring on, auth.json is not in the home: the dock keeps
# a live copy for Pi, which the agent folder links to.
auth_src="$real/auth.json"
[ -f "$auth_src" ] || auth_src="$real/agent/auth.json"
if [ -f "$auth_src" ]; then
  python3 - "$auth_src" "$home/auth.json" "${ARIA_PARITY_OAUTH:-0}" <<'PY'
import json, sys
auth = json.load(open(sys.argv[1]))
if sys.argv[3] != "1":
    auth = {k: v for k, v in auth.items() if isinstance(v, dict) and v.get("type") == "api_key"}
json.dump(auth, open(sys.argv[2], "w"), indent=2)
PY
fi
[ "${ARIA_PARITY_OAUTH:-0}" = 1 ] && [ -f "$real/mcp-auth.json" ] && cp "$real/mcp-auth.json" "$home/mcp-auth.json"
# The MCP list without Aria's Engram entry, which names the real database; the scratch Pi gets its own entry.
if [ -f "$real/mcp.json" ]; then
  python3 - "$real/mcp.json" "$home/mcp.json" "$real" <<'PY'
import json, sys
mcp = json.load(open(sys.argv[1]))
servers = mcp.get("mcpServers", mcp)
for name in [n for n, s in servers.items() if isinstance(s, dict) and (s.get("env") or {}).get("ENGRAM_DATA_DIR", "").startswith(sys.argv[3]) or (isinstance(s, dict) and str(s.get("command", "")).startswith(sys.argv[3]))]:
    del servers[name]
json.dump(mcp, open(sys.argv[2], "w"), indent=2)
PY
fi
for file in auth.json mcp.json mcp-auth.json; do [ -f "$home/$file" ] && chmod 600 "$home/$file"; done
# Pi's packages (pi-parallel-agents, superpowers, the Claude provider), copied so nothing writes into the real ones.
for dir in npm git; do
  [ -d "$real/agent/$dir" ] && { cp -R --reflink=auto "$real/agent/$dir" "$home/agent/$dir" 2>/dev/null || cp -R "$real/agent/$dir" "$home/agent/$dir"; }
done
[ -f "$real/agent/settings.json" ] && cp "$real/agent/settings.json" "$home/agent/settings.json"
ln -s "$real/pi" "$home/pi"
for tool in "$real"/bin/*; do
  [ -x "$tool" ] && [ "$(basename "$tool")" != pi ] && ln -s "$tool" "$home/bin/$(basename "$tool")"
done
git -C "$work" init -q

env=$(cd "$repo" && ARIA_HOME="$home" ARIA_PARITY_WORKDIR="$work" cargo run -q -p aria-core --example parity_env)
eval "$env"
export ARIA_HOME="$home"
export ARIA_REAL_AUTH="$auth_src"
cd "$work"
set +e
"$@"
code=$?
exit "$code"
