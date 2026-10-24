#!/bin/sh
# Regenerates docs/assets/aria-dock.png from the UI sandbox (fake content, no Pi, nothing of the machine's).
#   sh docs/assets/regenerate.sh        needs bun, ~/.aria/bin/agent-browser (or AGENT_BROWSER=...), and magick + optipng
set -eu
cd "$(dirname "$0")/../.."
port=${PORT:-8771}
ab=${AGENT_BROWSER:-$HOME/.aria/bin/agent-browser}
export AGENT_BROWSER_SESSION=aria-readme-shot
raw=$(mktemp -d)/dock.png
bun tools/ui-sandbox/serve.ts "$port" >/dev/null &
server=$!
trap '"$ab" close >/dev/null 2>&1 || true; kill "$server" 2>/dev/null || true' EXIT
until curl -fs "http://127.0.0.1:$port/" >/dev/null; do sleep 0.2; done
"$ab" set viewport 700 900 2 >/dev/null
"$ab" open "http://127.0.0.1:$port/" >/dev/null
"$ab" wait 1500 >/dev/null
"$ab" eval --stdin <docs/assets/dock-scene.js >/dev/null
"$ab" screenshot "$raw" >/dev/null
# The dock column plus a margin of its veil: 520 x 900 CSS px at 2x.
magick "$raw" -crop 1040x1800+360+0 +repage -strip docs/assets/aria-dock.png
optipng -o5 -quiet docs/assets/aria-dock.png
