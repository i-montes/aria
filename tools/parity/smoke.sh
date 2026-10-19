#!/bin/sh
# The setup's smoke test, through a scratch Aria home (a few cents of the daily model):
#   sh tools/parity/scratch-home.sh sh $PWD/tools/parity/smoke.sh
# ARIA_SMOKE_MODEL=<provider/id> asks a given model (a claude-bridge one checks the Max plan).
set -eu
[ -n "${ARIA_PI:-}" ] || { echo "run this through tools/parity/scratch-home.sh" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
if [ -n "${ARIA_SMOKE_MODEL:-}" ]; then
  answer=$("$ARIA_PI" -p --model "$ARIA_SMOKE_MODEL" "hola" </dev/null)
else
  answer=$("$ARIA_PI" -p "hola" </dev/null)
fi
printf '%s\n' "$answer"
printf '%s' "$answer" | bun "$here/smoke.ts"
