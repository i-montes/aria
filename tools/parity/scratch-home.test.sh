#!/bin/sh
# Tests for scratch-home.sh against a fake ~/.aria: sh tools/parity/scratch-home.test.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
script="$here/scratch-home.sh"
version=$(cat "$here/../../pi-version")
failures=0
fail() { echo "FAIL: $1"; failures=$((failures + 1)); }
pass() { echo "ok:   $1"; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
real="$tmp/real"
mkdir -p "$real/pi/$version/node_modules/.bin" "$real/bin"
printf '{"opencode-go":{"type":"api_key","key":"sk-fake-123456789"},"anthropic":{"type":"oauth","access":"a","refresh":"r","expires":1}}\n' > "$real/auth.json"
printf '{"notion":{"tokens":{"refresh_token":"r"}}}\n' > "$real/mcp-auth.json"
printf '{"mcpServers":{"notion":{"url":"https://mcp.notion.com/mcp"},"engram":{"command":"%s/bin/engram","args":["mcp"],"env":{"ENGRAM_DATA_DIR":"%s/engram"}}}}\n' "$real" "$real" > "$real/mcp.json"
mkdir -p "$real/agent/npm/node_modules/pi-parallel-agents" "$real/agent/git/github.com/obra/superpowers"
printf '{}\n' > "$real/agent/npm/node_modules/pi-parallel-agents/package.json"
printf 'x\n' > "$real/agent/git/github.com/obra/superpowers/README.md"
stub="$real/pi/$version/node_modules/.bin/pi"
printf '#!/bin/sh\necho "agent=$PI_CODING_AGENT_DIR settings=$ARIA_SETTINGS"\n' > "$stub"
chmod +x "$stub"
printf '#!/bin/sh\n' > "$real/bin/engram" && chmod +x "$real/bin/engram"
hash_real() { find "$real" -type f | sort | xargs sha256sum | sha256sum; }
before=$(hash_real)

out=$(ARIA_REAL_HOME="$real" sh "$script" sh -c 'echo "home=$ARIA_HOME"; stat -c %a "$ARIA_HOME/auth.json"; "$ARIA_PI"; ls "$ARIA_HOME/bin"' 2>&1)
home=$(printf '%s\n' "$out" | sed -n 's/^home=//p')
if [ -n "$home" ] && [ "$home" != "$real" ]; then pass "the command runs in a scratch home"; else fail "no scratch home: $out"; fi
printf '%s\n' "$out" | grep -qx 600 && pass "the copied auth.json is private" || fail "auth.json is not 600: $out"
printf '%s\n' "$out" | grep -q "agent=$home/agent settings=$home/settings.json" && pass "Pi gets the scratch home's session environment" || fail "Pi's environment is not the scratch home's: $out"
printf '%s\n' "$out" | grep -qx engram && pass "Aria's tools are linked" || fail "engram is not linked: $out"
[ -n "$home" ] && [ ! -e "$home" ] && pass "the scratch home is removed" || fail "the scratch home is left at $home"

out=$(ARIA_REAL_HOME="$real" sh "$script" sh -c 'echo "home=$ARIA_HOME"; exit 3' 2>&1)
code=$?
home=$(printf '%s\n' "$out" | sed -n 's/^home=//p')
[ "$code" = 3 ] && pass "the command's exit code is kept" || fail "exit code $code, not 3"
[ -n "$home" ] && [ ! -e "$home" ] && pass "a failing command's scratch home is removed" || fail "the scratch home is left at $home"

out=$(ANTHROPIC_API_KEY=z ARIA_REAL_HOME="$real" sh "$script" sh -c 'echo "key=${ANTHROPIC_API_KEY:-absent}"' 2>&1)
printf '%s\n' "$out" | grep -qx 'key=absent' && pass "ANTHROPIC_* variables are dropped" || fail "ANTHROPIC_API_KEY reached the command: $out"

out=$(ARIA_REAL_HOME="$real" sh "$script" sh -c 'cat "$ARIA_HOME/mcp.json"; echo; cat "$ARIA_HOME/auth.json"; echo; ls "$ARIA_HOME"; [ -f "$ARIA_HOME/agent/npm/node_modules/pi-parallel-agents/package.json" ] && [ ! -L "$ARIA_HOME/agent/npm" ] && echo packages-copied; [ -f "$ARIA_HOME/agent/git/github.com/obra/superpowers/README.md" ] && echo git-copied' 2>&1)
printf '%s\n' "$out" | grep -q "$real/engram" && fail "the scratch Pi's Engram would write the real database: $out" || pass "the real Engram entry is not copied"
printf '%s\n' "$out" | grep -q notion && pass "the other MCP servers are copied" || fail "notion is missing: $out"
printf '%s\n' "$out" | grep -q oauth && fail "OAuth sign-ins are copied without asking: $out" || pass "only API keys are copied by default"
printf '%s\n' "$out" | grep -qx mcp-auth.json && fail "mcp-auth.json is copied without asking" || pass "MCP sign-ins are not copied by default"
printf '%s\n' "$out" | grep -qx packages-copied && printf '%s\n' "$out" | grep -qx git-copied && pass "Pi's packages are copied, not linked" || fail "Pi's packages are missing: $out"
out=$(ARIA_PARITY_OAUTH=1 ARIA_REAL_HOME="$real" sh "$script" sh -c 'cat "$ARIA_HOME/auth.json"; ls "$ARIA_HOME"' 2>&1)
printf '%s\n' "$out" | grep -q oauth && printf '%s\n' "$out" | grep -qx mcp-auth.json && pass "ARIA_PARITY_OAUTH=1 copies the sign-ins" || fail "the sign-ins were not copied on request: $out"

[ "$(hash_real)" = "$before" ] && pass "the real home is unchanged" || fail "the real home changed"

[ "$failures" = 0 ] && echo "all passed" || { echo "$failures failed"; exit 1; }
