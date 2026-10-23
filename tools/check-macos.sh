#!/bin/sh
# Type-checks the macOS build from Linux: `cargo check`/`clippy` for aarch64-apple-darwin, with zig (through mise) as the
# C compiler that `ring` needs. It does not link, so nothing runs; it catches what only breaks the Mac build (code
# behind `cfg(target_os = "macos")`, imports that only exist on Linux).
#   sh tools/check-macos.sh            check every crate the Mac builds, tests included
set -e
cd "$(dirname "$0")/.."
rustup target add aarch64-apple-darwin >/dev/null
zig=$(mise x zig@0.14.1 -- which zig)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
wrapper="$tmp/zigcc"
# cc-rs passes Apple's flags (-arch, --target=arm64-apple-macosx, -mmacosx-version-min); zig wants its own target.
cat > "$wrapper" <<PY
#!/usr/bin/env python3
import os, sys
out = []
args = iter(sys.argv[1:])
for a in args:
    if a == '-arch':
        next(args, None)
        continue
    if a == '-gfull' or a.startswith('-mmacosx-version-min=') or a.startswith('--target='):
        continue
    out.append(a)
os.execv('$zig', ['$zig', 'cc', '-target', 'aarch64-macos'] + out)
PY
chmod +x "$wrapper"
# whisper.cpp (dictation) cannot be built for macOS from here (cmake, Metal, Apple's SDK). A type-check needs only its
# bindings: whisper-rs-sys copies the ones it ships and stops before cmake when it thinks it runs on docs.rs.
DOCS_RS=1 WHISPER_DONT_GENERATE_BINDINGS=1 CC_aarch64_apple_darwin="$wrapper" CXX_aarch64_apple_darwin="$wrapper" cargo clippy -q --target aarch64-apple-darwin -p aria -p aria-mac -p aria-core -p aria-engine --all-targets
echo "macOS check passed"
