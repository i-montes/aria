#!/bin/sh
# Builds `aria serve` for a server without a window: the cargo feature `gui` off, cross-compiled from this computer with
# zig (through mise) as the C compiler and linker, pinned to glibc 2.39 (Ubuntu 24.04) so the binary runs there.
#   sh tools/build-headless.sh [triple]   default aarch64-unknown-linux-gnu; x86_64-unknown-linux-gnu also works
# The last line of stdout is the path of the binary, so a caller can use it: sh install/install.sh --headless --local "$(sh tools/build-headless.sh | tail -1)"
set -e
cd "$(dirname "$0")/.."
triple=${1:-aarch64-unknown-linux-gnu}
case "$triple" in
  aarch64-unknown-linux-gnu) zig_target=aarch64-linux-gnu.2.39 ;;
  x86_64-unknown-linux-gnu) zig_target=x86_64-linux-gnu.2.39 ;;
  *) echo "build-headless: unsupported target $triple (use aarch64-unknown-linux-gnu or x86_64-unknown-linux-gnu)" >&2; exit 1 ;;
esac
rustup target add "$triple" >&2
zig=$(mise x zig@0.14.1 -- which zig)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
wrapper="$tmp/zigcc"
# cc-rs and rustc pass the Rust triple as --target=...; zig wants its own, with the glibc version after the dot. Zig is its
# own linker (lld) and has no libgcc_s (its unwinder is libunwind), so the flags that name the system's linker, its helper
# folder and a gcc-only erratum fix on aarch64 are dropped, and -lgcc_s becomes -lunwind.
cat > "$wrapper" <<PY
#!/usr/bin/env python3
import os, sys
drop = ('--target=', '-fuse-ld=', '-B', '-Wl,--fix-cortex-a53-843419')
out = ['-lunwind' if a == '-lgcc_s' else a for a in sys.argv[1:] if not a.startswith(drop)]
os.execv('$zig', ['$zig', 'cc', '-target', '$zig_target'] + out)
PY
chmod +x "$wrapper"
env_triple=$(printf '%s' "$triple" | tr '-' '_')
upper_triple=$(printf '%s' "$env_triple" | tr '[:lower:]' '[:upper:]')
export "CC_$env_triple=$wrapper" "CXX_$env_triple=$wrapper" "CARGO_TARGET_${upper_triple}_LINKER=$wrapper"
cargo build --release --no-default-features --target "$triple" -p aria >&2
target_dir=$(cargo metadata --format-version 1 --no-deps | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
echo "$target_dir/$triple/release/aria"
