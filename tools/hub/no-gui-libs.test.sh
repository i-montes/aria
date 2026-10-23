#!/bin/sh
# Checks tools/hub/no-gui-libs.sh with a fake `ldd` (no real binary needed).
#   sh tools/hub/no-gui-libs.test.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
failures=0
pass() { echo "ok:   $1"; }
fail() {
  echo "FAIL: $1"
  failures=$((failures + 1))
}

dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
# The fake `ldd` prints the file it is given.
printf '#!/bin/sh\ncat "$1"\n' >"$dir/ldd"
chmod +x "$dir/ldd"

printf '\tlinux-vdso.so.1 (0x0000)\n\tlibc.so.6 => /lib/libc.so.6 (0x0000)\n\tlibssl.so.3 => /lib/libssl.so.3 (0x0000)\n' >"$dir/plain"
out=$(PATH="$dir:$PATH" sh "$here/no-gui-libs.sh" "$dir/plain" 2>&1) && [ "$out" = "no GUI libraries linked" ] \
  && pass "a binary with plain libraries is accepted" || fail "a binary with plain libraries is accepted ($out)"

for lib in libgtk-3.so.0 libgdk-3.so.0 libwebkit2gtk-4.1.so.0 libvulkan.so.1 libasound.so.2 libjavascriptcoregtk-4.1.so.0; do
  printf '\tlibc.so.6 => /lib/libc.so.6 (0x0000)\n\t%s => /lib/%s (0x0000)\n' "$lib" "$lib" >"$dir/gui"
  out=$(PATH="$dir:$PATH" sh "$here/no-gui-libs.sh" "$dir/gui" 2>&1)
  status=$?
  if [ "$status" -eq 1 ] && printf '%s' "$out" | grep -q "$lib"; then
    pass "$lib is refused and named"
  else
    fail "$lib is refused and named (status $status: $out)"
  fi
done

out=$(PATH="$dir:$PATH" sh "$here/no-gui-libs.sh" 2>&1)
[ $? -eq 2 ] && pass "no argument is a usage error" || fail "no argument is a usage error ($out)"

[ "$failures" -eq 0 ] || { echo "$failures failed"; exit 1; }
echo "all passed"
