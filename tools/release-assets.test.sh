#!/bin/sh
# Checks tools/release-assets.sh in a scratch repository (never this one, never a real build): the names, the checksum
# the installer verifies, the refusals and the command it prints.
#   sh tools/release-assets.test.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
failures=0
# The scratch repositories must not depend on the machine's git config (signing, hooks).
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
pass() { echo "ok:   $1"; }
fail() {
  echo "FAIL: $1"
  failures=$((failures + 1))
}

base=$(mktemp -d)
trap 'cd / && rm -rf "$base"' EXIT

# The platform is whatever these say, so the test runs the same on Linux and on a Mac.
fake_uname() { # name, system, machine
  mkdir -p "$base/$1"
  printf '#!/bin/sh\ncase "$1" in -s) echo %s ;; -m) echo %s ;; *) echo %s ;; esac\n' "$2" "$3" "$2" >"$base/$1/uname"
  chmod +x "$base/$1/uname"
}
fake_uname linux Linux x86_64
fake_uname linux-arm Linux aarch64
fake_uname mac-arm Darwin arm64
fake_uname mac-intel Darwin x86_64
fake_uname windows MINGW64_NT x86_64
# A gh that only records that it was called: the script must print the command, never run it.
mkdir -p "$base/gh"
printf '#!/bin/sh\necho "gh $*" >> "%s/gh-calls"\n' "$base" >"$base/gh/gh"
chmod +x "$base/gh/gh"

printf '#!/bin/sh\necho "the aria binary"\n' >"$base/aria-binary"
chmod +x "$base/aria-binary"

scratch() {
  dir=$(mktemp -d "$base/repo.XXXXXX")
  mkdir -p "$dir/tools" "$dir/install"
  cp "$here/release-assets.sh" "$dir/tools/"
  cp "$root/install/install.sh" "$dir/install/"
  cd "$dir" || exit 1
  git init -q -b main
  git config user.email aria@localhost
  git config user.name Aria
  printf '[workspace]\nmembers = []\n\n[workspace.package]\nversion = "0.3.0"\nedition = "2021"\n' >Cargo.toml
  printf '# Releases\n\nIntro.\n\n## 0.3.0 — 2026-10-22\n\n### New\n\n- a new thing\n\n## 0.2.0 — 2026-10-21\n\nFirst.\n' >RELEASES.md
  printf '0.1.2\n' >pi-version
  printf 'git:example.invalid/pkg@abc\n' >pi-packages
  printf '1.0.0\nlinux_amd64 %064d\n' 0 >engram-version
  printf '0.9.9\n' >agent-browser-version
  printf '22.19.0\nlinux-x64 %064d\n' 0 >node-version
  printf '/target\n/dist\n' >.gitignore
  git add -A && git commit -q -m "release: 0.3.0"
  git tag -a v0.3.0 -m "Aria 0.3.0"
}
# $1 is the PATH directory with the fake uname; the rest are the script's arguments.
assets() { uname_dir=$1; shift; PATH="$base/$uname_dir:$base/gh:$PATH" ARIA_BINARY="${BIN:-$base/aria-binary}" sh tools/release-assets.sh "$@"; }

# Linux x86_64: the artifact, its checksum, the pins, the installer and the notes
scratch
out=$(assets linux "$base/out" 2>&1) && pass "assets are collected" || fail "the script failed: $out"
[ -f "$base/out/aria-x86_64-linux" ] && [ -x "$base/out/aria-x86_64-linux" ] && cmp -s "$base/aria-binary" "$base/out/aria-x86_64-linux" && pass "the binary is copied as aria-x86_64-linux" || fail "no executable aria-x86_64-linux"
ok=1
for f in pi-version pi-packages engram-version agent-browser-version node-version; do
  cmp -s "$f" "$base/out/$f" || { ok=0; fail "$f is not copied from the repository root"; }
done
[ "$ok" = 1 ] && pass "the five pin files are copied as they are"
cmp -s install/install.sh "$base/out/install.sh" && pass "install/install.sh is copied as install.sh" || fail "install.sh is missing or differs"
printf '%s\n' "$out" | grep -qF 'v0.3.0' && pass "it says which tag the assets are for" || fail "the tag is not printed: $out"
want=$(sha256sum "$base/aria-binary" 2>/dev/null | cut -d ' ' -f 1 || shasum -a 256 "$base/aria-binary" | cut -d ' ' -f 1)
[ "$(cut -d ' ' -f 1 "$base/out/aria-x86_64-linux.sha256")" = "$want" ] && pass "the first field of the .sha256 file is the digest" || fail "the .sha256 file's first field is not the digest: $(cat "$base/out/aria-x86_64-linux.sha256")"
[ "$(wc -l <"$base/out/aria-x86_64-linux.sha256")" -eq 1 ] && pass "the .sha256 file is one line" || fail "the .sha256 file is not one line"
# what the installer verifies: the real installer's download step against these files
fetched=$(TMPDIR="$base" ARIA_DOWNLOAD_URL="file://$base/out/aria-x86_64-linux" sh install/install.sh --fetch-binary 2>&1) && printf '%s\n' "$fetched" | grep -qF "fetched $want" && pass "the installer accepts the checksum file" || fail "the installer refused the assets: $fetched"
printf '%s\n' "$out" | grep -qF "$base/out/pi-packages" && pass "the pin files are in the printed list" || fail "the printed list lacks the pin files"
# the notes: the section of the tag, without its heading and without the others
[ "$(cat "$base/out/notes.md")" = "$(printf '### New\n\n- a new thing')" ] && pass "notes.md is the tag's section of RELEASES.md, without its heading" || fail "notes.md is wrong: $(cat "$base/out/notes.md")"
# the printed command, not run
printf '%s\n' "$out" | grep -qF "gh release create v0.3.0 $base/out/aria-x86_64-linux $base/out/aria-x86_64-linux.sha256" && pass "it prints gh release create with the tag and the artifact" || fail "no gh release create line: $out"
printf '%s\n' "$out" | grep -qF -- "--title \"Aria 0.3.0\"" && printf '%s\n' "$out" | grep -qF -- "--notes-file $base/out/notes.md" && printf '%s\n' "$out" | grep -qF -- '--verify-tag' && pass "the command has the title, the notes file and --verify-tag" || fail "the command lacks --title, --notes-file or --verify-tag: $out"
for f in install.sh pi-version pi-packages engram-version agent-browser-version node-version; do
  printf '%s\n' "$out" | grep -F 'gh release create' | grep -qF "$base/out/$f" || fail "the command does not upload $f"
done
[ ! -e "$base/gh-calls" ] && pass "gh was not run" || fail "gh was run: $(cat "$base/gh-calls")"
# a second run overwrites and is still clean (the default folder is ignored by git)
assets linux "$base/out" >/dev/null 2>&1 && pass "it can run again over the same folder" || fail "the second run failed"
# the default folder is dist/, and it is ignored (so the tree stays clean)
assets linux >/dev/null 2>&1 && [ -f dist/aria-x86_64-linux ] && [ -z "$(git status --porcelain)" ] && pass "the default folder is dist/ and git ignores it" || fail "the default folder is not an ignored dist/"
assets linux >/dev/null 2>&1 && pass "a run after dist/ exists is not refused as dirty" || fail "dist/ made the tree dirty"
cd "$root" && git check-ignore -q dist/aria-x86_64-linux && pass "this repository ignores dist/" || fail "this repository does not ignore dist/"

# the other platforms: the Mac names
scratch
assets mac-arm "$base/out-arm" >/dev/null 2>&1 && [ -f "$base/out-arm/aria-aarch64-apple-darwin" ] && [ -f "$base/out-arm/aria-aarch64-apple-darwin.sha256" ] && pass "Apple silicon: aria-aarch64-apple-darwin" || fail "no aria-aarch64-apple-darwin"
assets mac-intel "$base/out-intel" >/dev/null 2>&1 && [ -f "$base/out-intel/aria-x86_64-apple-darwin" ] && [ -f "$base/out-intel/aria-x86_64-apple-darwin.sha256" ] && pass "Intel Mac: aria-x86_64-apple-darwin" || fail "no aria-x86_64-apple-darwin"
[ ! -e "$base/out-arm/aria-x86_64-linux" ] && [ ! -e "$base/out-intel/aria-x86_64-linux" ] && pass "a Mac makes no Linux artifact" || fail "a Linux artifact appeared on a Mac"
# anything else is refused with a message, and nothing is written
if out=$(assets linux-arm "$base/out-arm64-linux" 2>&1); then fail "Linux aarch64 was accepted"
else printf '%s\n' "$out" | grep -qi 'aarch64' && printf '%s\n' "$out" | grep -qi 'x86_64' && [ ! -e "$base/out-arm64-linux" ] && pass "Linux aarch64 is refused, saying what is supported" || fail "the Linux aarch64 refusal is unclear: $out"; fi
if out=$(assets windows "$base/out-win" 2>&1); then fail "an unknown system was accepted"
else printf '%s\n' "$out" | grep -qF 'MINGW64_NT' && [ ! -e "$base/out-win" ] && pass "an unknown system is refused" || fail "the unknown system refusal is unclear: $out"; fi

# refusals: a dirty tree, a HEAD that is not a release tag, a version that is not the tag's
echo changed >>pi-version
if out=$(assets linux "$base/out-dirty" 2>&1); then fail "a dirty tree was accepted"
else printf '%s\n' "$out" | grep -qi 'clean\|changes' && [ ! -e "$base/out-dirty" ] && pass "a modified tracked file is refused" || fail "the dirty refusal is unclear: $out"; fi
git checkout -q -- pi-version
echo x >untracked.txt
assets linux "$base/out-dirty" >/dev/null 2>&1 && fail "an untracked file was accepted" || pass "an untracked file is refused too"
rm untracked.txt
echo y >log.txt && git add log.txt && git commit -q -m "feat: after the release"
if out=$(assets linux "$base/out-moved" 2>&1); then fail "a HEAD after the tag was accepted"
else printf '%s\n' "$out" | grep -qi 'tag' && [ ! -e "$base/out-moved" ] && pass "a HEAD that is not at a release tag is refused" || fail "the not-at-a-tag refusal is unclear: $out"; fi
git tag -a nightly -m "not a release"
assets linux "$base/out-moved" >/dev/null 2>&1 && fail "a tag that is not v* was accepted" || pass "a tag that is not v* is not a release tag"
git tag -a v0.4.0 -m "Aria 0.4.0"
if out=$(assets linux "$base/out-moved" 2>&1); then fail "a tag whose version is not Cargo.toml's was accepted"
else printf '%s\n' "$out" | grep -qF '0.3.0' && printf '%s\n' "$out" | grep -qF '0.4.0' && pass "a tag that is not the workspace version is refused, naming both" || fail "the version mismatch message is unclear: $out"; fi
git tag -d v0.4.0 >/dev/null
# RELEASES.md has no section for the tag
scratch
sed -i.bak '/^## 0.3.0/,/^## 0.2.0/{/^## 0.2.0/!d;}' RELEASES.md && rm -f RELEASES.md.bak
git commit -q -am "docs: lose the section" && git tag -f -a v0.3.0 -m "Aria 0.3.0" >/dev/null
if out=$(assets linux "$base/out-nonotes" 2>&1); then fail "a tag with no notes section was accepted"
else printf '%s\n' "$out" | grep -qF 'RELEASES.md' && pass "a tag with no section in RELEASES.md is refused" || fail "the missing-notes refusal is unclear: $out"; fi

# a missing binary hook, a missing pin file
scratch
if out=$(BIN="$base/nope" assets linux "$base/out-nobin" 2>&1); then fail "a missing ARIA_BINARY was accepted"
else printf '%s\n' "$out" | grep -qF 'ARIA_BINARY' && pass "a missing ARIA_BINARY is refused" || fail "the ARIA_BINARY refusal is unclear: $out"; fi
git rm -q engram-version && git commit -q -m "chore: drop a pin" && git tag -f -a v0.3.0 -m "Aria 0.3.0" >/dev/null
if out=$(assets linux "$base/out-nopin" 2>&1); then fail "a missing pin file was accepted"
else printf '%s\n' "$out" | grep -qF 'engram-version' && pass "a missing pin file is refused, naming it" || fail "the missing-pin refusal is unclear: $out"; fi

# with no ARIA_BINARY it builds: cargo build --release, default features, and takes target/release/aria
scratch
mkdir -p "$base/cargo"
printf '#!/bin/sh\necho "cargo $*" >> "%s/cargo-calls"\nmkdir -p target/release\ncp "%s/aria-binary" target/release/aria\n' "$base" "$base" >"$base/cargo/cargo"
chmod +x "$base/cargo/cargo"
PATH="$base/linux:$base/cargo:$PATH" sh tools/release-assets.sh "$base/out-built" >/dev/null 2>&1 && cmp -s "$base/aria-binary" "$base/out-built/aria-x86_64-linux" && pass "without ARIA_BINARY it builds and uses target/release/aria" || fail "the build route failed"
[ "$(cat "$base/cargo-calls")" = "cargo build --release" ] && pass "the build is exactly cargo build --release (default features)" || fail "cargo was called as: $(cat "$base/cargo-calls")"
# ARIA_TARGET cross-builds and names the artifact after the target, whatever this machine is
rm -f "$base/cargo-calls"
printf '#!/bin/sh\necho "cargo $*" >> "%s/cargo-calls"\nmkdir -p target/x86_64-apple-darwin/release\ncp "%s/aria-binary" target/x86_64-apple-darwin/release/aria\n' "$base" "$base" >"$base/cargo/cargo"
PATH="$base/mac-arm:$base/cargo:$PATH" ARIA_TARGET=x86_64-apple-darwin sh tools/release-assets.sh "$base/out-cross" >/dev/null 2>&1 && cmp -s "$base/aria-binary" "$base/out-cross/aria-x86_64-apple-darwin" && [ -f "$base/out-cross/aria-x86_64-apple-darwin.sha256" ] && [ ! -e "$base/out-cross/aria-aarch64-apple-darwin" ] && pass "ARIA_TARGET=x86_64-apple-darwin on an Apple silicon Mac makes the Intel artifact" || fail "the cross build failed or is named after this machine"
[ "$(cat "$base/cargo-calls")" = "cargo build --release --target x86_64-apple-darwin" ] && pass "a cross build is cargo build --release --target <target>" || fail "cargo was called as: $(cat "$base/cargo-calls")"
if out=$(PATH="$base/linux:$PATH" ARIA_TARGET=riscv64gc-unknown-linux-gnu sh tools/release-assets.sh "$base/out-riscv" 2>&1); then fail "an unknown ARIA_TARGET was accepted"; else printf '%s\n' "$out" | grep -qF 'riscv64gc-unknown-linux-gnu' && [ ! -e "$base/out-riscv" ] && pass "an unknown ARIA_TARGET is refused" || fail "the ARIA_TARGET refusal is unclear: $out"; fi
# a build that fails stops it, with nothing written
printf '#!/bin/sh\nexit 1\n' >"$base/cargo/cargo"
if PATH="$base/linux:$base/cargo:$PATH" sh tools/release-assets.sh "$base/out-failed" >/dev/null 2>&1; then fail "a failing build was ignored"; else [ ! -e "$base/out-failed/aria-x86_64-linux" ] && pass "a failing build stops it and leaves no artifact" || fail "an artifact appeared after a failing build"; fi

# the hub's binary: ARIA_HEADLESS=1 names it by CPU and builds without the desktop features
scratch
BIN="$base/aria-binary" ARIA_HEADLESS=1 assets linux "$base/out-hub" >/dev/null 2>&1 && [ -x "$base/out-hub/aria-x86_64-linux-headless" ] && [ -f "$base/out-hub/aria-x86_64-linux-headless.sha256" ] && [ ! -e "$base/out-hub/aria-x86_64-linux" ] && pass "ARIA_HEADLESS=1 on x86_64 Linux makes aria-x86_64-linux-headless" || fail "no aria-x86_64-linux-headless"
ARIA_HEADLESS=1 assets linux-arm "$base/out-hub-arm" >/dev/null 2>&1 && [ -x "$base/out-hub-arm/aria-aarch64-linux-headless" ] && pass "ARIA_HEADLESS=1 on aarch64 Linux makes aria-aarch64-linux-headless" || fail "no aria-aarch64-linux-headless"
fetched=$(TMPDIR="$base" ARIA_DOWNLOAD_URL="file://$base/out-hub-arm/aria-aarch64-linux-headless" sh install/install.sh --fetch-binary 2>&1) && printf '%s\n' "$fetched" | grep -qF "fetched " && pass "the installer accepts the hub's checksum file" || fail "the installer refused the hub's assets: $fetched"
if out=$(ARIA_HEADLESS=1 assets mac-arm "$base/out-hub-mac" 2>&1); then fail "a hub binary was made on a Mac"; else printf '%s\n' "$out" | grep -qF 'Darwin' && [ ! -e "$base/out-hub-mac" ] && pass "a hub binary is refused on a Mac" || fail "the Mac hub refusal is unclear: $out"; fi
rm -f "$base/cargo-calls"
mkdir -p "$base/cargo"
printf '#!/bin/sh\necho "cargo $*" >> "%s/cargo-calls"\nmkdir -p target/release\ncp "%s/aria-binary" target/release/aria\n' "$base" "$base" >"$base/cargo/cargo"
chmod +x "$base/cargo/cargo"
PATH="$base/linux:$base/cargo:$PATH" ARIA_HEADLESS=1 sh tools/release-assets.sh "$base/out-hub-built" >/dev/null 2>&1 && [ "$(cat "$base/cargo-calls")" = "cargo build --release --no-default-features -p aria" ] && pass "the hub is built with --no-default-features" || fail "the hub build was: $(cat "$base/cargo-calls" 2>/dev/null)"

# syntax, and shellcheck when it is installed
sh -n "$here/release-assets.sh" && pass "release-assets.sh parses" || fail "release-assets.sh has a syntax error"
if command -v shellcheck >/dev/null 2>&1; then shellcheck "$here/release-assets.sh" && pass "shellcheck is clean" || fail "shellcheck found problems"; else echo "skip: shellcheck is not installed"; fi

cd / || exit 1
[ "$failures" -eq 0 ] && echo "all checks passed" || {
  echo "$failures check(s) failed"
  exit 1
}
