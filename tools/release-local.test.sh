#!/bin/sh
# Checks tools/release-local.sh in a scratch repository with a bare "origin" and stubs for cargo, gh, uname and the hub's
# build (never this repository, never a real build, never GitHub): what it builds where, the guard against this machine's
# home folder, the checks on the tag, and what it hands to gh.
#   sh tools/release-local.test.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
failures=0
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
pass() { echo "ok:   $1"; }
fail() {
  echo "FAIL: $1"
  failures=$((failures + 1))
}

base=$(mktemp -d)
trap 'cd / && rm -rf "$base"' EXIT
home="$base/home"
mkdir -p "$home"

fake_uname() { # name, system, machine
  mkdir -p "$base/$1"
  printf '#!/bin/sh\ncase "$1" in -s) echo %s ;; -m) echo %s ;; *) echo %s ;; esac\n' "$2" "$3" "$2" >"$base/$1/uname"
  chmod +x "$base/$1/uname"
}
fake_uname linux Linux x86_64
fake_uname mac Darwin arm64
fake_uname pi Linux riscv64
# cargo: records how it was called and what the flags were, and leaves a binary (holding $LEAK when a test asks for it).
mkdir -p "$base/bin"
cat >"$base/bin/cargo" <<STUB
#!/bin/sh
echo "cargo \$* | RUSTFLAGS=\$RUSTFLAGS | CFLAGS=\$CFLAGS | TARGET=\$CARGO_TARGET_DIR" >> "$base/cargo-calls"
mkdir -p "\$CARGO_TARGET_DIR/release"
printf 'a binary %s\n' "\${LEAK:-clean}" > "\$CARGO_TARGET_DIR/release/aria"
STUB
# gh: records every call; "release view" says the release exists only when GH_EXISTS=1.
cat >"$base/bin/gh" <<STUB
#!/bin/sh
echo "gh \$*" >> "$base/gh-calls"
if [ "\$1 \$2" = "release view" ]; then [ "\${GH_EXISTS:-0}" = 1 ]; exit \$?; fi
STUB
chmod +x "$base/bin/cargo" "$base/bin/gh"

scratch() {
  dir=$(mktemp -d "$base/repo.XXXXXX")
  origin=$(mktemp -d "$base/origin.XXXXXX")
  git init -q --bare "$origin"
  mkdir -p "$dir/tools" "$dir/install"
  cp "$root/tools/release-assets.sh" "$root/tools/release-local.sh" "$dir/tools/"
  cp "$root/install/install.sh" "$dir/install/"
  # The hub's build is a stub: it leaves a binary where the real one would and prints where.
  cat >"$dir/tools/build-headless.sh" <<'STUB'
#!/bin/sh
mkdir -p "$CARGO_TARGET_DIR/$1/release"
printf 'a hub binary %s\n' "${LEAK:-clean}" > "$CARGO_TARGET_DIR/$1/release/aria"
echo "$CARGO_TARGET_DIR/$1/release/aria"
STUB
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
  git remote add origin "$origin"
  git push -q origin main v0.3.0
  rm -f "$base/cargo-calls" "$base/gh-calls"
}
# $1 is the PATH directory with the fake uname; the rest are the script's arguments.
release() { uname_dir=$1; shift; PATH="$base/$uname_dir:$base/bin:$PATH" HOME="$home" sh tools/release-local.sh "$@"; }

# Linux x86_64: the dock and the hub's two builds, each with its checksum, the pins and the installer
scratch
out=$(release linux v0.3.0 --no-publish 2>&1) && pass "it builds on Linux x86_64" || fail "the build failed: $out"
for f in aria-x86_64-linux aria-x86_64-linux-headless aria-aarch64-linux-headless; do
  [ -x "dist/v0.3.0/$f" ] && [ -f "dist/v0.3.0/$f.sha256" ] && pass "dist/v0.3.0 has $f and its checksum" || fail "dist/v0.3.0 lacks $f"
done
for f in install.sh pi-version pi-packages engram-version agent-browser-version node-version notes.md; do
  [ -f "dist/v0.3.0/$f" ] || fail "dist/v0.3.0 lacks $f"
done
[ -z "$(git status --porcelain)" ] && pass "the working tree is left as it was (the build is in a worktree)" || fail "the working tree changed: $(git status --porcelain)"
[ "$(git worktree list | wc -l)" -eq 1 ] && pass "the worktree is removed" || fail "a worktree is left: $(git worktree list)"
[ ! -e "$base/gh-calls" ] && pass "--no-publish never runs gh" || fail "gh ran with --no-publish: $(cat "$base/gh-calls")"
grep -q "RUSTFLAGS=.*--remap-path-prefix=$home=/build" "$base/cargo-calls" && pass "Rust paths under the home folder are rewritten" || fail "no remap in: $(cat "$base/cargo-calls")"
grep -q "CFLAGS=.*-ffile-prefix-map=$home=/build" "$base/cargo-calls" && pass "C paths under the home folder are rewritten too" || fail "no C remap"
grep -q "TARGET=.*/target/release-local" "$base/cargo-calls" && pass "it builds in its own target folder" || fail "the target folder is not target/release-local"
grep -q 'cargo build --release ' "$base/cargo-calls" && pass "the dock is cargo build --release" || fail "the dock build was: $(cat "$base/cargo-calls")"
# --only
scratch
release linux v0.3.0 --only hub --no-publish >/dev/null 2>&1 && [ ! -e "$base/cargo-calls" ] && [ -f dist/v0.3.0/aria-aarch64-linux-headless ] && [ ! -e dist/v0.3.0/aria-x86_64-linux ] && pass "--only hub builds no dock" || fail "--only hub built the dock"
scratch
release linux v0.3.0 --only dock --no-publish >/dev/null 2>&1 && [ -f dist/v0.3.0/aria-x86_64-linux ] && [ ! -e dist/v0.3.0/aria-aarch64-linux-headless ] && pass "--only dock builds no hub" || fail "--only dock built the hub"
release linux v0.3.0 --only both >/dev/null 2>&1 && fail "--only both was accepted" || pass "--only takes dock or hub"

# the guard: a binary that holds the home folder is not published
scratch
if out=$(LEAK="$home/.cargo/registry/src/x/lib.rs" release linux v0.3.0 --yes 2>&1); then fail "a binary with the home folder was accepted"
else printf '%s\n' "$out" | grep -qF "home folder" && printf '%s\n' "$out" | grep -qF "$home/.cargo/registry" && [ ! -e "$base/gh-calls" ] && pass "a binary that holds the home folder stops the run before gh, and says what it found" || fail "the guard is unclear or gh ran: $out"; fi

# the checks on the tag
scratch
release linux 0.3.0 --no-publish >/dev/null 2>&1 && fail "a version without v was accepted" || pass "a tag without v is refused"
release linux v9.9.9 --no-publish >/dev/null 2>&1 && fail "a tag that does not exist was accepted" || pass "a tag that does not exist is refused"
git tag -a v0.4.0 -m local-only
if out=$(release linux v0.4.0 --no-publish 2>&1); then fail "a tag that is not on GitHub was accepted"; else printf '%s\n' "$out" | grep -qF 'not on GitHub' && pass "a tag that is not on GitHub is refused, saying to push it" || fail "the unpushed-tag refusal is unclear: $out"; fi
release pi v0.3.0 --no-publish >/dev/null 2>&1 && fail "an unknown platform was accepted" || pass "a platform with no build is refused"

# publishing: gh create for a new release, upload for one that exists, nothing on a no
scratch
release linux v0.3.0 --yes >/dev/null 2>&1
grep -q '^gh release create v0.3.0 .*aria-x86_64-linux-headless.* --title Aria 0.3.0 --notes-file .*notes.md --verify-tag' "$base/gh-calls" && pass "a new release is gh release create, with the title, the notes and --verify-tag" || fail "gh was called as: $(cat "$base/gh-calls")"
grep '^gh release create' "$base/gh-calls" | grep -q ' [^ ]*/notes.md [^ ]*$\| [^ ]*/notes.md ' && grep '^gh release create' "$base/gh-calls" | sed 's/--notes-file [^ ]*//' | grep -q 'notes.md' && fail "notes.md is uploaded as a file" || pass "notes.md is the notes, not an asset"
rm -f "$base/gh-calls"
GH_EXISTS=1 release linux v0.3.0 --yes >/dev/null 2>&1
grep -q '^gh release upload v0.3.0 .* --clobber' "$base/gh-calls" && ! grep -q 'release create' "$base/gh-calls" && pass "a release that exists gets gh release upload --clobber" || fail "gh was called as: $(cat "$base/gh-calls")"
rm -f "$base/gh-calls"
if ( : </dev/tty ) 2>/dev/null; then echo "skip: the 'no' answer is read from the terminal, and this run has one"
else
  printf 'n\n' | release linux v0.3.0 >/dev/null 2>&1
  ! grep -q 'release create\|release upload' "$base/gh-calls" 2>/dev/null && pass "answering no publishes nothing" || fail "gh published on a no: $(cat "$base/gh-calls")"
fi

# a Mac builds its dock and no hub
scratch
release mac v0.3.0 --no-publish >/dev/null 2>&1 && [ -f dist/v0.3.0/aria-aarch64-apple-darwin ] && [ -f dist/v0.3.0/aria-aarch64-apple-darwin.sha256 ] && [ ! -e dist/v0.3.0/aria-aarch64-linux-headless ] && pass "a Mac makes aria-aarch64-apple-darwin and no hub binary" || fail "the Mac build is wrong"
release mac v0.3.0 --only hub --no-publish >/dev/null 2>&1 && fail "--only hub was accepted on a Mac" || pass "--only hub is refused on a Mac"

sh -n "$here/release-local.sh" && pass "release-local.sh parses" || fail "release-local.sh has a syntax error"
if command -v shellcheck >/dev/null 2>&1; then shellcheck "$here/release-local.sh" && pass "shellcheck is clean" || fail "shellcheck found problems"; else echo "skip: shellcheck is not installed"; fi

cd / || exit 1
[ "$failures" -eq 0 ] && echo "all checks passed" || {
  echo "$failures check(s) failed"
  exit 1
}
