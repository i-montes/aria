#!/bin/sh
# Makes the assets of a release on THIS machine and publishes them with `gh`: what .github/workflows/release.yml does,
# for when GitHub Actions cannot run (a billing lock, no runner). Run it on every machine that has a binary to give:
# Omarchy (the dock and the hub's two builds) and a Mac (its dock); the first creates the release, the next add to it.
#   sh tools/release-local.sh vX.Y.Z [--only dock|hub] [--no-publish] [--yes]
# The tag must exist (tools/release.sh) and be on GitHub (git push origin main vX.Y.Z). It builds from a clean worktree of
# the tag, never from your working tree, with every path under your home folder and the worktree rewritten, and it
# REFUSES to go on when a binary still holds your home folder: this repository is public and so are its binaries. It
# asks "Publish ...? [y/N]" before it uploads (--yes does not ask; --no-publish stops with the files in dist/<tag>/).
# Where each binary comes from:
#   Linux x86_64   aria-x86_64-linux (the dock, cargo build --release), and the hub's aria-x86_64-linux-headless and
#                  aria-aarch64-linux-headless (tools/build-headless.sh: zig, glibc 2.39, so they run on Ubuntu 24.04)
#   macOS          the dock of this Mac (aria-aarch64-apple-darwin or aria-x86_64-apple-darwin)
# The build goes to target/release-local (its own cache, kept between runs). Not signed: nothing here is, and a release made
# here is trusted as far as this machine is.
set -e

die() {
  echo "release-local: $*" >&2
  exit 1
}

tag=""
only=""
publish=1
yes=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only)
      [ $# -ge 2 ] || die "--only needs dock or hub"
      only=$2
      shift
      ;;
    --no-publish) publish=0 ;;
    --yes | -y) yes=1 ;;
    -*) die "unknown option: $1" ;;
    *) [ -z "$tag" ] || die "one tag only"; tag=$1 ;;
  esac
  shift
done
case "$only" in "" | dock | hub) ;; *) die "--only is dock or hub (got '$only')" ;; esac
case "$tag" in v[0-9]*.[0-9]*.[0-9]*) ;; *) die "usage: sh tools/release-local.sh vX.Y.Z [--only dock|hub] [--no-publish] [--yes]" ;; esac
case "$tag" in *[!A-Za-z0-9.+-]*) die "not a release tag: $tag" ;; esac

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
git rev-parse -q --verify "refs/tags/$tag" >/dev/null || die "there is no tag $tag here: run tools/release.sh first"
git ls-remote --exit-code --tags origin "refs/tags/$tag" >/dev/null 2>&1 || die "the tag $tag is not on GitHub: git push origin main $tag (a release is made for a tag that is there)"
[ "$publish" = 0 ] || command -v gh >/dev/null 2>&1 || die "gh is needed to publish (or pass --no-publish)"

work=$(mktemp -d)
cleanup() {
  git -C "$root" worktree remove --force "$work/src" >/dev/null 2>&1 || true
  rm -rf "$work"
}
trap cleanup EXIT
git worktree add -q --detach "$work/src" "$tag"
src=$work/src
out=$root/dist/$tag
rm -rf "$out"
mkdir -p "$out"

# Nothing of this machine in what goes out: the home folder (cargo's registry, rustup, the checkout) and the worktree are
# rewritten in the Rust paths and in the C and C++ ones (whisper.cpp), and the result is checked below.
remap="--remap-path-prefix=$HOME=/build --remap-path-prefix=$src=/src"
cremap="-ffile-prefix-map=$HOME=/build -ffile-prefix-map=$src=/src"
for dir in "${CARGO_HOME:-}" "${RUSTUP_HOME:-}"; do
  case "$dir" in
    "" | "$HOME"/*) ;;
    *) remap="$remap --remap-path-prefix=$dir=/build/$(basename "$dir")"; cremap="$cremap -ffile-prefix-map=$dir=/build/$(basename "$dir")" ;;
  esac
done
export RUSTFLAGS="${RUSTFLAGS:-} $remap"
export CFLAGS="${CFLAGS:-} $cremap"
export CXXFLAGS="${CXXFLAGS:-} $cremap"
export CARGO_TARGET_DIR="$root/target/release-local"

# A binary that still holds this machine's home folder is not published.
guard() {
  if grep -aqF "$HOME" "$1"; then
    echo "release-local: $1 still holds this machine's home folder, so it is not published:" >&2
    tr -c '[:print:]' '\n' <"$1" | grep -F "$HOME" | sort -u | head -3 | cut -c1-120 >&2
    exit 1
  fi
}

# collect <binary> [VAR=value...]: the file, its checksum, the pins, install.sh and the notes, by the script that names them.
collect() {
  binary=$1
  shift
  guard "$binary"
  (cd "$src" && env ARIA_BINARY="$binary" "$@" sh tools/release-assets.sh "$out" >/dev/null)
}

system=$(uname -s)
machine=$(uname -m)
case "$system/$machine" in
  Linux/x86_64)
    if [ "$only" != hub ]; then
      echo "release-local: building the dock"
      (cd "$src" && cargo build --release)
      collect "$CARGO_TARGET_DIR/release/aria"
    fi
    if [ "$only" != dock ]; then
      for triple in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu; do
        echo "release-local: building the hub for $triple"
        binary=$(cd "$src" && sh tools/build-headless.sh "$triple" | tail -1)
        [ -f "$binary" ] || die "the hub build for $triple left no binary"
        collect "$binary" ARIA_HEADLESS=1 ARIA_TARGET="$triple"
      done
    fi
    ;;
  Darwin/arm64 | Darwin/x86_64)
    [ "$only" != hub ] || die "a Mac makes no hub binary (--only hub)"
    echo "release-local: building the dock"
    (cd "$src" && cargo build --release)
    collect "$CARGO_TARGET_DIR/release/aria"
    ;;
  *) die "no release binary is made for $system $machine" ;;
esac

echo
echo "release-local: for $tag, in $out:"
for f in "$out"/*; do printf '  %s\n' "${f##*/}"; done
[ "$publish" = 1 ] || {
  echo "release-local: not published (--no-publish)"
  exit 0
}

if [ "$yes" != 1 ]; then
  printf 'Publish these to the GitHub release %s (public)? [y/N] ' "$tag"
  answer=""
  read -r answer </dev/tty 2>/dev/null || read -r answer || answer=""
  case "$answer" in y | Y | yes) ;; *) echo "release-local: nothing was published"; exit 0 ;; esac
fi

files=""
for f in "$out"/*; do
  case "$f" in */notes.md) ;; *) files="$files $f" ;; esac
done
if gh release view "$tag" >/dev/null 2>&1; then
  # A release that exists (another machine made it, or a half made one): add these files, replacing the ones of the same name.
  # shellcheck disable=SC2086
  gh release upload "$tag" $files --clobber
else
  # shellcheck disable=SC2086
  gh release create "$tag" $files --title "Aria ${tag#v}" --notes-file "$out/notes.md" --verify-tag
fi
echo "release-local: $tag is published with what this machine built"
