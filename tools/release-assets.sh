#!/bin/sh
# Collects what a GitHub release of Aria needs from THIS machine, and prints the command that publishes it. Nothing is
# pushed or published, and `gh` is never run.
#   sh tools/release-assets.sh [outdir]     outdir is dist/ by default (ignored by git)
# It refuses unless the working tree is clean and HEAD is exactly at the tag vX.Y.Z of Cargo.toml's version (so the
# assets are the ones of the tag). It builds `cargo build --release` (default features) unless ARIA_BINARY=<path> gives a
# binary that is already built (how the tests run), and cross-builds when ARIA_TARGET names a target (the workflow builds
# the Intel Mac's binary on an Apple silicon runner: ARIA_TARGET=x86_64-apple-darwin; the artifact is then the target's, not
# this machine's). ARIA_HEADLESS=1 makes the hub's binary instead (Linux only, `cargo build --release --no-default-features`,
# no desktop libraries) for this machine's CPU or ARIA_TARGET (x86_64-unknown-linux-gnu or aarch64-unknown-linux-gnu); the
# workflow runs it natively on an x86_64 and on an aarch64 runner, both glibc 2.39. The outdir gets:
#   aria-x86_64-linux | aria-aarch64-apple-darwin | aria-x86_64-apple-darwin   the binary, by this machine's platform
#   aria-x86_64-linux-headless | aria-aarch64-linux-headless                   the hub's binary (ARIA_HEADLESS=1)
#   <artifact>.sha256                       its digest in the format `sha256sum` and `shasum -a 256` print, which the installer checks
#   pi-version pi-packages engram-version agent-browser-version node-version   the pins, as in the repository root
#   install.sh                              install/install.sh, so the one-line install is a release asset too
#   notes.md                                the tag's section of RELEASES.md
# A release is made from the machines that have a binary to give: a Mac adds its binary to a release with
# `gh release upload <tag> <artifact> <artifact>.sha256`. tools/release.sh makes the tag; .github/workflows/release.yml
# does all of this from CI.
set -e

die() {
  echo "release-assets: $*" >&2
  exit 1
}

# An outdir given relative is relative to where the command was typed, not to the repository.
case "${1:-}" in
  "") outdir=dist ;;
  /*) outdir=$1 ;;
  *) outdir=$PWD/$1 ;;
esac
cd "$(dirname "$0")/.."
# The folder's name goes into a command for the owner to paste: it must need no quoting.
case "$outdir" in *[!A-Za-z0-9_./+~-]*) die "the output folder's name must be plain (letters, digits and _ . / + ~ -): $outdir" ;; esac

PINS="pi-version pi-packages engram-version agent-browser-version node-version"

[ -z "$(git status --porcelain)" ] || die "the working tree has changes: commit or stash them (or remove untracked files) first, so the assets are those of the tag"

version=$(sed -n '/^\[workspace.package\]/,/^\[/s/^version = "\(.*\)"/\1/p' Cargo.toml)
[ -n "$version" ] || die "no version in Cargo.toml's [workspace.package]"
tag=v$version
at_head=$(git tag --points-at HEAD --list 'v*')
[ -n "$at_head" ] || die "HEAD is not at a release tag (a v* tag such as $tag): run tools/release.sh first, and check out the tag if you moved on"
printf '%s\n' "$at_head" | grep -qx "$tag" || die "HEAD is tagged $(printf '%s' "$at_head" | tr '\n' ' ')but Cargo.toml's version is $version, so the tag should be $tag"

# The platform (or the target asked for) decides the artifact name, which is what the installer asks for.
if [ "${ARIA_HEADLESS:-}" = 1 ]; then
  if [ -n "${ARIA_TARGET:-}" ]; then
    case "$ARIA_TARGET" in
      x86_64-unknown-linux-gnu) artifact=aria-x86_64-linux-headless ;;
      aarch64-unknown-linux-gnu) artifact=aria-aarch64-linux-headless ;;
      *) die "ARIA_TARGET=$ARIA_TARGET has no hub artifact (x86_64-unknown-linux-gnu or aarch64-unknown-linux-gnu)" ;;
    esac
  else
    system=$(uname -s)
    machine=$(uname -m)
    case "$system/$machine" in
      Linux/x86_64) artifact=aria-x86_64-linux-headless ;;
      Linux/aarch64) artifact=aria-aarch64-linux-headless ;;
      *) die "no hub binary is made for $system $machine (a hub is Linux, x86_64 or aarch64)" ;;
    esac
  fi
elif [ -n "${ARIA_TARGET:-}" ]; then
  case "$ARIA_TARGET" in
    x86_64-unknown-linux-gnu) artifact=aria-x86_64-linux ;;
    aarch64-apple-darwin) artifact=aria-aarch64-apple-darwin ;;
    x86_64-apple-darwin) artifact=aria-x86_64-apple-darwin ;;
    *) die "ARIA_TARGET=$ARIA_TARGET has no release artifact (x86_64-unknown-linux-gnu, aarch64-apple-darwin or x86_64-apple-darwin)" ;;
  esac
else
  system=$(uname -s)
  machine=$(uname -m)
  case "$system/$machine" in
    Linux/x86_64) artifact=aria-x86_64-linux ;;
    Darwin/arm64) artifact=aria-aarch64-apple-darwin ;;
    Darwin/x86_64) artifact=aria-x86_64-apple-darwin ;;
    *) die "no release binary is made for $system $machine (it is Linux x86_64, or a Mac with Apple silicon or Intel)" ;;
  esac
fi

for pin in $PINS install/install.sh; do [ -f "$pin" ] || die "$pin is missing from the repository"; done
notes=$(awk -v v="$version" '$0 == "## " v || index($0, "## " v " ") == 1 { on = 1; next } /^## / && on { exit } on' RELEASES.md)
# Without the blank lines the heading leaves in front (the command substitution drops those at the end).
notes=$(printf '%s\n' "$notes" | sed '/./,$!d')
[ -n "$notes" ] || die "RELEASES.md has no section for $version: run tools/release.sh (or write the section) before the assets"

if [ -n "${ARIA_BINARY:-}" ]; then
  binary=$ARIA_BINARY
  [ -f "$binary" ] || die "ARIA_BINARY is not a file: $binary"
elif [ "${ARIA_HEADLESS:-}" = 1 ]; then
  if [ -n "${ARIA_TARGET:-}" ]; then
    cargo build --release --no-default-features -p aria --target "$ARIA_TARGET"
    binary=target/$ARIA_TARGET/release/aria
  else
    cargo build --release --no-default-features -p aria
    binary=target/release/aria
  fi
  [ -f "$binary" ] || die "the build left no $binary"
elif [ -n "${ARIA_TARGET:-}" ]; then
  cargo build --release --target "$ARIA_TARGET"
  binary=target/$ARIA_TARGET/release/aria
  [ -f "$binary" ] || die "the build left no $binary"
else
  cargo build --release
  binary=target/release/aria
  [ -f "$binary" ] || die "the build left no $binary"
fi

mkdir -p "$outdir"
cp "$binary" "$outdir/$artifact"
chmod 755 "$outdir/$artifact"
# From inside the folder, so the file name in the line is the artifact's, as in `sha256sum <artifact> > <artifact>.sha256`.
# The installer reads the first field; sha256sum and shasum print the digest, two spaces and the name.
if command -v sha256sum >/dev/null 2>&1; then
  (cd "$outdir" && sha256sum "$artifact" >"$artifact.sha256")
else
  (cd "$outdir" && shasum -a 256 "$artifact" >"$artifact.sha256")
fi
for pin in $PINS; do cp "$pin" "$outdir/$pin"; done
cp install/install.sh "$outdir/install.sh"
printf '%s\n' "$notes" >"$outdir/notes.md"

echo "release-assets: $artifact for $tag is in $outdir"
echo
echo "The tag must be on GitHub first (git push origin $tag). Then publish the release with:"
echo
files="$outdir/$artifact $outdir/$artifact.sha256"
for pin in $PINS install.sh; do files="$files $outdir/$pin"; done
echo "gh release create $tag $files --title \"Aria $version\" --notes-file $outdir/notes.md --verify-tag"
echo
echo "A release that already exists gets this machine's binary with: gh release upload $tag $outdir/$artifact $outdir/$artifact.sha256"
