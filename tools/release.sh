#!/bin/sh
# Makes a release of Aria: the version in the workspace's Cargo.toml, its notes in RELEASES.md, a commit and a tag.
# Nothing is pushed or published.
#   sh tools/release.sh X.Y.Z [--skip-checks]
set -e
cd "$(dirname "$0")/.."

die() {
  echo "release: $*" >&2
  exit 1
}

version=
checks=1
for arg in "$@"; do
  case "$arg" in
    --skip-checks) checks=0 ;;
    -*) die "unknown option $arg" ;;
    *) version=$arg ;;
  esac
done
echo "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' || die "usage: sh tools/release.sh X.Y.Z [--skip-checks]"

[ "$(git rev-parse --abbrev-ref HEAD)" = main ] || die "releases are made on main"
[ -z "$(git status --porcelain)" ] || die "the working tree has changes: commit or stash them first"
git rev-parse -q --verify "refs/tags/v$version" >/dev/null && die "v$version already exists"

current=$(sed -n '/^\[workspace.package\]/,/^\[/s/^version = "\(.*\)"/\1/p' Cargo.toml)
[ -n "$current" ] || die "no version in Cargo.toml's [workspace.package]"
# Never backwards; the current version itself only while it was never tagged (the first release).
newest=$(printf '%s\n%s\n' "$current" "$version" | sort -t. -k1,1n -k2,2n -k3,3n | tail -n 1)
[ "$newest" = "$version" ] || die "$version is before $current"
if [ "$version" = "$current" ] && git rev-parse -q --verify "refs/tags/v$current" >/dev/null; then
  die "$version is the current version and already tagged"
fi

if [ "$checks" = 1 ]; then
  sh tools/parity/suites.sh
fi

# From here on files change: a failure puts them back, so the tree is as it was.
trap 'git checkout -q HEAD -- Cargo.toml RELEASES.md; [ -f Cargo.lock ] && git checkout -q HEAD -- Cargo.lock; rm -f RELEASES.md.new; echo "release: stopped, nothing changed" >&2' EXIT

# The version: Cargo.toml, and Cargo.lock's entries for the workspace's own crates (offline when the registry is cached).
sed -i.bak "/^\[workspace.package\]/,/^\[/s/^version = \".*\"/version = \"$version\"/" Cargo.toml && rm -f Cargo.toml.bak
if [ -f Cargo.lock ]; then
  cargo update --workspace --offline -q 2>/dev/null || cargo update --workspace -q
fi

# The notes: a section written by hand stays; otherwise one from the feat and fix commits since the last tag.
if ! grep -Eq "^## $version( |$)" RELEASES.md; then
  last=$(git describe --tags --abbrev=0 --match 'v*' 2>/dev/null || true)
  range=${last:+$last..}HEAD
  # `feat: x`, `feat(scope): x`; with `!` before the colon, a breaking change.
  subjects() { git log --no-merges --format='%s' "$range" | sed -n "s/^$1\(([^)]*)\)\{0,1\}$2: /- /p"; }
  section=$(mktemp)
  {
    echo "## $version — $(date +%Y-%m-%d)"
    echo
    breaking=$(subjects feat '!'; subjects fix '!')
    features=$(subjects feat '')
    fixes=$(subjects fix '')
    if [ -n "$breaking" ]; then printf '### Breaking\n\n%s\n\n' "$breaking"; fi
    if [ -n "$features" ]; then printf '### New\n\n%s\n\n' "$features"; fi
    if [ -n "$fixes" ]; then printf '### Fixed\n\n%s\n\n' "$fixes"; fi
    if [ -z "$breaking$features$fixes" ]; then printf 'No feature or fix commits since %s.\n\n' "${last:-the start}"; fi
  } >"$section"
  # After the introduction (everything before the first "## " heading).
  awk -v file="$section" 'BEGIN { while ((getline line < file) > 0) text = text line "\n" }
    /^## / && !done { printf "%s", text; done = 1 } { print }
    END { if (!done) printf "\n%s", text }' RELEASES.md >RELEASES.md.new
  mv RELEASES.md.new RELEASES.md
  rm -f "$section"
fi

git add Cargo.toml RELEASES.md
[ -f Cargo.lock ] && git add Cargo.lock
# Nothing to commit when the version and its notes were already in place (the first release): the tag goes on HEAD.
git diff --cached --quiet || git commit -q -m "release: $version"
trap - EXIT
notes=$(awk -v v="$version" '$0 == "## " v || index($0, "## " v " ") == 1 { on = 1; next } /^## / && on { exit } on' RELEASES.md)
# `whitespace`: the default cleanup would drop the notes' `###` headings as comments.
git tag -a --cleanup=whitespace "v$version" -m "Aria $version" -m "$notes"
echo "release: v$version is tagged; push it with: git push origin main v$version"
