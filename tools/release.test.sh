#!/bin/sh
# Checks tools/release.sh in a scratch repository (never this one): the version, the notes, the commit and the tag.
#   sh tools/release.test.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
failures=0
pass() { echo "ok:   $1"; }
fail() {
  echo "FAIL: $1"
  failures=$((failures + 1))
}

scratch() {
  dir=$(mktemp -d)
  mkdir -p "$dir/tools"
  cp "$here/release.sh" "$dir/tools/"
  cd "$dir" || exit 1
  git init -q -b main
  git config user.email aria@localhost
  git config user.name Aria
  printf '[workspace]\nmembers = []\n\n[workspace.package]\nversion = "0.2.0"\nedition = "2021"\n' >Cargo.toml
  printf '# Releases\n\nIntro.\n\n## 0.2.0 — 2026-10-21\n\nFirst.\n' >RELEASES.md
  git add -A && git commit -q -m "start"
  git tag -a v0.2.0 -m "Aria 0.2.0"
  for subject in "feat(chat): a new thing" "fix: a broken thing" "docs: words" "feat: another thing"; do
    echo "$subject" >>log.txt && git add -A && git commit -q -m "$subject"
  done
}

scratch
sh tools/release.sh 0.3.0 --skip-checks >/dev/null 2>&1 && pass "a release runs" || fail "the release failed"
grep -q '^version = "0.3.0"' Cargo.toml && pass "Cargo.toml has the new version" || fail "Cargo.toml was not bumped"
git rev-parse -q --verify refs/tags/v0.3.0 >/dev/null && pass "v0.3.0 is tagged" || fail "no tag"
[ "$(git log -1 --format=%s)" = "release: 0.3.0" ] && pass "the release commit" || fail "wrong commit subject"
section=$(awk '/^## 0.3.0 /{on=1;next} /^## /&&on{exit} on' RELEASES.md)
echo "$section" | grep -q '^- a new thing$' && echo "$section" | grep -q '^- another thing$' && pass "features are listed" || fail "features missing"
echo "$section" | grep -A2 '^### Fixed' | grep -q '^- a broken thing$' && pass "fixes are listed" || fail "fixes missing"
echo "$section" | grep -q 'words' && fail "docs commits are listed" || pass "docs commits are left out"
[ "$(grep -n '^## ' RELEASES.md | head -1 | cut -d: -f2-)" = "## 0.3.0 — $(date +%Y-%m-%d)" ] && pass "the newest section comes first" || fail "section order"
grep -q 'Intro.' RELEASES.md && pass "the introduction stays" || fail "introduction lost"
git tag -n20 v0.3.0 | grep -q 'a new thing' && pass "the tag carries the notes" || fail "tag message without notes"

sh tools/release.sh 0.3.0 --skip-checks >/dev/null 2>&1 && fail "a version tagged twice" || pass "an existing tag is refused"
sh tools/release.sh 0.2.5 --skip-checks >/dev/null 2>&1 && fail "an older version was accepted" || pass "a version before the current one is refused"
sh tools/release.sh 1.0 --skip-checks >/dev/null 2>&1 && fail "a malformed version was accepted" || pass "a malformed version is refused"
echo dirty >>log.txt
sh tools/release.sh 0.4.0 --skip-checks >/dev/null 2>&1 && fail "a dirty tree was released" || pass "a dirty tree is refused"
git checkout -q -- log.txt
git checkout -q -b other
sh tools/release.sh 0.4.0 --skip-checks >/dev/null 2>&1 && fail "released from another branch" || pass "only main releases"
git checkout -q main

# A section written by hand before the release is kept as it is.
printf '# Releases\n\nIntro.\n\n## 0.4.0 — by hand\n\nWritten first.\n\n' >RELEASES.new && tail -n +4 RELEASES.md >>RELEASES.new && mv RELEASES.new RELEASES.md
git commit -q -am "docs: notes for 0.4.0"
sh tools/release.sh 0.4.0 --skip-checks >/dev/null 2>&1 && pass "a release with notes written first" || fail "the release with hand notes failed"
[ "$(grep -c '^## 0.4.0' RELEASES.md)" = 1 ] && grep -q 'Written first.' RELEASES.md && pass "hand-written notes are kept" || fail "hand notes replaced or doubled"

cd / && rm -rf "$dir"
[ "$failures" -eq 0 ] && echo "all checks passed" || {
  echo "$failures check(s) failed"
  exit 1
}
