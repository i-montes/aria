#!/bin/sh
# Checks tools/hooks/pre-push in a scratch repository (never this one): personal data and an unreleased version stop a
# push of main; a released one goes through.
#   sh tools/hooks/pre-push.test.sh
# shellcheck disable=SC2015 # pass/fail never fail themselves
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
failures=0
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
pass() { echo "ok:   $1"; }
fail() {
  echo "FAIL: $1"
  failures=$((failures + 1))
}

dir=$(mktemp -d)
git init -q --bare "$dir/remote.git"
git init -q -b main "$dir/work"
cd "$dir/work" || exit 1
mkdir -p tools/hooks
cp "$here/release.sh" tools/
cp "$here/hooks/pre-push" tools/hooks/
git config user.email aria@localhost
git config user.name Aria
git config core.hooksPath tools/hooks
git remote add origin "$dir/remote.git"
printf '[workspace]\nmembers = []\n\n[workspace.package]\nversion = "0.2.0"\nedition = "2021"\n' >Cargo.toml
printf '# Releases\n\nIntro.\n\n## 0.2.0 — 2026-10-21\n\nFirst.\n' >RELEASES.md
git add -A && git commit -q -m "start" && git tag -a v0.2.0 -m "Aria 0.2.0"
git push -q origin main v0.2.0 2>/dev/null && pass "a released main goes through" || fail "a released main was refused"

echo "words" >notes.md && git add -A && git commit -q -m "docs: words"
git push -q origin main 2>/dev/null && pass "a docs commit needs no release" || fail "a docs commit was refused"

echo "x" >a.txt && git add -A && git commit -q -m "feat: a thing"
out=$(git push origin main 2>&1) && fail "an unreleased feat went through" || {
  echo "$out" | grep -q 'call for v0.3.0' && pass "an unreleased feat is refused, naming the version" || fail "no version in: $out"
}
sh tools/release.sh --skip-checks >/dev/null 2>&1
git push -q origin main v0.3.0 2>/dev/null && pass "after the release it goes through" || fail "the released feat was refused"

echo "see $HOME/secret" >b.txt && git add -A && git commit -q -m "docs: a path"
git push -q origin main 2>/dev/null && fail "a home path went through" || pass "a home path is refused"
git reset -q --hard HEAD~1

echo "acme-corp" >"$(git rev-parse --git-dir)/info/personal-data"
echo "works at ACME-corp" >c.txt && git add -A && git commit -q -m "docs: an employer"
git push -q origin main 2>/dev/null && fail "a listed pattern went through" || pass "a listed pattern is refused"

rm -rf "$dir"
[ "$failures" = 0 ] && echo "all checks passed" || { echo "$failures check(s) failed"; exit 1; }
