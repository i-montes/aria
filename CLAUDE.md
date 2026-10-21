# Working on Aria

Aria is developed on more than one machine (Omarchy and macOS). These rules live here, not in any one machine's memory,
so every session follows them.

## Before a push of main

The repository is public. In this order:

1. **The hook is on**: `git config core.hooksPath tools/hooks` (once per clone). `tools/hooks/pre-push` refuses a push
   of main that carries personal data or commits that call for a release without one.
2. **No personal data** in what goes out: no real home paths, employer or client names, project or task names; test
   fixtures use neutral ones (`/home/ana`, `acme/checkout-widget`). Each clone keeps its own patterns, one extended
   regex per line, in `.git/info/personal-data` (never committed); the hook adds the machine's home path.
3. **The version**: `sh tools/release.sh --next`. If it prints a version, run `sh tools/release.sh` (it runs the suites,
   bumps Cargo.toml, writes the RELEASES.md section, commits and tags), read the new section, and push the tag with main:
   `git push origin main vX.Y.Z`. If it prints nothing, push main alone.
4. Ask the owner before pushing.

## Commits

Conventional subjects decide the version: `feat(scope): …` (minor), `fix(scope): …` (patch), `docs:`, `style:`,
`test:`, `chore:` (no release); `!` only for a real break. Everything is in English: code, names, messages, docs, commits.

## Both platforms

A change made on one platform must keep the other working: `sh tools/check-macos.sh` type-checks the macOS build from
Linux; on a Mac, run the Linux-only suites' counterparts or say they were not run. Behaviour that depends on the
platform (keys, user agent, service manager, keyring tool) is tested for both, or the test says which one it covers.
