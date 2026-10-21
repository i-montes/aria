# Releases

What changed in each version of Aria, newest first. Versions follow [SemVer](https://semver.org): while Aria is `0.x`,
the middle number goes up for new features or anything that changes how it behaves, the last one for fixes. Each version
is a git tag `vX.Y.Z` on `main`; `tools/release.sh` makes one (see the end of this file). `aria --version` and the
settings overview say which one is running.

## 0.2.0 — 2026-10-21

The first tagged version: everything since the project started.

### The dock and the setup

- A dock on the right of the screen, opened with `SUPER + A` on Omarchy (bound at run time, no edits to the Hyprland
  config) or `Option + Space` on macOS; the shortcut can be chosen in the settings, with fallbacks when it is taken.
- First-run setup and a settings screen: provider (OpenCode Go, an Anthropic API key, or a Claude sign-in), models
  (daily and heavy; MiMo V2.6 Flash and Pro by default on OpenCode Go), the router, effort, context, memory,
  integrations (Notion, Tavily, the MCP servers other apps already have, each connector's access level), safety and
  agents.
- An installer for Omarchy and macOS that installs the pinned Pi, Engram and the packages, and updates without cutting a
  run.

### The chat

- One Pi per tab (`pi --mode rpc`), the conversation and its history, markdown in the bubbles, the `/` menu with Pi's
  own commands and the skills, the model picker, the quota card, messages queued or put in at the run's next step.
- The owner's extensions inside Aria: the sandbox (bubblewrap) with its permission cards, the router (Jev), the quota
  fallback, autocompact, questions, subagents and their watch, the clock.
- Background work: never-ending commands and long jobs run in the background, owned by the dock, as pills above the
  composer and in the Alt+A list; subagents in their own git worktree when the repository is taken.
- Connectors ask before they write (read only, ask or allow, per connector).
- Attachments: files and images by the paperclip, drop or paste; images go to models that can see them.
- A turn that stops answering is stopped, and the person told.

## Making a release

```sh
sh tools/release.sh --next    # which version the commits since the last tag call for (nothing: no release needed)
sh tools/release.sh           # checks, bumps to that version, writes the notes, commits "release: X.Y.Z", tags vX.Y.Z
git push origin main vX.Y.Z   # when the notes read right
```

The version is decided from the commit subjects since the last tag, by SemVer: a breaking change (`feat!:`, `fix!:`,
or `BREAKING CHANGE:` in a commit body) raises the major number, or the middle one while Aria is `0.x`; a `feat`
raises the middle one; a `fix` the last one. Commits of other kinds (`docs`, `style`, `test`, `chore`, `perf`…) make no
release on their own. `sh tools/release.sh X.Y.Z` picks a version by hand instead.

The script stops on a dirty tree, a branch other than `main`, a version that is already tagged or comes before the
current one. When `RELEASES.md` has no section for the version yet, it writes one from the `feat` and `fix` commits
(edit it before pushing); a section written by hand is kept as it is. A failure puts the files back.
`--skip-checks` leaves out the test suites.
