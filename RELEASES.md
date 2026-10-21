# Releases

What changed in each version of Aria, newest first. Versions follow [SemVer](https://semver.org): while Aria is `0.x`,
the middle number goes up for new features or anything that changes how it behaves, the last one for fixes. Each version
is a git tag `vX.Y.Z` on `main`; `tools/release.sh` makes one (see the end of this file). `aria --version` and the
settings overview say which one is running.

## 0.5.0 — 2026-10-21

### New

- verify-in-browser, checking a page in a headless browser in one sandboxed call before saying it works
- the installer brings the pinned agent-browser and its browser, for the agents' browser check

### Fixed

- the browser check sets a 2-minute idle timeout and quotes each batch step for the shell, so a cut-off check stops by itself and evals keep their strings
- the browser test finds a leftover by its AGENT_BROWSER_SESSION environment, and closes from the host only after a failed run

## 0.4.0 — 2026-10-21

### New

- open the keyring vault at start, sync it every second, close it on quit; aria secret and aria keyring
- the setup, the import and the quota read and write through the keyring
- move plain keys into the keyring, verified, and the way back
- sealed credential files with a live copy outside ~/.aria
- credentials.keyring, on by default
- the system keyring behind one interface (security on macOS, secret-tool on Linux)
- aria quit and aria start; Cmd+Q asks first and quits on a second press

### Fixed

- Linux keeps long values in pieces, as secret-tool cuts a secret at 8192 bytes and still exits 0; the Cmd+Q check expects Super+Q left alone off a Mac
- the keyring sync runs off the UI thread and backs off when the keyring fails
- a vanished live copy is written back, never deleted; plain copies are merged into the sealed one; live files always exist; a migrated home keeps its keyring
- long values are kept in pieces below security's line limit, and names that could break its input are refused
- no access list on macOS items, so an update never raises a keychain dialog
- a migration seals every file before it removes any
- the page's defaults carry credentials.keyring too
- Quit Aria (Cmd+Q) stops the background work and the Pis first, as a signal does; macOS cannot end them with the dock
- the writable and protected folders are also compared by their real paths, so /tmp, /var and linked project folders work on macOS

## 0.3.0 — 2026-10-21

### New

- the next version is decided from the commits since the last tag (SemVer), and --next prints it

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
