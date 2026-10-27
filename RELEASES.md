# Releases

What changed in each version of Aria, newest first. Versions follow [SemVer](https://semver.org): while Aria is `0.x`,
the middle number goes up for new features or anything that changes how it behaves, the last one for fixes. Each version
is a git tag `vX.Y.Z` on `main`; `tools/release.sh` makes one (see the end of this file). `aria --version` and the
settings overview say which one is running.

## 0.22.0 — 2026-10-26

### New

- a server updates itself through the restricted key, so aria hub update needs no admin login
- aria hub update brings a server to this Aria's version, and the terminal says when the versions differ
- installing the dock also brings the linked servers to the same version
- aria hub init draws a guided panel with ratatui on a terminal
- aria hub init is a guided screen in plain words, with the technical plan behind --details

### Fixed

- the closing line names a binary that works, ssh-keygen is quiet, and the connectors count as one thing
- run by aria hub init, the installer is one compact step of a longer screen
- the screen fits a narrow terminal, every step ends with a check mark, and a hub gives an address that works

## 0.21.0 — 2026-10-26

### New

- aria hub init needs no checkout, makes its key and remembers the hub
- the installer asks for this computer or a hub, and sets a hub up by itself
- a release carries the installer, the dock and the hub's binaries

## 0.20.2 — 2026-10-26

### Fixed

- the model picker reads the live credentials, filters where you type and hides the input

## 0.20.1 — 2026-10-23

### Fixed

- the store refuses a kept writable grant that was swapped for a link out; docs state the grant-rebinding limit
- hub init accepts only from=, expiry-time= and verify-required after the forced command
- secret_import opens the file once and judges what it opened
- the permission footer names exactly the letters whose option exists; the docs say which folder @ can list
- hub init sends and plans from what the bundle holds, and says a /proc mount failure is not the AppArmor one
- hub init allows only from=, expiry-time= and no-* options after the forced command
- the sandbox and credentials texts say what an off toggle and a missing keyring do; ask-user drops an Other in more languages
- every granted writable root keeps its git hooks, git config, .pi and .aria read-only
- the language written into aria.json is one line of at most 40 characters
- a run judges each writable grant by its real path; a task names at most 16; a projects folder of / or a relative one allows nothing
- Alt+A and Alt+D fire on the key labelled with the letter on any layout, never with Ctrl or Cmd
- secret_import judges a file by its real path against the real project and projects folder
- the settings and setup texts say what the code does, and the welcome screen counts its steps from the real list
- the tool asks in the conversation's language, named from aria.json, not in Spanish
- Alt+A is read by the key's place, so Option+A works on a Mac
- Enter on a / word no command matches sends it as a message, as the dock does
- @ lists the session's folder, not the projects folder
- a permission card's footer names the keys that work (a number or a click), not y/a/n
- aria start starts the unit that is installed, so it is right on a hub too
- hub init checks the forced command, names the projects folder only when it sends settings, and says it reads the keyring
- the bwrap probes mount /dev and /proc like the real call
- a remote sessionClose for an unknown session is refused
- secret_import reads only inside the project or the projects folder
- a task may write only inside the projects folder or /tmp
- only the dock page's Approve approves a waiting task; the service refuses it from every client

## 0.20.0 — 2026-10-23

### New

- aria serve exits 1 when terminals cannot connect
- sign in from a terminal by pasting the redirect address
- /schedules lists, pauses and deletes the scheduled tasks
- aria hub init installs a hub over ssh and says what it could not do
- the bundle a hub is set up from, and the import that writes it
- a headless install, a private Node, and a cross-build for the server
- aria serve runs the service with no window, behind the gui feature
- aria connect opens the terminal on another computer's service over ssh
- aria client-stdio, a bridge from ssh to the clients socket
- a hello with a protocol number, and what a remote client may not do

### Fixed

- the terminal says what a hub refuses, a banner before the hello is skipped, and the docs say what is real
- a non-login ssh gets the user manager's directory, a hub without linger is not called ready, and a server skips the browser
- running hub init again keeps the service running and what the hub already has
- the quick start uses a dedicated key, and hub init prints the ssh entry that makes connect use it
- where the box does not work every command asks, and the client key waits for a working sandbox
- a remote client can talk only to sessions the hub has
- a remote client's Pi commands are the terminal's own, and the whole surface is pinned
- a remote client never chooses a new session's folder
- texts for the sign-in errors the dock can now emit
- a resume in /schedules is checked against a fresh list and never approves a waiting task
- the client key is added whole and restricted, a failed ss is said, and a failing step says what is on the hub
- a credential file left out because the keyring could not be opened is reported
- the private Node is checked against a hash pinned in the repo
- aria connect gives ssh 60 s to log in, and shows no local git for a remote session

## 0.19.0 — 2026-10-23

### New

- a slice phase's tasks, live in the dock and the terminal
- the slice plan as a card, approved in the thread
- slice instead of Superpowers
- plan_ready approves a slice plan and starts its build
- after a slice job, the next step
- the slice skill
- plan mode is slice's phase A
- read slice plans and brief their phases
- a slice phase's subagent commits in its folder
- a subagent may be allowed to commit
- Ponytail in every session; from Superpowers only systematic-debugging
- Jev through TypeSafe when its key is the one stored
- TypeSafe as a provider for Jev; Jev stays on under Claude with a Go or TypeSafe key

### Fixed

- slice.ts no longer loads modes.ts, so a test's projects folder is not fixed by whoever imports it first
- texts that still described /plan as Superpowers' spec and plan
- the slice skill's dirty-tree check accepts its own leftovers
- the build starts in the plan's repo, unclassified, with a marker that survives an empty base
- leaving plan mode cancels an approved build that has not started
- a fresh onboarding turns on only systematic-debugging
- no Jev key at finish turns Jev's sandbox review off with the router

## 0.18.0 — 2026-10-23

### New

- notices come back when a session is opened again
- the footer says while Jev cannot route
- Jev steps aside model by model
- the paid Jev answers while the free one's quota is spent

### Fixed

- Aria's notices carry a mark and a colour of their own

## 0.17.1 — 2026-10-23

### Fixed

- a failing check first only for code with tests

## 0.17.0 — 2026-10-23

### New

- Rewind, as Claude Code draws it
- focused work starts from its outcome and its check
- a notification when a turn ends where nobody looks
- /context and checks after editing
- tasks above the editor
- the changes of each instruction, and undo
- checkpoints of a session's folder
- what each session is doing reaches the terminal

### Fixed

- no system folder keeps checkpoints, nor one with too many files
- Esc Esc and /undo say why there is nothing to undo
- a notification per session that never touches another's; undo says when it could not
- undo finds its place first, holds the tab, and checkpoints go with their session
- undo deletes only what the turn made; checkpoints ignore the person's git setup
- checks never hold the session, and the model cannot name them
- only what a terminal needs, an old dock said at once, reconnecting
- pasting, Ctrl+L, full outputs, clearer words

## 0.16.0 — 2026-10-23

### New

- the brief view lists each step, as the dock's accordion
- select with the mouse and it is copied; a frame only when something changed
- the queue over the editor, and Ctrl+Enter interrupts
- suggestions float above the editor
- the working line, as the dock's
- aria in a terminal, attached to the dock's sessions
- the editor, pickers and dialogs
- the session view
- the conversation as data
- the client's link to the dock
- display.verbose, on by default
- the service is `aria dock`; plain `aria` in a terminal is the client
- the dock serves its sessions to terminal clients
- the dock can open and close tabs, and never leaves none
- a session can work in its own folder
- the model picker lists only connected providers, newest model first

### Fixed

- the effort shown follows Pi's thinking level
- a message a terminal sends shows up in the dock
- the page gets its replies again, whatever its ids
- a tab a terminal closes never leaves the dock without an active one
- scroll the conversation, a cursor where the editor's is, the terminal given back on errors
- a dock the terminal starts outlives that terminal
- only a terminal can run a !command, never the dock's page

## 0.15.0 — 2026-10-22

### New

- several providers connected at once, one of them the main one

### Fixed

- the jobs client listens for errors before it connects
- the browser checks use Chrome for Testing and never leave it running

## 0.14.0 — 2026-10-22

### New

- passwords, keys and tokens the model asks for and never sees
- Jev offers to plan sizable work, and /plan goes straight from questions to the plan

### Fixed

- a resumed session gets the tools a newer Aria added
- Linux builds and runs it: whisper-rs 0.16, Vulkan, one thread per core

## 0.13.2 — 2026-10-22

### Fixed

- Aria is signed ad hoc as com.aria.dock on macOS, its Info.plist bound
- the microphone takes one click: its button changes in place as the seconds tick
- a second or less of voice, or silence, is not transcribed; Whisper's made-up closings are dropped

## 0.13.1 — 2026-10-22

### Fixed

- while a dictation is written, the microphone says so with a spinner

## 0.13.0 — 2026-10-22

### New

- dictate with the microphone button
- the dock dictates for the tab that asks
- record, transcribe with Whisper, unload when idle
- audio to 16 kHz mono, and the voice model's file

### Fixed

- whisper.cpp's load lines stay out of the dock's log; a live check of the microphone
- /debug's wide surface keeps the monitor's height on Omarchy

## 0.12.2 — 2026-10-22

### Fixed

- the dock after WebKitGTK 2.54: shared-memory buffers, and nothing animates it on Omarchy

## 0.12.1 — 2026-10-22

### Fixed

- the wing reads as part of the notch: it runs on under it, flares into the top edge, rounds its bottom corner

## 0.12.0 — 2026-10-22

### New

- Aria's face beside the notch
- where Aria's face goes beside the notch
- tell listeners when the face changes
- the face's motion as a pure engine

### Fixed

- the face's listeners are macOS's; Linux builds without a warning

## 0.11.0 — 2026-10-22

### New

- Aria's face in the bar follows what Aria is doing, on its own

## 0.10.0 — 2026-10-22

### New

- Aria's face in the Omarchy bar, put there by the installer

## 0.9.0 — 2026-10-22

### New

- /debug opens a live trace of the turn
- the trace waterfall part
- widen the surface for the trace panel
- trace a tab's turns on request
- keep each subagent's spans for the trace
- send ids, times and usage for the trace
- build a turn's trace from Pi's events

### Fixed

- a fresh page tells the dock it is not wide

## 0.8.1 — 2026-10-22

### Fixed

- a task may not write in Linux's keyring or Hyprland's config, as on macOS it may not in Keychains

## 0.8.0 — 2026-10-22

### New

- scheduled tabs and the /schedules panel
- run scheduled tasks in their own tabs
- schedule tools
- schedule ops on the session socket
- a scheduled tab's fixed permissions (ARIA_GRANTS)
- scheduled tabs beside the eight
- scheduled tasks and when they run

### Fixed

- scheduled tabs sleep until their run, and /schedules approves a task that writes
- nothing is ever asked in a scheduled tab (noAsk): unlisted connectors read, ask_user and confirmations decline
- a scheduled task's background agents run under its grants
- a waiting run reads its task again, and a run whose Pi cannot start or dies is recorded as failed
- a scheduled tab never changes tasks, and a task that writes waits for the person's approval
- a scheduled run whose prompt Pi refuses is recorded as failed and notified

## 0.7.2 — 2026-10-21

### Fixed

- only the group being worked on is live; a message to the owner folds the chain before it and stops its pulse

## 0.7.1 — 2026-10-21

### Fixed

- a rotation's start never decides, even when nothing was handed over
- a restart into a new version reaps the old page's WebKit processes as they end, instead of leaving them as zombies
- a session rotation hands the switch to the free model over, so a renewed quota still brings the paid model back

## 0.7.0 — 2026-10-21

### New

- show the earlier-conversation summary and page back through older history
- rotate a tab's Pi session after a compaction and answer chatEarlier
- the rotate extension and its /aria-rotate command
- tab segments in sessions.json and the earlier-history reader

### Fixed

- a compaction Aria asks for with no run shows as working, and a later state no longer hides it
- the session grants survive a rotation
- earlier pages keep custom messages and branch summaries; a bad cursor reads nothing
- a rotation keeps the tab's model, records a switch whose answer was lost, and messages wait through compactions
- the earlier summary as markdown, and paging review fixes
- a closed or stopped tab drops what it held during a rotation
- hold the tab's messages and restarts during a rotation
- carry the whole kept context and Pi's summary framing

## 0.6.0 — 2026-10-21

### New

- the macOS dock reloads its page when the web view's process dies
- the Omarchy dock reloads its page when the web view's process dies

### Fixed

- reload follow-ups: unique request ids, a reload log line, docs
- a reloaded page gets every tab's conversation back

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
