# Pending: what the chat shows but cannot feed yet

The chat has every element of the design (`design/Aria.html`, `design/components/`). These parts are built and
tested as UI, with their data contract written at the top of each module in `ui/chat/parts/`, but nothing feeds
them yet. Each row says what is missing and which part of the roadmap brings it.

## Fed by the owner's extensions (phase 2, part 2b — done)

The permission and write cards (sandbox, with its four choices), the quota card (quota-fallback), the question card
(`ask_user`), the router's route line (jev-router) and the subagents panel (Aria's own watch socket) are fed. Left:

- The quota card's countdown is shown as a number, not counting down; Pi's own timeout decides.
- An agent's session view has no composer (the design has one): there is no way to steer a subagent.
- `/agents` (the terminal overlay) does nothing in the dock; Alt+A is the dock's way.

## Needs the goals engine (phase 3)

| What the design shows | What is missing |
|---|---|
| Task dock above the composer and the inline task log | the goals/tasks engine (fork of `pi-goal-x`) feeding `chat.setTasks(tasks, kind)` |
| Document cards (spec, plan) with approval status | the planning flow publishing documents (`renderDocument` contract) |
| Phases card | the planning flow's phases (`renderPhases` contract) |

## Needs an Engram feed (phase 2, part 2c)

| What the design shows | What is missing |
|---|---|
| Memory panel (`/memory`): list, search, filters, pin, forget with undo | a read of Engram's memories into `chat.setMemory([...])` and the actions (pin, forget) written back; the contract is in `ui/chat/parts/memory.js` |

## Settings drive Pi (phase 2, part 2c — done)

The settings reach Pi and the extensions (`~/.aria/agent/aria.json`, the rendered persona and experts, Aria's Engram
entry) and a change applies when each session's run settles. Left:

- A release must publish `pi-version`, `pi-packages` and `engram-version` next to the binary: the `curl | sh` install
  reads them from there. There is no release workflow in the repo yet.
- The TypeSafe router backend is not built: without OpenCode Go (whose account includes Jev) the router is off, and
  the setup and the settings say so instead of offering it.
- Pi packages come only from a local folder (`integrations.packages.path`); npm, git, a gallery and trusting
  third-party packages are not built, and the setup no longer offers them (`thirdPartyTrusted` is kept in the file).
- A settings change restarts the running Pis only when it touches what Pi reads at start (persona, experts, its
  settings with the compaction reserves, `mcp.json`, the shim, the working folder), each when its run settles. A new
  compaction percent from the settings screen is written as Pi's reserves before that restart; `/autocompact` in a
  session applies it at once and needs no restart.
- The settings' list of turned-on services (`integrations.services`) only drives the setup's screens: turning a
  service off does not remove its entry from `mcp.json`.
- The settings screen shows an outside change (`/effort`, `/autocompact`) at once, unless it has changes of the
  reader's own that are not saved yet: those stay on screen, and saving them writes them over the outside change.

## Parity (phase 2, part 2d — done)

`docs/parity.md` walks every item of `ARIA-SETUP.md` with its check and its last result (2026-10-19): every offline
suite passes, the sandbox acceptance is 7/7, Jev's review eval 10/10 and the smoke test passes. Left:

- The router eval is 93/94: one expert case ("performance opportunities in the bundle") sits on the line between
  `reviewer` and `frontend` (0.48 against 0.52) and flips between runs; the experts are the owner's, word for word.
- The manual rows wait for the owner: Claude Max in the dock, Notion and Tavily, a subagent's `## Learnings`, the
  `ask_user` answer lifting read-only mode, and the `PI_*` switches through the dock's environment.
- Aria's memory starts with a single memory of its own (Aria saves it so Engram knows the project): its Engram database is its own (the owner's decision), so the memories in `~/.engram`
  are not in it. Importing them would be a new feature.
- Parity is accepted by the owner using Aria for real work, not by the matrix.

## The Claude provider

- `/connect` installs the Claude provider (`claude-bridge`) for the sign-in route and needs Claude Code on the machine. Closing the card does not cancel a sign-in still waiting in the browser (it times out after five minutes).
- Showing the sign-in's email or plan in the card needs a profile call the sign-in does not make today.
- Disconnecting a provider (removing its credential or the package) has no command yet.

## The quota bars

- OpenCode Go's windows (5h, Week, Month) are read from `GET /zen/go/v1/usage` and were checked against the real answer. A Claude plan's windows (5h, Week and any per-model one such as Fable) are read from `GET /api/oauth/usage` with the sign-in's token, an endpoint Anthropic does not document: its answer format is parsed defensively and has **not** been checked against a real account (none is connected yet).
- An API key, OpenCode Zen and an expired sign-in token show no bars. Aria does not refresh an expired Claude token (Pi does when it uses it); the bars come back after the next run.
- The card is one component (`renderQuotaCard`): the header chip opens it and each session tab shows it on hover (led by the tab's title and state, with that session's model and context). Beside the name only the 5-hour chip shows; hovering it (or focusing it with the keyboard) opens a card with every window and when each resets. With several sessions the header shows tabs, not the bar.

## macOS (phase 5, first pass)

The dock, setup, chat, session tabs and installer run on macOS (`Option + Space`). Left:

- Click-through: the whole 700 px panel takes clicks; a click on the veil closes the dock instead of passing through to the app below.
- Desktop notifications come from `osascript`; clicking one does not open the dock (an unbundled app has no notification action). The chat still posts one when a prompt is hidden.
- Checked on the Mac (2026-10-21): the dock runs, `cargo test --workspace` passes, and so does
  `install/install-mac.test.sh`. Still open:
  - A missing page file is served as an empty 200 (`dock/mac.rs`, logged as "missing page asset"); GTK answers
    NotFound.
  - The `aria-mac` hotkey test checks its own constants; the screen tests check the geometry only.
  - The installer gives up waiting for the old agent after 5 s, and `bootstrap` then fails after the binary was
    replaced.
- `crates/aria/src/dock/mac.rs` repeats the page queue, the open/close generations and the notice timing of `dock/gtk.rs`; they can be shared now that the Mac build is at hand.
- The owner's `sandbox` extension uses `bwrap` (Linux). On macOS there is no box yet: every command asks first with the permission card (`extensions/sandbox/route.ts`). A Seatbelt (`sandbox-exec`) profile with the same limits is the macOS sandbox still to design; some of its tests assume Linux paths.
- No `.app` bundle, signing or notarization, so the standard "open at login" UI and notification actions wait.
- The panel opens on the screen under the pointer and joins all spaces; multi-monitor and full-screen behavior needs manual testing on the owner's layout.
- `~/.local/bin` is not on the default macOS `PATH`, so the CLI is `~/.local/bin/aria toggle` (the hotkey does not need it).

## Smaller gaps

- The agent-session view has no composer (the design has one): there is no way to steer a subagent yet.
- Clicking the "needs your answer" desktop notification opens the dock on the active tab, not on the tab that is waiting.
- Question card: the sliding animation between questions and the "Answers sent" / "Open questions" states were not ported (the card is removed when answered).
- Message index: chips grow with the pointer but do not fade by distance (the owner's solid-fill rule).

- A session's images (screenshots the model read) are not sent to the page, which never shows them: a session full of
  them made its history tens of megabytes.
- A tab used to come back empty after a restart when its session ran on another model than the daily one: the page
  noted a "model change" before the history arrived and then dropped the history. Notices that come before the history
  now stay after it.
- Updating Aria waits for the dock: the installer tells it a new version is installed (`aria update-ready`), and it
  restarts into the new binary (same process, so systemd and launchd keep it) once it is hidden and no session has a
  run going or a question open. A session in a long run delays the update until it settles. A changed service file,
  `--restart-now`, or a dock older than this restarts at once, cutting a run as before. The page does not say an
  update is waiting; only the log does (`~/.aria/logs/aria.log` on macOS).
- The dock wakes every saved tab's Pi when it starts (the active one first), so the conversations are there before it
  opens; each open tab keeps a Pi and its MCP servers running. Until a tab's history arrives it shows "Loading the
  conversation". Closing a tab stops its Pi and everything in that Pi's process group; a program that leaves the group
  on purpose (`setsid`, a daemon) is not followed.

## Background work

Processes and subagents run in the dock while the conversation goes on (`crates/aria/src/jobs.rs`, the `background`
extension, the strip and the Alt+A list; see the [architecture](architecture.md#background-work)). Left:

- A background subagent cannot be steered: its Session view shows its steps, but has no composer. Stopping it and
  starting another is the only way to change its course.
- macOS has no `PR_SET_PDEATHSIG`: if the dock crashes (rather than quits), the processes it started keep running and
  nothing stops them; the next start marks them `cut` without killing them. On Linux the signal reaches only the job's
  own leader, so a child it forked outlives a crash unless systemd's cgroup takes it (the installed service does).
- The limits (3 subagents at once, 10 processes, 1000 lines of output, 30 minutes before the update notice) are
  constants in `jobs.rs`, not settings.
- A worktree has the person's committed and uncommitted tracked changes, not their untracked files: a subagent that
  needs a new file the person has not added yet will not see it.
- When a worktree's changes cannot come back because git itself failed (not a clash), it is reported as a clash, the
  worktree is kept, and git's message is only in the log.
- If the dock dies in the moment between making a worktree and recording it (the job is saved right after), the
  worktree stays in `<projects dir>/.aria-worktrees/` without a job naming it; `git worktree list` shows it. An empty
  `.aria-worktrees/` folder is left in the projects folder after the last one is removed.
- A subagent stopped while it is still starting is shown as failed, not stopped.
- The "update is waiting" notice is in English only, and reaches only the tabs whose Pi is running.
- A broken `sessions.json` (moved aside at start) makes the dock forget every session's pending end notes.
- If the dock never answers a `jobOutput`, that row's output stops refreshing until it is folded and opened again, or
  the list is closed.
- `background_output` for a subagent says its state, not what its watch says it is doing now (the jobs socket does not
  see the watch).
- Two background subagents in one session with the same title and task share one watch agent in the list.
- The browser check (`tools/ui-sandbox/check.ts`) prints `PAGE ERROR ReferenceError: process is not defined` a few
  times; it happens on `main` too, and every check still passes.

## Known limits of the page (from the review, deferred)

- The thread is redrawn on every streamed fragment, so text cannot be selected or copied while a run streams; the other regions only redraw when they change.
- The quota card's countdown ("Times out in 300s") is static; the real timeout is Pi's.
- Choosing `model` from the command picker only types `/model `; the model list opens once a space follows (or from the chip under the input).
- Closing the active tab does not save its unsent draft; a language change drops unsent drafts.
- `rows` of the composer counts only explicit newlines; a long wrapped line does not grow it (WebKitGTK's `field-sizing` support was not checked).
- The working phase keeps the last tool's name ("Reading files") while the answer streams.
- The browser checks (`tools/ui-sandbox/check.ts`) cover the controller's riskiest flows but are not part of `bun test` (they need Chrome).

## Deliberate differences from the design

- Esc never stops a run (the owner's choice, 2026-10-19): it closes a picker or a panel, and otherwise closes the dock while the run goes on. The stop button stops it; the composer no longer shows "esc stop".
- The working folder row and the settings button (owner's decisions).
- The header has no model button or meter (owner's decision; the design has one beside the name): the model is the chip under the input; the quota chip beside the name and each tab's hover open the same quota card, with the model and the context.
- Buttons and cards use solid fills; only the border takes the accent (owner's taste).
