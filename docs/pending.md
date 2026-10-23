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

## Needs an Engram feed (phase 2, part 2c)

| What the design shows | What is missing |
|---|---|
| Memory panel (`/memory`): list, search, filters, pin, forget with undo | a read of Engram's memories into `chat.setMemory([...])` and the actions (pin, forget) written back; the contract is in `ui/chat/parts/memory.js` |

## Settings drive Pi (phase 2, part 2c — done)

The settings reach Pi and the extensions (`~/.aria/agent/aria.json`, the rendered persona and experts, Aria's Engram
entry) and a change applies when each session's run settles. Left:

- A release must publish `pi-version`, `pi-packages`, `engram-version` and `agent-browser-version` next to the
  binary: the `curl | sh` install reads them from there. `tools/release.sh` tags a version and writes its notes
  (`RELEASES.md`), but nothing is published yet: no binaries, no hosted `install.sh`.
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
  - The page reload after the web content process dies (`webViewWebContentProcessDidTerminate`, `dock/mac.rs`) is
    type-checked only: killing the WebContent process on a Mac is the check still to run.
- The chosen shortcut on macOS (2026-10-21): `aria-mac` registers the `shortcut` setting with Carbon, falls back to
  `Option + Space`, `Control + Option + Space`, `Command + Option + Space` when `RegisterEventHotKey` refuses one, and
  registers again (new first, then the old one unregistered) when the setting changes. The key codes and the order are
  tested on Linux (`aria-mac/src/carbon.rs`) and the Carbon calls type-check (`tools/check-macos.sh`), but nothing of it
  has run on a Mac yet. Registration is exclusive, and the system's enabled shortcuts (`CopySymbolicHotKeys`) count as
  taken. To try there: Command + Space (Spotlight: it should fall back, the row saying "In use: …"), Control + Space
  (input sources), a combination another app holds (another hotkey app, or a second Aria-like registration), a change
  in the settings (the new one works at once, the old one no longer opens the dock), every candidate taken (the banner,
  "None available", a retry once one is freed), a fallback in use (freeing the chosen one: within 10 s the dock takes
  it back), and a letter on a non-US layout (Carbon codes are key positions). A failed `InstallEventHandler` (its own
  banner, `hotkey-handler-failed`, retried every 10 s) cannot be provoked on purpose.
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
- The sandbox keeps a repository's `.git/config` and `hooks` read-only, but the rest of `.git` stays writable, and git
  follows a `commondir` file in any git dir: a boxed command can point a repository at a config of its own, which the
  person's own git (outside the box) then reads. Repositories nested under a folder that is not one are not protected
  at all. The dock's own git commands neutralize hooks, fsmonitor and filter drivers, but a boxed background process
  that swaps `commondir` between the dock's read of the filter drivers and its `git add -A` could slip one in. To do:
  protect `commondir` and `gitdir` (or the whole git dir but what git must write) in `extensions/sandbox/policy.ts`, and
  give the dock's git an empty attributes source.
- If the projects folder is itself inside a git repository, `.aria-worktrees/` lands inside that repository.
- A slice job that ends while the main agent is in the middle of a turn gets no `[slice]` hint and no tasks-feed finish:
  Pi queues the `aria-background` message as a follow-up (agent-session `followUp`) and `before_agent_start` does not
  fire for it. The design accepts it; the later fix is to emit the hint from a `message_end` hook for `aria-background`.

## Connectors and attachments

Connector approvals (the `connectors` extension, the access rows in Settings → Integrations) and attachments (the
paperclip, drop and paste; `~/.aria/attachments/`). Left:

- Nothing removes old attachments: every file sent stays under `~/.aria/attachments/` until the person deletes it.
- Pi's write and edit can never touch `aria.json`, the settings file or the attachments folder, whatever was granted;
  a bash command the person let run outside the sandbox still can (as before this branch).
- The macOS dock has no navigation guard yet (GTK refuses anything outside `aria://`); the page stops file drops on
  both platforms.
- A message with files that the dock refuses right after a run settled can come back without its files (the page
  lets the data go when a run settles, and keeps at most three messages' data).
- The thread shows an attached image as a chip (name and kind), never as a preview: the page never receives image data.
- A connector card's preview shows the arguments as they are, not redacted; a secret in them is on the card.
- Paste and drop were checked in headless Chrome only, not on the real dock (WebKitGTK): whether it hands files from
  the file manager or the clipboard over as files is untried.
- A queued message that is re-queued (another queued one cancelled) keeps its path lines, but an image that went to
  the model as an image does not go again as one; the file stays on disk and its line names it.
- Two queued messages with the same text and different files can swap their chips until Pi's next queue update puts
  each one right.

## The turn watchdog, MCP import and the shortcut

The dock stops a run that has gone silent (`models.turnTimeoutMinutes`), Settings → Integrations and the setup import
the MCP servers other apps have, and the shortcut is chosen in the settings (the macOS side is above). Left:

- No MCP reconnect in Aria: Pi has no way for Aria to ask it to reconnect a server, so a connector that failed to
  start waits for its next use (or a Pi restart). Retrying with backoff stays Pi's.
- A tool that never reports its end keeps the watchdog paused for the rest of that run (a running tool pauses it on
  purpose, so a long silent build is not cut); the watchdog only covers the model being silent.
- An imported server whose env var is empty in the other app reaches Pi without that variable.
- What Pi cannot read as the other app did is imported as it is and only warned about in the row ("check it after
  importing"): a variable in a command, argument, folder or URL (Pi expands none there), Claude Code's
  `${NAME:-default}`, an editor's `${workspaceFolder}`, a `~` argument, VS Code's `envFile`. The person fixes those in
  `~/.aria/mcp.json` by hand.
- The test `an_import_that_cannot_write_says_so` fails when the tests run as root (a read-only folder does not stop
  root).
- In the import list, the checks are lost when the setup goes Back and returns to Integrations.
- The editor variables (`${userHome}`, `${workspaceFolder}`…) are translated whatever app the server came from: a
  Claude Code entry whose env uses a variable with one of those names gets the editor's meaning, not Claude Code's.
  Claude Desktop is assumed to expand `${X}` in env like Claude Code.
- After a "nothing to import" refusal, the note shows on the Import button; when the new scan leaves nothing to
  import, there is no button and the note is not shown.
- On the Mac, a later retry that fails for another reason keeps the first problem's banner.
- On Hyprland, a combination already bound (or Aria's own) never reaches the page, so the Shortcut row cannot record
  it by pressing it; it can be typed in the field under the row.

## Keys in the keyring

- A command in the sandbox can still run `aria secret` (or `security`, `secret-tool`) and read a key: the keyring keeps
  keys out of plain files, not out of reach of the person's own processes ([design](superpowers/specs/2026-10-21-keyring-design.md)).
- Claude Code's own credentials (the Claude sign-in through `claude-bridge`) stay where Claude Code keeps them.
- The settings screen has no row for the keyring: `credentials.keyring`, `aria keyring on|off|status` and the dock's log.
- On Omarchy the keyring needs a running Secret Service (gnome-keyring); without one the dock keeps plain files and says
  why in its log. Not yet tried on the Omarchy machine.
- No Touch ID: the keyring opens with the session, as the owner chose (2026-10-21).

## Session rotation

A tab continues in a new Pi session file after each compaction (see the
[spec](superpowers/specs/2026-10-21-session-rotation-design.md)). Known limits:

- A rotation reloads the extensions in-process: the session grants are kept, but the MCP servers reconnect, so a
  stateful MCP session (a browser, a login held by the server) starts over.
- A segment whose tokens exceed the model's window cannot compact (the summary call itself does not fit), so the
  compaction Aria asks for over 8 MB fails and is tried again once per Pi start; the file keeps growing meanwhile.
- While Pi compacts, the page's messages wait for the compaction's end; a compaction over a large context takes a
  model call's time, and the composer says nothing about why the message has not started.

## Scheduled tasks

- A task runs only while Aria runs; a sleeping computer is never woken (`pmset`), so a missed time runs when it is back,
  once, and only on the same day.
- On macOS a click on the notification does not open the dock (an unbundled app has no notification action); the tab is
  marked unread for when it opens.
- No intervals ("every 2 hours") and no results sent outside Aria (email, Telegram).
- Tasks are changed by asking Aria; the `/schedules` panel only pauses, resumes and deletes them.
- A successful scheduled run was checked live only with the fake Pi (a real one needs a key); the late start, the tab,
  Pi's grants and a refused run's record were checked live on macOS. Not yet tried on Omarchy.

## The live trace (`/debug`)

- The wide surface was not seen live: on Omarchy the GTK side (`set_size_request` and the input region over both
  columns) was written on the Mac and not compiled there yet; on macOS the panel's frame change was built and type-checked
  but not opened by hand. The panel itself was checked in headless Chrome at 1,060 px, and the fields the trace reads were
  checked against Pi 1.3.0's event types; a turn with a real model was not traced yet (the scratch home has no keys
  since they moved to the keyring).
- Only the current turn is kept: no saved traces, no past turns, no comparing runs, no export.
- A subagent's own subagents appear only as far as they reach the session's watch socket.

## Aria's face on macOS

- Checked live on one notched MacBook screen (the wing at the notch's right edge, 38 x 32 pt, under 1 % CPU); the
  drawing was checked by rendering every face to images. A click on the wing and a setup with an external display were
  not tried by hand yet.
- No setting hides the wing; no right-click menu or hover details.
- Omarchy's bar keeps its QML widget; `face_motion.rs` is not used there.

## Dictation

- Linux builds Whisper on the CPU: its GPU backend (CUDA or Vulkan) is Linux's next step, and the microphone path
  (cpal on ALSA/PipeWire) has not been run there yet.
- A denied macOS microphone is told by exact silence (cpal gives zeros, not an error).
- Not tried live in the dock yet: the first permission prompt from Aria itself, and whether macOS asks again after an
  update (Aria is not signed with a stable identity). The engine was checked live on the Mac: the real microphone
  (5 s at 16 kHz), and a 59 s recording transcribed in about 2 s warm.
- No live text while speaking, no custom vocabulary (Supabase, Vercel…), no reading replies aloud.

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

## The terminal

Milestones 1 and 2 are done (docs/superpowers/specs/2026-10-23-terminal-design.md, …-terminal-m2-design.md). Left:

- The task strip shows a subagent's name, not its model: the watch does not report the model.
- Verify on macOS runs only `.aria/verify.json`'s commands (there is no box there).
- The dock's own page does not show tasks, changes, `/context` or the checks yet.
