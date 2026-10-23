# The terminal — design

Aria in a terminal: `aria` opens a client of the dock's own sessions. The work belongs to the service, never to the window
that shows it: leaving the terminal (Ctrl+C, closing it) stops nothing, the same session can be open in the dock and in
the terminal at once, and a new session started from the terminal works in the folder it was started from. The look is
the canvas "Aria Terminal" (2026-10-23): a character grid, a zen layout, one accent.

Decided with the owner (2026-10-23):

- **The service owns every session.** The terminal and the dock are two clients of the same sessions (`sessions.json`):
  the same list, the same conversations, live in both.
- **`aria`** in a terminal lists the open sessions and offers a new one **in the current folder**. A session started in
  the dock keeps using the projects folder of the settings.
- **Ctrl+C, Ctrl+D or closing the terminal only leave**: the session and its work go on. **Esc** stops the running turn
  (as in the dock). **`/exit`** closes the session for real: its Pi stops and its tab closes in the dock too.
- **There is always a session**: closing the last one opens a new one.
- **Ratatui** (with crossterm) draws the client, inside the `aria` binary: no Node to start a client, the same binary on
  Linux and macOS.
- **Verbose by default** (`display.verbose: true`), `/verbose [on|off]` and a key switch it. The terminal only, for now;
  the dock does not change in this work.
- **The method (outcome-driven or spec-driven) is the router's business**, never shown: the screen shows the tasks.
- **Per-turn changes**: after an instruction that changed files, one line says how many (files, +added −removed), git or
  not. **Double Esc** (idle) or **`/undo`** puts files and conversation back to before the last message.
- **`/context`** draws what fills the context, like Claude Code's.
- **Questions** answer with a click or the option's number (1–9); permissions with `y` / `a` / `n` or a click.
- **One desktop notification**, only when neither the terminal nor the dock shows that session.

## The service

- **`aria dock`** runs the service (the systemd unit and the macOS LaunchAgent; the installer rewrites both). Plain
  `aria` with a terminal on stdin and stdout is the client; without one (an old unit that still runs `aria`) it runs the
  service, so an update never leaves a machine without its dock. `aria toggle`, `show`, `quit`, `start` are unchanged.
- **The clients socket**: `<the control socket's folder>/clients.sock`, mode `600`, JSON lines. A client speaks the
  page's messages (`docs/architecture.md`, "The messages"): `sessionsLoad`, `sessionClose`, `session { chatSend,
  chatAbort, chatLoad, uiAnswer, listCommands, listModels, setModel, listFiles, … }`, and gets the dock's replies and
  events (`sessions`, `session { chatState, chatEvent, chatDialog, chatDialogClosed, chatQuota, … }`, `jobs`). Nothing
  new to learn on either side: what reaches the page reaches every client.
  - `chat.rs`'s `emit` goes to the page and to every client attached to that session (a client attaches with
    `session { chatLoad { fresh: true } }`, which already sends a running Pi's conversation again; it detaches by leaving
    or attaching elsewhere). A client that cannot keep up is dropped, never waited for.
  - A dialog goes to every client; the first answer wins and the others get `chatDialogClosed`.
  - Attaching starts the session's Pi if it is not running, as showing a tab does.
- **A session's folder**: `SessionEntry` gains `workdir` (absolute; none means the settings' projects folder). The
  client's new session sets it to its current folder; Pi starts there (`inner.workdir`). The sandbox's writable folder
  follows the session's folder.
- **Never zero sessions**: closing the last one (from either side) creates a new one in its place.
- **Seen**: a session is seen when the dock shows it (dock visible, chat shown, that tab active) or when a client is
  attached to it and its terminal has the focus (the client reports focus changes: crossterm's focus events; a terminal
  that does not report focus counts as focused while attached). The desktop notification that exists for "a prompt
  nobody can see" uses this, and so does the new "finished" notification: one per event, never one per client.

## The client — `crates/aria/src/term/`

- **Start**: the list of sessions (title, folder's last component, state: working / waiting for you / done, how long
  ago), newest first, and `+ new in <current folder>` last. Enter attaches; the mouse works too. With no service running,
  `aria` starts it first (`control::command_or_start`).
- **The session view** (the canvas): the conversation, a task strip above the editor when there are tasks, the editor
  between two thin rules, one footer line.
  - User messages start with `›`; the assistant's markdown is rendered (headings, lists, code, emphasis; code blocks
    unhighlighted at first).
  - **Verbose on**: the thinking in grey; one line per tool (`✓ edit src/coupon.ts +14 −2`) with its output in grey under
    it (cut to 12 lines, the rest behind a key). **Verbose off**: no thinking, one line per turn
    (`✓ 3 reads · 2 edits · tests 43/43`; the texts follow the assistant's language).
  - **The footer**: left, model, effort, `ctx used / compacts at`, the quota of the provider in use with the time to its
    reset (`chatQuota`); right, the git branch with its changed files count (none outside git) and the folder. Quota
    colours: normal below 80 %, amber from 80 %, coral from 95 %; the context turns amber 5 points before compacting.
  - **The task strip**: `✓✓✓●●○○ 3/7`, then the running tasks (with their subagent, its model and time) and the pending
    ones; done tasks only count. A key and a click on the strip fold it to its first line.
  - **The editor**: multi-line, history (↑ on an empty editor), `@` opens a fuzzy file list (`listFiles`), `!command`
    runs in the session (Pi's `bash`) and its output joins the context, `/` lists commands (`listCommands`).
  - **Keys**: Enter sends (queued while a turn runs, as in the dock), Esc stops the turn, Esc Esc (idle) opens the undo
    list, Ctrl+C / Ctrl+D leave, Ctrl+L redraws. The verbose key is picked in the plan against Pi's and the terminal's
    own keys.
- **Dialogs** in place of the editor: `ask_user` questions (options by number 1–9 or click, a free answer, several
  questions one after the other), the sandbox's permissions (`y` once, `a` always in this repo, `n`, or a click),
  `request_secret` (masked input; the value goes to the dock as the page sends it, never to the model), the quota card.
- **Commands the client keeps for itself**: `/exit`, `/verbose`, `/undo`, `/context`, `/model` (a picker), `/sessions`
  (back to the list). Everything else goes to the session as the dock sends it.
- **Mouse**: on for the strip, options and lists. Selecting text then needs Shift (Option on macOS), as in any terminal
  app that reads the mouse; the client says so once, the first time.
- **Theme**: the dock's palette (ground `#0A0A0C`, text `#D6D6DC`, grey `#8A8A94`, accent `#B9A8FF`, green `#8BE0B0`,
  amber `#F0B44C`, coral `#FF9A8B`) in 24-bit colour, falling back to the nearest 256 colours.

## The data

- **Tasks**: a new extension, `tasks`, with the tool `tasks_update { tasks: [{ id, title, status: todo|active|done,
  agent? }] }` (the whole list each time). Its result's `details` carry the list, so the client reads the tasks from the
  conversation itself: they come back when attaching, without new messages or files. The agent and its model come from
  the session's subagents (`chatAgents`), joined by the task's `agent`. The router tells the model when to keep the
  list (any work with more than one step).
- **Checkpoints**: before each user message reaches Pi, the service snapshots the session's folder into a shadow
  repository (`~/.aria/checkpoints/<session>/`, git with its own `GIT_DIR`, the folder as work tree, the project's
  `.gitignore` respected, files over 5 MB skipped, 50 checkpoints kept per session). After the turn it compares the folder
  with that snapshot and sends `chatEvent { type: 'changes', files, added, removed }`; nothing is sent when nothing
  changed. **Undo** (`session { chatUndo }`): files back to the last snapshot, and the conversation forked before that
  message (Pi's `fork`, the tab's segments updated as a rotation does); the client shows the message's text back in the
  editor. The list (Esc Esc, `/undo`) offers the last 10.
- **`/context`**: Pi's `get_session_stats` (the context window and its use) and a breakdown from the `context`
  extension, which measures at `before_agent_start` the system prompt, the tools (built-in, extensions, MCP), the memory
  hint and the messages; drawn as Claude Code does: a grid of cells by category, then the list with tokens and percent,
  free space and the compaction reserve.
- **Verification after editing**: a new extension, `verify`, on every `edit` and `write` result: runs the project's lint,
  typecheck and affected tests (found from `package.json`, `Cargo.toml`, `pyproject.toml`; overridable in
  `.aria/verify.json` in the project), inside the sandbox, at most once per turn after the last edit; the result goes
  back to the model and into the footer (`tests ✓ 43`).
- **The method**: `jev-router` sends focused work through outcome plus check, and work to split through the `/plan`
  flow, moving up mid-way when what it read is bigger than estimated; the hint never names a method.

## Both platforms

The client, the socket and the checkpoints are the same code on Linux and macOS. Platform pieces: the LaunchAgent
(`aria dock`), focus reporting (Terminal.app and iTerm2 report it; checked on both), the Shift/Option hint, the
notification command. `tools/check-macos.sh` type-checks every change; the client's tests that depend on keys say which
platform they cover.

## Out of scope

- The dock's look (no task strip, no verbose there yet).
- Code highlighting in the terminal.
- More than one client per terminal tab, or a remote client: the socket is local, for the owner's user.
