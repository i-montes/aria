# The terminal, milestone 2 — design

Milestone 1 (docs/superpowers/specs/2026-10-23-terminal-design.md, v0.16.0) made `aria` a client of the dock's
sessions. This milestone brings the rest of that spec and what the owner asked while trying it. The rules of
milestone 1 hold: the dock owns every session, the terminal is zen (a character grid, one accent), and every piece of
data comes from one source that both the dock and the terminal can read.

Decided with the owner (2026-10-23), from the list they agreed to:

- The session list says what each session is doing and how long ago, newest first.
- Tasks show as a strip above the editor, folded with a key or a click.
- After an instruction that changed files, one line says how many, git or not; **Esc Esc** or **`/undo`** puts files
  and conversation back to before the last message.
- **`/context`** draws what fills the context, as Claude Code does.
- After editing, the project's lint, typecheck and affected tests run, and their result reaches the model and the
  footer.
- **One desktop notification** when a turn ends or needs an answer, only when neither the terminal nor the dock shows
  that session.
- The method (outcome first, or a spec and a plan) is the router's business, never shown.
- The dock's look does not change in this milestone.

## 0. What the review left (first, small)

- Pasted `\r` and `\r\n` are line breaks, never characters in the text.
- **Ctrl+L** redraws the whole screen.
- A call's output cut at 12 lines: a click on `… N more lines` shows it all (and again folds it).
- Clearer words: a session closed elsewhere says "This session was closed" (the dock or another terminal); a client
  the dock dropped for not reading says so and reconnects; an old dock without `clients.sock` (its control socket
  answers, its clients socket does not) is reported as "Aria's dock is older than this `aria`: restart it with
  `aria quit` and `aria`", at once, never after a 3 s wait.
- What a client may send is a list of its own: `sessionsLoad`, `sessionNew`, `sessionClose`, `secretSave`, and inside
  `session`: `chatSend`, `chatAbort`, `chatLoad`, `uiAnswer`, `listCommands`, `listModels`, `setModel`, `listFiles`,
  `piCommand`, plus the new ones below. Anything else is `not-for-clients`.
- A panic on a background thread of the client (the git watcher, the link) ends the client with its message, after
  the terminal is given back.
- A message Pi refused (`command_failed`) leaves the client's list of its own echoes.

## 1. What each session is doing

- The dock keeps, per open session, its **status** and the time it changed: `working` (a run), `waiting` (a dialog
  is open), `done` (a run ended and nobody looked since), `idle`. A session is looked at while the dock shows it or a
  focused terminal is attached to it (milestone 1's "seen"); looking turns `done` into `idle`.
- `DockMessage::SessionStatus { session, status, at }` (`"sessionStatus"`, `at` in ms) goes to every client when a
  status changes, and for every open session right after a client's `sessionsLoad` reply. The page ignores it.
- The list: title, folder, then `working` / `waiting for you` / `done` and how long ago (`just now`, `3 min ago`,
  `2 h ago`, `3 d ago`; in the assistant's language); sessions newest first by that time; `+ new in <folder>` stays last.

## 2. Tasks

- A new extension, `tasks`: the tool `tasks_update { tasks: [{ id, title, status: "todo" | "active" | "done",
  agent? }] }` (the whole list each time, at most 20). Its result's `details` carry the list; the client reads the
  last one from the conversation, so tasks come back with the history. The system prompt line the extension adds:
  keep a task list for any work with more than one step, update it as steps start and end.
- The strip, above the queue and the working line: `tasks  ✓✓✓●●○○  3/7`, then one row per active task (with its
  subagent and model, from `chatAgents`, when `agent` names one) and per pending one; done tasks only count. At most
  6 rows, then `… N more`.
- **Ctrl+T** or a click on the strip folds it to its first row, and back. Folded is remembered per session while the
  client runs.

## 3. Changes and undo

- **Checkpoints** (the dock, `crates/aria/src/checkpoints.rs`): before a person's message reaches Pi (`chatSend`
  without `behavior`, and a steer or follow-up when Pi takes it), the session's folder is snapshotted into
  `~/.aria/checkpoints/<session>/`: a git repository of its own (`GIT_DIR` there, the folder as work tree), its
  `info/exclude` holding `.git` and every file over 5 MB, the project's own `.gitignore` respected. At most 50
  checkpoints per session; older ones are dropped (`git replace`-free: a new orphan root every 50). A folder that is
  not a directory, or `$HOME` itself, gets no checkpoints (a note says why, once).
- **The changes of an instruction**: when the run settles, the dock compares the folder with the checkpoint taken
  before it and sends `chatEvent { type: "changes", files, added, removed }` (nothing when nothing changed). The
  client shows one line after the answer: `3 files changed  +40 −12`.
- **Undo** (`session { chatUndo { id, checkpoint } }`, answered `saved` or `error`): files back to that checkpoint
  (files changed since are restored, files created since are deleted, files deleted since come back; nothing outside
  what the checkpoint covers is touched); the conversation is forked before that message (Pi's `fork` with the
  message's entry id, the tab's segments appended as a rotation does); the message's text goes back to the editor.
  Refused while a run is going (`error: busy`).
- **Esc Esc** (idle, two presses within 500 ms) or **`/undo`** opens the list of the last 10 messages, newest first
  (`chatCheckpoints { id }` → `checkpoints { id, list: [{ checkpoint, text, at, files }] }`), in the editor's place;
  Enter undoes to before the chosen one.

## 4. `/context`

- A new extension, `context`, measures at every `before_agent_start` what Pi will send: the system prompt, the tools
  (built-in, extensions', MCP), the skills' descriptions, the memory hint, and the messages; in tokens, estimated at
  4 characters each unless Pi reports better. It publishes the breakdown as the status `aria-context` (JSON), which the
  client keeps.
- `/context` in the client: Pi's `get_session_stats` (the window and its use) with that breakdown, drawn as Claude
  Code draws it: a grid of cells (one per 1 % of the window) coloured by category, then a list with each category's
  tokens and percent, the free space and the compaction reserve (`compaction.percent`). It shows in the editor's place;
  Esc closes it.

## 5. Verification after editing

- A new extension, `verify`. When a run that edited or wrote files is about to end (`agent_end` with edits in it), it
  runs, inside the sandbox, the project's checks: the commands in `.aria/verify.json` (`{ "lint": "…", "types": "…",
  "test": "…" }`) when the project has one, else the ones it finds (`package.json` scripts `lint`, `typecheck`, `test`;
  `Cargo.toml`: `cargo clippy`, `cargo test`; `pyproject.toml`: `ruff check`, `pytest`), each at most 5 minutes.
- All passed: the status `aria-verify` says so (`tests ✓ 43`) and the run ends. One failed: the model gets a follow-up
  with the command and the last 40 lines of its output, once per run (a second failure after its fix is reported, not
  fed back again), and the footer says `tests ✗`.
- `/verify off` and `/verify on` per session; off by default for a folder with none of the above.

## 6. Notifications

- When a run settles in a session nobody looks at (as above), one desktop notification: `Aria · <session title>` with
  the answer's first line, or "It did not finish" with why. The one for a dialog stays as it is. A notification is
  replaced by the session's next one, never piled; looking at the session takes it away.

## 7. The method by load

- `jev-router`: a focused task gets the hint to state the outcome and the check that proves it first, then work
  test-first; work to split goes through the `/plan` flow as today (offered once for sizable work). The hint never
  names a method. The owner rules later whether sizable work starts planning without the offer.

## Both platforms

Checkpoints use the system's `git` on both; the notifications use the dock's notifier on both; the client's keys are
tested as before. `tools/check-macos.sh` for every change.

## Out of scope

The dock's look (no strip, no changes line, no `/context` there), code highlighting, remote clients.
