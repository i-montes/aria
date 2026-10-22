# One continuous conversation: session rotation and paged history

Idea 7 of `docs/ideas.md`. A tab is meant to be the one conversation the owner keeps talking in ("one session should be
enough", the background-work design). Two things stop that today:

- **The file never stops growing.** Each tab is one Pi session file (`~/.aria/agent/sessions/aria/<time>_<tab id>.jsonl`).
  Pi's compaction appends a summary entry and never removes anything, and every saved tab's Pi is woken at start, so
  each start parses every file whole (the largest on the owner's machine is 26.9 MB and has never been compacted).
- **What was compacted disappears from the chat.** `get_messages` returns what the model sees (a `compactionSummary`
  plus the kept tail), the page drops the summary, and nothing reads the older part again.

## The model

A tab owns a chain of **segments**: Pi session files, oldest first, the last one being the one Pi runs on. The tab id
stays the tab's identity everywhere it is today (`sessions.json`, background jobs, the jobs and watch sockets, the
page); only the Pi session id Aria passes as `--session-id` becomes the tab's **current segment**.

`sessions.json` entries gain `segments: [<pi session id>, …]` (oldest first). An entry without it (every tab today)
reads as `segments: [<tab id>]`, so nothing migrates. Closing a tab forgets its whole chain (the files stay on disk, as
Pi's files do today).

## Rotation

- **When.** Right after a compaction (`compaction_end`, automatic or `/compact`), once the tab is quiet: no run, no
  queued prompt, no open dialog. Rotating at the compaction is natural: the model's context is then exactly "summary +
  kept tail", which is what the next segment starts from.
- **Size.** When the current segment's file is larger than **8 MB** and the tab is quiet, Aria asks Pi to compact
  (`compact`), which then leads to a rotation. Checked when a run settles and when the tab's Pi starts.
- **How.** Aria sends the prompt `/aria-rotate`, a command of a new Aria extension `extensions/rotate`. The command:
  1. reads the current context through its `ctx.sessionManager` (the latest compaction's summary and the messages from
     its `firstKeptEntryId` to the leaf, on the active path);
  2. calls `ctx.newSession({ parentSession: <current file>, setup })`, where `setup` appends to the new session a
     `custom_message` of type `aria-earlier` (content: the summary; `display: true`; details: `{ from: <old session
     id> }`) and then every kept message, in order (`appendMessage`);
  3. answers nothing to the model (no turn).
  A command with no compaction in the current segment does nothing (nothing to carry).
- **After.** Aria asks `get_state`, takes `sessionId`, and appends it to the tab's `segments` in `sessions.json`. The
  page is not told to reload: its thread already shows the conversation. If the command fails, Aria logs it and
  tries again at the next quiet moment, never in a loop (one try per compaction).
  - Pi's `sessionId` is read after the command whatever its answer, and every later `get_state` (a Pi announcing
    itself, a page reload) records it too when it is not the tab's current segment: a switch whose answer was lost
    or late is never left out of the chain.
  - Pi's new session starts from the settings' default model and thinking level, so Aria reads both before the
    command and sets them again (`set_model`, then `set_thinking_level`) where they differ, then sends the page the
    state and the context's fill.
  - Pi reloads the extensions in-process with the new session. The person's session grants (the sandbox's, the
    connectors') are kept: the command marks the switch on `globalThis` while `newSession` runs, and each extension
    hands its grants over at `session_shutdown` and takes them back at the `session_start` (reason `new`) that
    follows (`extensions/lib/rotation.ts`). The MCP servers reconnect, so a stateful MCP session (a browser) starts
    over.
- **Held messages.** While a rotation runs, the page's sends, `/` commands, model changes, queue edits, restarts and
  `chatLoad` wait and then run in order; a second held prompt goes as a follow-up. A stop gives the held prompts back
  (and, once they are being sent, stops Pi too). A send counts as waiting from the moment it arrives, so no rotation
  starts under it. While Pi compacts (and the tab has no prompt of its own waiting on it), the same messages wait for
  the compaction's end: Pi refuses prompts meanwhile.

## Paged history

- **The summary is visible.** A `compactionSummary` message and an `aria-earlier` custom message both render as one
  divider, "Earlier in this conversation", with the summary folded under it (open on click), and a **"Show earlier"**
  button above it while there is more.
- **Earlier pages.** The page asks `chatEarlier { cursor }` (no cursor: the newest earlier page). Aria reads the
  segment files itself (JSONL, no Pi involved) and answers a chat event `{ type: 'earlier', messages, cursor, more }`:
  the previous **40** messages before the cursor, in order, in the same shape as `get_messages` (and through the same
  `for_page`, which strips image data).
- **What counts as earlier.** For each segment, oldest first: the `message` entries on its active path (walked from
  its last entry up through `parentId`) that come before its latest compaction's `firstKeptEntryId`. A segment without
  a compaction contributes nothing unless it is not the current one (then all of it, for a chain made by hand). The
  `aria-earlier` seed messages are not repeated.
- **The page** prepends the page it gets above what it shows (`fromHistory` handles the messages), keeps the scroll
  where the reader was, and shows "Show earlier" while `more` is true.

## Out of scope

Searching old segments, deleting old files, a setting for the 8 MB or the 40, rotating a tab that never compacts
(below 8 MB), and showing a compaction's summary that only the live `compaction_end` event carried before this change.

## Checks

- `sessions.json` reads old entries as one segment and round-trips `segments` (aria-core).
- The earlier-history reader: fixtures with a tree (an abandoned branch), a compaction, two segments; pages of 40,
  the cursor, `more`, no seed repeated (aria-core).
- The `rotate` extension builds the seed from a session with a compaction and does nothing without one (bun).
- The chat: after `compaction_end` and a settled run it sends `/aria-rotate` once, records the new segment from
  `get_state`, starts a later Pi with `--session-id` of the current segment, and over 8 MB it sends `compact` first
  (fake Pi, chat.rs).
- The page: the divider, "Show earlier", prepending a page (ui tests; the ui-sandbox check).
- Live, with the real Pi in a scratch `ARIA_HOME`: a short conversation, `/compact`, the rotation, and the new segment's
  `get_messages` beginning with the `aria-earlier` message (an `--ignored` test, like `live_pi`).
