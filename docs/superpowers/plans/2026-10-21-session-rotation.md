# One continuous conversation — plan

Spec: `docs/superpowers/specs/2026-10-21-session-rotation-design.md` (the authority).

## Global constraints

- Everything in English (code, names, messages, docs, commits); the page's strings in `ui/chat/texts.js` in English and
  Spanish. Match the surrounding style of each file.
- Commit subjects are conventional and end with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- No personal data in anything committed; fixtures use neutral names (`/home/ana`, `acme`).
- Exact values: rotate when the segment file is larger than **8 MB** (8 * 1024 * 1024 bytes); earlier pages of **40**
  messages; the command is `/aria-rotate`; the seed's custom type is `aria-earlier`; the new field is `segments`.
- The tab id stays the identity of a tab everywhere (jobs, sockets, page); only `--session-id` uses the current segment.
- Never kill or restart the installed dock or its Pis; never write into `~/.aria` (tests use temp dirs or `ARIA_HOME`).
- At the end of each task these pass: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo fmt --check`, `bun test ui`, `bun test extensions`, `sh tools/check-macos.sh`.

## Task 1: segments in `sessions.json`, and the earlier-history reader (aria-core)

Files: `crates/aria-core/src/sessions.rs`, a new `crates/aria-core/src/history.rs` (registered in `lib.rs`),
`crates/aria-core/src/bridge.rs`.

1. `sessions.rs`: a session entry gains `segments: Vec<String>` (serde default empty, skipped when empty on write is
   NOT wanted: write it always once set). A helper `segments_of(entry) -> Vec<String>` returns `segments`, or
   `[id]` when empty; `current_segment(entry)` is its last. A function to append a segment to a tab by id and save.
   Tests: an old file (no field) reads as `[id]`; append + reload round-trips; the id rules for a segment are Pi's
   session id rules (accept a UUID; reject a path separator).
2. `history.rs`: given the sessions folder and a tab's segment ids (oldest first), and a cursor, returns
   `EarlierPage { messages: Vec<serde_json::Value>, cursor: Option<EarlierCursor>, more: bool }`:
   - finds each segment's file in the folder by its id (`*_<id>.jsonl`, the header's `id`); a missing file is skipped;
   - per segment: parse JSONL lines (a bad line is skipped), walk the active path from the last entry up through
     `parentId`, keep `message` entries in order; the cut is the latest `compaction` entry's `firstKeptEntryId` on that
     path; the earlier part is the messages before the cut; with no compaction, the earlier part is empty for the
     current (last) segment and the whole path for any other;
   - drop `custom_message` entries of type `aria-earlier` (each segment's seed) — keep every other message as Pi wrote
     it (`entry.message`);
   - the earlier list is the concatenation over segments, oldest first; a page is the 40 messages just before the
     cursor (no cursor = the end of the list); `cursor` points at the first message of the page; `more` says whether
     anything is before it. `EarlierCursor { segment: usize, index: usize }` (Serialize/Deserialize, camelCase as the
     bridge uses).
   Tests with JSONL fixtures written in the test: an abandoned branch is not shown; the compaction cut; two segments;
   paging 40/40/rest with `more`; the seed is never repeated; a missing segment file.
3. `bridge.rs`: `PageMessage::ChatEarlier { cursor: Option<EarlierCursor> }` (`{"type":"chatEarlier"}` and with a
   cursor); parse tests like the others. (The reply goes out as a `ChatEvent` in Task 3; nothing new on `DockMessage`.)

## Task 2: the `rotate` extension

Files: `extensions/rotate/index.ts`, `extensions/rotate/seed.ts`, `extensions/rotate/seed.test.ts` (bun), and the
extensions README/list if the repo keeps one (`extensions/README.md`).

1. Read how Aria's other extensions are written and loaded (`extensions/background/index.ts`, `extensions/install/`)
   and Pi's extension API (`~/.aria/pi/1.3.0/node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/types.d.ts`,
   `newSession` with `setup`, and `SessionManager` in `dist/core/session-manager.d.ts`).
2. `seed.ts` (pure): from the entries on the active path of a session (as `SessionManager` returns them), returns
   `null` when there is no compaction, else `{ summary, from, messages }`: the latest compaction's summary, the session
   id, and the `message` entries' messages from its `firstKeptEntryId` to the leaf (skipping earlier `aria-earlier`
   seeds is not needed: they are custom messages and stay as they are).
3. `index.ts`: registers the command `aria-rotate`; it builds the seed from `ctx.sessionManager`; with `null` it does
   nothing; otherwise `ctx.newSession({ parentSession: <current session file>, setup: async (sm) => {
   sm.appendCustomMessageEntry('aria-earlier', summary, true, { from }); for (m of messages) sm.appendMessage(m) } })`.
   It never starts a model turn.
4. Tests (bun, pure `seed.ts`): no compaction → null; the summary and the kept messages in order from a fixture with an
   abandoned branch and two compactions (the latest wins).

## Task 3: the chat rotates, starts on the current segment, and answers `chatEarlier`

Files: `crates/aria/src/chat.rs` (and its tests), `crates/aria/src/handlers.rs` if routing needs it,
`crates/aria/tests/live_rotation.rs` (new, `#[ignore]`), `docs/architecture.md` (the new message and the segments).

1. Pi starts with `--session-id <current segment>` (from `sessions.json`) instead of the tab id; everything else keyed
   by the tab id stays.
2. Rotation state per tab: `compaction_end` sets `rotate_due`; when the tab is quiet (the run settled: no run, no queued
   prompt, no open dialog — use the same notion the code already has for "settled"), if `rotate_due`, clear it and send
   the prompt `/aria-rotate`; then `get_state` and, when its `sessionId` differs from the current segment, append it to
   the tab's `segments`. One try per compaction: a failure is logged (`note`) and waits for the next compaction.
3. Size: when the tab is quiet (after a run settles and after its Pi starts) and the current segment's file
   (`get_state.sessionFile`) is larger than 8 MB, and no rotation is due, send `compact` (once per Pi process; the
   `compaction_end` it causes leads to the rotation).
4. `PageMessage::ChatEarlier { cursor }`: reads the tab's segments and answers with
   `DockMessage::ChatEvent { event: for_page({"type":"earlier","messages":…,"cursor":…,"more":…}) }`, off the UI thread
   like other reads. Route it like the other per-session chat messages.
5. Tests with the existing fake Pi harness: after `compaction_end` and a settled run, `/aria-rotate` is sent once and
   the new `sessionId` lands in `segments`; a second settle without a new compaction sends nothing; a later Pi start
   uses the new segment id; over 8 MB `compact` is sent; `chatEarlier` answers an `earlier` event from fixture files.
6. `crates/aria/tests/live_rotation.rs`, `#[ignore]` like `tests/live_pi.rs` (read it and follow it): a scratch
   `ARIA_HOME`, the real pinned Pi, two short prompts, `/compact`, the rotation, and the new segment's `get_messages`
   starts with the `aria-earlier` custom message. It runs only by hand (it spends a little model use).

## Task 4: the page shows the summary and pages back

Files: `ui/chat/state.js`, `ui/chat/thread.js`, `ui/chat/view.js`, `ui/chat/texts.js`, their tests,
`tools/ui-sandbox/` (stub + check).

1. `fromHistory`: a `compactionSummary` message and a custom `aria-earlier` message become one entry
   `{ kind: 'earlier', summary }` (rendered as a divider "Earlier in this conversation" / "Antes en esta conversación",
   the summary folded under it, opened on click; reuse the existing divider and fold components — never a look-alike).
2. "Show earlier" / "Ver lo anterior": a button above the first message while the state says there is more (true when
   the history started with an `earlier` entry, then from the last `earlier` reply's `more`). Clicking posts
   `chatEarlier` with the last cursor (none the first time) and shows it is loading.
3. The reducer handles `{ type: 'earlier', messages, cursor, more }`: prepends `fromHistory(messages)` above the current
   entries, stores `cursor` and `more`. The view keeps the reader's scroll position when content is prepended.
4. Tests: the divider from both message kinds; the button shows and hides; a reply prepends in order and keeps the
   rest; the ui-sandbox stub answers `chatEarlier` with two pages and the check clicks through them.
