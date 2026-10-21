# Connector Approvals and Attachments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every MCP write asks first (or is refused or allowed by the connector's access level), and the chat takes
files and images as attachments.

**Architecture:** Part 1 is a Pi extension (`extensions/connectors/`) on Pi's `tool_call` hook, which also sees codemode's
nested calls; its levels come from `settings.json` → `aria.json`; it asks with the existing permission card (new
`kind: "connector"`). Part 2 adds attachments to `chatSend`; the dock saves them under `~/.aria/attachments/` and sends
images through Pi's `prompt.images` when the model can see them, paths otherwise; the composer gets a paperclip, drop
and paste, and the bubble shows chips.

**Tech Stack:** TypeScript Pi extensions (`bun test`), Rust (`aria-core`, `aria`), plain ES modules UI (`bun test ui`),
headless Chrome checks (`tools/ui-sandbox`).

**Spec:** `docs/superpowers/specs/2026-10-21-connector-approvals-and-attachments-design.md`

## Global Constraints

- Everything in English (code, names, JSON, docs, commits); visible strings in English and Spanish with correct accents.
- Access levels: exactly `"read" | "ask" | "allow"`, default `"ask"`, setting `integrations.access` (object, default `{}`),
  `aria.json` key `connectors`. `engram` is never gated.
- Read words (first word of the tool name, server prefix removed): `search, fetch, get, list, read, query, find,
  retrieve, lookup, describe, view, show, count, check`. Resource tools `list_mcp_resources`,
  `list_mcp_resource_templates`, `read_mcp_resource` are reads.
- Card options, in order: `Yes, just this once` (primary), `Yes, and don't ask again for <connector> <action> this session`, `No`.
- Preview: at most 12 lines, values cut to 300 characters.
- Attachments: at most 10 per message, 20 MB each, 50 MB together; saved under
  `~/.aria/attachments/<session>/<yyyy-mm-dd>/<n>-<safe name>` (file 0600, folders 0700); safe name `[A-Za-z0-9._-]`,
  at most 80 characters. Message lines: `[attached: <name> (<mime>, <size>) at <path>]` and
  `[attached image: <name> at <path>]`.
- UI house rule: solid fills that never change on hover/focus; only borders take the accent. Reuse parts, never copy.
- Public repo: no personal paths, names or employers in code, tests or docs.
- Commit per task, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push.

## Review Focus

1. A codemode script that calls a write tool: it is gated exactly like a direct call — Task 2 test `a_nested_call_is_gated_the_same`.
2. A connector whose tools have no annotations and odd names (`createPage`, `notion-update-page`, `rea_open_binary`) — Task 2 test `names_without_annotations`.
3. A file named `../../etc/passwd` or with spaces and accents — Task 4 test `unsafe_names_are_made_safe`.
4. An image attached while the model cannot see images — Task 4 test `an_image_for_a_text_model_goes_as_a_path`.
5. A message with attachments sent while Aria works (queued or steering) — Task 5 test `a_queued_message_keeps_its_attachments`.

---

### Task 1: The access setting

**Files:** `crates/aria-core/src/settings.rs`, `crates/aria-core/src/agent_config.rs`, `crates/aria-core/tests/golden/*.json`
(regenerate), `ui/onboarding/steps.js`, `extensions/lib/config.ts` (+ tests), `docs/settings.md`.

**Interfaces — Produces:** `Integrations.access: BTreeMap<String, String>` (serde `access`, default empty);
`aria.json` gains `"connectors": { … }` (same map); `extensions/lib/config.ts` `AriaConfig.connectors: Record<string, 'read'|'ask'|'allow'>`
(DEFAULTS `{}`), an unknown value read as `'ask'`; JS defaults `integrations.access: {}`.

- [ ] **Step 1: Tests** — Rust: `access_defaults_to_empty`, `access_round_trips`, `aria_json_carries_the_connectors`
  (golden updated with `cargo test -p aria-core regenerate_golden -- --ignored`); JS: `steps.test.js` defaults include
  `access: {}`; `config.test.ts` `connectors default to empty`, `an unknown level reads as ask`.
- [ ] **Step 2: Run** — `cargo test -p aria-core && bun test ui/onboarding extensions/lib` → new tests FAIL.
- [ ] **Step 3: Implement**, update `docs/settings.md`.
- [ ] **Step 4: Run** — same commands → PASS.
- [ ] **Step 5: Commit** — `feat(settings): an access level per connector (read only, ask, allow)`

### Task 2: The connectors extension

**Files:** Create `extensions/connectors/index.ts`, `extensions/connectors/classify.ts`, `extensions/connectors/card.ts`,
and their `*.test.ts`; modify `extensions/README.md` (table row).

**Interfaces — Consumes:** `ariaConfig().connectors` (Task 1), `canAsk` from `../sandbox/card.ts`.
**Produces:**
- `classify.ts`: `export function serverOf(toolName: string): string | null` (`mcp__<server>__<tool>`, resource tools →
  `'*'`), `export function isWrite(toolName: string, annotations?: { readOnlyHint?: boolean; destructiveHint?: boolean }): boolean`,
  `export function actionOf(toolName: string, title?: string): string` ("notion-create-pages" → "create pages").
- `card.ts`: `export function preview(input: unknown): string`, `export function connectorCard(connector: string, action: string, detail: string, options: string[]): string`
  (`{"aria":1,"kind":"connector",summary,detail,options}`, first option primary).
- `index.ts`: `pi.on('tool_call', …)` gating as the spec says; grants per session (`Set` of `server\0tool`), cleared on
  `session_start`; annotations looked up with `pi.getAllTools()` by name (cached per name, refreshed when missing).

- [ ] **Step 1: Tests** — `classify.test.ts`: `annotations_decide_first`, `names_without_annotations` (`notion-search` read,
  `notion-create-pages` write, `createPage` write, `getPage` read, `rea_open_binary` write, `list_mcp_resources` read),
  `the_server_is_the_middle_part`. `card.test.ts`: `the_preview_is_short_and_readable` (12 lines max, 300 chars,
  nested as compact JSON), `the_card_is_a_connector_card`. `index.test.ts` (a fake `pi`): `a_read_runs_without_asking`,
  `a_write_on_ask_asks_and_runs_on_yes`, `no_refuses_with_the_reason`, `session_grant_skips_the_next_question`,
  `read_only_refuses_without_asking`, `allow_runs_without_asking`, `nobody_to_ask_refuses`, `engram_is_never_gated`,
  `a_nested_call_is_gated_the_same`, `grants_clear_on_session_start`.
- [ ] **Step 2: Run** — `bun test extensions/connectors` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — `bun test extensions` → PASS.
- [ ] **Step 5: Commit** — `feat(extensions): connectors ask before they write, by the access level of each`

### Task 3: The card and the settings rows

**Files:** `ui/chat/parts/permission.js` (+ test, css if needed), `ui/chat/parts/cards.js` (+ test), `crates/aria/src/chat.rs`
(`dialog_body` for `kind: "connector"`: its `summary`), `ui/onboarding/view.js` (+ test), `ui/i18n/en.json`, `ui/i18n/es.json`,
`tools/ui-sandbox/serve.ts`, `tools/ui-sandbox/check.ts`.

**Interfaces — Produces:** `renderPermission({ kind: 'connector', … })` with a plug icon and default title
`chat.perm.connector` ("A connector wants to act" / "Un conector quiere actuar"); the Integrations section lists every
`mcp:<server>` from `connected` except `engram` with a three-way choice (`ob.integrations.access.read|ask|allow`:
Read only / Ask / Allow — Solo lectura / Preguntar / Permitir) writing `integrations.access[server]`.

- [ ] **Step 1: Tests** — `permission.test.js` `a connector card has its icon and title`; `cards.test.js` `kind connector routes to the permission card`;
  Rust `a_connector_card_notifies_with_its_summary`; `view.test.js` `integrations list each connected server with its access`, `choosing a level saves it`;
  sandbox "Connector wants to write" simulation and check `the connector card sends the chosen option`.
- [ ] **Step 2: Run** — `bun test ui && cargo test -p aria chat::` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — same + `bun tools/ui-sandbox/serve.ts 8770 &` / `bun tools/ui-sandbox/check.ts` → PASS.
- [ ] **Step 5: Commit** — `feat(chat): the connector card, and each connector's access level in the settings`

### Task 4: Attachments in the dock

**Files:** `crates/aria-core/src/bridge.rs`, `crates/aria-core/src/paths.rs` (`pub fn attachments(&self) -> PathBuf`),
create `crates/aria/src/attachments.rs`, modify `crates/aria/src/chat.rs`, `crates/aria/src/main.rs` (`mod attachments;`).

**Interfaces — Produces:**
- `PageMessage::ChatSend { text, behavior, #[serde(default)] attachments: Vec<Attachment> }`,
  `pub struct Attachment { pub name: String, pub mime: String, pub data: String /* base64 */ }`.
- `attachments.rs`: `pub struct Saved { pub name: String, pub mime: String, pub size: u64, pub path: PathBuf, pub image: Option<(String /*mime*/, String /*base64*/)> }`;
  `pub fn save(root: &Path, session: &str, today: &str, list: &[Attachment]) -> Result<Vec<Saved>, String>` (limits,
  safe names, modes; `image` set for `image/png|jpeg|gif|webp`); `pub fn safe_name(name: &str) -> String`;
  `pub fn message(text: &str, saved: &[Saved], images_ok: bool) -> (String, Vec<Value>)` (the lines and the Pi `images`).
- `chat.rs`: `send_prompt` takes the attachments; images go as images only when the current model's `input` includes
  `image` (from `get_available_models`, cached per model); a save error emits the same `command_failed` the page
  already handles (text goes back).

- [ ] **Step 1: Tests** — `unsafe_names_are_made_safe` (`../../etc/passwd` → `etc_passwd`-like, accents/spaces, 80 chars),
  `files_are_private` (0600/0700), `limits_are_enforced` (11 files, 21 MB, 51 MB → Err), `an_image_for_a_text_model_goes_as_a_path`,
  `an_image_for_a_vision_model_goes_as_an_image_and_a_path_line`, `other_files_go_as_path_lines`; bridge
  `chat_send_without_attachments_still_parses`; chat `the_prompt_carries_images_for_a_vision_model` (fake Pi records
  the prompt command), `a_failed_save_gives_the_text_back`.
- [ ] **Step 2: Run** — `cargo test -p aria-core bridge && cargo test -p aria attachments:: chat::` → FAIL.
- [ ] **Step 3: Implement.** Base64 decode: no new dependency if a small decoder is simple; otherwise add `base64 = "0.22"` to `crates/aria`.
- [ ] **Step 4: Run** — Rust suites one crate at a time, fmt, clippy `--workspace --all-targets` → PASS, no warnings.
- [ ] **Step 5: Commit** — `feat(chat): the dock saves attachments and sends images to models that can see them`

### Task 5: Attachments in the composer and the thread

**Files:** `ui/chat/parts/composer.js` (+ css, test), `ui/chat/parts/user.js` (+ test), create `ui/chat/parts/attachments.js`
(+ css, test) for the chips (reusing the pill part `ui/chat/parts/pill.js` for each chip), `ui/chat/view.js`, `ui/chat/state.js`
(+ test), `ui/chat/texts.js`, `tools/ui-sandbox/serve.ts`, `tools/ui-sandbox/check.ts`.

**Interfaces — Produces:** composer prop `attachments: [{ id, name, mime, size }]`, a paperclip button
(`data-chat="attach"`, hidden `<input type=file multiple>`), chips with remove (`data-chat="attach-remove" data-id`);
drop on the chat view and paste in the input add files; limits refused with a note (`chat.attach.tooMany`,
`chat.attach.tooBig`); `send` includes `attachments: [{ name, mime, data }]` (base64 via `FileReader`); the user bubble
renders `[attached: …]` / `[attached image: …]` lines as chips (`renderUser`), live and from history; queued and steering
messages keep their attachments (the dock gets them with the queued text).

- [ ] **Step 1: Tests** — `attachments.test.js` `chips with names, sizes and remove`; `composer.test.js` `the paperclip and the chips`;
  `user.test.js` `attached lines become chips`, `the raw lines are not shown`; `state.test.js` `a_queued_message_keeps_its_attachments`;
  view-level via sandbox checks: `attach adds a chip`, `remove drops it`, `send carries the attachments`, `a dropped file is attached`.
- [ ] **Step 2: Run** — `bun test ui` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — `bun test ui` and the browser checks → PASS.
- [ ] **Step 5: Commit** — `feat(chat): attach files and images from the composer, by drop or by paste`

### Task 6: Docs and the whole suite

**Files:** `docs/architecture.md`, `extensions/README.md`, `docs/pending.md`, `CHANGELOG.md`, `docs/ideas.md` (mark 1 and 2 done).

- [ ] **Step 1: Docs** — architecture: `chatSend.attachments`, `~/.aria/attachments`; extensions README: `connectors`;
  pending: old attachments are never removed, image previews not in the thread, connector previews not redacted;
  CHANGELOG: `~/.aria/attachments/` row with revert; ideas: 1 and 2 marked done (2026-10-21).
- [ ] **Step 2: Whole suite** — `sh tools/parity/suites.sh` → `all suites passed`; fmt; clippy; browser checks.
- [ ] **Step 3: Commit** — `docs: connector approvals and attachments`
