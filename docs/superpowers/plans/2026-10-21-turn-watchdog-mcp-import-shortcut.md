# Turn Watchdog, MCP Import and Shortcut Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop a turn that stops answering; import the MCP servers the person already has, from the onboarding and
the settings; let the person choose the shortcut, with fallbacks when it is taken.

**Architecture:** A watchdog in the dock's `Chat` aborts a silent run. A new `aria-core` module reads other apps' MCP
configs and writes the chosen ones into Aria's `mcp.json`; the page lists them in Integrations. A platform-free
shortcut module parses and orders fallbacks; `aria-hyprland` binds the chosen one and checks `hyprctl -j binds`;
`aria-mac` registers it with Carbon.

**Tech Stack:** Rust (`aria-core`, `aria-hyprland`, `aria-mac`, `aria`), plain ES modules UI (`bun test ui`), headless
Chrome checks (`tools/ui-sandbox`).

**Spec:** `docs/superpowers/specs/2026-10-21-turn-watchdog-mcp-import-shortcut-design.md`

## Global Constraints

- English code, names, JSON and docs; visible strings in English and Spanish with correct accents.
- Settings: `models.turnTimeoutMinutes` (default 10, 0 = off) and `shortcut` (default `""` = platform default),
  mirrored in Rust, JS and the golden JSON (`cargo test -p aria-core regenerate_golden -- --ignored`).
- Watchdog check every 30 s; abort once per run; paused while a dialog waits; notice key `chat.watchdog.stopped` with `minutes`.
- Import sources and keys exactly as the spec's table; `id` = `source:name`; secrets never reach the page; names
  deduplicated with `-2`, `-3`…; `mcp.json` stays mode 0600; never write other apps' files.
- Shortcut format `MOD + MOD + KEY`; Linux modifiers `SUPER CTRL ALT SHIFT`, macOS `Command Control Option Shift`;
  keys A–Z, 0–9, `Space`, `F1`–`F12`; fallbacks: chosen, default, then `SUPER + SHIFT + A`, `SUPER + ALT + A`
  (macOS `Control + Option + Space`, `Command + Option + Space`).
- UI house rule: solid fills, only borders take the accent; reuse onboarding components, never copy.
- Public repo: nothing personal in code, tests, fixtures or docs (fixtures use neutral names and `/home/ana`-style paths).
- Commit per task ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push.

## Review Focus

1. A run waiting on the person (a card open) for an hour is never aborted — Task 1 test `a_waiting_dialog_pauses_the_watchdog`.
2. A Claude Code config with dozens of unrelated keys and a broken entry — Task 2 test `odd_entries_are_skipped_and_the_rest_found`.
3. Importing a server whose name collides twice — Task 2 test `names_are_deduplicated`.
4. A shortcut the person types lower case or with extra spaces (`super+shift+a`) — Task 4 test `parsing_is_forgiving_and_canonical`.
5. The chosen shortcut and the default both taken — Task 4 test `the_first_free_fallback_wins`.

---

### Task 1: The turn watchdog

**Files:** `crates/aria-core/src/settings.rs` (+ golden), `ui/onboarding/steps.js` (+ test), `crates/aria/src/chat.rs`,
`crates/aria/src/main.rs` (the 30 s tick), `ui/chat/state.js` or the notice texts (`chat.watchdog.stopped`), `ui/onboarding/view.js`
(Models section field), `ui/i18n/{en,es}.json`, `docs/settings.md`.

**Interfaces — Produces:** `Models.turn_timeout_minutes: u32` (serde `turnTimeoutMinutes`, default 10);
`Chat::check_stalled(&self, now: Instant, limit: Duration) -> bool` (aborts and notices at most once per run; true when it did);
`Sessions::check_stalled(now)` called from the tick every 30 s with the settings' limit.

- [ ] **Step 1: Tests** — Rust: `turn_timeout_defaults_to_ten`, `a_silent_run_is_stopped_once` (fake Pi starts a run and stays silent; limit 1 s; one `abort` sent, one notice), `a_waiting_dialog_pauses_the_watchdog`, `zero_turns_it_off`, `events_keep_it_alive`; JS: steps defaults, the Models field writes `turnTimeoutMinutes`.
- [ ] **Step 2: Run** — `cargo test -p aria-core && cargo test -p aria chat:: && bun test ui` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — same + fmt + clippy → PASS.
- [ ] **Step 5: Commit** — `feat(chat): a turn that stops answering is stopped, and the person told`

### Task 2: Finding and importing MCP servers (core)

**Files:** create `crates/aria-core/src/mcp_import.rs` (+ `pub mod`), modify `crates/aria-core/src/bridge.rs`,
`crates/aria/src/handlers.rs` (`is_blocking` for both), fixtures under `crates/aria-core/tests/fixtures/mcp/` (neutral content).

**Interfaces — Produces:**
- `pub struct Found { pub id: String, pub name: String, pub source: String /* claude-code|claude-desktop|cursor|vscode */, pub kind: String /* stdio|http */, pub target: String, pub secrets: Vec<String>, config: Value /* Pi format, not serialized to the page */ }`;
- `pub fn find(home: &Path, existing: &[String]) -> Vec<Found>` (macOS paths under `cfg(target_os = "macos")`);
- `pub fn import(paths: &AriaPaths, found: &[Found], ids: &[String]) -> Result<Vec<String> /* names written */, ImportError>`;
- `PageMessage::McpScan { id }`, `PageMessage::McpImport { id, ids: Vec<String> }`, `DockMessage::McpFound { id, servers: Value }`.
  The handler for `McpImport` also triggers the engines restart (as `enginesRestart` does) on success.

- [ ] **Step 1: Tests** — `claude_code_user_servers_are_found`, `vscode_uses_servers`, `http_and_sse_are_http`, `odd_entries_are_skipped_and_the_rest_found`, `secrets_are_names_only` (serialized `Found` has no values), `existing_and_engram_are_hidden`, `names_are_deduplicated`, `import_keeps_the_others_and_the_mode`, `nothing_to_import_is_an_error`; bridge round trips; handler `scan_then_import_writes_mcp_json`.
- [ ] **Step 2: Run** — `cargo test -p aria-core mcp_import bridge && cargo test -p aria handlers::` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Rust suites, fmt, clippy → PASS.
- [ ] **Step 5: Commit** — `feat(integrations): find the MCP servers other apps have, and import the ones the person picks`

### Task 3: The import section (onboarding and settings)

**Files:** `ui/onboarding/view.js` (+ test), `ui/onboarding/components.js` only if a checkbox list widget is missing
(reuse what exists first), `ui/app.js` (request plumbing if needed), `ui/i18n/{en,es}.json`, `tools/ui-sandbox/{serve,check}.ts`.

**Interfaces — Consumes:** Task 2 messages. **Produces:** the "Your other MCP servers" section in the Integrations step
and in Settings → Integrations: scan on open, checkboxes, Import (disabled when none checked), a result line
("Imported 2: rea, github" / "Importados 2: …"), the nothing-found line.

- [ ] **Step 1: Tests** — `view.test.js`: `found servers are listed with source and target`, `secrets show only their names`, `import sends the checked ids`, `nothing found says so`, `already imported servers are not offered again after a refresh`; browser check `importing a found server`.
- [ ] **Step 2: Run** — `bun test ui` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — `bun test ui` + browser checks → PASS.
- [ ] **Step 5: Commit** — `feat(onboarding): the MCP servers the person already has, a step of the setup and in the settings`

### Task 4: The shortcut (core and Linux)

**Files:** create `crates/aria-core/src/shortcut.rs` (+ `pub mod`), `crates/aria-core/src/settings.rs` (+ golden),
`ui/onboarding/steps.js`, `crates/aria-hyprland/src/lib.rs`, `crates/aria/src/main.rs` (apply on settings change,
send `Platform` again), `docs/settings.md`.

**Interfaces — Produces:**
- `pub struct Shortcut { pub mods: Vec<String>, pub key: String }`, `pub fn parse(text: &str, platform: Platform) -> Option<Shortcut>`,
  `impl Shortcut { pub fn label(&self) -> String /* canonical "SUPER + SHIFT + A" */ }`, `pub fn candidates(chosen: &str, platform: Platform) -> Vec<Shortcut>` (chosen, default, fallbacks; deduplicated);
- `aria_hyprland`: `register_hotkey(exe, &Shortcut)` (Lua unbinds Aria's previous shortcut and binds the new one), `fn taken(binds_json: &Value, s: &Shortcut, exe: &Path) -> bool`, `keep_hotkey_registered` takes a shortcut source so a settings change re-binds; returns which shortcut is in use.
- `Settings.shortcut: String` (default `""`).

- [ ] **Step 1: Tests** — `parsing_is_forgiving_and_canonical`, `invalid_shortcuts_are_refused`, `the_first_free_fallback_wins`, `the_bind_lua_for_a_shortcut`, `taken_reads_hyprctl_binds` (fixture), `aria_s_own_bind_is_not_taken`, `a_settings_change_rebinds` (with the existing fake Hyprland socket tests' pattern).
- [ ] **Step 2: Run** — `cargo test -p aria-core shortcut && cargo test -p aria-hyprland` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Rust suites, fmt, clippy → PASS.
- [ ] **Step 5: Commit** — `feat(shortcut): choose the shortcut, with fallbacks when it is taken (Omarchy)`

### Task 5: The shortcut row and macOS

**Files:** `ui/onboarding/view.js` (+ test), `ui/app.js` (Platform updates reach the settings row), `ui/i18n/{en,es}.json`,
`crates/aria-mac/src/hotkey.rs`, `crates/aria/src/main.rs` (macOS path), `tools/ui-sandbox/{serve,check}.ts`, `docs/pending.md`.

**Interfaces — Produces:** the "Shortcut" row (key capture, Esc cancels, Backspace resets, shows `<kbd>`s, "In use: X (Y was taken)");
`aria_mac::install_hotkey_for(&Shortcut, callback) -> Result<(), String>` and unregister of the previous one, with the
modifier/key mapping in a pure function tested on every platform (`carbon_codes(&Shortcut) -> Option<(u32, u32)>`).

- [ ] **Step 1: Tests** — `view.test.js` `the shortcut row records keys`, `escape cancels`, `backspace resets to the default`, `in use shows a fallback`; Rust `carbon_codes_for_option_space_and_letters` (compiled on Linux: keep the mapping out of `cfg(macos)`); browser check `recording a shortcut`.
- [ ] **Step 2: Run** — `bun test ui && cargo test -p aria-mac` → FAIL (aria-mac tests compile on Linux? check how it builds today; if aria-mac is mac-only, put `carbon_codes` in `aria-core::shortcut`).
- [ ] **Step 3: Implement**; the Carbon registration code is written but not compiled here — note it in `docs/pending.md`.
- [ ] **Step 4: Run** — `bun test ui`, Rust suites, browser checks → PASS.
- [ ] **Step 5: Commit** — `feat(settings): record the shortcut in the settings; macOS registers the chosen one`

### Task 6: Docs and the whole suite

- [ ] **Step 1** — docs per the spec's Docs section; `docs/ideas.md`: 3, 5, 6 marked done (3 with "the dock stops a turn that stops answering; MCP reconnects stay Pi's").
- [ ] **Step 2** — `sh tools/parity/suites.sh` → `all suites passed`; fmt; clippy; browser checks.
- [ ] **Step 3: Commit** — `docs: turn watchdog, MCP import, the shortcut`
