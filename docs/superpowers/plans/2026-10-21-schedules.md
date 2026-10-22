# Scheduled Tasks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The person asks Aria in the chat for a recurring task ("every day at 7:30 give me my Notion tasks"); the dock
runs it by itself in the task's own tab, with fixed permissions, and posts a notification.

**Architecture:** `aria-core/src/schedules.rs` holds the tasks (`~/.aria/schedules.json`) and the pure timing rules. The
dock checks every enabled task once a second, creates the task's tab (a `sessions.json` entry with `schedule`), shows a
`scheduled_run` event and sends the prompt through the chat's ordered worker; the tab's Pi runs with `ARIA_GRANTS`. A Pi
extension `schedule` creates/lists/changes/deletes tasks through new ops on the session's jobs socket. The page shows
scheduled tabs outside the 8 and a `/schedules` panel.

**Tech Stack:** Rust (aria-core, aria; `libc` for local time), TypeScript Pi extensions (bun tests), plain JS page.

**Spec:** `docs/superpowers/specs/2026-10-21-schedules-design.md`

## Global Constraints

- File `~/.aria/schedules.json`, mode 600, `{ "tasks": [...] }`, camelCase keys as in the spec; a broken file moves to `schedules.json.broken`.
- Task id `s-` + 12 lowercase alphanumerics; at most 50 tasks; `at` is `HH:MM` 00:00–23:59; `days` ⊆ `mon tue wed thu fri sat sun`, non-empty unless `date` (`YYYY-MM-DD`) is set.
- Grants: connector levels `read` | `allow` only, unknown connector names refused; `writable` absolute, existing, not a secret path.
- Due = enabled ∧ occurrence ≤ now ∧ occurrence on today's local date ∧ occurrence > max(lastRun, created). Late = now − occurrence > 120 000 ms. `lastRun` written before the prompt is sent.
- Env var `ARIA_GRANTS` JSON `{ "connectors": {name: level}, "sandbox": { "askToLeave": false, "extraWritable": [...] } }`.
- Page messages: `schedulesList {id}`, `scheduleUpdate {id, schedule, enabled}`, `scheduleDelete {id, schedule}`; dock messages: `schedules {id?, tasks}`, `scheduleTab {id, title, schedule}`; chat event `scheduled_run {text, at, late}`.
- Socket ops `schedule_create|schedule_list|schedule_update|schedule_delete`, answered `{re, ok, …}` like the other ops.
- Notification title `Aria · <task title>`; summary = first non-empty line of the last answer, ≤ 140 chars.
- English everywhere in code and docs; UI strings in `en` and `es`; match surrounding style; never touch the real `~/.aria` in tests.

## Review Focus

1. A task created at 07:40 with `at` 07:30 must not run today (created after the occurrence) — Task 1 test.
2. A task whose time passes while its tab is mid-run waits for the run to settle and then runs once — Task 6 test.
3. The page saving its tabs while a scheduled tab exists only on disk must not drop it — Task 2 test.
4. Deleting a task while it runs: the run is stopped, the tab and entry removed, no notification after — Task 6 test.
5. `ARIA_GRANTS` that is not valid JSON must leave `aria.json`'s config untouched (never widen access) — Task 3 test.

---

### Task 1: The tasks and their timing (`aria-core/src/schedules.rs`)

**Files:** Create `crates/aria-core/src/schedules.rs`; modify `crates/aria-core/src/lib.rs`, `crates/aria-core/src/paths.rs` (`pub fn schedules(&self) -> PathBuf`).

**Interfaces — Produces:**
- `Task { id, title, prompt, when: When, grants: Grants, enabled: bool, created: i64, last_run: Option<i64>, last_result: Option<RunResult> }`, `When { at: String, days: Vec<String>, date: Option<String> }`, `Grants { connectors: BTreeMap<String,String>, writable: Vec<String> }`, `RunResult { at: i64, ok: bool, summary: String }` (serde camelCase, defaults).
- `pub trait Clock { fn local(&self, epoch_ms: i64) -> Local; fn epoch(&self, y: i32, m: u32, d: u32, h: u32, min: u32) -> i64; }`, `Local { y, m, d, h, min, weekday: u32 /*0=mon*/ }`; `pub struct SystemClock` (libc `localtime_r`/`mktime`).
- `pub fn occurrence(task, now, clock) -> Option<i64>`, `pub fn due(task, now, clock) -> bool`, `pub fn late(occurrence, now) -> bool`, `pub fn next_run(task, now, clock) -> Option<i64>`.
- `pub fn validate(input: &TaskInput, servers: &[String]) -> Result<TaskInput, String>` with `TaskInput` = the create/update fields; `pub fn new_id() -> String`.
- `pub fn load(paths) -> Vec<Task>` (broken → moved aside, empty), `pub fn save(paths, &[Task]) -> io::Result<()>`.

- [ ] **Step 1: failing tests** with a fixed-offset `FakeClock`: `every_day_runs_at_its_time`, `weekdays_skip_the_weekend`, `a_one_off_runs_on_its_date_only`, `created_after_todays_time_waits_for_tomorrow`, `ran_already_today_is_not_due`, `slept_across_the_time_runs_late_once`, `three_days_off_runs_only_today`, `disabled_never_runs`, `late_is_over_two_minutes`, `next_run_is_tomorrow_after_todays_run`, `validate_refuses_bad_times_days_connectors_and_paths`, `store_round_trips_with_mode_600`, `a_broken_store_is_moved_aside`, plus `system_clock_round_trips_a_local_time` (libc).
- [ ] **Step 2:** `cargo test -p aria-core schedules` — fails to compile.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** `cargo test -p aria-core` — pass.
- [ ] **Step 5:** commit `feat(schedules): scheduled tasks and when they run`.

### Task 2: Scheduled tabs in `sessions.json`

**Files:** `crates/aria-core/src/sessions.rs`, `crates/aria/src/chat.rs` (`SessionsSave` handler).

**Interfaces — Produces:** `SessionEntry.schedule: Option<String>` (`#[serde(default, skip_serializing_if = "Option::is_none")]`); `sanitized()` keeps ≤ 8 regular + all scheduled; `pub fn merge_scheduled(from_page: Vec<SessionEntry>, on_disk: &SessionList, tasks: &[String]) -> Vec<SessionEntry>` (adds on-disk scheduled entries the page did not send if their task id is in `tasks`; drops scheduled entries whose task is gone); `pub fn add_scheduled(paths, id, title, schedule) -> io::Result<()>`.

- [ ] Tests: `scheduled_tabs_do_not_count_against_eight`, `a_page_save_keeps_a_scheduled_tab_it_did_not_send`, `a_scheduled_tab_of_a_deleted_task_is_dropped`, `add_scheduled_is_idempotent`; then the chat.rs `SessionsSave` path uses `merge_scheduled` (a test in chat.rs). Commit `feat(sessions): scheduled tabs beside the eight`.

### Task 3: `ARIA_GRANTS` overlay

**Files:** `extensions/lib/config.ts`, `extensions/lib/config.test.ts`.

**Interfaces — Produces:** `ariaConfig()` overlays `process.env.ARIA_GRANTS` (`connectors` replaced whole; `sandbox.askToLeave`, `sandbox.extraWritable` replaced) after merging defaults and `aria.json`; invalid JSON or wrong types are ignored (config as without it).

- [ ] Tests: `grants_set_connector_levels_and_turn_asking_off`, `invalid_grants_change_nothing`, `a_level_outside_read_allow_ask_becomes_ask` (existing rule still applies). Commit `feat(extensions): a scheduled tab's fixed permissions (ARIA_GRANTS)`.

### Task 4: Socket ops

**Files:** `crates/aria/src/jobs_socket.rs`, `crates/aria/src/schedule_store.rs` (new: `#[derive(Clone)] pub struct ScheduleStore { paths, tasks: Arc<Mutex<Vec<Task>>> }` with `create`, `list`, `update`, `delete`, `take_due(now, clock)`, `record(id, RunResult)`, `servers()` from `mcp.json`).

**Interfaces — Produces:** `Session` gains `schedules: ScheduleStore`; ops answer `{task}` / `{tasks: [{…task, next}]}` / `{}`; errors as `Err(String)`.

- [ ] Tests over the socket: `create_lists_and_deletes_a_task`, `create_refuses_an_unknown_connector`, `update_pauses_a_task`, `more_than_fifty_is_refused`; `ScheduleStore` unit tests: `take_due_writes_last_run_first`. Commit `feat(jobs): schedule ops on the session socket`.

### Task 5: The `schedule` extension

**Files:** create `extensions/schedule/index.ts`, `extensions/schedule/index.test.ts`; register it where extensions are listed (`extensions/README.md`, `crates/aria-core/src/agent.rs` if it enumerates them).

**Interfaces — Produces:** tools `schedule_create`, `schedule_list`, `schedule_update`, `schedule_delete` (JSON-schema params via the `str`/`object` helpers of `extensions/background/run.ts`); description says to state schedule and permissions and wait for a yes before `schedule_create`; refuse in subagents and without a dock (reuse `DOCK_REFUSAL`).

- [ ] Tests against the fake dock: `create_sends_the_fields`, `list_shows_next_runs`, `refuses_in_a_subagent`, `refuses_without_a_dock`. Commit `feat(extensions): schedule tools`.

### Task 6: The dock runs tasks

**Files:** `crates/aria/src/scheduler.rs` (new), `crates/aria/src/chat.rs`, `crates/aria/src/main.rs`, `crates/aria-core/src/bridge.rs`, `crates/aria/src/handlers.rs`.

**Interfaces — Produces:**
- Bridge: `PageMessage::{SchedulesList{id}, ScheduleUpdate{id, schedule, enabled}, ScheduleDelete{id, schedule}}`, `DockMessage::{Schedules{id: Option<u64>, tasks: Value}, ScheduleTab{id, title, schedule}}`.
- `Sessions::run_scheduled(&self, task: &Task, late: bool)`: ensures the entry (`add_scheduled`) and emits `ScheduleTab` the first time, sets the chat's grants (env `ARIA_GRANTS` at its next spawn), emits `scheduled_run`, sends `ChatSend` — or queues it until the chat is not busy.
- `Chat` gains `scheduled: Option<(String /*task id*/, String /*title*/)>`; on `agent_settled` of a scheduled run it calls back `ScheduleStore::record` and the notifier (unless the tab is on screen).
- `scheduler::tick(store, sessions, clock, now)` called once a second from main's tick, off the UI thread.

- [ ] Tests (fake Pi): `a_due_task_creates_its_tab_and_sends_its_prompt`, `the_settled_run_is_recorded_and_notified`, `a_task_due_mid_run_waits_for_the_run`, `deleting_a_running_task_stops_it_and_removes_its_tab`, `the_tab_pi_gets_its_grants`. Commit `feat(dock): run scheduled tasks in their own tabs`.

### Task 7: The page

**Files:** `ui/chat/sessions.js`, `ui/chat/state.js`, `ui/chat/view.js`, `ui/chat/parts/` (new `schedules.js` panel part + css + test), `ui/app.js`, `ui/chat/texts.js`, `ui/i18n/*.json`, `tools/ui-sandbox/serve.ts` + `check.ts`.

**Interfaces — Consumes:** Task 6 messages. **Produces:** tab entries carry `schedule`; `addSession` count ignores them; `visibleScheduled` rule (active or unread); `scheduled_run` reduces to a user entry with `scheduled: {at, late}`; panel `/schedules`.

- [ ] Tests: sessions.test (`scheduled_tabs_are_outside_the_eight`, `a_scheduled_tab_shows_only_with_news`), state.test (`scheduled_run_is_a_bubble_and_the_echo_is_dropped`), part test for the panel (rows, pause, delete, escaping), check.ts (a `scheduleTab` + `scheduled_run` from the stub shows the tab and bubble; `/schedules` lists). Commit `feat(chat): scheduled tabs and the /schedules panel`.

### Task 8: Docs and the live check

**Files:** `README.md` (what it does, where things live), `docs/architecture.md`, `docs/settings.md` (schedules.json), `docs/pending.md`, `docs/ideas.md` (#14 note: a scheduled task can be the morning brief).

- [ ] Live check per the spec's Testing section with a throwaway `ARIA_HOME`; full suites; commit `docs: scheduled tasks`.
