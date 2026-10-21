# Background Work Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aria runs never-ending commands and long jobs in the background, owned by the dock, visible as a strip of
pills above the composer and in the Alt+A list, and never forgotten by Aria (hidden context line, end-of-job messages).

**Architecture:** A `Jobs` manager in the dock (`crates/aria/src/jobs.rs`, with `worktree.rs` and `jobs_socket.rs`)
spawns and supervises processes and `pi --mode json` subagents from one spawner thread, keeps their output, persists
their state to `~/.aria/jobs.json`, and keeps per-session events until acknowledged. Each session's Pi reaches it
through a per-session Unix socket (`ARIA_JOBS_SOCK`) from a new extension (`extensions/background/`); the sandbox
extension registers `background_run` so commands keep the sandbox route. The page gets a global `jobs` message and
draws the strip (new `background.js` part, pills from a shared pill part) and the list (inside the subagents view).

**Tech Stack:** Rust (std, `libc`, `serde_json`), TypeScript Pi extensions run by Pi 1.3.0 (`bun test`), plain ES
modules UI (`bun test ui`), headless Chrome checks (`tools/ui-sandbox`).

**Spec:** `docs/superpowers/specs/2026-10-21-background-work-design.md`

## Global Constraints

- Everything in English: code, names, JSON keys, messages, docs, commits. Visible strings in English and Spanish.
- Limits (constants in `jobs.rs`): `MAX_AGENTS_RUNNING = 3`, `MAX_PROCESSES = 10`, `OUTPUT_LINES = 1000`,
  `LINE_CHARS = 2000`, `RESULT_CHARS = 8000`, `FINISHED_KEPT = 20`, `STOP_GRACE = 3 s`, `UPDATE_NOTICE_AFTER = 30 min`,
  page updates at most every `250 ms`.
- The jobs socket lives beside the watch socket: `$XDG_RUNTIME_DIR/aria/jobs-<session>.sock` (long paths:
  `j-<16 hex>.sock`, same rule as `watch::socket_path`), mode `0600`.
- Every job and subagent is spawned from **one long-lived spawner thread**; each child is the leader of its own
  process group; on Linux it gets `PR_SET_PDEATHSIG(SIGTERM)`.
- Stop = `SIGTERM` to the group, `SIGKILL` after 3 s.
- Worktrees at `<git root>/.aria-worktrees/<job id>`, `.aria-worktrees/` appended once to `<git common dir>/info/exclude`.
- Changes come back with plain `git apply` (working tree only). Nothing is ever committed, staged in the person's
  folder, or removed from a kept worktree without the person.
- Extension off in subagents: `ctx.mode === 'json'`. Without `ARIA_JOBS_SOCK` the tools refuse with
  "Background work needs Aria's dock."
- Rules for when to use the background go in the tool **descriptions** (Pi ignores `promptGuidelines` with Aria's
  custom `SYSTEM.md`).
- UI house rule: solid backgrounds that never change on hover/focus; only the border takes the accent. Reuse parts,
  never copy them.
- Commit after each task (`feat(...)`/`fix(...)`/`docs(...)`), ending with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push.

## Review Focus

1. A process that prints a lot quickly (e.g. `yes`): the dock's memory stays bounded and the page is not flooded —
   Task 1 test `output_is_bounded_by_lines_and_chars`.
2. A Pi that restarts while a job ends: the end message still reaches the new Pi exactly once — Task 2 test
   `events_stay_until_acked_and_survive_reconnect` and Task 7 test `acks_only_after_send_message`.
3. A dev server whose children ignore SIGTERM (or a `bash -c` that forks): Stop leaves nothing behind — Task 1 test
   `stop_kills_the_whole_group`.
4. A tab closed while its subagent is queued: it never starts later — Task 4 test `closing_a_session_drops_its_queue`.
5. Two writing subagents asked for the same non-git folder: the second waits and starts when the first ends — Task 4
   test `second_writer_in_a_plain_folder_waits`.

---

### Task 1: Jobs core — processes

**Files:**
- Create: `crates/aria/src/jobs.rs`
- Modify: `crates/aria/src/main.rs` (`mod jobs;`), `crates/aria/Cargo.toml` (`libc = "0.2"`),
  `crates/aria-core/src/paths.rs` (`pub fn jobs(&self) -> PathBuf` = `root/jobs.json`)
- Test: unit tests in `crates/aria/src/jobs.rs` (`#[cfg(test)] mod tests`, take `crate::SERIAL`)

**Interfaces:**
- Produces:
  - `pub enum Kind { Process, Agent }`, `pub enum JobState { Queued, Running, Exited, Failed, Stopped, Done, Cut }`
    (serde `lowercase`).
  - `pub struct Job` (serde `camelCase`): `id: String` (`"j<number>"`), `number: u64`, `session: String`, `kind`,
    `title: String`, `command: Option<String>` (what the model asked, never the boxed line), `task: Option<String>`,
    `folder: String` (shown with `~`), `state`, `started_at: i64` (ms), `ended_at: Option<i64>`,
    `url: Option<String>`, `exit: Option<i32>`, `isolation: Option<String>` (`"folder"|"worktree"`),
    `worktree: Option<String>`, `writes: bool`, `result: Option<String>`, `conflict: bool`, `dismissed: bool`.
  - `pub struct ProcessStart { pub session: String, pub title: String, pub command: String, pub line: String, pub folder: PathBuf, pub env: Vec<(String, String)> }`
    (`line` is what `bash -c` runs).
  - `pub struct Jobs` (cheap `Clone`, `Arc` inside) with:
    `pub fn new(store: PathBuf, on_change: Arc<dyn Fn() + Send + Sync>) -> Jobs`,
    `pub fn start_process(&self, start: ProcessStart) -> Result<Job, String>`,
    `pub fn output(&self, id: &str, lines: usize) -> Option<Vec<String>>`,
    `pub fn stop(&self, id: &str) -> bool`, `pub fn get(&self, id: &str) -> Option<Job>`,
    `pub fn list(&self, session: Option<&str>) -> Vec<Job>` (running and queued first, then finished newest first),
    `pub fn active(&self) -> usize` (running + queued), `pub fn stop_session(&self, session: &str)`,
    `pub fn stop_all(&self)` (blocks until every group is gone, at most `STOP_GRACE` + 1 s),
    `pub fn dismiss(&self, id: &str)`.
  - `fn detect_url(line: &str) -> Option<String>`, `fn strip_ansi(line: &str) -> String` (private, tested).

- [ ] **Step 1: Write the failing tests**

```rust
#[test] fn a_process_runs_and_its_output_is_kept() // start `echo hi; echo there; sleep 5`; within 2 s output(id, 10) == ["hi", "there"]; get(id).state == Running
#[test] fn exit_code_and_state_are_recorded() // `exit 3` -> state Exited, exit Some(3) within 2 s; `true` -> Exited, exit Some(0)
#[test] fn stop_kills_the_whole_group() // `sh -c 'trap "" TERM; sleep 60' & sleep 60` ; stop(id) == true; within 5 s state Stopped and `kill(pgid, 0)` fails (ESRCH)
#[test] fn the_first_local_url_is_the_jobs_url() // detect_url("  ▲ ready on http://0.0.0.0:3000/x") == Some("http://localhost:3000/x"); "https://127.0.0.1:8443" kept; "http://example.com" -> None; "[::1]:5173" with scheme kept
#[test] fn ansi_escapes_are_removed() // strip_ansi("\u{1b}[32mready\u{1b}[0m") == "ready"
#[test] fn output_is_bounded_by_lines_and_chars() // `yes | head -n 5000; sleep 5` -> output(id, 5000).len() == 1000; a 5000-char line is kept clipped to 2000 chars
#[test] fn an_eleventh_process_is_refused() // ten `sleep 30` running; the 11th Err containing "10 background processes"
#[test] fn the_folder_and_env_are_the_ones_given() // `pwd; echo $X` with folder tempdir and env X=1 -> output == [tempdir, "1"]
#[test] fn changes_are_reported_at_most_four_times_a_second() // on_change counter after a burst of 100 lines in < 1 s is <= 5
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p aria jobs::`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement `Jobs` processes in `crates/aria/src/jobs.rs`**

A spawner thread owns every `Command::spawn` (requests sent over an `mpsc` channel, the reply over a one-shot
channel); each child: `bash -c <line>`, `stdin(null)`, stdout and stderr into one pipe (`os_pipe` is not a dependency:
use `libc::pipe` or two readers that both append under the lock), `process_group(0)`, and on Linux
`pre_exec(|| { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM); Ok(()) })`. A reader thread per job appends
lines (ANSI stripped, clipped) to a `VecDeque` capped at `OUTPUT_LINES` and sets the URL once; a waiter thread records
the exit. A throttle thread calls `on_change` at most every 250 ms when dirty (the `watch.rs` pattern).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p aria jobs::`
Expected: PASS, and `cargo clippy -p aria --all-targets` clean.

- [ ] **Step 5: Commit** — `feat(jobs): the dock runs background processes, keeps their output and finds their URL`

### Task 2: Jobs core — events and persistence

**Files:**
- Modify: `crates/aria/src/jobs.rs`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `pub enum EndedBy { Itself, Stopped }` (serde: `"self"`, `"stop"`).
  - `pub struct JobEvent` (serde `camelCase`, tag `event`): `Ended { seq: u64, job: Job, by: EndedBy, last_lines: Vec<String> }`
    (last 20 lines, processes only), `Cut { seq: u64, jobs: Vec<Job> }`.
  - `pub fn events(&self, session: &str) -> Vec<JobEvent>` (undelivered, in `seq` order),
    `pub fn ack(&self, session: &str, seq: u64)`,
    `pub fn subscribe(&self, session: &str, tx: Sender<JobEvent>)` (new events pushed while subscribed; one
    subscriber per session, a new one replaces the old).
  - `Jobs::new` loads `store`: every job still `Running` or `Queued` there becomes `Cut` (with `ended_at` = load time)
    and its session gets one `Cut` event; finished jobs are kept (at most `FINISHED_KEPT` per dock).
  - The store is rewritten (`aria_core::json_file::write_json_atomic`, mode `0600`) on every state change, never on
    output lines.

- [ ] **Step 1: Write the failing tests**

```rust
#[test] fn a_process_that_ends_by_itself_leaves_an_event() // `echo bye; exit 1` -> events("s1") == [Ended { by: Itself, job.exit: Some(1), last_lines: ["bye"] }]
#[test] fn a_stopped_process_says_it_was_stopped() // stop(id) -> Ended { by: Stopped }
#[test] fn events_stay_until_acked_and_survive_reconnect() // two ends; events() has both; ack(seq1); events() has only the second; subscribe() then a third end arrives on the channel and is in events() until acked
#[test] fn running_jobs_in_the_store_come_back_cut() // write a store with one Running and one Exited job; Jobs::new -> list() has both, the first Cut; events() == [Cut { jobs: [first] }]
#[test] fn the_store_has_no_output() // after `echo secret-ish`, the store file does not contain "secret-ish"
#[test] fn stop_session_stops_only_that_session() // s1 and s2 each one sleep; stop_session("s1") -> s1 job Stopped, s2 Running
```

- [ ] **Step 2: Run them to verify they fail** — `cargo test -p aria jobs::` → the new tests FAIL.
- [ ] **Step 3: Implement** the event queue (per session `Vec<JobEvent>`, global `seq` counter persisted in the
  store as `nextSeq`), the subscriber map, load-and-cut in `new`, and the store rewrite in the state transitions.
- [ ] **Step 4: Run the tests to verify they pass** — `cargo test -p aria jobs::` → PASS.
- [ ] **Step 5: Commit** — `feat(jobs): job ends become events kept until acknowledged, and jobs cut by a stop are remembered`

### Task 3: Worktrees

**Files:**
- Create: `crates/aria/src/worktree.rs` (+ `mod worktree;` in `main.rs`)

**Interfaces:**
- Produces:
  - `pub fn git_root(folder: &Path) -> Option<PathBuf>` (`git -C folder rev-parse --show-toplevel`).
  - `pub struct Worktree { pub path: PathBuf, pub base: String }` (`base` = the commit it started from).
  - `pub fn create(root: &Path, id: &str) -> io::Result<Worktree>`: `git stash create` (empty output → `HEAD`),
    `git worktree add --detach <root>/.aria-worktrees/<id> <base>`, and `.aria-worktrees/` appended to
    `<git rev-parse --git-common-dir>/info/exclude` unless a line already equals it.
  - `pub enum Back { Applied, Empty, Conflict(String) }` and
    `pub fn bring_back(root: &Path, wt: &Worktree) -> io::Result<Back>`: in the worktree `git add -A` then
    `git diff --binary --cached <base>`; empty → remove worktree, `Empty`; else `git apply` in `root` (stdin = patch);
    success → `git worktree remove --force`, `Applied`; failure → keep, `Conflict(stderr)`.
  - `pub fn remove(root: &Path, wt_path: &Path) -> io::Result<()>`.

- [ ] **Step 1: Write the failing tests** (temp repos via `git init`, `git -c user.email=a@b -c user.name=a commit`)

```rust
#[test] fn the_worktree_starts_from_the_persons_uncommitted_changes() // repo with a.txt committed "1", working copy "2"; create -> worktree's a.txt == "2"; person's folder still "2" and `git status --porcelain` unchanged
#[test] fn a_clean_repo_starts_from_head()
#[test] fn the_exclude_line_is_written_once() // create twice -> info/exclude has exactly one ".aria-worktrees/" line; `git status --porcelain` in root does not list .aria-worktrees
#[test] fn changes_come_back_unstaged() // edit a.txt and add new.txt in the worktree; bring_back == Applied; root has both changes, `git diff --cached` empty, worktree dir gone
#[test] fn a_clash_keeps_the_worktree() // worktree edits line 1; person edits line 1 differently after create; bring_back == Conflict(_); worktree dir exists; person's file untouched
#[test] fn nothing_changed_is_empty() // bring_back == Empty and the worktree is removed
#[test] fn a_plain_folder_has_no_git_root()
```

- [ ] **Step 2: Run them to verify they fail** — `cargo test -p aria worktree::` → FAIL.
- [ ] **Step 3: Implement** with `std::process::Command::new("git")`, `-C <dir>`, checking `status.success()`.
- [ ] **Step 4: Run the tests to verify they pass** — `cargo test -p aria worktree::` → PASS.
- [ ] **Step 5: Commit** — `feat(jobs): worktrees for subagents, made from the person's uncommitted changes and brought back unstaged`

### Task 4: Jobs core — subagents

**Files:**
- Modify: `crates/aria/src/jobs.rs`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `pub struct AgentStart { pub session: String, pub title: String, pub task: String, pub folder: PathBuf, pub writes: bool, pub isolation: String /* auto|folder|worktree */, pub program: PathBuf /* the pi shim */, pub args: Vec<String> /* --model/--thinking pairs */, pub env: Vec<(String, String)>, pub unset: Vec<String>, pub before_spawn: Option<Arc<dyn Fn(&Job) + Send + Sync>> }`.
  - `pub fn start_agent(&self, start: AgentStart) -> Result<Job, String>`: returns `Running` or `Queued`.
  - The child is `<program> --mode json -p --no-session <args…> "Task: <brief>"` where `<brief>` is the task plus,
    in order: `"\n\nWork in <folder>."`, `"\nDo not commit."` when it writes, `"\nDo not change any file."` when
    `writes` is false. `ARIA_JOBS_SOCK` is in `unset`.
  - Result: text of the last `message_end` whose `message.role == "assistant"` (text blocks joined), clipped to
    `RESULT_CHARS`; `stopReason == "error"` or a non-zero exit → `Failed`, else `Done`.
  - Isolation: `writes && git_root(folder)` is `Some(root)`: `worktree` asked → worktree; `auto` → worktree when
    another writing agent of any session is `Running` in the same root, else folder; `folder` → folder (waits when
    the root is taken). Not git: folder, keyed by the canonical folder. A folder-mode writer holds the key until it
    ends; another folder-mode writer for that key is `Queued`. `writes == false` takes no key.
  - On end of a worktree agent that is `Done`: `bring_back`; `Conflict` sets `job.conflict = true` and keeps
    `job.worktree`. Stopped/failed/cut worktree agents keep `job.worktree`, nothing applied.
  - Queue: FIFO; a queued agent starts when the running count drops below `MAX_AGENTS_RUNNING` and its key is free.
    `stop` on a queued job makes it `Stopped` without starting it; `stop_session` drops that session's queue.

- [ ] **Step 1: Write the failing tests** (a fake `pi`: a shell script in a tempdir that prints
  `{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"all green"}]}}`, optionally
  `sleep $FAKE_SLEEP`, optionally writes a file, then exits `$FAKE_EXIT`)

```rust
#[test] fn an_agent_ends_with_its_last_answer() // Done, result Some("all green"), Ended { by: Itself } event
#[test] fn an_error_answer_or_exit_fails() // stopReason "error" -> Failed; exit 2 -> Failed
#[test] fn the_brief_says_where_and_how() // fake pi echoes its last arg into a file; contains "Task: fix it", "Work in", "Do not commit."; writes=false -> "Do not change any file."
#[test] fn at_most_three_run_and_the_rest_queue() // 4 sleeping read-only agents -> 3 Running, 1 Queued; the first ends -> the fourth Running
#[test] fn second_writer_in_a_plain_folder_waits() // two writers, same non-git tempdir -> Running, Queued; first ends -> second Running
#[test] fn auto_uses_a_worktree_when_the_repo_is_taken() // git repo; writer A folder mode Running; writer B auto -> isolation "worktree", its cwd under .aria-worktrees
#[test] fn a_done_worktree_agent_brings_its_changes_back() // fake pi writes b.txt in its cwd; after Done the person's folder has b.txt unstaged and job.worktree is None
#[test] fn a_stopped_worktree_agent_keeps_its_worktree() // stop -> Stopped, job.worktree Some(path) and path exists
#[test] fn closing_a_session_drops_its_queue() // s1 fills the 3 slots plus one queued; stop_session("s1") -> the queued one Stopped, never spawned (fake pi touches a marker file per run: 3 markers)
#[test] fn before_spawn_runs_with_the_job() // the hook gets the job with its title and task before the child starts
#[test] fn the_child_does_not_get_the_jobs_socket() // env ARIA_JOBS_SOCK set in the dock's env for the test; fake pi writes `${ARIA_JOBS_SOCK:-none}`; file == "none"
```

- [ ] **Step 2: Run them to verify they fail** — `cargo test -p aria jobs::` → the new tests FAIL.
- [ ] **Step 3: Implement** the agent spawn through the same spawner thread, the stdout JSON reader, the key map, the
  FIFO, and the end handling above.
- [ ] **Step 4: Run the tests to verify they pass** — `cargo test -p aria jobs:: worktree::` → PASS.
- [ ] **Step 5: Commit** — `feat(jobs): background subagents: three at a time, one writer per folder, a worktree when the repository is taken`

### Task 5: The jobs socket

**Files:**
- Create: `crates/aria/src/jobs_socket.rs` (+ `mod jobs_socket;`)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - `pub fn socket_path(dir: &Path, session: &str) -> PathBuf` (`jobs-<session>.sock`, else `j-<16 hex>.sock` past 100 bytes).
  - `pub trait Launcher: Send + Sync { fn agent(&self, model: &str /* "session"|"heavy" */) -> Option<(PathBuf, Vec<String>, Vec<(String, String)>, Vec<String>)>; fn before_spawn(&self, job: &Job); }`
    (program, extra args, env, unset; `None` = cannot start one now).
  - `pub struct JobsSocket` with `pub fn bind(path: &Path, session: String, jobs: Jobs, launcher: Arc<dyn Launcher>) -> io::Result<JobsSocket>`,
    `pub fn path(&self) -> &Path`; `Drop` removes the file and closes clients.
  - Protocol exactly as the spec's tables. Requests: `{"id":n,"op":"hello"|"start"|"list"|"output"|"stop"|"ack", …}`;
    answers `{"re":n,"ok":true,…}` / `{"re":n,"ok":false,"error":"…"}`; pushed `{"event":"ended"|"cut",…}` (the
    `JobEvent` JSON). `start` process fields: `command`, `line`, `title`, `folder`, `env` (object). `start` agent
    fields: `task`, `title`, `folder`, `writes`, `isolation`, `model`. `hello` answers and then writes every pending
    event of the session and subscribes the connection. Lines capped at 1 MiB (the `watch.rs` reader).

- [ ] **Step 1: Write the failing tests** (connect with `UnixStream`, write JSON lines, read answers)

```rust
#[test] fn hello_then_pending_events() // an ended job before connecting; hello -> {"re":1,"ok":true} then {"event":"ended",…,"seq":…}
#[test] fn start_list_output_stop_a_process() // start (command "echo hi; sleep 30") -> ok + job.id; output -> lines ["hi"]; list -> 1 job; stop -> ok; then an ended event with "by":"stop" is pushed
#[test] fn an_agent_start_uses_the_launcher() // a fake launcher returning a fake pi; start agent -> ok, job.kind "agent"; launcher.before_spawn called once
#[test] fn without_a_launcher_answer_an_agent_is_refused() // launcher.agent returns None -> ok:false, error mentions "cannot start"
#[test] fn a_session_sees_only_its_own_jobs() // list on s1 socket does not return s2's job
#[test] fn the_socket_is_private() // mode 0600
#[test] fn a_bad_line_is_ignored() // "not json\n" then a valid list -> answered
```

- [ ] **Step 2: Run them to verify they fail** — `cargo test -p aria jobs_socket::` → FAIL.
- [ ] **Step 3: Implement** with a non-blocking accept loop and one blocking thread per connection (as
  `watch.rs`), a writer half guarded by a `Mutex` shared by answers and pushed events.
- [ ] **Step 4: Run the tests to verify they pass** — `cargo test -p aria jobs_socket::` → PASS.
- [ ] **Step 5: Commit** — `feat(jobs): a socket per session for Pi to start, list, read and stop background work`

### Task 6: Wiring the dock

**Files:**
- Modify: `crates/aria-core/src/bridge.rs`, `crates/aria/src/handlers.rs` (`is_chat`), `crates/aria/src/chat.rs`,
  `crates/aria/src/main.rs`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces:
  - `DockMessage::Jobs { jobs: Value }` (`"type":"jobs"`), `DockMessage::JobOutput { id: u64, lines: Vec<String> }`
    (`"jobOutput"`); `PageMessage::JobStop { job: String }` (`"jobStop"`), `PageMessage::JobOutput { id: u64, job: String, lines: Option<usize> }`
    (`"jobOutput"`, default 80), `PageMessage::JobDismiss { job: String }` (`"jobDismiss"`). All are chat messages
    (`is_chat`), handled by `Sessions`.
  - `ChatCtx` gains `pub jobs: Jobs`. `Sessions::new` builds nothing new; `main.rs` makes `Jobs::new(paths.jobs(), …)`
    whose `on_change` sends `DockMessage::Jobs { jobs: json!(jobs.list(None)) }`; the page also gets it after
    `ready` (`Sessions::handle(PageMessage::Ready)` is not routed today: send it from the `chatLoad` of any session).
  - `Chat` opens its `JobsSocket` lazily like `watch()` and adds `ARIA_JOBS_SOCK` to Pi's env; its `Launcher`:
    program = `paths.bin().join("pi")`; env = `pi::session_env(…, workdir)` + `PI_WATCH_SOCK`; unset =
    `pi::unset_for_pi` + `ARIA_JOBS_SOCK`; `model "session"` → `--model <inner.model>` and
    `--thinking <inner.thinking>` when known; `"heavy"` → `settings.models.planning` and `planning_effort`;
    `before_spawn` → `watch.expect_tasks(&json!({"task": job.task, "agent": job.title}))`.
  - `Chat::busy` unchanged; `Sessions::busy` = any chat busy **or** `jobs.active() > 0`.
    `Sessions::busy_without_jobs()` for the update notice. `SessionClose` also calls `jobs.stop_session`.
  - `main.rs`: when `update_ready` and blocked only by jobs for `UPDATE_NOTICE_AFTER`, once: each session with an
    active job gets `ui_notice` (warning) "An Aria update is waiting for the background work to finish; stop it from
    the background list to update now." Before `control::reexec()`: `jobs.stop_all()`.

- [ ] **Step 1: Write the failing tests**

```rust
// bridge.rs
#[test] fn job_messages_round_trip() // page(r#"{"type":"jobStop","job":"j3"}"#) == PageMessage::JobStop{job:"j3"}; jobOutput default lines None; value(DockMessage::Jobs{jobs:json!([])}) == {"type":"jobs","jobs":[]}
// chat.rs
#[test] fn pi_gets_the_jobs_socket() // the fake Pi's env has ARIA_JOBS_SOCK ending in "jobs-s1.sock"
#[test] fn closing_a_tab_stops_its_jobs() // start a process job for s1 through ctx.jobs; SessionClose s1 -> job Stopped
#[test] fn background_jobs_keep_the_dock_busy() // idle sessions + one running job -> busy(); busy_without_jobs() false
#[test] fn a_session_launcher_uses_the_sessions_model() // inner.model "opencode-go/x", thinking "high" -> args ["--model","opencode-go/x","--thinking","high"]; heavy -> settings planning model
```

- [ ] **Step 2: Run them to verify they fail** — `cargo test -p aria-core bridge && cargo test -p aria chat::` → FAIL.
- [ ] **Step 3: Implement** the messages, the `ChatCtx` field (tests' `ctx()` helpers build a `Jobs` on a temp
  store), the socket and launcher in `Chat`, the `Sessions` handlers, and the `main.rs` notice and `stop_all`.
- [ ] **Step 4: Run all Rust suites** — `for c in aria-engine aria-core aria aria-hyprland; do cargo test -q -p $c; done`
  and `cargo clippy --workspace --all-targets` → PASS, no warnings.
- [ ] **Step 5: Commit** — `feat(jobs): sessions reach their background work, the page sees every job, and updates wait for it`

### Task 7: The background extension

**Files:**
- Create: `extensions/background/index.ts`, `extensions/background/client.ts`, `extensions/background/text.ts`,
  `extensions/background/client.test.ts`, `extensions/background/text.test.ts`, `extensions/background/index.test.ts`

**Interfaces:**
- Consumes: the socket protocol (Task 5).
- Produces:
  - `client.ts`: `export class JobsClient { constructor(path: string); request(op: string, fields?: object, timeoutMs?: number): Promise<any>; onEvent(fn: (e: any) => void): void; close(): void }`
    (reconnects every 1 s while closed; sends `hello` on every connect) and
    `export const jobsClient = (): JobsClient | null` (one per process, `null` without `ARIA_JOBS_SOCK`).
  - `export async function startProcess(o: { command: string; line: string; title: string; folder: string }): Promise<{ content: any[]; isError?: boolean; details?: any }>`:
    sends `start` (env = `process.env`), then polls `output` every 300 ms for up to 3 s (stops early once a URL
    shows), and returns text: `Started in the background as #<n> (<title>). Output so far:\n<lines>\nURL: <url>`
    plus `Use background_output, background_stop.`; without the dock: the refusal text, `isError: true`.
  - `text.ts`: `export function nowLine(jobs: Job[]): string | null` (the `[background] running: …` line of the spec,
    `null` when nothing is running or queued); `export function endedMessage(e: any): { text: string; triggerTurn: boolean }`
    (`by: "self"` or any agent end → `triggerTurn: true`; `by: "stop"` and `cut` → false; agent done text ends with
    `Tell the person what it did, and save its Learnings to memory.`; a conflict names the kept worktree and says to
    ask the person how to merge; a stopped/failed/cut worktree agent names its worktree).
  - `index.ts`: tools are registered at load (the mode is not known yet); in a subagent (`ctx.mode === 'json'`)
    every tool's `execute` refuses with "Background work is not available inside a subagent.", the client is never
    opened, and nothing is added to the context. Tools: `background_agent`, `background_list`, `background_output`, `background_stop`,
    and `background_run` **only when `process.env.PI_SANDBOX === '0'`** (plain: `line = command`). Their
    descriptions carry the spec's "Deciding" rules (when to use the background, `writes`, `isolation: worktree` for
    a large job, not editing a folder a writing subagent holds, `model: heavy` for heavy work). `folder` is resolved
    (`~`, relative to `ctx.cwd`) and checked with `subagentProblem({ cwd: folder }, ctx.cwd)` from
    `../sandbox/policy.ts`; a problem refuses. `before_agent_start` for user messages adds
    `{ customType: 'aria-background-now', content: nowLine(jobs), display: false }` when not null (jobs from `list`,
    150 ms timeout, skipped on timeout). Events: `pi.sendMessage({ customType: 'aria-background', content: text, display: true, details: { job, by, event } }, triggerTurn ? { triggerTurn: true, deliverAs: 'followUp' } : undefined)`, then `ack`.

- [ ] **Step 1: Write the failing tests**

```ts
// client.test.ts — a fake dock: node:net server on a tmp socket
test('hello on connect and answers matched by id')
test('pushed events reach onEvent')
test('reconnects and says hello again after the dock comes back')
test('a request without a dock rejects within its timeout')
// text.test.ts
test('nothing running: no line') // nowLine([]) === null; only finished -> null
test('the line names number, kind, title, folder, url, minutes and a subagent's status and isolation')
test('a process that died by itself starts a turn and shows its last lines')
test('a stopped job does not start a turn')
test('a done agent asks to tell the person and save the learnings')
test('a conflict names the worktree and says to ask how to merge')
// index.test.ts — a fake `pi` (registerTool/on/sendMessage recorders) and the fake dock
test('background_agent sends the resolved folder and refuses one outside the projects folder')
test('without ARIA_JOBS_SOCK every tool refuses with "Background work needs Aria\'s dock."')
test('in a subagent (mode json) the tools refuse and no context line is added')
test('acks_only_after_send_message') // the ack line reaches the fake dock after sendMessage was recorded
test('background_run exists only with PI_SANDBOX=0')
```

- [ ] **Step 2: Run them to verify they fail** — `bun test extensions/background` → FAIL.
- [ ] **Step 3: Implement** the three modules.
- [ ] **Step 4: Run** — `bun test extensions` → PASS (all extensions).
- [ ] **Step 5: Commit** — `feat(extensions): background tools for Aria: subagents, the list, output and stop, and what Aria is told`

### Task 8: `background_run` in the sandbox

**Files:**
- Modify: `extensions/sandbox/index.ts`
- Test: `extensions/sandbox/background.test.ts` (create)

**Interfaces:**
- Consumes: `startProcess` (Task 7), `bashRoute`, `wrapCommand`, the existing `askOutside`.
- Produces: tool `background_run({ command, title, folder?, outsideSandbox? })`, registered by the sandbox only when
  `ctx.mode` is not `json` at execute time (same refusal as Task 7 in json mode). Route exactly as `bash.execute`:
  `plain` → `line = command`; `boxed` → `line = wrapCommand(command, folder)`; `ask` → `askOutside(ctx, command, title)`
  then plain; refusals → the same texts `bash` returns. Then `startProcess({ command, line, title, folder })`.
  `folder` resolved and checked as in Task 7. Description: never-ending commands only; `title` ≤ 6 words in the
  person's language.

- [ ] **Step 1: Write the failing tests** (export a small `backgroundLine(route, command, folder)` from `sandbox/route.ts`
  to test the mapping without Pi)

```ts
test('boxed: the line is the bwrap line for the folder') // backgroundLine('boxed','bun dev','/p') === wrapCommand('bun dev','/p')
test('plain and allowed-after-asking: the command as is')
test('refusals stay refusals') // backgroundLine('refuse-nobody', …) === null
```

- [ ] **Step 2: Run** — `bun test extensions/sandbox` → FAIL.
- [ ] **Step 3: Implement** `backgroundLine` in `route.ts` and the tool in `index.ts`.
- [ ] **Step 4: Run** — `bun test extensions` → PASS; `sh extensions/sandbox/accept.sh` if it runs offline (skip if it needs a model).
- [ ] **Step 5: Commit** — `feat(sandbox): background_run takes the same sandbox route as bash`

### Task 9: The strip

**Files:**
- Create: `ui/chat/parts/pill.js`, `ui/chat/parts/pill.css`, `ui/chat/parts/pill.test.js`,
  `ui/chat/parts/background.js`, `ui/chat/parts/background.css`, `ui/chat/parts/background.test.js`
- Modify: `ui/chat/parts/memory.js`, `ui/chat/parts/memory.css` (use the pill part), `ui/chat/view.js`,
  `ui/chat/state.js`, `ui/chat/parts/composer.js`, `ui/chat/texts.js`, `ui/app.js`, `ui/chat.css`, the stylesheet list
  that loads part CSS (`ui/index.html` or `ui/chat/parts/index.css`, wherever `memory.css` is linked)

**Interfaces:**
- Produces:
  - `pill.js`: `export const renderPill = ({ label, meta, dot, tone, pressed, action, id, extra }, ui, t) => string`
    — the markup and look of today's `.p-memory-chip` (class `p-pill`, dot `p-pill-dot`), `tone` adds `tone-<x>`;
    memory's chips call it (memory tests unchanged).
  - `background.js`: `export function renderStrip({ jobs, now }, ui, t)` — active session's jobs that are running,
    queued, or failed and not dismissed; at most 4 pills + `+N` (`data-chat="bg-list"`); each pill
    `data-chat="bg-open" data-id`, stop `data-chat="bg-stop" data-id`, failed dismiss `data-chat="bg-dismiss"`;
    `''` when empty. `export function jobMeta(job, now, t)` (process: `:<port> · <n> min`, agent: status word via
    `statusLabel`, queued: `queued`).
  - `state.js`: custom messages `role === 'custom' && customType === 'aria-background' && display` → note
    `{ kind: 'note', label: 'background', key: 'chat.bg.note.<outcome>', vars: { title } }` (outcome from
    `details`: `exited`, `failed`, `stopped`, `done`, `conflict`, `cut`), live (`message_start`) and in `fromHistory`.
  - `view.js`: `chat.setJobs(jobs)` (all sessions), `#chat-background` placed right above `#chat-tasks`, redrawn when
    jobs change or the active tab changes; clicks: `bg-stop` → `{type:'jobStop', job}` (not session-wrapped),
    `bg-dismiss` → `jobDismiss`, `bg-open`/`bg-list` → open the agents view with `ui.job = id` (Task 10 draws it).
  - `composer.js`: prop `background: boolean`; hint `alt+a background` (`chat.composer.hint.background`) instead of
    `alt+a subagents` when true.
  - `app.js`: handler `jobs: (m) => booted.then(() => chat.setJobs(m.jobs))`.
  - Strings (en/es): `chat.bg.title` Background/En segundo plano, `chat.bg.more` +{n}, `chat.bg.stop` Stop/Detener,
    `chat.bg.dismiss`, `chat.bg.queued` queued/en cola, `chat.bg.minutes` {n} min, `chat.bg.note.exited`
    `{title} ended (exit {exit})`, `chat.bg.note.failed`, `chat.bg.note.stopped`, `chat.bg.note.done`,
    `chat.bg.note.conflict`, `chat.bg.note.cut`, `chat.composer.hint.background`.

- [ ] **Step 1: Write the failing tests**

```js
// pill.test.js
test('a pill has the memory chip look: dot, label, meta, pressed') 
// background.test.js
test('only the active session, only running, queued and failed-not-dismissed')
test('five jobs: four pills and +1')
test('a process shows its port and minutes; an agent its status word; queued says queued')
test('stop, open and dismiss carry the job id')
test('nothing to show renders nothing')
// state.test.js
test('an aria-background custom message becomes a background note, live and from history')
test('a hidden custom message (display false) adds nothing')
// composer.test.js
test('the hint says alt+a background while the session has background jobs')
```

- [ ] **Step 2: Run** — `bun test ui` → the new tests FAIL.
- [ ] **Step 3: Implement** the parts, the extraction (memory's look unchanged), state, view and app wiring.
- [ ] **Step 4: Run** — `bun test ui` → PASS.
- [ ] **Step 5: Commit** — `feat(chat): background work shows as pills above the composer, and its ends as notes in the thread`

### Task 10: The background list (Alt+A)

**Files:**
- Modify: `ui/chat/parts/subagents.js`, `ui/chat/parts/subagents.css`, `ui/chat/parts/subagents.test.js`,
  `ui/chat/parts/background.js` (+ test), `ui/chat/view.js`, `ui/chat/parts/session.js` (only if a `result` step is needed)

**Interfaces:**
- Consumes: Task 9 (`jobMeta`, `setJobs`, `ui.job`), Task 6 messages.
- Produces:
  - `background.js`: `export function renderJobRows({ jobs, sessions, active, open, output, now }, ui, t)` — sections
    "Running" (running + queued) grouped by session (headed by the tab title only when more than one session has
    jobs; active first) and "Finished" (newest first). A row reuses `p-agents-row` markup (dot, kind as the mono
    role, title as the task, meta as the status) followed by actions: process → `bg-url` (when `url`),
    `bg-output` (toggles; the expanded row shows `output` lines in a mono block), `bg-stop`; agent → `bg-session`,
    `bg-stop`; finished agent → `bg-session` labelled Result; finished rows show how they ended (`chat.bg.end.*`) and
    a kept worktree path.
  - `subagents.js`: `renderSubagents` gains `props.background` (HTML from `renderJobRows`) drawn above the agents
    list; the title is `chat.bg.title` when there are jobs; `parallel` agents linked to a job (same `task` as an
    agent job of that session) are left out of its list.
  - `view.js`: `bg-url` → `openUrl`; `bg-output` → toggles `ui.output[id]` and requests `{type:'jobOutput', id, job, lines: 80}`
    every 1 s while expanded (stopped when collapsed or the view closes); `bg-session` → the session view on the
    linked watch agent (found by task), or a result view built with `renderAgentSession({ agent: { role: 'subagent', task: title, status, kind: 'done', steps: [{kind:'msg', text: result}] } })`
    when none is linked; Alt+A opens with the background section whenever there are jobs even with no subagents
    (`chat.agents.empty` only when both are empty); opening from a pill scrolls to and outlines that row.
  - Strings: `chat.bg.running`, `chat.bg.finished`, `chat.bg.kind.process`, `chat.bg.kind.agent`, `chat.bg.open`
    Open/Abrir, `chat.bg.output` Output/Ver salida, `chat.bg.session` Session/Ver sesión, `chat.bg.result`
    Result/Ver resultado, `chat.bg.worktree` `worktree kept at {path}`, `chat.bg.end.exited` `exited {exit}`,
    `chat.bg.end.stopped`, `chat.bg.end.done`, `chat.bg.end.failed`, `chat.bg.end.cut` `cut when Aria stopped`.

- [ ] **Step 1: Write the failing tests**

```js
test('rows are grouped by session only when several have jobs, the active one first')
test('a process row has Open only with a URL, Output and Stop; an agent row Session and Stop')
test('finished rows say how they ended and name a kept worktree')
test('an expanded process row shows its output lines escaped')
test('the subagents list leaves out the agent a background job runs, and keeps parallel ones')
test('with jobs and no subagents the view shows the jobs, not the empty text')
```

- [ ] **Step 2: Run** — `bun test ui` → FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** — `bun test ui` → PASS.
- [ ] **Step 5: Commit** — `feat(chat): Alt+A lists the background work of every session, with its output, its URL and a way to stop it`

### Task 11: Sandbox check, docs, the whole suite, install

**Files:**
- Modify: `tools/ui-sandbox/serve.ts`, `tools/ui-sandbox/check.ts`, `docs/architecture.md`, `extensions/README.md`,
  `docs/pending.md`, `CHANGELOG.md`

- [ ] **Step 1: Sandbox** — `serve.ts`: a "Background work" button that sends a `jobs` message with a process
  (`bun dev`, url `http://localhost:3000`), a running agent and a queued one for the active session, answers
  `jobStop` by moving that job to Finished, and `jobOutput` with three lines. `check.ts`: checks
  `the strip shows the session's background work`, `stop sends jobStop with the job id`,
  `a pill opens the background list on that job`, `Output shows the job's lines`.
- [ ] **Step 2: Run** — `bun tools/ui-sandbox/serve.ts 8770 &` then `bun tools/ui-sandbox/check.ts` → `all checks passed`.
- [ ] **Step 3: Docs** — architecture: the module trio, the socket, the four messages, lifetimes; extensions README:
  the `background` row and the sandbox's `background_run`; pending: steering a background subagent, macOS leftover
  processes after a crash (no pdeathsig there), the limits as settings, untracked files are not in a worktree;
  CHANGELOG: a dated section with `~/.aria/jobs.json`, `<repo>/.aria-worktrees/` and the `.git/info/exclude` line,
  each with how to revert.
- [ ] **Step 4: Whole suite** — `sh tools/parity/suites.sh` → `all suites passed`; `cargo fmt --all --check`;
  `cargo clippy --workspace --all-targets` with no warnings.
- [ ] **Step 5: Commit** — `docs: background work in the architecture, the extensions and what is left`
- [ ] **Step 6: Install on the dock** — `cargo build --release -p aria && sh install/install.sh --local target/release/aria`;
  `systemctl --user is-active aria` → `active`; the journal shows `aria: ready`.
