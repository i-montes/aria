# Aria phase 2a — the engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (the owner chose native execution) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aria runs its own pinned Pi as a child process (`pi --mode rpc`) and the dock becomes a chat: you send a message and see the streamed reply, the tool chips, the working folder and the extension dialogs, even when the dock was hidden while Pi worked.

**Architecture:** A new crate `aria-engine` speaks Pi's RPC protocol (JSONL framing, commands, responses, events, extension-UI dialogs) and supervises one child. `aria-core` gains the pinned-Pi location, the agent directory (`~/.aria/agent/`) and the `pi` shim. The `aria` binary gets a `chat` module that owns the engine, translates page messages into commands and Pi's events into dock messages, and raises a persistent desktop notification when a dialog needs an answer while the dock is hidden. The page gets a pure chat reducer (tested) and a chat view built from the design components.

**Tech Stack:** Rust (std threads and `mpsc`, `serde_json`), GTK/WebKitGTK as in phase 1, plain ES modules tested with `bun test`, shell for the installer.

**Spec:** `docs/superpowers/specs/2026-10-18-aria-phase2-parity-design.md` (sections "Engine (2a)", "The dock as chat (2a)", "Notifications for a hidden dock (2a)", "Calls made after the builder check"). Pi's protocol: `~/.pi/agent/install/releases/1.3.0/node_modules/@earendil-works/pi-coding-agent/docs/{rpc,rpc-commands,rpc-extension-ui,json}.md`.

## Global Constraints

- Everything built is in English; the chat with the owner is Spanish.
- Pi is the npm package `@earendil-works/pi-coding-agent`, Node ≥ 22.19.0; the pinned version lives in a repo file `pi-version` (first release: `1.3.0`) and is installed under `~/.aria/pi/<version>/`.
- The agent directory is `~/.aria/agent/`, passed to Pi as `PI_CODING_AGENT_DIR`; Pi's own `settings.json` lives there, never next to Aria's.
- Pi's working folder is `sandbox.projectsDir` (default `~/Projects`, `~` expanded); the dock shows that path at all times.
- RPC framing is strict JSONL: split only on LF, strip an optional CR, never use a Unicode-aware line reader.
- A secret never reaches a log, `settings.json` or a message to the page (`auth.json` is only linked, never read by the engine).
- Buttons and cards keep solid backgrounds that never change opacity on hover/focus/selection; only the border takes the accent `#B9A8FF`.
- Notifications for a hidden dock use `omarchy-notification-send` (critical urgency) and `omarchy-notification-dismiss`.
- Out of this part: the seven extensions and their structured cards (2b), settings→Pi sync beyond provider/model (2c), the parity matrix (2d). Dialogs the extensions send are rendered as plain dialogs here.
- Every out-of-repo change (files under `~/.aria`, the installed binary, the service) is logged in `CHANGELOG.md`.

## Review Focus

- Pi's stdout carrying a record split across two reads, a CRLF line, or `U+2028` inside a string: the framing must neither lose nor split a record (Task 1).
- Pi dying mid-run: the dock must say so, not freeze, and the next message must start a fresh Pi (Task 2, Task 6).
- A dialog still open when the dock hides, closes or Pi exits: the notification must appear once, and must be removed when the dialog is answered or dies (Task 6).
- Pi or Node missing on the machine: Aria must start, show a plain problem, and never crash (Task 3, Task 8).
- A page message arriving while no run exists (`abort`, a dialog answer for an unknown id) must be ignored, not crash the dock (Task 6).

---

### Task 1: aria-engine — framing and protocol types

**Files:**
- Create: `crates/aria-engine/Cargo.toml`, `crates/aria-engine/src/lib.rs`, `crates/aria-engine/src/framing.rs`, `crates/aria-engine/src/protocol.rs`
- Modify: `Cargo.toml` (workspace members)
- Test: unit tests inside both modules

**Interfaces:**
- Produces:
  - `framing::LineSplitter::new() -> LineSplitter`; `LineSplitter::push(&mut self, bytes: &[u8]) -> Vec<String>` (complete records, LF-split, CR stripped, partial tail kept, invalid UTF-8 replaced lossily).
  - `protocol::Incoming` enum: `Response { id: Option<String>, command: String, success: bool, data: Value, error: Option<String> }`, `UiRequest { id: String, method: String, raw: Value }`, `Event { kind: String, raw: Value }`, `Unparsable(String)`.
  - `protocol::parse_line(line: &str) -> Incoming` (`type: "response"` → Response; `type: "extension_ui_request"` → UiRequest; any other object with a string `type` → Event; anything else → Unparsable).
  - `protocol::command(kind: &str, id: &str, fields: Value) -> String` (one JSON line without the LF, `type` and `id` merged with `fields`).
  - `protocol::UiAnswer` enum `Value(String) | Confirmed(bool) | Cancelled` and `protocol::ui_response(id: &str, answer: &UiAnswer) -> String`.

- [ ] **Step 1: Write the failing tests** in `framing.rs`: `splits_on_lf_only` (`b"{\"a\":1}\n{\"b\":2}\n"` → two records), `keeps_a_partial_record_until_it_completes` (`push(b"{\"a\"")` → `[]`, then `push(b":1}\n")` → `["{\"a\":1}"]`), `strips_a_carriage_return` (`"x\r\n"` → `"x"`), `does_not_split_on_unicode_separators` (`"{\"t\":\"a\u{2028}b\"}\n"` → one record), `a_multibyte_char_split_across_reads_survives` (the two bytes of `é` pushed separately). In `protocol.rs`: `parses_a_response`, `parses_an_event`, `parses_a_ui_request_with_its_method`, `garbage_is_unparsable`, `command_merges_type_id_and_fields` (`command("prompt","1",json!({"message":"hi"}))` parses back to `{"type":"prompt","id":"1","message":"hi"}`), `ui_response_shapes` (`Value("Allow")` → `{"type":"extension_ui_response","id":"u","value":"Allow"}`, `Confirmed(true)` → `confirmed:true`, `Cancelled` → `cancelled:true`).
- [ ] **Step 2: Run** `cargo test -p aria-engine` — Expected: FAIL (crate/functions missing).
- [ ] **Step 3: Implement** the crate (deps `serde_json = "1"`, `serde = { version = "1", features = ["derive"] }`, `thiserror = "1"`) and both modules; `LineSplitter` buffers raw bytes and only decodes a complete record.
- [ ] **Step 4: Run** `cargo test -p aria-engine` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(engine): Pi RPC framing and protocol types`.

---

### Task 2: aria-engine — the child process

**Files:**
- Create: `crates/aria-engine/src/engine.rs`, `crates/aria-engine/tests/fake_pi.sh` (a POSIX shell script playing Pi), `crates/aria-engine/tests/engine.rs`
- Modify: `crates/aria-engine/src/lib.rs`
- Test: `crates/aria-engine/tests/engine.rs`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `engine::SpawnSpec { program: PathBuf, args: Vec<String>, env: Vec<(String,String)>, cwd: PathBuf }`.
  - `engine::EngineEvent` enum: `Event(Value)`, `UiRequest { id: String, method: String, raw: Value }`, `Exited { code: Option<i32> }`.
  - `engine::Engine::spawn(spec: SpawnSpec) -> Result<(Engine, mpsc::Receiver<EngineEvent>), EngineError>`.
  - `Engine::send(&self, kind: &str, fields: Value) -> Result<String, EngineError>` (fire a command, returns its id).
  - `Engine::request(&self, kind: &str, fields: Value, timeout: Duration) -> Result<Response, EngineError>` where `Response { success: bool, data: Value, error: Option<String> }`; waits for the matching `response`.
  - `Engine::answer_ui(&self, id: &str, answer: UiAnswer) -> Result<(), EngineError>`.
  - `Engine::is_alive(&self) -> bool`; `Engine::shutdown(self)` (closes stdin, waits up to 3 s, then kills).
  - `EngineError` (`thiserror`): `Spawn(io::Error)`, `Gone`, `Timeout`.

- [ ] **Step 1: Write `fake_pi.sh`**: reads stdin lines; for `get_state` replies `{"type":"response","id":<id>,"command":"get_state","success":true,"data":{"isStreaming":false}}`; for `prompt` replies `started` then prints `agent_start`, a `message_update` text delta `"hi"`, and `agent_settled`; for `ask` (test-only command) prints an `extension_ui_request` select with id `u1` and, on receiving the `extension_ui_response`, prints an event `{"type":"answered","value":<value>}`; for `crash` exits with code 3; ids are extracted with `sed`.
- [ ] **Step 2: Write the failing tests** in `tests/engine.rs` (each spawns `fake_pi.sh` via `/bin/sh`): `request_gets_its_matching_response`, `prompt_streams_events_in_order` (receiver yields `agent_start`, the delta, `agent_settled`), `ui_request_arrives_and_answer_reaches_pi` (answer `Value("Allow")`, then the `answered` event carries `"Allow"`), `exit_is_reported` (`send("crash")` → `Exited{code:Some(3)}` and `is_alive()` false), `request_after_exit_is_gone`, `request_times_out_when_pi_stays_silent` (a `silent` command; timeout 200 ms → `Timeout`), `a_missing_program_is_a_spawn_error`.
- [ ] **Step 3: Run** `cargo test -p aria-engine --test engine` — Expected: FAIL (module missing).
- [ ] **Step 4: Implement** `Engine`: spawn with piped stdin/stdout (stderr to `Stdio::null()`, a thread drains it), a reader thread runs `LineSplitter` over stdout and routes `Response`s to waiting `request`s (a `Mutex<HashMap<String, mpsc::Sender<Response>>>`) and everything else to the event channel; when stdout closes the thread waits for the child and sends `Exited`; ids are a process-wide counter as strings.
- [ ] **Step 5: Run** `cargo test -p aria-engine` — Expected: PASS (all).
- [ ] **Step 6: Commit** `feat(engine): supervise a Pi child over RPC`.

---

### Task 3: Pinned Pi, the agent directory and the shim (aria-core)

**Files:**
- Create: `pi-version` (content `1.3.0\n`), `crates/aria-core/src/pi.rs`, `crates/aria-core/src/agent.rs`
- Modify: `crates/aria-core/src/paths.rs`, `crates/aria-core/src/lib.rs`
- Test: unit tests in `pi.rs`, `agent.rs`, `paths.rs`

**Interfaces:**
- Consumes: `AriaPaths`, `settings::Settings` (fields `models.daily`, `sandbox.projects_dir`), `json_file`.
- Produces:
  - `AriaPaths::agent() -> PathBuf` (`<root>/agent`), `bin() -> PathBuf` (`<root>/bin`), `pi_root() -> PathBuf` (`<root>/pi`), `pi_install(version: &str) -> PathBuf` (`<root>/pi/<version>`).
  - `pi::PINNED_VERSION: &str` (`include_str!("../../../pi-version").trim()`).
  - `pi::PiInstall { pub dir: PathBuf, pub binary: PathBuf }`; `pi::locate(paths: &AriaPaths, version: &str) -> Result<PiInstall, PiError>` (binary is `<dir>/node_modules/.bin/pi`, must exist and be executable); `PiError::{NotInstalled, NodeMissing}` with `PiError::problem_code(&self) -> &'static str` returning `pi-missing` / `node-missing`.
  - `pi::node_ok(path_var: &str) -> bool` (finds `node` on the given `PATH` string and checks `node --version` ≥ 22.19); a `pi::parse_node_version(&str) -> Option<(u32,u32,u32)>` helper.
  - `pi::write_shim(paths: &AriaPaths, install: &PiInstall) -> io::Result<PathBuf>` writes `<root>/bin/pi` (mode 0755): `#!/bin/sh\nexec '<binary>' "$@"\n`.
  - `agent::prepare(paths: &AriaPaths, settings: Option<&Settings>) -> io::Result<()>`: creates `agent/` (0700); symlinks `auth.json`, `mcp.json`, `mcp-auth.json` to the files in `paths.root` when those exist (replacing a stale link, never overwriting a real file); writes Pi's `settings.json` (`defaultProvider`, `defaultModel` split from `models.daily` at the first `/`; `quietStartup: true`) preserving any other keys already in it; writes a minimal `SYSTEM.md` only when absent.
  - `agent::workdir(settings: Option<&Settings>) -> PathBuf` (`sandbox.projectsDir` with `~` expanded to `$HOME`, default `~/Projects`).

- [ ] **Step 1: Write the failing tests**: `paths` (`agent_and_pi_paths`); `pi::locate` (`locates_an_installed_binary` with a fake tree in a tempdir, `missing_binary_is_not_installed`, `problem_codes`); `parse_node_version` (`v22.23.2` → `(22,23,2)`, `22.18.0` rejected by `node_ok` logic through a `meets_minimum` helper: `(22,19,0)` ok, `(22,18,9)` no, `(24,0,0)` ok); `write_shim_is_executable_and_execs_the_binary`; `agent::prepare`: `links_credentials_that_exist`, `does_not_link_missing_files`, `replaces_a_stale_link_but_keeps_a_real_file`, `sets_default_provider_and_model_from_the_daily_model` (`opencode-go/deepseek-v4.1-flash` → provider `opencode-go`, model `deepseek-v4.1-flash`), `keeps_unknown_keys_in_pi_settings`, `leaves_an_existing_system_md_alone`, `agent_dir_is_private`; `workdir_expands_tilde_and_defaults`.
- [ ] **Step 2: Run** `cargo test -p aria-core pi:: agent:: paths::` — Expected: FAIL.
- [ ] **Step 3: Implement** both modules and the path helpers (use `json_file` for the settings read/write so atomicity and 0600 handling match phase 1).
- [ ] **Step 4: Run** `cargo test -p aria-core` — Expected: PASS (all).
- [ ] **Step 5: Commit** `feat(core): pinned Pi location, agent directory and pi shim`.

---

### Task 4: The installer puts Node and Pi in place

**Files:**
- Modify: `install/install.sh`, `install/install.test.sh`
- Test: `install/install.test.sh`

**Interfaces:**
- Consumes: `pi-version` (Task 3).
- Produces: installer behaviour — `install.sh` reads the pinned version (from the repo with `--local`, from the downloaded release otherwise), checks `node` ≥ 22.19 and prints `Node 22.19 or newer is required (install it with your package manager or mise)` and exits 1 when missing; installs `npm install --prefix ~/.aria/pi/<version> --no-audit --no-fund @earendil-works/pi-coding-agent@<version>` when `~/.aria/pi/<version>/node_modules/.bin/pi` is absent; `--dry-run` prints those steps without running them; `--uninstall` keeps `~/.aria` (as today).

- [ ] **Step 1: Write the failing checks** in `install.test.sh`: `--dry-run --local` output contains `npm install --prefix` and the version read from `pi-version`; with a fake `PATH` holding a `node` that prints `v20.0.0` the installer exits non-zero with the Node message; with `node` printing `v22.23.2` and the pinned Pi already present (fake tree in a temp `ARIA_HOME`) no `npm install` line appears.
- [ ] **Step 2: Run** `sh install/install.test.sh` — Expected: FAIL on the three new checks.
- [ ] **Step 3: Implement** the Node check and the Pi install step in `install.sh`, respecting `--dry-run` and the existing `safe_to_delete` guard.
- [ ] **Step 4: Run** `sh install/install.test.sh` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(install): ensure Node and install the pinned Pi`.

---

### Task 5: Chat messages in the bridge

**Files:**
- Modify: `crates/aria-core/src/bridge.rs`
- Test: `crates/aria-core/src/bridge.rs` tests

**Interfaces:**
- Produces, added to `PageMessage`: `ChatSend { text: String, behavior: Option<String> }` (`behavior`: `steer` | `followUp` when a run is active), `ChatAbort`, `ChatNew`, `ChatLoad` (page asks for the state on open), `UiAnswer { id: String, value: Option<String>, confirmed: Option<bool>, cancelled: Option<bool> }`, `ListCommands { id: u64 }`, `ListModels { id: u64 }`, `SetModel { id: u64, model: String }`, `ListFiles { id: u64, query: String }`.
- Produces, added to `DockMessage`: `ChatState { workdir: String, running: bool, model: Option<String>, thinking: Option<String>, problem: Option<String> }`, `ChatEvent { event: Value }` (a Pi session event verbatim), `ChatDialog { id: String, method: String, request: Value, hidden: bool }`, `ChatDialogClosed { id: String }`, `ChatEngine { state: String }` (`starting` | `ready` | `exited`), `Commands { id: u64, commands: Value }`, `Models { id: u64, models: Value }`, `Files { id: u64, files: Vec<String> }`.

- [ ] **Step 1: Write the failing tests**: every new variant parses/serializes with the exact JSON tags above (`chatSend`, `chatAbort`, `chatNew`, `chatLoad`, `uiAnswer`, `listCommands`, `listModels`, `setModel`, `listFiles`; `chatState`, `chatEvent`, `chatDialog`, `chatDialogClosed`, `chatEngine`, `commands`, `models`, `files`), asserted like the existing tests.
- [ ] **Step 2: Run** `cargo test -p aria-core bridge` — Expected: FAIL.
- [ ] **Step 3: Implement** the variants (camelCase field names, `Option`s as `null`).
- [ ] **Step 4: Run** `cargo test -p aria-core` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(core): chat messages in the bridge`.

---

### Task 6: The chat coordinator

**Files:**
- Create: `crates/aria/src/chat.rs`, `crates/aria-core/src/files.rs`
- Modify: `crates/aria/src/main.rs`, `crates/aria/src/handlers.rs`, `crates/aria/Cargo.toml`, `crates/aria-core/src/lib.rs`
- Test: unit tests in `chat.rs` and `files.rs`

**Interfaces:**
- Consumes: `aria_engine::{Engine, SpawnSpec, EngineEvent, UiAnswer}` (Task 2), `pi::locate`, `agent::{prepare, workdir}`, `pi::write_shim` (Task 3), the Task 5 messages.
- Produces:
  - `files::list_files(root: &Path, query: &str, limit: usize) -> Vec<String>`: relative paths, `.gitignore`-aware (reads `.gitignore` patterns of `root` only: plain names, `dir/`, `*.ext`, leading `/`; always skips `.git`), skips symlink loops, case-insensitive substring match on the path, shallowest first.
  - `chat::Notifier` trait (`fn send(&self, id: &str, title: &str, body: &str)`, `fn dismiss(&self, id: &str)`) with `OmarchyNotifier` (runs `omarchy-notification-send`/`-dismiss`, ignoring failures) and a test double.
  - `chat::Chat::new(ctx: ChatCtx, emit: Arc<dyn Fn(DockMessage) + Send + Sync>) -> Chat` where `ChatCtx { paths: AriaPaths, notifier: Arc<dyn Notifier>, dock_visible: Arc<AtomicBool>, spawner: Arc<dyn Fn(&SpawnSpec) -> Result<(Engine, Receiver<EngineEvent>), EngineError> + Send + Sync> }`.
  - `Chat::handle(&self, message: PageMessage)` for the chat messages: starts Pi on first need (`ensure_engine`: `agent::prepare`, `write_shim`, spawn `pi --mode rpc` with `PI_CODING_AGENT_DIR=<agent>`, `PATH=<bin>:$PATH`, cwd = workdir; emits `ChatEngine{starting}` then `ChatEngine{ready}`), `ChatSend` → `prompt` (with `streamingBehavior` when `behavior` is set), `ChatAbort` → `clear_queue` then `abort`, `ChatNew` → `new_session`, `ChatLoad` → emits `ChatState` (workdir, running, model/thinking from `get_state`, `problem` from `pi::locate`/`node_ok` failures), `ListCommands`/`ListModels` → `get_commands`/`get_available_models`, `SetModel` → `set_model` (`provider`/`modelId` split at the first `/`), `ListFiles` → `files::list_files(workdir, …)`, `UiAnswer` → `answer_ui` and `ChatDialogClosed`.
  - Behaviour: every Pi event is forwarded as `ChatEvent`; a `UiRequest` with method `select|confirm|input|editor` is forwarded as `ChatDialog { hidden: !dock_visible }` and, when hidden, `notifier.send(id, "Aria needs an answer", <title>)`; `notify` requests become a `ChatEvent` of type `ui_notice`; the notification is dismissed once on answer, on `Exited`, and when the dialog's own timeout elapses (the dock closes it when it receives `ChatDialogClosed`); `Exited` emits `ChatEngine{exited}` and a `ChatEvent{type:"engine_exited"}`, clears the engine so the next `ChatSend` spawns again; a message with no engine and no way to start one emits `ChatState.problem`.

- [ ] **Step 1: Write the failing tests** for `files` (`lists_matching_files_shallowest_first`, `respects_gitignore_patterns`, `skips_dot_git_and_caps_the_limit`, `handles_a_symlink_loop`) and for `chat` using a fake spawner pointing at a script (like Task 2's `fake_pi.sh`, copied to `crates/aria/tests/fake_pi.sh`) and the test notifier: `first_send_starts_pi_and_forwards_events`, `second_send_reuses_the_engine`, `send_while_running_passes_streaming_behavior`, `dialog_while_hidden_notifies_once_and_answer_dismisses`, `dialog_while_visible_does_not_notify`, `exit_clears_the_engine_dismisses_and_next_send_respawns`, `ui_answer_for_an_unknown_id_is_ignored`, `abort_with_no_engine_is_ignored`, `missing_pi_reports_the_problem_code`, `list_files_uses_the_workdir`.
- [ ] **Step 2: Run** `cargo test -p aria chat:: && cargo test -p aria-core files::` — Expected: FAIL.
- [ ] **Step 3: Implement** `files.rs` and `chat.rs`; engine events are consumed on a thread per engine; `Chat` is `Send + Sync` behind a `Mutex<Option<Running>>`.
- [ ] **Step 4: Wire** `main.rs`/`handlers.rs`: chat messages are routed to `Chat::handle` (off the GTK thread), `dock_visible` follows `visibility::State`, the dock exposes `is_visible()`; clicking the notification must open the dock (the notifier's action runs `aria toggle` when the dock is hidden).
- [ ] **Step 5: Run** `cargo test --workspace` — Expected: PASS (all).
- [ ] **Step 6: Commit** `feat: the chat coordinator (engine, dialogs, notifications, file picker)`.

---

### Task 7: The chat reducer (pure JS)

**Files:**
- Create: `ui/chat/state.js`, `ui/chat/state.test.js`
- Test: `bun test ui/chat/state.test.js`

**Interfaces:**
- Produces:
  - `createChat() -> ChatState` with `{ workdir: '', running: false, engine: 'idle', model: null, messages: [], dialogs: [], problem: null }`.
  - `reduce(state, message) -> ChatState` (pure; `message` is a dock message of Task 5). Message shapes in `messages`: `{ kind: 'user', text }`, `{ kind: 'assistant', text, thinking, done }`, `{ kind: 'tools', calls: [{ id, name, args, status: 'running'|'done'|'error', summary }] }` (consecutive tool calls grouped, in execution order), `{ kind: 'notice', tone, text }`.
  - `addUser(state, text) -> ChatState`, `answerDialog(state, id) -> ChatState`.
  - `toolSummary(name, args) -> string` (`bash` → the command's first line, `read`/`write`/`edit` → the path, anything else → the tool name).

- [ ] **Step 1: Write the failing tests**: `agent_start_sets_running_and_agent_settled_clears_it`; `text_deltas_build_one_assistant_message` (two `message_update` deltas of block 0 → one message with the joined text; `message_end` replaces it with the authoritative content); `thinking_deltas_go_to_thinking`; `tool_calls_group_in_order` (`tool_execution_start` ×2 then `tool_execution_end` → one `tools` entry with both calls and their statuses; `isError` → `error`); `an_assistant_message_after_tools_starts_a_new_entry`; `a_dialog_request_is_queued_and_closed_on_chatDialogClosed`; `engine_exited_adds_a_notice_and_stops_running`; `auto_retry_start_adds_a_notice`; `chatState_sets_workdir_model_and_problem`; `tool_summary_cases`; `unknown_events_are_ignored`.
- [ ] **Step 2: Run** `bun test ui/chat/state.test.js` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `state.js` (no DOM, no I/O; return new objects).
- [ ] **Step 4: Run** `bun test` — Expected: PASS (all suites).
- [ ] **Step 5: Commit** `feat(ui): chat state reducer`.

---

### Task 8: The chat view

**Files:**
- Create: `ui/chat/view.js`, `ui/chat/view.test.js`, `ui/chat.css`
- Modify: `ui/index.html`, `ui/app.js`, `ui/i18n/en.json`, `ui/i18n/es.json`, `ui/dock.css`
- Test: `bun test ui/chat/view.test.js`

**Interfaces:**
- Consumes: Task 7 (`createChat`, `reduce`, `addUser`, `answerDialog`), Task 5 messages through `window.aria.send`/`request`.
- Produces: `createChatView({ root, t, send, request }) -> { receive(message), show(), hide(), render() }`: header with the working path (always visible, `title` attribute with the full path), the message list (user bubble, assistant bubble with `white-space: pre-wrap`, tool-chip group that expands on click, notice rows, the working indicator while `running`), a composer (Enter sends, Shift+Enter newline; while running Enter sends as `steer` and a Stop button sends `chatAbort`), the pickers (`/` opens the command list from `listCommands`, `@` the file list from `listFiles`, a model button listing `listModels`), the generic dialog card (`select` options as buttons, `confirm` Yes/No, `input`/`editor` a text field and OK/Cancel, always a Cancel), the problem banner for `pi-missing`/`node-missing`, and a "New chat" button (`chatNew`). Exported pure helpers `renderMessages(state, t) -> string` and `renderDialog(dialog, t) -> string` (HTML strings, every dynamic value escaped with `esc`), tested without a DOM.
- app.js: after a finished setup the dock shows the chat; the onboarding stays reachable from a settings button in the chat header (it reuses `onboarding.onOpen`/`start`). The page posts `chatLoad` when the dock opens.

- [ ] **Step 1: Write the failing tests** with the same DOM-less approach as `ui/onboarding/view.test.js`: `renders_user_and_assistant_messages_escaped` (a `<script>` in the text is escaped), `renders_the_working_path_in_the_header`, `tool_group_shows_a_count_and_expands_its_rows`, `select_dialog_renders_one_button_per_option_plus_cancel`, `confirm_dialog_renders_yes_no`, `the_working_indicator_shows_only_while_running`, `a_problem_shows_the_install_hint`, `empty_chat_shows_a_start_hint`.
- [ ] **Step 2: Run** `bun test ui/chat/view.test.js` — Expected: FAIL.
- [ ] **Step 3: Implement** view, CSS (design tokens from `ui/dock.css`: solid backgrounds, accent border only, 14.5px body, Geist/Geist Mono fallbacks) and the i18n keys in both languages (`chat.placeholder`, `chat.send`, `chat.stop`, `chat.new`, `chat.working`, `chat.path`, `chat.cancel`, `chat.yes`, `chat.no`, `chat.ok`, `chat.problem.pi-missing`, `chat.problem.node-missing`, `chat.engine.exited`, `chat.hint.empty`, `chat.settings`).
- [ ] **Step 4: Wire** `app.js`/`index.html`: route `chatState|chatEvent|chatDialog|chatDialogClosed|chatEngine` to the chat view; keep Escape closing the dock; arrow-key navigation keeps working inside the onboarding only.
- [ ] **Step 5: Run** `bun test` — Expected: PASS (all suites, including the i18n key-parity test).
- [ ] **Step 6: Commit** `feat(ui): the dock as a chat`.

---

### Task 9: Live check against the real Pi, docs and changelog

**Files:**
- Modify: `CHANGELOG.md`, `README.md`, `docs/superpowers/specs/2026-10-18-aria-phase2-parity-design.md` (status line only)
- Test: manual run + full suites

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Install and run the pinned Pi** with `sh install/install.sh --local` (installs `~/.aria/pi/1.3.0`, rebuilds and installs the binary, restarts `aria.service`); Expected: `~/.aria/pi/1.3.0/node_modules/.bin/pi --version` prints `1.3.0`.
- [ ] **Step 2: Drive the real engine** with a small integration test marked `#[ignore]` in `crates/aria/tests/live_pi.rs` (spawns the pinned Pi in `--mode rpc` with `PI_CODING_AGENT_DIR` pointing at a temp agent dir prepared by `agent::prepare` from the owner's real `~/.aria/auth.json`, sends `get_state`, then a one-word prompt, waits for `agent_settled`); Expected when run with `cargo test -p aria --test live_pi -- --ignored`: PASS and a non-empty assistant text.
- [ ] **Step 3: Check the dock by eye** in headless Chrome with the stubbed bridge (screenshot of an empty chat, a streamed reply with tool chips, a select dialog, the problem banner) and fix visual problems.
- [ ] **Step 4: Check the persistent notification** on the owner's shell: `omarchy-notification-send` with critical urgency and no expiry stays on screen; record whether the explicit expiry of `0` is needed in the notifier and in `CHANGELOG.md`.
- [ ] **Step 5: Run** `cargo test --workspace && bun test && sh install/install.test.sh` — Expected: PASS (all).
- [ ] **Step 6: Update** `CHANGELOG.md` (new files under `~/.aria/{pi,agent,bin}`, installed binary, shim) and `README.md` (Node requirement, what the dock does now); **Commit** `docs: phase 2a changelog and readme`.

---

## Addendum: sessions as tabs (owner's decision after Task 8)

The owner pointed out that the design's `+` opens a **new session as a tab**, with several sessions at once. This replaces the plan's single "New chat" button (Task 5/6/8). Decisions (mine, from the design component `session-tabs.dc.html`): one Pi process per session (`--session-id <id>`), the page owns the tab list and statuses, Rust owns the engines and persists the list in `~/.aria/sessions.json`, a notification is raised per session but dismissed only when no session has one left.

### Task 10: Sessions in Rust

**Files:**
- Create: `crates/aria-core/src/sessions.rs`
- Modify: `crates/aria-core/src/{bridge.rs,paths.rs,lib.rs}`, `crates/aria/src/{chat.rs,main.rs,handlers.rs}`
- Test: unit tests in those files

**Interfaces:**
- Produces:
  - `AriaPaths::sessions() -> PathBuf` (`<root>/sessions.json`).
  - `sessions::SessionList { sessions: Vec<SessionEntry { id: String, title: String }>, active: Option<String> }`; `sessions::load(&AriaPaths) -> SessionList` (a missing or broken file is an empty list); `sessions::save(&AriaPaths, &SessionList) -> io::Result<()>` (atomic, mode 0600, at most 8 entries, ids kept only if `[A-Za-z0-9-]{1,64}`).
  - Bridge: `PageMessage::Session { session: String, message: Box<PageMessage> }` (carries every chat message to one session), `SessionsLoad { id }`, `SessionsSave { sessions, active }`, `SessionClose { session }`; `DockMessage::Session { session: String, message: Box<DockMessage> }`, `Sessions { id, sessions, active }`. `ChatNew` is removed (a new session is a new tab).
  - `chat::Sessions::new(ctx: ChatCtx, emit: Emit) -> Sessions` with `handle(&self, PageMessage)`: routes a `Session` message to that session's `Chat` (created on first use; its engine runs `--session-id <session>`), handles `SessionsLoad/Save/Close` (close drops that session's engine: stdin closed, dialogs closed, its notification dismissed), and `dock_visibility_changed()` for all sessions. Every `Chat` emits through a wrapper that tags `DockMessage::Session`.
  - `Notifier::send(&self, key: &str, title: &str, body: &str)` and `dismiss(&self, key: &str)`; a counting wrapper dismisses the real notification only when no key is left.
- Tests: bridge shapes; `sessions` round trip, cap of 8, bad ids dropped, broken file → empty; `two_sessions_run_their_own_pi_at_once`; `a_dialog_in_one_session_does_not_dismiss_anothers_notification`; `closing_a_session_stops_its_pi_and_closes_its_dialogs`; `a_session_message_for_an_unknown_id_creates_it`; `sessions_list_persists_across_restart`.

### Task 11: Tabs in the page

**Files:**
- Create: `ui/chat/sessions.js`, `ui/chat/sessions.test.js`
- Modify: `ui/chat/view.js`, `ui/chat/view.test.js`, `ui/chat.css`, `ui/app.js`, `ui/i18n/{en,es}.json`
- Test: `bun test ui`

**Interfaces:**
- Produces: `sessions.js` — pure tab state: `createTabs(saved?) -> { sessions: [{ id, title, status, unread }], active, closed }`, `addSession(tabs, id) -> tabs` (max 8, new one active, title `''` until the first message), `selectSession`, `closeSession` (keeps the entry for undo; the last session is never closed), `undoClose`, `renameSession`, `titleFromMessage(text) -> string` (first line, 28 chars max with an ellipsis), `applyStatus(tabs, id, chatState, isActive) -> tabs` (`working` while running, `needs` while a dialog is open, `done` + `unread` when a background run settles, otherwise `idle`), `visibleTabs(tabs, max = 4) -> { visible, overflow }` (the active one is always visible), `toSaved(tabs)`.
- The chat view keeps one `ChatState` per session, draws the tab bar (`renderTabs`, `renderOverflow`), the `+` button, per-tab close buttons, double-click rename, the "closed X · Undo" bar, and wraps every message it sends in `{type:'session', session, message}`. `app.js` unwraps `session` dock messages (a reply with an `id` resolves its pending request; the rest goes to the chat view with its session id) and loads/saves the tab list through `sessionsLoad`/`sessionsSave`.
- Tests: tab reducer cases above; `renderTabs` escapes titles and marks the active tab and statuses; overflow keeps the active tab visible.

### Task 12: Live check and docs

- Run two sessions at once against the real Pi (`--session-id` per tab), check a background tab raises "needs you", update `README.md`, `CHANGELOG.md` (`~/.aria/sessions.json`), then one more focused review of the new commits.
