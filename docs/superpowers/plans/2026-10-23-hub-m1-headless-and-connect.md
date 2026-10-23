# Hub M1 — headless service and `aria connect` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aria's service runs without a window on a Linux server, and `aria connect <host>` opens the terminal client on it over ssh, from Linux or macOS.

**Architecture:** The service already owns every session and speaks the page's messages to terminal clients on `clients.sock`. M1 adds (1) a handshake with a protocol number and a *remote* mark that narrows what a client may do, (2) `aria client-stdio`, a stdio bridge to `clients.sock` that ssh runs, (3) the terminal client over a child process instead of a socket, (4) the service extracted from the dock so `aria serve` can run it with no GTK, behind a `gui` cargo feature, (5) a headless install, `aria hub init`, schedules and browser-less sign-in from a terminal.

**Tech Stack:** Rust workspace (`aria`, `aria-core`, `aria-engine`), ratatui terminal client, `sh` installer, zig as the cross C compiler.

**Spec:** `docs/superpowers/specs/2026-10-23-hub-and-machines-design.md` (M1 and the rulings 1–19). Read `docs/architecture.md` ("The terminal") and `crates/aria/src/clients.rs` first.

## Global Constraints

- Everything in English: code, names, messages, docs, commits. Chat texts of the terminal go in `term/texts.rs` in English and Spanish.
- Commits are conventional: `feat(hub): …`, `fix(…)`, `test:`, `docs:`, `chore:`; each ends with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Never push, never run `tools/release.sh`, never edit `Cargo.toml`'s version.
- The repository is public: no real hostnames, IPs, home paths or project names anywhere (fixtures use `ana`, `/home/ana`, `hub.example`, `ana@hub.example`).
- Both platforms: `sh tools/check-macos.sh` passes at the end of every task that touches code with default features; code behind `not(feature = "gui")` is Linux-only; a macOS build without `gui` is a `compile_error!`.
- Nothing of Aria listens on a new port: the transport is `ssh <host> <command>`. The hub never edits `sshd_config`, the firewall or Tailscale.
- `clients.rs`'s whitelist stays the only gate: a remote client (hello with `"remote": true`) cannot send `secretSave` or a `!command` (`piCommand` `bash`), whatever else a local terminal may.
- Protocol: `aria_core::bridge::PROTOCOL: u32 = 1`; first line of every connection is `{"type":"hello","protocol":N,"version":"x.y.z","remote":bool}`; the answer is a `hello` without `remote`; a different `protocol` is answered with `{"type":"error","id":0,"code":"protocol-mismatch"}` and the connection closes. Error codes new in this plan: `protocol-mismatch`, `not-for-remote`, `hub-down`, `hub-too-old`.
- Data stays under `~/.aria`; files holding keys are mode 600; writes are atomic (`aria_core::json_file`).
- The ssh program is `ssh`, or the program named by `ARIA_SSH` (tests and custom setups). A host that starts with `-` is refused.

## Review Focus

Inputs and conditions the spec implies and no single task's happy path exercises, most likely first:

1. **The ssh connection drops mid-turn:** the session's run goes on in the service and the terminal reconnects to it (Task 3: `a_dropped_ssh_reconnects_and_attaches_again`).
2. **A stolen client key tries more than chatting:** a second `hello` that says `remote: false`, a `secretSave`, a `!command`, a line over the size cap, bytes that are not JSON (Tasks 1 and 2).
3. **The hub is down, or older, or newer:** `aria client-stdio` and `aria connect` say which in one line and end with a distinct exit code instead of hanging (Tasks 2 and 3).
4. **A service started by systemd with no desktop:** no `XDG_RUNTIME_DIR` in an ssh session, no `secret-tool`, no display, no `xdg-open` (Tasks 2, 5 and 10).
5. **The sandbox cannot work on this host** (Ubuntu 24.04 restricts user namespaces even with bwrap installed): `hub init` says so and sessions ask every time instead of failing silently (Task 8).

---

### Task 1: The handshake and the remote mark in the hub

**Files:**
- Modify: `crates/aria-core/src/bridge.rs` (add `PROTOCOL`)
- Modify: `crates/aria/src/clients.rs` (hello, `remote`, schedule messages allowed, `not-for-remote`)

**Interfaces:**
- Produces: `aria_core::bridge::PROTOCOL: u32 = 1`; in `clients.rs` the private `Client` gains `remote: bool` and `spoke: bool`; `Hub::take` handles a `hello` line; `allowed(&PageMessage) -> bool` now also accepts `SchedulesList`, `ScheduleUpdate`, `ScheduleDelete` (top level, not wrapped).
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write the failing tests** in `clients.rs`'s test module (use its `fixture()`, `connect()`, `Peer`):
  - `hello_is_answered_with_the_protocol`: send `{"type":"hello","protocol":PROTOCOL,"version":"9.9.9"}`; `next()` equals `{"type":"hello","protocol":PROTOCOL,"version":env!("CARGO_PKG_VERSION")}`.
  - `a_different_protocol_is_refused_and_the_client_dropped`: send protocol `PROTOCOL + 1`; the next two lines are the `hello` and `{"type":"error","id":0,"code":"protocol-mismatch"}`; `wait_until("dropped", || f.hub.count() == 0)`.
  - `a_remote_client_cannot_save_secrets_or_run_bash`: after `hello` with `"remote": true`, `secretSave` (id 5) answers `{"type":"error","id":5,"code":"not-for-remote"}` and nothing reaches `f.handled`; `piCommand` with `command: "bash"` (id 6) answers the same with id 6 and nothing reaches `f.chat`; `chatAbort` for a session still reaches `f.chat`.
  - `a_second_hello_cannot_lift_the_remote_mark`: `hello remote:true`, then `hello remote:false`, then `secretSave` → `not-for-remote`.
  - `a_local_client_without_hello_keeps_working`: the existing local tests pass unchanged (no hello): `secretSave` reaches `f.handled`.
  - `clients_may_manage_schedules`: `schedulesList`, `scheduleUpdate`, `scheduleDelete` reach `f.chat` as the matching `PageMessage`s with remapped ids. Edit `clients_may_send_only_their_list`: drop `scheduleDelete` from its refused list.
- [ ] **Step 2: Run** `cargo test -p aria clients::` — Expected: the new tests FAIL.
- [ ] **Step 3: Implement** `PROTOCOL` in `bridge.rs`; in `Hub::take` handle `"hello"` before the other client-only lines: ignored unless it is the client's first line (`spoke` false), sets `remote` from `value["remote"] == true`, replies via `send_to` with the hello, and on a protocol mismatch sends the error then drops the client (`drop_client`). Mark `spoke` on any first line. After the `PiCommand`→`ChatBash` conversion, refuse `ChatBash` and `SecretSave` from a remote client with `error { id: own.unwrap_or(0), code: "not-for-remote" }`. Add the three schedule messages to `allowed`.
- [ ] **Step 4: Run** `cargo test -p aria clients:: && cargo test -p aria-core bridge` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(hub): a hello with a protocol number, and what a remote client may not do`

### Task 2: `aria client-stdio`, the bridge ssh runs

**Files:**
- Create: `crates/aria/src/stdio_bridge.rs`
- Modify: `crates/aria/src/control.rs` (`runtime_dir`), `crates/aria/src/main.rs` (command `client-stdio`, usage text)

**Interfaces:**
- Consumes: Task 1's hello (the bridge sends `remote: true` itself).
- Produces: `pub const MAX_LINE: usize = 8 * 1024 * 1024;` `pub enum Exit { Done = 0, HubDown = 1, Mismatch = 3, Garbage = 4 }`; `pub fn run(input: impl Read + Send + 'static, output: impl Write + Send + 'static, socket: &Path) -> Exit`; `pub fn command() -> i32` (real stdin, stdout and `clients::socket_path()`); in `control.rs` `pub fn runtime_dir() -> PathBuf` built on the pure `fn runtime_dir_from(xdg: Option<OsString>, uid: u32, exists: impl Fn(&Path) -> bool) -> PathBuf`, used by `control_socket_path` on Linux.

- [ ] **Step 1: Write the failing tests.** In `control.rs`: `without_xdg_the_users_run_dir_is_used_when_it_exists` (`runtime_dir_from(None, 1001, |p| p == Path::new("/run/user/1001"))` is `/run/user/1001`; with `exists` false it is `/tmp`; a set `XDG_RUNTIME_DIR` wins). In `stdio_bridge.rs` (a real `Hub` on a tempdir socket via `clients::bind` + `Hub::serve`; the "client" end is a `UnixStream::pair()`):
  - `the_client_gets_the_hubs_hello_and_the_bridge_marks_it_remote`: client writes its hello, reads a `hello`; then writes `secretSave` and reads `not-for-remote`.
  - `a_clients_own_hello_is_not_forwarded`: the hub's `count()` is 1 and its first received line is the bridge's hello with `"remote": true` (assert through a fake hub socket that records lines).
  - `a_first_line_that_is_not_a_hello_ends_with_mismatch`: returns `Exit::Mismatch`, output holds `protocol-mismatch`.
  - `a_different_protocol_ends_with_mismatch`.
  - `a_hub_that_is_down_says_so`: socket path missing → output `{"type":"error","id":0,"code":"hub-down"}`, `Exit::HubDown`.
  - `a_line_over_the_cap_ends_the_bridge_and_drops_the_hub_side`: `MAX_LINE + 1` bytes with no newline → `Exit::Garbage`, hub `count()` back to 0.
  - `lines_that_are_not_json_are_dropped_not_relayed`, and `a_later_hello_from_the_client_is_dropped`.
- [ ] **Step 2: Run** `cargo test -p aria stdio_bridge:: control::` — Expected: FAIL (no module).
- [ ] **Step 3: Implement** `run`: read the client's first line (bounded by `MAX_LINE`, via a reader that errors past the cap), require `type == "hello"` and `protocol == PROTOCOL`; connect to `socket` (on failure write the `hub-down` error and return `HubDown`); write `{"type":"hello","protocol":PROTOCOL,"version":CARGO_PKG_VERSION,"remote":true}` to it; relay the hub's lines to `output` on one thread and the client's lines to the hub on this one, parsing each as JSON and dropping non-JSON and any `hello`; a closed side closes the other. `command()` maps `Exit` to its code. Wire `Some("client-stdio")` in `main.rs` and the usage string.
- [ ] **Step 4: Run** the same command — Expected: PASS. Also `cargo clippy -p aria --all-targets` clean.
- [ ] **Step 5: Commit** `feat(hub): aria client-stdio, a bridge from ssh to the clients socket`

### Task 3: The terminal over ssh — `aria connect`

**Files:**
- Create: `crates/aria/src/connect.rs`
- Modify: `crates/aria/src/term/link.rs`, `term/mod.rs`, `term/screen.rs`, `term/app.rs`, `term/texts.rs`, `main.rs`

**Interfaces:**
- Consumes: Task 2's `client-stdio` and the hello.
- Produces: `Link::over(reader: impl Read + Send + 'static, writer: impl Write + Send + 'static, remote: bool) -> io::Result<Link>` (sends the hello, reads the reply; errors are `io::ErrorKind::InvalidData` with the message `protocol-mismatch`, `hub-too-old` or `hub-down`); `Link::connect(path)` becomes `over` on the socket with `remote: false`; `Link::spawn(command: &mut std::process::Command) -> io::Result<Link>` (child's stdout/stdin, killed and waited on drop); `connect::ssh_command(host: &str) -> Result<std::process::Command, ConnectError>` running `<ssh> -T -o ServerAliveInterval=15 -o ServerAliveCountMax=3 <host> ~/.local/bin/aria client-stdio` with `<ssh>` = `ARIA_SSH` or `ssh`; `ConnectError::{OptionLikeHost, NoSsh}`; `term::run_remote(host: &str) -> i32`; `app::Config.remote: Option<String>`; `screen::run`'s last parameter becomes `reconnect: &dyn Fn() -> io::Result<Link>`.

- [ ] **Step 1: Write the failing tests.** `link.rs` (socket pairs, as its `FakeDock`): `the_hello_is_sent_first_and_a_matching_reply_opens_the_link`; `an_old_hub_that_answers_not_for_clients_is_hub_too_old`; `a_hello_with_another_protocol_is_a_mismatch`; `a_bridge_that_says_hub_down_is_hub_down`; keep the existing four tests green. `connect.rs`: `a_host_that_looks_like_an_option_is_refused` (`-oProxyCommand=x`); `the_command_is_ssh_with_keepalive_and_the_remote_binary` (assert `get_program`, `get_args`); `ARIA_SSH_names_another_program` (pure function taking the env value). `app.rs`: `over_connect_a_new_session_has_no_folder` (with `Config.remote = Some("hub.example".into())` the "new" row sends `SessionNew { workdir: None }` and its label is the `term.newRemote` text with the host); `a_dropped_ssh_reconnects_and_attaches_again` (the pure state machine: after `Input::Dock(Error{code:"gone"})` it emits `Effect::Reconnect`, and after `Input::Reconnected` it re-attaches the open session, as it does today for the socket).
- [ ] **Step 2: Run** `cargo test -p aria link:: connect:: term::app` — Expected: new tests FAIL.
- [ ] **Step 3: Implement** the pieces named above. `run_remote` builds the config like `run` (language and `verbose` from the local settings; `connected` is not read locally), connects with `Link::spawn(&mut connect::ssh_command(host)?)`, passes a `reconnect` closure that does the same, and prints one plain line for each error (`term.noSsh`, `term.hubDown`, `term.hubTooOld`, `term.protocolMismatch` with both numbers) returning 1 or 3. `run` (local) is unchanged apart from passing `&|| Link::connect(&clients)`. `main.rs`: `Some("connect")` with the host in `args[2]` (usage `aria connect [user@]host`; a missing host prints usage and exits 2).
- [ ] **Step 4: Run** `cargo test -p aria` — Expected: PASS. `sh tools/check-macos.sh` — Expected: `macOS check passed`.
- [ ] **Step 5: Commit** `feat(hub): aria connect opens the terminal on another computer's service over ssh`

### Task 4: Extract the service from the dock

**Files:**
- Create: `crates/aria/src/service.rs`
- Modify: `crates/aria/src/main.rs` (`run_dock` becomes the window's wiring around the service)

**Interfaces:**
- Consumes: nothing new.
- Produces: `service::Service` holding what `run_dock` builds today without a window (paths and vault, `Sessions`, `clients::Hub`, `Jobs`, `ScheduleStore`, settings watch, reply channel); `Service::start(options: Options) -> Service` with `Options { notifier: Arc<dyn chat::Notifier>, dock_visible: Arc<AtomicBool> }`; `Service::handle_page_message(&self, message: PageMessage)` (what the dock's `on_message` closure does: chat messages to the chat worker, blocking handlers on a thread); `Service::tick(&mut self, visible: bool) -> Tick` with `Tick { quit: bool, settings_changed: bool, to_page: Vec<DockMessage> }`, doing the quit check, the settings watch, the watchdog, the scheduler, the jobs-wait notice, the update restart and the reply routing through `hub.route` that the closure in `run_dock` does now; `Service::update_ready(&mut self)`; `Service::shutdown(&self)`.

- [ ] **Step 1: Write the characterization test first** in `service.rs`: `a_reply_for_a_terminal_does_not_reach_the_page` (a client's request id routed through the hub is absent from `Tick::to_page`; the page's own reply is present). Run it — Expected: FAIL (no module).
- [ ] **Step 2: Move the code.** Cut from `run_dock` the construction and the tick body that touch no `dock`, `hotkey`, `face_wing` or `registration_receiver`, into `Service`; `run_dock` keeps the window, the hotkey, the face wing, the control-socket thread and the platform messages, and feeds `visible` and the replies to/from `Service`. Behaviour must not change: the order of the checks in the tick is kept.
- [ ] **Step 3: Run** `cargo test --workspace` — Expected: PASS with no test edited except to follow moved names. `sh tools/check-macos.sh` — Expected: passes.
- [ ] **Step 4: Run the real dock once** in a scratch home to see it still starts: `tools/term/live-check.sh` — Expected: its checks pass.
- [ ] **Step 5: Commit** `refactor(hub): the service is its own type, the dock is its window`

### Task 5: The `gui` feature and `aria serve`

**Files:**
- Create: `crates/aria/src/serve.rs`, `crates/aria/tests/serve.rs`, `tools/hub/no-gui-libs.sh`
- Modify: `crates/aria/Cargo.toml`, `crates/aria/src/main.rs`, `chat.rs` (notifier, dictation), `ui.rs` and `dock/` (module gates)

**Interfaces:**
- Consumes: Task 4's `Service`.
- Produces: cargo feature `gui` (default) that gates `gtk`, `gdk`, `glib`, `gio`, `gtk-layer-shell`, `webkit2gtk`, `javascriptcore-rs`, `aria-hyprland`, `whisper-rs`, `cpal`, `sha2` (made `optional`); modules `dock`, `ui`, `dictation` only with `gui`; `chat::NullNotifier` (does nothing) as `PlatformNotifier` without `gui`; `Mode::Serve` for `aria serve`; `serve::run() -> i32`.

- [ ] **Step 1: Write the failing tests.** `main.rs`: `serve_is_the_service_without_a_window` (`mode(Some("serve"), true) == Mode::Serve`). `tests/serve.rs` (starts `env!("CARGO_BIN_EXE_aria") serve` with `ARIA_HOME` and `XDG_RUNTIME_DIR` in tempdirs and no `DISPLAY`/`WAYLAND_DISPLAY`): `serve_answers_a_terminal_without_a_display` (waits for `clients.sock`, connects with `Link::over`-equivalent raw lines: hello then `sessionsLoad` id 1, expects `hello` then `sessions` with id 1); `aria_quit_stops_serve` (`aria quit` prints `Aria is stopping…`, the process exits 0); `a_second_serve_says_already_running`; `serve_keeps_keys_in_plain_files_when_there_is_no_keyring` (with `PATH` holding no `secret-tool`, the log line `keys stay in plain files` appears and `auth.json` written by a `saveKey`-free import stays mode 600 — assert the log only). Gate these tests with `#[cfg(not(feature = "gui"))]` where they need the headless build, and run them in Step 4 with `--no-default-features`.
- [ ] **Step 2: Run** `cargo test -p aria --no-default-features` — Expected: FAIL to compile (feature missing).
- [ ] **Step 3: Implement.** Make the listed dependencies `optional` in every table they appear (`whisper-rs` is declared three times: keep the platform features, mark each optional) and define `gui = [...]` with `dep:` entries; `default = ["gui"]`; add `#[cfg(all(target_os = "macos", not(feature = "gui")))] compile_error!(...)`. Gate modules; without `gui`, dictation messages answer the page's existing "unavailable" shape (read `dictation_message` in `chat.rs` for the right variant) and `PlatformNotifier` is `NullNotifier`. `serve::run`: bind the control socket (`bind_single_instance`, same as the dock), `install_quit_handlers`, `Service::start` with `NullNotifier` and `dock_visible` false, then loop every 16 ms calling `tick(false)` and answering the control commands `quit` and `update-ready` (`toggle` and `show` log "no window here" and are ignored), exiting 0 after `shutdown`.
- [ ] **Step 4: Run** `cargo test -p aria --no-default-features && cargo test --workspace && cargo build -p aria --no-default-features --release && sh tools/hub/no-gui-libs.sh target/release/aria` — Expected: all PASS, and the script (which fails if `ldd` on the binary lists any of `gtk|gdk|webkit|vulkan|asound|javascriptcore`) prints `no GUI libraries linked`. `sh tools/check-macos.sh` — Expected: passes.
- [ ] **Step 5: Commit** `feat(hub): aria serve runs the service with no window, behind the gui feature`

### Task 6: A server install — unit, Node, cross-build

**Files:**
- Create: `install/aria-serve.service`, `node-version`, `tools/build-headless.sh`
- Modify: `install/install.sh` (`--headless`), `install/install.test.sh`

**Interfaces:**
- Consumes: Task 5's `aria serve`.
- Produces: `sh install/install.sh --headless [--local <binary>]` (Linux only): no GUI packages, no bar, no Hyprland, the user unit `aria-serve.service` (`ExecStart=%h/.local/bin/aria serve`, `Restart=on-failure`, `WantedBy=default.target`, a `PATH` that starts with `%h/.aria/node/bin`), `loginctl enable-linger` (printed when it needs privileges), `bubblewrap` and `iproute2` through `apt` or `pacman` (printed when sudo is refused), and a Node of the pinned `node-version` under `$DATA_DIR/node/` when the system's is older than 22.19 or missing, checked against nodejs.org's `SHASUMS256.txt`; `link_node` points `$DATA_DIR/bin/node` at it. `sh tools/build-headless.sh [triple]` (default `aarch64-unknown-linux-gnu`) prints the path of the binary it built.

- [ ] **Step 1: Write the failing tests** in `install.test.sh` (its existing style: `--dry-run` output assertions, the `--check-node` and `--link-node` hooks): `headless_dry_run_installs_no_gui_packages_and_the_serve_unit` (output has `aria-serve.service`, has `bubblewrap`, lacks `gtk3` and `webkit2gtk` and `omarchy`); `headless_refuses_macos` (with `uname` stubbed as `Darwin` it dies with a clear message); `an_old_node_is_replaced_by_a_private_one` (stubbed `node --version` printing `v18.19.1` makes the dry run say it would install Node `$(cat node-version)` into `$DATA_DIR/node`); `a_new_enough_node_is_kept`; `the_node_download_is_checked_against_shasums` (dry-run text names `SHASUMS256.txt`).
- [ ] **Step 2: Run** `sh install/install.test.sh` — Expected: the new cases FAIL.
- [ ] **Step 3: Implement** the flag and the helpers in `install.sh` following its `run`/`say`/`die` style; `node-version` holds the newest Node 22.x LTS at or above 22.19 that exists on nodejs.org (the implementer checks the index and writes that version). `build-headless.sh` follows `tools/check-macos.sh` (zig through mise as the C compiler, `rustup target add`, `cargo build --release --no-default-features --target <triple> -p aria`).
- [ ] **Step 4: Run** `sh install/install.test.sh && sh tools/build-headless.sh` then `file <the printed path>` — Expected: tests PASS; `file` says `ELF 64-bit … ARM aarch64`; `objdump -T <path> | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1` prints `GLIBC_2.39` or lower.
- [ ] **Step 5: Commit** `feat(hub): a headless install, a private Node, and a cross-build for the server`

### Task 7: `hub-import` and the bundle `hub init` sends

**Files:**
- Create: `crates/aria/src/hub_admin.rs`
- Modify: `crates/aria/src/main.rs` (command `hub-import`, usage)

**Interfaces:**
- Consumes: `aria_core::{settings, auth, mcp, paths, json_file}`.
- Produces: `pub struct Bundle { pub files: BTreeMap<String, String> }` (JSON on one line: `{"files":{name: content}}`); `pub const BUNDLE_FILES: [&str; 4] = ["settings.json", "auth.json", "mcp.json", "mcp-auth.json"]`; `pub fn export_bundle(paths: &AriaPaths, hub_home: &str, projects: Option<&str>) -> Result<Bundle, String>` (reads credentials the way Pi's shim does, writes nothing locally, sets `credentials.keyring` to false and `sandbox.projectsDir` to `projects` or `<hub_home>/Projects`); `pub fn import(paths: &AriaPaths, bundle: &str) -> Result<Vec<String>, String>`; `pub fn parse_listeners(ss_output: &str) -> Vec<Listener>` with `Listener { addr: String, port: u16 }` from `ss -ltnH`; `pub fn public_ssh(listeners: &[Listener], port: u16) -> bool`; `pub fn private_address(ip: IpAddr) -> bool` (loopback, RFC1918, `100.64.0.0/10`, `fc00::/7`).

- [ ] **Step 1: Write the failing tests** (fixtures with `/home/ana`): `export_sets_the_hubs_projects_folder_and_turns_the_keyring_off`; `export_carries_only_the_four_files`; `import_writes_the_four_files_with_mode_600`; `import_refuses_a_name_with_a_path` (`../x`, `a/b`); `import_backs_up_what_it_replaces` (`settings.json.bak`); `import_rejects_a_bundle_that_is_not_json_without_touching_files`; `parse_listeners_reads_ss_output` (a fixture with `0.0.0.0:2222`, `[::]:22`, `127.0.0.1:3000`, `100.64.0.9:41641`); `public_ssh_is_true_for_a_wildcard_and_false_for_loopback_and_private_only`; `private_address_covers_tailscale_and_rfc1918_and_not_a_public_ip`.
- [ ] **Step 2: Run** `cargo test -p aria hub_admin::` — Expected: FAIL.
- [ ] **Step 3: Implement** the functions above; `import` goes through `json_file`'s atomic write and sets the mode before the rename; `hub-import` in `main.rs` reads one line from stdin, calls `import`, prints each written name, exits 1 with the reason on an error.
- [ ] **Step 4: Run** `cargo test -p aria hub_admin::` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(hub): the bundle a hub is set up from, and the import that writes it`

### Task 8: `aria hub init`

**Files:**
- Modify: `crates/aria/src/hub_admin.rs`, `main.rs`
- Create: `crates/aria/tests/fake_ssh.sh`, `tests/hub_init.rs`

**Interfaces:**
- Consumes: Task 6's installer and binary, Task 7's bundle and listeners.
- Produces: `aria hub init <user@host> [--binary <path>] [--projects <dir>] [--client-key <pubkey file>] [--yes]`; internal `fn remote(host: &str, command: &str, stdin: Option<&[u8]>) -> io::Result<std::process::Output>` using `ARIA_SSH` like `connect`; `fn plan(probe: &Probe, args: &InitArgs) -> Vec<String>` (the numbered lines printed before anything runs); `struct Probe { arch, os, sshd_port, listeners, has_bwrap, bwrap_works, node_version }`.

- [ ] **Step 1: Write the failing tests.** `fake_ssh.sh` records every remote command it is given to a file and answers canned probe output. In `hub_init.rs` and unit tests: `the_plan_is_printed_before_anything_runs` (the recorded commands before the confirmation are probes only: `uname -m`, `/etc/os-release`, `ss -ltnH`, `ssh -G`, `command -v`, the bwrap probe); `a_no_answer_changes_nothing` (no write command recorded); `a_yes_uploads_installs_and_imports_in_order` (recorded order: upload of the tree and binary, `sh … install.sh --headless --local …`, `aria hub-import` with the bundle on stdin, `systemctl --user restart aria-serve`, `aria --version`); `a_public_sshd_is_warned_once_and_does_not_block` (the warning text names the port, exit code 0); `a_sandbox_that_cannot_work_is_said_with_the_fix_printed_not_applied` (bwrap probe fails → output names `kernel.apparmor_restrict_unprivileged_userns` and no command containing `sysctl -w` or `apparmor` is recorded); `a_client_key_is_installed_restricted_and_once` (the `authorized_keys` line starts `restrict,command="<hub home>/.local/bin/aria client-stdio" ` and a second run does not duplicate it); `a_binary_of_another_architecture_is_refused` (`file` of `--binary` says x86-64 while the probe says `aarch64`).
- [ ] **Step 2: Run** `cargo test -p aria hub_admin:: --test hub_init` — Expected: FAIL.
- [ ] **Step 3: Implement.** The upload is `tar` of `install/install.sh`, `pi-version`, `pi-packages`, `engram-version`, `agent-browser-version`, `node-version` and the binary, streamed to `~/.aria/hub-src` on the host; every write step is skipped without `--yes` or an interactive `y`. The listeners check uses `public_ssh` with the port from `ssh -G`.
- [ ] **Step 4: Run** `cargo test -p aria hub_admin:: --test hub_init` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(hub): aria hub init installs a hub over ssh and says what it could not do`

### Task 9: `/schedules` in the terminal

**Files:**
- Modify: `crates/aria/src/term/app.rs`, `term/view.rs`, `term/texts.rs`

**Interfaces:**
- Consumes: Task 1 (schedule messages allowed); `PageMessage::{SchedulesList, ScheduleUpdate, ScheduleDelete}` and `DockMessage::Schedules { id, tasks }` as in `bridge.rs`.
- Produces: the client command `/schedules`, a list in the editor's place (the `picker.rs` pattern): each row `title`, when (`next`), paused marker; Enter pauses or resumes, `d` then `y` deletes, Esc closes.

- [ ] **Step 1: Write the failing tests** in `app.rs` (pure state machine): `schedules_asks_the_dock_for_its_list`; `a_list_reply_opens_the_panel_with_a_row_per_task`; `enter_pauses_a_running_task_and_resumes_a_paused_one` (sends `ScheduleUpdate` with `enabled` flipped); `delete_needs_a_confirmation` (`d` alone sends nothing, `d` then `y` sends `ScheduleDelete`, `d` then `n` sends nothing); `an_empty_list_says_so`.
- [ ] **Step 2: Run** `cargo test -p aria term::app` — Expected: FAIL.
- [ ] **Step 3: Implement** with texts `term.schedules.*` in both languages (copy the dock's `/schedules` wording from `ui/i18n` where it exists).
- [ ] **Step 4: Run** `cargo test -p aria term::` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(term): /schedules lists, pauses and deletes the scheduled tasks`

### Task 10: Sign in without a browser on the service's computer

**Files:**
- Modify: `crates/aria-core/src/bridge.rs` (`OAuthCode`), `crates/aria-core/src/oauth/{loopback,anthropic,mcp}.rs`, `crates/aria/src/handlers.rs`, `clients.rs`, `term/app.rs`, `term/texts.rs`

**Interfaces:**
- Consumes: Task 1's remote gate.
- Produces: `PageMessage::OAuthCode { id: u64, redirect: String }` (`"oauthCode"`); `oauth::loopback::parse_pasted(text: &str, callback_path: &str, expected_state: &str) -> Result<String, OAuthError>` (a pasted full redirect address, or only its query); `wait_for_code(..., paste: Option<&Receiver<String>>)` checks the channel on every poll; the handlers' `Ctx` keeps `pending: Mutex<HashMap<u64, Sender<String>>>` keyed by the sign-in's request id; clients (local and remote) may send `StartOAuth`, `StartMcpOAuth` and `OAuthCode`, routed to the handlers like `SecretSave`; the hub keeps a client's id mapping after an `oauthUrl` and drops it after the final reply (`oauthResult` or `error`); the terminal command `/signin anthropic|notion` shows the URL and takes the pasted address.

- [ ] **Step 1: Write the failing tests.** `loopback.rs`: `a_pasted_redirect_gives_its_code`, `a_pasted_query_alone_gives_its_code`, `a_pasted_address_with_the_wrong_state_is_refused`, `a_pasted_address_for_another_path_is_refused`, `a_paste_while_waiting_ends_the_wait_before_the_timeout`. `handlers.rs`: `an_oauth_code_for_a_pending_sign_in_finishes_it` (fake provider as the existing OAuth tests use, `testing` feature) and `an_oauth_code_for_nothing_is_an_error` (`error { code: "no-sign-in" }`). `clients.rs`: `an_oauth_sign_in_keeps_the_clients_id_until_its_result` (`oauthUrl` then `oauthResult` both reach the asker with its own id). `app.rs`: `signin_shows_the_url_then_sends_the_pasted_address`.
- [ ] **Step 2: Run** `cargo test -p aria-core oauth && cargo test -p aria handlers:: clients:: term::app` — Expected: FAIL.
- [ ] **Step 3: Implement.** The opener (`xdg-open`) failing on a server must not fail the sign-in: its error is ignored and the URL is still sent.
- [ ] **Step 4: Run** the same command — Expected: PASS. `sh tools/check-macos.sh` — Expected: passes.
- [ ] **Step 5: Commit** `feat(hub): sign in from a terminal by pasting the redirect address`

### Task 11: Acceptance and docs

**Files:**
- Create: `tools/hub/accept.sh`
- Modify: `docs/architecture.md`, `docs/pending.md`, `README.md`, `docs/superpowers/specs/2026-10-23-terminal-design.md` ("Out of scope" line), `tools/parity/suites.sh`

- [ ] **Step 1: Write `tools/hub/accept.sh`**: in scratch homes, starts `aria serve` (hub), then drives `aria connect hub.example` with `ARIA_SSH` pointing at a script that runs `aria client-stdio` locally against the hub's scratch runtime dir; checks, printing `ok`/`FAIL` per line: the list shows the hub's sessions; a session opened over connect has no folder from the client; a run started over connect is still going after the client is killed and shows again from a second connect; `schedulesList` works; `aria quit` stops the hub.
- [ ] **Step 2: Run** `sh tools/hub/accept.sh` — Expected: every line `ok`.
- [ ] **Step 3: Docs.** `architecture.md`: a "The hub" section (service vs window, `aria serve`, `client-stdio`, hello and the remote set, `connect`, `hub init`); `pending.md`: what M1 leaves (M2 machines, web client, push notifications, the public ssh port, bwrap on Ubuntu 24.04, `aria update` on a hub); README: install a hub in four lines; the terminal design's out-of-scope line points to the new spec. Add the acceptance script to `tools/parity/suites.sh`.
- [ ] **Step 4: Run** `sh tools/parity/suites.sh` — Expected: every offline suite passes (the Linux-only ones, counterparts on a Mac, are named in the output as before). `sh tools/check-macos.sh` — Expected: passes.
- [ ] **Step 5: Commit** `docs: the hub, its commands and what M1 leaves`

---

## Self-review

- **Spec coverage (M1):** headless build and `aria serve` (Tasks 4, 5), `client-stdio` and `connect` (2, 3), hub init with install, Node, sandbox probe, public-ssh warning (6, 8), bundle and settings for the hub (7), sign-ins without a browser (10), schedule list/pause/delete (9), hello and protocol (1), narrower remote set (1), restricted keys (8's `--client-key`), runtime-dir fallback (2), docs (11). Not in M1: machines, shared memory, web client.
- **Type consistency:** `PROTOCOL`, `Exit`, `Link::over`, `ConnectError`, `Service::tick`/`Tick`, `Bundle`, `Listener` are defined once and used by later tasks as written.
- **Proportion:** tasks give signatures, tests and values; bodies are left to the implementer.
