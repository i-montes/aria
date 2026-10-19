# Aria phase 2b — the owner's extensions — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (the owner works native) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The owner's persona and his seven Pi extensions run inside Aria's own Pi, unchanged where possible, and feed what the chat already draws: the permission, write, quota and question cards, the Jev route line, the answer-style rule, and the subagents panel.

**Architecture:** The persona (`SYSTEM.md`, the expert files) and the extensions are copied into the repo (`agent-files/`, `extensions/`), embedded in the binary and unpacked into `~/.aria/agent/` by `agent::prepare`, next to the two packages they rely on (`pi-parallel-agents`, the superpowers skills), installed pinned. Each extension gets the smallest diff that lets it run in RPC mode: paths through one `agentDir()` helper, "is there a UI" instead of "is this the terminal", and dialogs whose `title` is the JSON card the chat already parses (`ui/chat/parts/cards.js`). Rust hosts the subagent socket (`PI_WATCH_SOCK`) per session and sends the agents to the page.

**Tech Stack:** TypeScript extensions loaded by Pi 1.3.0 (tested with `bun test`), Rust (`aria-core`, `aria`), the existing chat UI.

**Spec:** `docs/superpowers/specs/2026-10-18-aria-phase2-parity-design.md` ("Extensions (2b)", "Calls made after the builder check"). Source of the extensions: `~/.pi/agent/extensions/` (read-only: Aria never edits `~/.pi`). Card contract: header of `ui/chat/parts/cards.js`.

## Global Constraints

- Everything built is in English; chat with the owner in Spanish.
- Small diffs: an extension's logic stays word for word; only paths, mode checks and the way it asks change. Every change is listed in `extensions/README.md`.
- Aria never reads or writes `~/.pi`; every path an extension uses comes from `process.env.PI_CODING_AGENT_DIR` (Aria sets it to `~/.aria/agent`), falling back to `~/.pi/agent` so the files still run in the owner's own Pi.
- The extensions still behave as today in Pi's terminal (`mode === 'tui'`); RPC (`mode === 'rpc'`, `hasUI === true`) gets the cards; `json`/`print` children keep today's behaviour.
- Cards ride in the dialog `title` as `{"aria":1,"kind":…}`; the dialog's `options` are what the extension accepts back, unchanged.
- Packages are pinned: `npm:pi-parallel-agents@0.2.0`, superpowers `git:github.com/obra/superpowers` at commit `bb92a77` (skills `systematic-debugging`, `verification-before-completion`, as in the owner's `settings.json`).
- The owner's rule on UI: reuse existing parts (`ui/chat/parts/`); extend a part with optional props instead of making a new one.
- Out-of-repo changes go to `CHANGELOG.md`.

## Review Focus

- A dialog title that is not JSON (an extension or a package we did not touch) must still render as the generic dialog, never break the chat (Task 4 test `a plain title stays a plain dialog`).
- The sandbox's four choices (once / this program or folder for the session / everything this session / block) must all reach the owner; losing "everything" or the scoped grant would be less than today's Pi (Task 4 tests).
- An answer that arrives after the dialog's timeout (quota countdown) must not be applied twice or flip the decision (Task 5 test `a late answer after the timeout is ignored`).
- A subagent that connects, crashes or never sends `end` must end as failed or done in the panel, never "running" forever (Task 7 tests).
- The same files must keep working in the owner's own Pi (`~/.pi/agent`): the env fallback and `mode === 'tui'` paths (Task 2 test `agentDir falls back to ~/.pi/agent`).

## Done before starting (review of what was built, 2026-10-19)

A fresh review of `main` found problems this plan builds on; they were fixed first (branch `pre-2b-fixes`): Pi processes are really stopped (`Engine::close`), a restart after `/connect` waits for a run to settle, dialogs Pi drops when a run settles are closed (with their notification), a notification for a card reads its summary (not the JSON), sessions live in one folder (`agent::sessions_dir`, `--session-dir`) so a changed projects folder keeps every tab's history, no `$HOME` fallback for the working folder (`workdir-unavailable`), `prepare` is safe with several sessions starting at once, a broken Pi `settings.json` is reported (`pi-settings-invalid`) instead of wiping `packages`, a refused prompt goes back to the composer, history that arrives late goes in front of what was typed, the chat counts as unseen while the settings are open, late replies for a closed tab are dropped, the duplicate `meter` part is gone. Consequences for the tasks below:

- Task 1: rewrite `leaves_an_existing_system_md_alone` for the hash rule; `prepare` already holds a global lock.
- Task 3: install the packages from the installer (the dock's `PATH` may not have `npm`); `ARIA_PROJECTS_DIR` is the real working folder (never `$HOME`); the page's command cache is already cleared when a Pi restarts.
- Tasks 4–6: the old `aria_jev` reducer case was removed; Task 6 adds the `setStatus` path. Esc stops a run first: a question card stays answerable only while its run is going (Pi drops it on abort).
- Task 7: `setAgents` is shared by all tabs today; make it per session.

---

### Task 1: The persona (fixes "it answered with a table")

**Files:**
- Create: `agent-files/SYSTEM.md`, `agent-files/agents/{frontend,backend,security,reviewer,researcher}.md` (copied word for word from `~/.pi/agent/`), `agent-files/README.md` (what each file is, that 2c turns them into templates)
- Modify: `crates/aria-core/src/agent.rs` (embed `agent-files/` with `include_dir`, write into `~/.aria/agent/` on `prepare`)
- Test: `crates/aria-core/src/agent.rs` tests

**Interfaces:**
- Produces: `agent::prepare` writes `SYSTEM.md` and `agents/*.md` from the embedded copy when the file is missing **or was written by an earlier Aria** (a marker file `agent/.aria-files` lists the sha256 of what Aria wrote; a file whose hash differs was edited by the owner and is kept).

- [ ] **Step 1: Failing tests** `writes_the_owners_persona` (SYSTEM.md equals the embedded one, the five experts exist), `replaces_the_placeholder_persona` (a SYSTEM.md equal to today's one-line placeholder is replaced), `keeps_a_persona_the_owner_edited` (edited file + recorded hash → untouched), `the_marker_lists_what_was_written`.
- [ ] **Step 2: Run** `cargo test -p aria-core agent::` — Expected: FAIL.
- [ ] **Step 3: Implement** (drop `DEFAULT_SYSTEM`; the placeholder's hash is a constant so the upgrade replaces it).
- [ ] **Step 4: Run** `cargo test --workspace` — Expected: PASS.
- [ ] **Step 5: Commit** `feat: Aria runs with the owner's persona and experts`.

### Task 2: The extensions in the repo, unpacked into Aria's agent

**Files:**
- Create: `extensions/` (copy of `~/.pi/agent/extensions/` with tests, minus `eval.ts`/`review-eval.ts` which need the owner's install), `extensions/lib/agent-dir.ts`, `extensions/lib/agent-dir.test.ts`, `extensions/README.md`, `extensions/package.json` (only if a test needs a runtime dependency; versions equal to Pi 1.3.0's)
- Modify: every `homedir() … '.pi', 'agent'` and `~/Projects` in `jev-router/{index,usage,modes}.ts`, `autocompact/autocompact.ts`, `sandbox/policy.ts` → `agentDir()`; `crates/aria-core/src/agent.rs` (embed and unpack `extensions/` to `agent/extensions/`, same hash rule as Task 1)
- Test: `bun test extensions`, `cargo test -p aria-core agent::`

**Interfaces:**
- Produces: `extensions/lib/agent-dir.ts`: `agentDir(): string` (`process.env.PI_CODING_AGENT_DIR || join(homedir(), '.pi', 'agent')`), `projectsDir(): string` (`process.env.ARIA_PROJECTS_DIR || join(homedir(), 'Projects')`); Aria sets `ARIA_PROJECTS_DIR` to the workdir when it spawns Pi.
- Pi must not load `extensions/lib/` as an extension: check how Pi 1.3.0 discovers extensions (`docs/extensions.md`); if a folder without `index.ts` is loaded, move the helper to a file each extension imports relatively (Ruling).

- [ ] **Step 1: Failing tests** `agentDir uses PI_CODING_AGENT_DIR`, `agentDir falls back to ~/.pi/agent`, `projectsDir uses ARIA_PROJECTS_DIR`; Rust `unpacks_the_extensions` (each `*/index.ts` exists under `agent/extensions/`, no `*.test.ts`), `keeps_an_extension_the_owner_edited`.
- [ ] **Step 2: Run** `bun test extensions && cargo test -p aria-core agent::` — Expected: FAIL (helper and unpack missing); the copied extension tests must already PASS (run them first to know the baseline).
- [ ] **Step 3: Implement**, then `grep -rn "'.pi', 'agent'\|homedir(), 'Projects'" extensions --include=*.ts` must print only `lib/agent-dir.ts` and the eval scripts left out.
- [ ] **Step 4: Run** `bun test extensions && cargo test --workspace` — Expected: PASS; add `bun test extensions` to the README's development commands.
- [ ] **Step 5: Live check** start the dock, send "hola": `~/.aria/agent/extensions/` exists, the answer is short, no table (STYLE_NOTE from `jev-router` reaches the model only once Task 3 makes RPC a routed mode — note the result).
- [ ] **Step 6: Commit** `feat: the owner's extensions run in Aria's Pi`.

### Task 3: The packages they rely on, and RPC as a routed mode

**Files:**
- Modify: `crates/aria-core/src/agent.rs` (`required_packages()`), `crates/aria/src/handlers.rs` or the installer (install them), `extensions/jev-router/index.ts:126` (`routed`), `crates/aria/src/chat.rs` (env `ARIA_PROJECTS_DIR`)
- Test: Rust unit tests, `bun test extensions`, live test

**Interfaces:**
- Produces: `agent::REQUIRED_PACKAGES: [&str; 2]` = `["npm:pi-parallel-agents@0.2.0", "git:github.com/obra/superpowers@bb92a77"]` with the superpowers entry carrying the owner's filter (`extensions: []`, `skills: [systematic-debugging, verification-before-completion]`); `agent::packages_installed(&AriaPaths) -> bool`; the installer runs `pi install` for any missing one (same command shape as `claude_bridge_install_command`). Check Pi's syntax for a pinned git package in `docs/packages.md` and write the filter into Pi's `settings.json` as the owner's has it.
- `jev-router`: `routed(ctx)` is true for `tui`, `print` **and** `rpc`.

- [ ] **Step 1: Failing tests**: `required_packages_are_pinned`, `prepare_lists_the_superpowers_filter`, installer dry-run lists both `pi install` lines; `routed` test in `jev-router` (export `routedMode(mode)` from `modes.ts` and test it: tui/print/rpc true, json false).
- [ ] **Step 2: Run** them — Expected: FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** all suites — Expected: PASS. Live (`#[ignore]` test in `crates/aria/tests/live_pi.rs`): after install, `get_commands` lists `plan`, `execute`, `effort`, `autocompact` and the skills; `get_available_models`… unchanged.
- [ ] **Step 5: Commit** `feat: the packages the extensions need, and the router in RPC`.

### Task 4: Sandbox cards (leave the sandbox, write outside)

**Files:**
- Modify: `extensions/sandbox/index.ts` (lines ~87 and ~147: ask in RPC), `ui/chat/parts/permission.js` (+ test: optional `options`), `ui/chat/parts/cards.js` (+ test: `options` field), `ui/chat/thread.js`, `ui/chat/state.js` (note text for arbitrary choices)
- Test: `bun test extensions/sandbox ui/chat`

**Interfaces:**
- Produces: in RPC the sandbox calls `ctx.ui.select(cardTitle, OPTIONS)` where `cardTitle = JSON.stringify({ aria: 1, kind: 'sandbox' | 'write', summary, detail, path?, jev?, options: [{ value, label, primary? }] })` and `OPTIONS` are the same four strings as today (`ONCE`, `forUnits ?? forExact` / `forDir`, `ALL`, `NO`); the card's buttons send the option string back, so the extension's code after `select` is unchanged. Helper `sandboxCard(kind, fields, options): string` in `extensions/sandbox/card.ts` (+ test).
- `renderPermission({ …, options })`: when `options` is given, one button per option (first `primary`, the rest secondary) with `data-choice="<value>"`; without it, today's allow/always/deny.
- `answerDialog`: the note's text is the chosen option's label; `key` only for the known codes (`allow|always|deny|free|keep`).

- [ ] **Step 1: Failing tests**: `sandboxCard` round-trips through `parseCardTitle` (copy the parser expectations into a Bun test that imports `../../ui/chat/parts/cards.js`); permission part `renders the given options in order`, `the first option is the primary`, `values are escaped`; thread `a sandbox card with four options shows four buttons`; state `an answered card with a free-text choice notes its label`; `a plain title stays a plain dialog`.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement**; the terminal path (`mode === 'tui'`) keeps its plain title.
- [ ] **Step 4: Run** `bun test extensions ui && bun tools/ui-sandbox/check.ts` (sandbox "Permission request" now uses the four-option card) — Expected: PASS.
- [ ] **Step 5: Commit** `feat: the sandbox asks with the chat's permission card`.

### Task 5: Quota card

**Files:**
- Modify: `extensions/quota-fallback/ask.ts:17` (ask when there is a UI and the mode is `tui` or `rpc`), the title as card JSON in RPC; `ui/chat/parts/quota.js` if the option values need mapping (reuse, no new part)
- Test: `bun test extensions/quota-fallback`

**Interfaces:**
- Produces: RPC title `{"aria":1,"kind":"quota","percent","freeModel","countdown":<seconds>,"window","options":[{value: SWITCH, label, primary:true},{value: STAY, label}]}`; the select keeps `{ timeout: ASK_TIMEOUT_MS }`; `choiceFrom` unchanged.
- `renderQuota` accepts the same optional `options` as the permission part (Task 4 pattern) and sends the option string.

- [ ] **Step 1: Failing tests**: `asks in rpc with a quota card`, `still answers switch without a UI`, `a late answer after the timeout is ignored` (the chat side: a `uiAnswer` for a closed dialog does nothing — already covered in Rust by `ui_answer_for_an_unknown_id_is_ignored`; add the UI side: the card disappears on `chatDialogClosed`).
- [ ] **Step 2–4:** run (FAIL), implement, run all (PASS).
- [ ] **Step 5: Commit** `feat: the quota fallback asks with the chat's quota card`.

### Task 6: Questions (`ask_user`) and the Jev line

**Files:**
- Modify: `extensions/ask-user/index.ts:58` (RPC path), `extensions/ask-user/questions.ts` (`parseCardAnswers(text, questions): Answer[] | null`), `extensions/jev-router/index.ts` (after classification), `ui/chat/state.js` (status → jev), `crates/aria/src/chat.rs` only if `setStatus` needs mapping
- Test: `bun test extensions/ask-user extensions/jev-router ui/chat`

**Interfaces:**
- Produces: ask_user in RPC: `const reply = await ctx.ui.input(JSON.stringify({ aria: 1, kind: 'question', questions }))`; the chat answers with `JSON.stringify([{ question, answers, talk? }])` (already what `questionAnswers` returns); `parseCardAnswers` maps it to the extension's `Answer[]`; cancelled or unparsable → today's `CANCELLED_TEXT` result.
- Jev line: after a successful classification jev-router calls `ctx.ui.setStatus('aria-jev', JSON.stringify({ mode, action }))` with `mode` from `{quick:'quick', focused:'focused task', parallel:'work to split', think:'needs thinking'}` and `action` = the flags joined with ` · ` (`searches memory`, `searches the web`, `checks Notion`, `read-only`); only when `ctx.mode === 'rpc'`. The reducer turns a `ui_notice` with `method: 'setStatus'`, `statusKey: 'aria-jev'` into `{ kind: 'jev', mode, action }` (and ignores it when `statusText` is empty or not JSON).

- [ ] **Step 1: Failing tests**: `parseCardAnswers` (good JSON, talk-only, garbage → null), `rpc asks through input with a question card` (mock ctx), `jevStatus(hints)` pure helper in `jev-router/route.ts` (four routes, flags, read-only); reducer `a jev status becomes a route line`, `a broken jev status is ignored`.
- [ ] **Step 2–4:** run (FAIL), implement, run all (PASS) plus `bun tools/ui-sandbox/check.ts`.
- [ ] **Step 5: Commit** `feat: ask_user and the router's route line in the chat`.

### Task 7: The subagents panel (Rust hosts the socket)

**Files:**
- Create: `crates/aria/src/watch.rs` (socket server, protocol parsing, agent list)
- Modify: `extensions/subagent-watch/index.ts` (load the parent half only in `tui`; the child half unchanged), `crates/aria/src/chat.rs` (one socket per session, `PI_WATCH_SOCK` in the spawn env, `parallel` tool events name the agents), `crates/aria-core/src/bridge.rs` (`DockMessage::ChatAgents { agents: Value }`), `ui/chat/state.js` + `ui/chat/view.js` (`chatAgents` → `setAgents` for the active session; Alt+A badge)
- Test: `crates/aria/src/watch.rs` tests, chat tests, UI tests

**Interfaces:**
- Produces: `watch::Watch::bind(path: &Path) -> io::Result<Watch>`; `Watch::agents(&self) -> Vec<Agent>`; `Agent { id: u64, role: String, task: String, status: String, kind: "active"|"done"|"queued", steps: Vec<Step> }` serialized to the contract in `ui/chat/parts/session.js` (`tool {verb, target}`, `msg {text}`, `live {text, since}`); protocol lines `{"t":"prompt"|"text"|"thinking"|"turn"|"tool"|"result"|"end", …}` exactly as `subagent-watch/index.ts` sends them; naming from the latest `parallel` call's tasks (same rule as the extension's `nameFor`). Socket path `$XDG_RUNTIME_DIR/aria/watch-<session>.sock`, mode 0600, removed on shutdown. Updates are throttled to one `ChatAgents` per 250 ms per session.

- [ ] **Step 1: Failing tests**: `a child that connects is an active agent`, `prompt names it from the pending parallel task`, `text deltas join into one msg step`, `tool and result become steps`, `end marks it done`, `a socket that closes without end is failed` (or done when it had entries, as the extension does), `more than 20 agents keep the latest 20`, `garbage lines are skipped`, chat `pi is spawned with PI_WATCH_SOCK`, UI `chatAgents feeds the panel of the active session only`.
- [ ] **Step 2–4:** run (FAIL), implement, run all (PASS).
- [ ] **Step 5: Live check**: ask Aria for a task that uses `parallel` (two small read-only subagents); Alt+A shows them running, then done; their session view shows the steps.
- [ ] **Step 6: Commit** `feat: the subagents panel, fed by Aria's own watch socket`.

### Task 8: Docs, pending list, final review

- [ ] Update `docs/pending.md` (cards, Jev line, subagents are fed; what is still unfed: tasks/documents/phases = phase 3, memory = 2c), `README.md`, `CHANGELOG.md` (`~/.aria/agent/extensions`, `agents/`, packages), `extensions/README.md` (every diff from the owner's copy).
- [ ] Run `cargo test --workspace && bun test && bun test extensions && sh install/install.test.sh && bun tools/ui-sandbox/check.ts`.
- [ ] Whole-branch review (fresh reviewer, the session's model), fix pass, merge decision with the owner.
