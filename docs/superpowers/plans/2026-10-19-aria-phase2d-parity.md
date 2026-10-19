# Aria Phase 2d — Parity Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Walk `ARIA-SETUP.md` item by item against Aria. Each item gets an automated check or a short manual step, recorded in an acceptance matrix in the repo. Close the gaps the walk found.

**Architecture:** The matrix is `docs/parity.md`, with one row or more per item. A bun test keeps the matrix honest: every item has a row, and every file or test a row names exists. The checks that need a real model are the owner's evals, ported into the repo. They run against Aria's pinned Pi on a scratch copy of `~/.aria`, never on the installed dock. The walk found three gaps, fixed here:
- the installer does not install `bubblewrap` and `iproute2`;
- Pi inherits `ANTHROPIC_*` variables from the dock's environment;
- the owner's evals are not in the repo.

**Tech Stack:** Rust (`aria-core`, `aria-engine`, `aria`), TypeScript on bun (extensions, tools), POSIX sh (installer, scripts).

**Spec:** `docs/superpowers/specs/2026-10-18-aria-phase2-parity-design.md` ("The rule", "Goals, non-goals, anti-goals", "Parity check (2d)"); parity source `ARIA-SETUP.md` (items 1–37).

## Global Constraints

- "Parity" means:
  - every item in sections B–C works from Aria;
  - the items in sections D–E are handled by the setup and the installer;
  - section F's checks pass against Aria.
- "Nothing new is added in this phase."
- Out of scope, by the owner's decision: `rea`, OpenPencil and Figma (MCP servers) and the local `meeting` package. The matrix marks those items out of scope.
- "Parity is accepted by the owner using Aria for real work, not by the checklist alone." The matrix never claims acceptance.
- Everything built is in English. The eval cases keep the owner's Spanish messages, because they are the user's input under test.
- The repo is public:
  - no personal paths (`/home/<user>`), employer or client names, or keys in any file, eval cases included;
  - scan before every commit that touches `tools/` or `extensions/*/eval*`.
- Never modify `~/.pi`, and never reinstall, restart or write into the installed Aria (`~/.aria`). Live checks run on a scratch `ARIA_HOME` (Task 2), which reads `~/.aria` and writes nothing there.
- Never put secrets in logs or output. The sandbox acceptance compares a key prefix without printing it.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. The dock is started from a shell that exports `ANTHROPIC_API_KEY` or another `ANTHROPIC_*` variable. Pi must not see them (item 29: they break the Claude provider). Test in Task 1.
2. A live check copies `auth.json` into the scratch home. The copy must be mode 600 and removed when the script ends, even when it fails. Test in Task 2.
3. A live check must leave `~/.aria` byte for byte as it was, including sessions, `agent/`, the Engram data and `settings.json`. Test in Task 2.
4. A test is renamed and the matrix row that names it goes stale. The guard must fail. Test in Task 5.
5. On a fresh Omarchy without `bubblewrap`, the sandbox refuses every command (`refuse-no-bwrap`). The installer must install it, and on a machine that already has it the step must change nothing. Test in Task 3.

---

### Task 1: Pi's environment, shared, without `ANTHROPIC_*`

**Files:**
- Modify: `crates/aria-engine/src/engine.rs` (`SpawnSpec`, `Engine::spawn`)
- Modify: `crates/aria-core/src/pi.rs`
- Modify: `crates/aria/src/chat.rs:412-437` (the spawn spec)
- Modify: the other `SpawnSpec { … }` literals (`grep -rn 'SpawnSpec {' crates`, 6 today)
- Test: `crates/aria-engine/tests/engine.rs`, `crates/aria-core/src/pi.rs` tests, `crates/aria/src/chat.rs` tests

**Interfaces:**
- Produces:
  - `pub unset: Vec<String>` on `SpawnSpec`: names removed from the inherited environment before `env` is applied.
  - `pub fn session_env(paths: &AriaPaths, path_var: &str, workdir: &Path) -> Vec<(String, String)>` in `pi.rs`. It returns today's five pairs, in today's order: `PI_CODING_AGENT_DIR`, `PATH`, `PI_SKIP_VERSION_CHECK=1`, `ARIA_PROJECTS_DIR`, `ARIA_SETTINGS`.
  - `pub fn unset_for_pi(vars: impl Iterator<Item = (String, String)>) -> Vec<String>` in `pi.rs`: the names starting with `ANTHROPIC_`.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing tests**
  - `engine.rs` test `an_unset_variable_does_not_reach_the_child`:
    - set `ARIA_ENGINE_TEST_SECRET=x` in the test process;
    - spawn a `sh -c 'printenv ARIA_ENGINE_TEST_SECRET || echo absent'` child with `unset: vec!["ARIA_ENGINE_TEST_SECRET".into()]`;
    - assert its stdout line is `absent`.
    - Follow the file's existing pattern for reading a child's output, or use `fake_pi.sh` if it already prints its environment.
  - `pi.rs` test `anthropic_variables_are_unset_for_pi`: for `[("ANTHROPIC_API_KEY","k"),("ANTHROPIC_BASE_URL","u"),("PATH","/bin"),("MY_ANTHROPIC","z")]`, assert the result is `["ANTHROPIC_API_KEY","ANTHROPIC_BASE_URL"]`.
  - `pi.rs` test `session_env_lists_what_pi_needs`: assert the five keys in that order, and that `ARIA_SETTINGS` is `paths.settings()`.
- [ ] **Step 2: Run them and watch them fail**
  - Run: `cargo test -p aria-engine --test engine an_unset && cargo test -p aria-core pi::`
  - Expected: compile errors, because there is no `unset` field, no `session_env` and no `unset_for_pi`.
- [ ] **Step 3: Implement**
  - Add the field. `Engine::spawn` calls `command.env_remove(name)` for each name before the `env` loop.
  - Add both functions.
  - `chat.rs` builds `env` with `pi::session_env(…)` (then pushes `PI_WATCH_SOCK` as today) and sets `unset: pi::unset_for_pi(std::env::vars())`.
  - Every other literal gets `unset: Vec::new()`.
  - `claude_bridge_install_command` runs Pi too, but returns an `InstallCommand`, not a spawn spec. Add `pub unset: Vec<String>` to `InstallCommand`, fill it the same way, and remove those names where the command is run (`grep -rn 'claude_bridge_install_command' crates`).
- [ ] **Step 4: Run the crates' suites**
  - Run: `cargo test -p aria-engine && cargo test -p aria-core && cargo test -p aria`
  - Expected: all pass. Under machine load, the engine's timing tests may need a rerun per crate.
  - `chat.rs`'s `pi_gets_aria_settings_in_its_env` keeps passing.
- [ ] **Step 5: Commit**
  - `fix: Pi never inherits ANTHROPIC_* variables from the dock's environment; one function builds Pi's session environment`

### Task 2: A scratch Aria home for live checks

**Files:**
- Create: `crates/aria-core/examples/parity_env.rs`
- Create: `tools/parity/scratch-home.sh`
- Create: `tools/parity/scratch-home.test.sh`
- Create: `tools/parity/README.md`

**Interfaces:**
- Consumes: `pi::session_env`, `pi::unset_for_pi` (Task 1), `agent::prepare_reporting`, `pi::write_shim`, `pi::locate`.
- Produces:
  - `sh tools/parity/scratch-home.sh <command> [args…]` runs the command in Pi's exact session environment for a scratch home, then deletes that home.
  - Inside the command:
    - `ARIA_HOME` is the scratch home;
    - `ARIA_PI` is the pinned Pi binary;
    - `ARIA_REAL_AUTH` is the path of the real `~/.aria/auth.json`, read-only, so that Task 4 can compare a key prefix;
    - `ARIA_PI_DIST` is the `@earendil-works/pi-coding-agent/dist` folder of the pinned Pi, so that the evals can import its `ModelRuntime`;
    - the working folder is a fresh `mktemp -d` git repo, as `accept.sh` had.

- [ ] **Step 1: Write the failing test `tools/parity/scratch-home.test.sh`**
  - It uses a fake real home (`ARIA_REAL_HOME=$tmp/real`) holding:
    - `settings.json`;
    - `auth.json` with `{"opencode-go":{"type":"api_key","key":"sk-fake-123456789"}}`;
    - a `pi/<pi-version>/node_modules/.bin/pi` stub that prints `$PI_CODING_AGENT_DIR` and `$ARIA_SETTINGS`.
  - It asserts four things:
    1. `scratch-home.sh sh -c 'echo $ARIA_HOME; stat -c %a $ARIA_HOME/auth.json; "$ARIA_PI"'` prints a home that is not `$tmp/real`, the mode `600`, and the stub prints paths inside that scratch home.
    2. After the run, that home no longer exists, and neither does it after a command that exits 3. The script's exit code is 3.
    3. A hash of every file under `$tmp/real` (`find … -type f | sort | xargs sha256sum`) is the same before and after.
    4. With `ANTHROPIC_API_KEY=z` exported, `scratch-home.sh sh -c 'echo ${ANTHROPIC_API_KEY:-absent}'` prints `absent`.
- [ ] **Step 2: Run it and watch it fail**
  - Run: `sh tools/parity/scratch-home.test.sh`
  - Expected: FAIL, because there is no `scratch-home.sh`.
- [ ] **Step 3: Implement**
  - `parity_env.rs`:
    - from `ARIA_HOME` (the scratch home), it loads the settings, runs `prepare_reporting` and writes the shim with the settings' `save_sessions`;
    - it prints shell lines: `unset NAME` for each name from `unset_for_pi(std::env::vars())`, `export K='V'` for each `session_env` pair with a temp working folder, then `export ARIA_PI='<binary>'` and `export ARIA_PI_DIST='<dist>'`;
    - values are single-quoted, with `'` escaped as `'\''`.
  - `scratch-home.sh`:
    - `real=${ARIA_REAL_HOME:-$HOME/.aria}`; `tmp=$(mktemp -d)`; `trap 'rm -rf "$tmp"' EXIT INT TERM`;
    - copy `settings.json`, `auth.json` (then `chmod 600`), and `mcp.json` and `mcp-auth.json` when they exist (`chmod 600`);
    - symlink `$real/pi` and every executable in `$real/bin` except `pi` (the shim is written fresh);
    - `git init -q` the working folder;
    - `eval "$(cargo run -q -p aria-core --example parity_env)"`, then `cd` into the working folder and `"$@"`; exit with its code.
    - The test passes `ARIA_PARITY_ENV=<cmd>` to replace the `cargo run` call only when that is needed to keep the test fast; otherwise it uses cargo.
  - `README.md` says what each script in `tools/parity/` checks, what it costs (Jev is free; `accept.sh` and the smoke test spend a few cents of the Go quota), and that nothing writes into `~/.aria`.
- [ ] **Step 4: Run it**
  - Run: `sh tools/parity/scratch-home.test.sh`
  - Expected: all pass.
- [ ] **Step 5: Commit**
  - `feat(tools): a scratch Aria home that runs a command in Pi's session environment and leaves ~/.aria untouched`

### Task 3: The installer installs the sandbox's system packages

**Files:**
- Modify: `install/install.sh:24` (`PACKAGES`)
- Test: `install/install.test.sh`

**Interfaces:** none.

- [ ] **Step 1: Write the failing test**
  - In `install.test.sh`, assert that the Omarchy package step's dry-run output (the `would:` line or the faked `pacman` call log) names `bubblewrap` and `iproute2`. Keep the existing case where `pacman -Qq` succeeds and nothing is installed.
- [ ] **Step 2: Run it and watch it fail**
  - Run: `sh install/install.test.sh`
  - Expected: the new assertion fails.
- [ ] **Step 3: Implement**
  - `PACKAGES="gtk3 webkit2gtk-4.1 gtk-layer-shell bubblewrap iproute2"`. The comment says that the sandbox runs every command in `bwrap`, and that the process guard tells the model to use `ss`. `bun` is not added: only the checks need it, not Aria.
- [ ] **Step 4: Run both installer tests**
  - Run: `sh install/install.test.sh && sh install/install-mac.test.sh`
  - Expected: all pass.
- [ ] **Step 5: Commit**
  - `fix(install): install bubblewrap and iproute2, which the sandbox and the process guard need`

### Task 4: The owner's evals, ported to Aria

**Files:**
- Create: `extensions/jev-router/eval.ts`, from `~/.pi/agent/extensions/jev-router/eval.ts`
- Create: `extensions/sandbox/review-eval.ts`, from `~/.pi/agent/extensions/sandbox/review-eval.ts`
- Create: `extensions/sandbox/accept.sh`, from `~/.pi/agent/extensions/sandbox/accept.sh`
- Create: `extensions/lib/runtime.ts`
- Modify: `extensions/README.md` (the "Left out" line becomes a "Checks" line)

**Interfaces:**
- Consumes: `ARIA_PI`, `ARIA_PI_DIST`, `ARIA_REAL_AUTH` and the working folder from Task 2. The evals run as `sh tools/parity/scratch-home.sh bun <repo>/extensions/jev-router/eval.ts`.
- Produces: `export async function modelRuntime(): Promise<any>` in `extensions/lib/runtime.ts`. It imports `${process.env.ARIA_PI_DIST}/index.js` and returns `ModelRuntime.create()`. Without `ARIA_PI_DIST` it throws `run this through tools/parity/scratch-home.sh`.

- [ ] **Step 1: Write the failing test `extensions/lib/runtime.test.ts`**
  - `modelRuntime rejects without ARIA_PI_DIST`: with the variable deleted, assert it rejects with a message containing `scratch-home.sh`.
- [ ] **Step 2: Run it and watch it fail**
  - Run: `bun test extensions/lib/runtime.test.ts`
  - Expected: FAIL, because the module is missing.
- [ ] **Step 3: Port the three files**
  - Keep each file's cases and logic word for word, except for these changes:
    - The Pi runtime comes from `modelRuntime()`.
    - Experts are read from the repo's `agent-files/agents/` (`new URL('../../agent-files/agents', import.meta.url)`), not from `~/.pi/agent/agents`.
    - Personal data is replaced with neutral equivalents that keep the case's meaning:
      - `/home/ana/Projects/acme/checkout-widget` → `~/Projects/acme/checkout-widget`, written through `projectsDir()` where code builds a path;
      - `checkout-api` → `checkout-api`;
      - `~/Projects/acme` → `~/Projects/acme`;
      - `elespectador.com` → `elespectador.com`;
      - the Notion task "Rifatón" → "Rifatón".
    - The bash classifier's `cwd` becomes `${projectsDir()}/acme/checkout-widget`.
  - In `accept.sh`:
    - `pi` is `"$ARIA_PI"`;
    - the repo is the current working folder that Task 2 provides, not its own `mktemp`;
    - the escape file is `$HOME/aria-sbx-escape.txt`;
    - case 3 reads the key prefix from `"$ARIA_REAL_AUTH"` and asks the model to read `~/.aria/auth.json` (Aria's secrets file).
- [ ] **Step 4: Run the unit test and the extension suite, then scan**
  - Run: `bun test extensions && git diff --cached --stat && grep -rnE 'ana|/home/[a-z]+/|acme|silla' extensions/jev-router/eval.ts extensions/sandbox/review-eval.ts extensions/sandbox/accept.sh`
  - Expected: the suite passes and the grep prints nothing. `bun test` does not pick up `eval.ts` or `review-eval.ts`, because their names have no `.test.`.
- [ ] **Step 5: Commit**
  - `feat(extensions): the owner's router and sandbox evals, run against Aria's pinned Pi through a scratch home`

### Task 5: The acceptance matrix and its guard

**Files:**
- Create: `docs/parity.md`
- Create: `tools/parity/matrix.ts`, `tools/parity/matrix.test.ts`

**Interfaces:**
- Produces:
  - `export type Row = { item: number; what: string; check: 'test' | 'eval' | 'script' | 'manual' | 'out-of-scope' | 'difference' | 'n/a'; where: string; result: string }`.
  - `export function parseMatrix(markdown: string): Row[]`: the table rows whose first cell is a number.
  - `export function setupItems(markdown: string): number[]`: the numbers of `ARIA-SETUP.md`'s `N. **…**` items.
  - `export function problems(rows: Row[], items: number[], exists: (path: string) => boolean, read: (path: string) => string): string[]`: one line per problem, empty when the matrix is sound.

- [ ] **Step 1: Write the failing tests** in `matrix.test.ts`, with in-memory inputs:
  - `every setup item needs a row`: items `[1,2]`, rows for `1` only → `["item 2 has no row"]`.
  - `a test row names a file and a test that exist`: `where: "crates/x.rs::the_name"`.
    - A missing file gives `item 1: crates/x.rs does not exist`.
    - A file without the string `the_name` gives `item 1: crates/x.rs has no test the_name`.
  - `eval and script rows name a file that exists`.
  - `manual, difference, out-of-scope and n/a rows need a where text`: an empty one gives `item 1: no reason or steps`.
  - `out-of-scope is only for the owner's exclusions`: allowed for items `5`, `6`, `17` and `32` only.
  - `the real matrix is sound`: runs `problems` on `docs/parity.md` and `ARIA-SETUP.md` from disk, and asserts `[]`.
- [ ] **Step 2: Run them and watch them fail**
  - Run: `bun test tools/parity`
  - Expected: FAIL, because the module and the matrix are missing.
- [ ] **Step 3: Implement `matrix.ts` and write `docs/parity.md`**
  - The page opens with how to read and rerun it, and the rule from the spec: acceptance is the owner's real use.
  - Then comes one table: `| Item | What | Check | Where | Result |`.
  - An item with several parts gets one row per part.
  - Each `test` row names an existing test that proves the behaviour. Find it with `grep -rn` in the file given below; when none exists, write it in this task, test-first.
  - Rows, with the check kind and where to look:

| Item | Check: where |
|---|---|
| 1 | test: `crates/aria-core/src/pi.rs` (pinned version is `pi-version`); test: `install/install.test.sh` (Pi installed at that version) |
| 2 | test: `install/install.test.sh` (Task 3); n/a for Node via mise for `rea`/`pi-acp` (out of scope) and bun (checks only) |
| 3 | test: `install/install.test.sh` (Engram 3.3.2 with checksum) |
| 4 | test: the installer's `claude` link (`install/install.test.sh`) and the start-up check (`grep -rn claude crates/aria/src`) |
| 5 | test: `crates/aria-core/src/agent.rs` (`with_package_filters`: superpowers filtered to the two skills; `pi-parallel-agents` listed; `CLAUDE_BRIDGE_PACKAGE` at 0.9.2); out-of-scope: `meeting` |
| 6 | out-of-scope: OpenPencil |
| 7 | test: `extensions/lib/config.test.ts` (daily default); test: `ui/chat/parts/models.test.js` (pay per use hidden) |
| 8 | test: `extensions/lib/config.test.ts` (`opencode/jev-1.13-free`) |
| 9 | test: `extensions/jev-router/modes.test.ts` (plan model and `high`) |
| 10 | test: `extensions/quota-fallback/fallback.test.ts` and `ask.test.ts` (98 / 90 / ask first); test: `crates/aria-core/src/providers.rs` (only OpenCode Go has a fallback) |
| 11 | test: `crates/aria-core/src/agent.rs` (`claude_bridge_installed`); manual: smoke test with a `claude-bridge` model (item 37) |
| 12 | test: `extensions/autocompact/autocompact.test.ts` (40 default, the setting) |
| 13 | test: `extensions/jev-router/effort.test.ts` (auto without a setting) |
| 14 | n/a: Pi 1.3.0 asks for project trust only in interactive mode (`dist/core/trust-manager.js` is used by `modes/interactive` only); the dock runs RPC |
| 15 | test: the persona golden test in `crates/aria-core/src/persona.rs` |
| 16 | test: `crates/aria-core/src/mcp.rs` (`sync_memory`: `--project`, 7 direct tools); difference: own database in `~/.aria/engram` (owner, 2026-10-19), so the 53 memories in `~/.engram` are not in Aria's memory |
| 17 | manual: Notion OAuth and the Tavily key from the setup's services step, then ask for a Notion task and a web search in the dock; out-of-scope: rea, openpencil, figma |
| 18 | test: `extensions/jev-router/route.test.ts`, `process-guard.test.ts`; eval: `extensions/jev-router/eval.ts` |
| 19 | test: `extensions/jev-router/effort.test.ts` |
| 20 | test: `extensions/jev-router/modes.test.ts` (`planMayWrite`, `notAPlan`) |
| 21 | test: `crates/aria-core/src/persona.rs` (`render_expert` read-only line) |
| 22 | test: `extensions/subagent-mode/limit.test.ts`; test: `crates/aria-core/src/pi.rs` (no saved sessions by default); test: `crates/aria/src/watch.rs`; difference: `/agents` does nothing in the dock, Alt+A opens the panel (`docs/pending.md`) |
| 23 | test: `extensions/ask-user/questions.test.ts`; manual: in a read-only turn, an answered `ask_user` lets the next write through (`jev-router/index.ts:376`) |
| 24 | test: `extensions/sandbox/*.test.ts`; script: `extensions/sandbox/accept.sh`; eval: `extensions/sandbox/review-eval.ts` |
| 25 | test: `extensions/quota-fallback/ask.test.ts` (one decision per window, shared through the decision file, countdown) |
| 26 | test: `extensions/autocompact/autocompact.test.ts` (`/autocompact` writes Aria's settings) |
| 27 | test: the persona golden test |
| 28 | manual: `PI_SANDBOX=0`, `PI_QUOTA_FALLBACK=0`, `PI_SUBAGENT_LEAN=0` and `CLAUDE_BRIDGE_DEBUG=1` reach Pi only through the dock's environment; the code reads them unchanged (`extensions/*/index.ts`) |
| 29 | test: `crates/aria-core/src/pi.rs::anthropic_variables_are_unset_for_pi` (Task 1); test: Aria's auth tests (`crates/aria-core/src/auth.rs`) |
| 30 | test: `crates/aria-core/src/agent_config.rs` (no key in Aria's config) and the setup's key test for `tavily` (`grep -rn tavily crates ui`) |
| 31 | test: `crates/aria-core/src/paths.rs` or the OAuth module (`mcp-auth.json` in Aria's home) |
| 32 | test: `crates/aria-core/src/mcp.rs` (the engram command is `~/.aria/bin/engram`); out-of-scope: rea |
| 33 | test: `crates/aria/src/chat.rs` (`ARIA_PROJECTS_DIR` is the working folder); `extensions/lib/agent-dir.test.ts` |
| 34 | n/a: the ported eval uses neutral paths built from `projectsDir()` (Task 4) |
| 35 | script: `cargo test` per crate, `bun test ui`, `bun test extensions`, `bun test tools/parity` |
| 36 | eval: `extensions/jev-router/eval.ts`, `extensions/sandbox/review-eval.ts`; script: `extensions/sandbox/accept.sh` |
| 37 | script: `tools/parity/smoke.sh` (Task 6); manual: the same with a `claude-bridge` model when one is connected |

  - The `Result` column is `—` until Task 7 fills it.
- [ ] **Step 4: Run them**
  - Run: `bun test tools/parity`
  - Expected: all pass, including `the real matrix is sound`.
- [ ] **Step 5: Commit**
  - `docs: the parity matrix, one row per ARIA-SETUP item, and a test that keeps every row pointing at a real check`

### Task 6: The smoke test

**Files:**
- Create: `tools/parity/smoke.sh`, `tools/parity/smoke.ts`, `tools/parity/smoke.test.ts`

**Interfaces:**
- Consumes: `ARIA_PI` from Task 2.
- Produces:
  - `export function judge(answer: string): string[]` in `smoke.ts`: the reasons an answer to "hola" fails. It fails when the answer is empty, has more than 4 sentences, has a table, a heading or a bullet list, or is not Spanish. "Spanish" means it contains `¡` or `¿`, or one of `hola`, `qué`, `en qué`, `puedo`, `ayudarte`, `ayudar`, `estás`, `cómo` (case-insensitive).
  - `smoke.sh` runs `"$ARIA_PI" -p "hola" </dev/null`, prints the answer, and exits non-zero with the reasons from `bun tools/parity/smoke.ts` when they are not empty. Run it as `sh tools/parity/scratch-home.sh sh tools/parity/smoke.sh`, optionally with `ARIA_SMOKE_MODEL=<provider/id>` passed as `--model`.

- [ ] **Step 1: Write the failing tests** in `smoke.test.ts`:
  - `¡Hola! ¿En qué te ayudo?` → `[]`;
  - `Hello! How can I help?` → `['not in Spanish']`;
  - an answer with `## Resumen` → contains `heading`;
  - six sentences → contains `more than 4 sentences`;
  - `""` → `['empty']`.
- [ ] **Step 2: Run them and watch them fail**
  - Run: `bun test tools/parity/smoke.test.ts`
  - Expected: FAIL, because the module is missing.
- [ ] **Step 3: Implement** `judge`, a `smoke.ts` main that reads the answer on stdin when run directly (`import.meta.main`), and `smoke.sh`.
- [ ] **Step 4: Run them**
  - Run: `bun test tools/parity`
  - Expected: all pass.
- [ ] **Step 5: Commit**
  - `feat(tools): the parity smoke test: Aria's Pi answers "hola" short and in Spanish`

### Task 7: The live walk and the record

**Files:**
- Modify: `docs/parity.md` (the `Result` column)
- Modify: `README.md` (one line: the parity matrix and how to rerun it)
- Modify: `docs/pending.md` (a "Parity (phase 2, part 2d)" section with what failed or stays manual)
- Modify: `docs/superpowers/specs/2026-10-18-aria-phase2-parity-design.md:3` (status: 2d implemented; acceptance waits for the owner's real use)
- Modify: `CHANGELOG.md` (a 2d section)

**Interfaces:** consumes everything above.

- [ ] **Step 1: Run every automated check and record each result with today's date**
  - Run, saving each output under the scratchpad and reading the tail:
    - `cargo test -p aria-engine`, `-p aria-core`, `-p aria`, `-p aria-hyprland`
    - `bun test ui`, `bun test extensions`, `bun test tools/parity`
    - `sh install/install.test.sh`, `sh install/install-mac.test.sh`, `sh tools/parity/scratch-home.test.sh`
  - Expected: all pass.
- [ ] **Step 2: Run the live checks through the scratch home**
  - Run:
    - `sh tools/parity/scratch-home.sh bun $PWD/extensions/jev-router/eval.ts`
    - `sh tools/parity/scratch-home.sh bun $PWD/extensions/sandbox/review-eval.ts`
    - `sh tools/parity/scratch-home.sh sh $PWD/extensions/sandbox/accept.sh`
    - `sh tools/parity/scratch-home.sh sh $PWD/tools/parity/smoke.sh`
  - Before and after, hash `~/.aria` (`find ~/.aria -type f -not -path '*/pi/*' | sort | xargs sha256sum | sha256sum`).
  - Expected:
    - the hashes match;
    - `accept.sh` shows `passed=7 failed=0`;
    - the smoke test passes;
    - the two evals print a score, which goes into the matrix as `N/M` with each failure's one-liner in `docs/pending.md`.
  - An eval failure is a finding, not a reason to change a case. A failure that reproduces against the owner's Pi is noted as such and is not a parity gap.
- [ ] **Step 3: Record**
  - Fill `Result`: `pass 2026-10-19`, `N/M 2026-10-19`, `manual — owner`, `out of scope`, or `difference`.
  - Write the docs lines listed under **Files**.
  - Run `bun test tools/parity`. Expected: pass.
- [ ] **Step 4: Scan and commit**
  - Run: `git diff --cached | grep -nEi 'ana|/home/[a-z]+/|sk-[a-z0-9]{12,}'`. Expected: nothing.
  - Commit: `docs: phase 2d walked: parity matrix results, pending gaps, spec status`
