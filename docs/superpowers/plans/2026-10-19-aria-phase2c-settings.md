# Aria phase 2c — settings drive Pi — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** What the setup stores in `~/.aria/settings.json` (models, quota, router, effort, compaction, persona, sandbox, git, subagents, memory, packages) becomes the configuration of Aria's Pi and of the owner's extensions, and a change applies without editing files and without cutting a run.

**Architecture:** `aria_core::agent::prepare` (already run before every Pi start) gains three outputs: one config file for the extensions (`agent/aria.json`), the persona and experts rendered from templates, and the Engram entry in `mcp.json`. The extensions read `aria.json` through one helper (`extensions/lib/config.ts`) whose defaults are today's constants, so the owner's own Pi, which has no `aria.json`, behaves exactly as before. The dock watches `settings.json` and restarts each session's Pi when its run settles (`Chat::restart_when_idle`, from 2a); `/effort` and `/autocompact` write Aria's settings instead of their own files when they run under Aria.

**Tech Stack:** Rust (`aria-core`, `aria`), TypeScript extensions run by Pi 1.3.0 (Bun for tests), POSIX `sh` installer.

**Spec:** `docs/superpowers/specs/2026-10-18-aria-phase2-parity-design.md` ("Settings drive Pi (2c)", "Calls made after the builder check"); parity source `ARIA-SETUP.md` (sections B–C).

## Global Constraints

- Everything built is in English; the conversation follows the user's language.
- The sandbox is always on for every Pi run, with the same policy as today (`bwrap`; writes only in the project, `/tmp` and download caches).
- Agents Aria launches use the chat's model; the quota, effort and compaction behaviour of today's setup is kept exactly.
- "Settings changes never restart a run": a change is applied when the current run settles; what Pi reads only at start (`SYSTEM.md`, models) is applied by restarting Pi at that moment.
- "The owner's prose becomes templates": `agents/*.md`, `SUBAGENT.md` and `SYSTEM.md` are templatized only where a setting varies them; everything else stays word for word.
- "Aria never edits or removes entries it did not write in `mcp.json`."
- Rewriting the owner's extensions beyond what running under Aria needs is an anti-goal: every change is listed in `extensions/README.md`, and with no `aria.json` every extension behaves as in `~/.pi/agent/extensions/`.
- No secret in a log, in `settings.json` or in `aria.json`.
- The TypeSafe router backend is not built (spec): with a provider that has no Jev the router stays off.
- TDD; reuse components; the repo is public (scan before committing); out-of-repo changes go in `CHANGELOG.md`.

## Review Focus

- A `settings.json` from an older or newer Aria (a section missing, an unknown key, a wrong type): the extensions fall back to today's defaults field by field and never throw at load.
- A setting changed while a run is going (from the settings screen or from `/effort` in another tab): the run finishes untouched, then every session picks the change up; nothing is lost when the settings screen saves right after `/effort` wrote.
- The owner's terminal Pi with these same extension files and no `aria.json`: identical behaviour (route, effort file, autocompact file, plan model, free model, thresholds, experts).
- Turning an expert or memory off and on again: it comes back (the "deleted stays deleted" rule of `install_embedded` must not apply to files Aria removed itself).
- `mcp.json` holding servers Aria did not write (`rea`, `figma`, a hand-written `engram`): untouched by memory on or off.

---

### Task 1: The extensions' config file

**Files:**
- Modify: `crates/aria-core/src/agent.rs` (`prepare_reporting`), new `crates/aria-core/src/agent_config.rs`
- Create: `extensions/lib/config.ts`, `extensions/lib/config.test.ts`

**Interfaces:**
- Produces (Rust): `pub fn extension_config(settings: &Settings) -> serde_json::Value` in `agent_config.rs`; `prepare` writes it to `agent/aria.json` (mode 600, atomic) when settings exist and removes it when they do not.
- Produces (TS): `ariaConfig(): AriaConfig` and `type AriaConfig` in `extensions/lib/config.ts`, reading `join(agentDir(), 'aria.json')` on every call (no cache: Pi restarts on change anyway, and tests write files).

`aria.json` shape (camelCase, every field optional on read):

```json
{
  "version": 1,
  "models": { "plan": { "provider": "opencode-go", "id": "glm-5.3" }, "planThinking": "high", "allowPayPerUse": false },
  "quota": { "switchAt": 98, "planSwitchAt": 90, "freeModel": { "provider": "opencode-go", "id": "cat-jam-free" },
             "askFirst": true, "countdownMs": 300000, "onNoAnswer": "switch" },
  "router": { "enabled": true },
  "effort": { "mode": "auto", "fixedLevel": null, "maxOnFree": true },
  "compaction": { "percent": 40 },
  "style": "very-short",
  "sandbox": { "extraWritable": [], "askToLeave": true, "jevReview": true },
  "subagents": { "saveSessions": false, "turnLimit": null, "planExecute": true }
}
```

`provider.quota` absent (a provider without a free tier) → no `quota` key. `router.enabled` is false when `router.backend` is not `opencode` (TypeSafe is not built). `extraWritable` entries are expanded (`~` → home) and kept only if absolute.

- [ ] **Step 1: Write the failing tests**

Rust, in `agent_config.rs`:
- `the_golden_settings_give_todays_values`: `extension_config(golden)` equals the JSON above (golden = `tests/golden/settings-provider-opencode-go.json`).
- `a_provider_without_a_free_tier_has_no_quota_section`
- `the_typesafe_backend_turns_the_router_off`
- `a_relative_extra_folder_is_dropped_and_tilde_is_expanded`
- `the_config_never_holds_a_secret`: settings with a Tavily service and a TypeSafe key stored in `auth.json` → the serialized config contains neither key.

Rust, in `agent.rs`: `prepare_writes_aria_json_and_removes_it_without_settings`.

TS, in `config.test.ts` (write files under a temp `PI_CODING_AGENT_DIR`):
- `without aria.json every value is today's default` — `ariaConfig()` deep-equals the JSON above.
- `a missing or wrong-typed field falls back field by field` — `{"quota":{"switchAt":"lots"},"effort":{"maxOnFree":false}}` gives `switchAt 98` and `maxOnFree false`.
- `a file that does not parse is ignored` — returns the defaults, does not throw.

- [ ] **Step 2: Run, expect FAIL** — `cargo test -p aria-core agent_config` (unresolved module) and `bun test extensions/lib` (cannot find `./config.ts`).
- [ ] **Step 3: Implement** both sides. The TS defaults are exactly today's constants (`PLAN_MODEL`, `PLAN_THINKING`, `QUOTA_THRESHOLD`, `FREE_MODEL`, `THRESHOLD`, `ASK_TIMEOUT_MS`, `DEFAULT_PERCENT`); import them from their modules so there is one source.
- [ ] **Step 4: Run, expect PASS**, then `cargo test --workspace` and `bun test extensions`.
- [ ] **Step 5: Commit** `feat: prepare writes the extensions' config (aria.json) from Aria's settings`.

### Task 2: Models, quota, effort and compaction follow the config

**Files:** `extensions/jev-router/{index,modes}.ts`, `extensions/quota-fallback/{fallback,ask,index}.ts`, `extensions/autocompact/{autocompact,index}.ts`, their tests; `extensions/README.md`.

**Interfaces:**
- Consumes: `ariaConfig()` (Task 1).
- Produces: `planModel()`, `planThinking()`, `planQuotaThreshold()` in `modes.ts`; `freeModel()`, `switchThreshold()`, `askTimeoutMs()` in `fallback.ts`; `compactPercent()` in `autocompact.ts`. The old constants stay exported as the defaults.

Behaviour:
- `/plan` uses `config.models.plan` at `config.models.planThinking`; the plan quota threshold is `quota.planSwitchAt`.
- quota-fallback switches at `quota.switchAt` to `quota.freeModel`, asks only when `askFirst` (otherwise switches and says so with `notify`), waits `countdownMs`, and on no answer does `onNoAnswer` (`switch` | `stay`).
- Effort: under Aria `effort.mode` `auto` follows the route; `fixed` uses `fixedLevel` for every message; free models run at their top level only when `maxOnFree`. Without `aria.json`, `jev-router-effort.json` decides as today.
- Compaction: under Aria the percent is `compaction.percent`; without `aria.json`, `autocompact.json` as today.
- Pay per use (ARIA-SETUP item 7): the dock's model picker (`ui/chat/parts/models.js` filter) leaves out `opencode/*` models unless `models.allowPayPerUse`; the current model always stays listed. Test in `models.test.js`: `pay-per-use models are hidden unless allowed`.

- [ ] **Step 1: Failing tests**, each with a temp `aria.json`:
  - `modes.test.ts`: `the plan model and effort come from the config`; `without a config the plan model is glm-5.3 at high`.
  - `fallback.test.ts`: `the free model and threshold come from the config`; `with askFirst off it switches without asking`; `on no answer, stay keeps the model`.
  - jev-router effort (new `effort.test.ts` over a pure `effortFor(route, model, config, fixedFile)`): `fixed mode ignores the route`; `maxOnFree off lets a free model follow the route`; `without a config the effort file decides`.
  - `autocompact.test.ts`: `the percent comes from the config`; `without a config autocompact.json decides`.
- [ ] **Step 2: Run** `bun test extensions`, expect FAIL on the missing functions.
- [ ] **Step 3: Implement**; move the effort decision out of `jev-router/index.ts` into `effortFor` (pure) so it can be tested.
- [ ] **Step 4: Run** `bun test extensions`, PASS; `bun build` each changed `index.ts` with `--external '*'` to check it loads.
- [ ] **Step 5: Commit** `feat: the plan model, quota, effort and compaction follow Aria's settings`; list each change in `extensions/README.md`.

### Task 3: Settings apply when the run settles; `/effort` and `/autocompact` write Aria's settings

**Files:** `crates/aria/src/chat.rs` (`Sessions`), `crates/aria/src/main.rs`, `crates/aria/src/handlers.rs`; `extensions/lib/config.ts`, `extensions/jev-router/index.ts`, `extensions/autocompact/index.ts`; `crates/aria-core/src/json_file.rs` if a helper is missing.

**Interfaces:**
- Produces (Rust): `Sessions::settings_changed(&self)` — prepares once, then `restart_when_idle()` on every open chat. `main.rs` polls `paths.settings()` mtime+len every second on the UI timer and calls it on a change; `SaveSettings` success calls it too (no double restart: a change already applied is ignored by comparing the file's content hash).
- Produces (TS): `writeAriaSetting(path: string[], value: unknown): boolean` in `config.ts`: when `ARIA_SETTINGS` is set (Aria's spawn env, the path of `~/.aria/settings.json`), reads it, sets the camelCase path, keeps every other key, writes a temp file and renames it (mode 600); returns false when not under Aria.

Behaviour: `/effort <level>` under Aria writes `effort.mode = "fixed"`, `effort.fixedLevel = level`; `/effort auto` writes `mode "auto"`, `fixedLevel null`. `/autocompact <n>` writes `compaction.percent`. Outside Aria they write their own files as today.

- [ ] **Step 1: Failing tests**
  - Rust `chat.rs`: `a_settings_change_waits_for_the_run_then_restarts_pi` (fake Pi in a run; change the file; no restart until `agent_settled`; then a new spawn); `a_change_while_idle_restarts_now`; `the_same_content_saved_twice_restarts_once`; `pi_gets_aria_settings_in_its_env`.
  - TS `config.test.ts`: `writeAriaSetting keeps unknown keys and writes atomically`; `outside Aria it writes nothing`.
  - TS autocompact/effort: `/autocompact 70 under Aria writes compaction.percent and not autocompact.json`.
- [ ] **Step 2: Run** `cargo test -p aria chat::` and `bun test extensions`, expect FAIL.
- [ ] **Step 3: Implement.** The page reloads its settings when the dock sends `Settings` after a change it did not save itself (the existing `loadSettings` reply is enough; send it unprompted).
- [ ] **Step 4: Run** all suites, PASS.
- [ ] **Step 5: Commit** `feat: settings changes apply when the run settles; /effort and /autocompact write Aria's settings`.

### Task 4: Router, sandbox and subagents follow the config

**Files:** `extensions/jev-router/index.ts`, `extensions/sandbox/{index,policy}.ts`, `extensions/subagent-mode/index.ts`, `crates/aria-core/src/agent.rs` (`SUPERPOWERS_SKILLS`, `with_package_filters`), `crates/aria-core/src/pi.rs` (the `pi` shim in `~/.aria/bin`), tests.

**Interfaces:** Consumes `ariaConfig()`. Produces `with_package_filters(packages: Value, skills: &[String]) -> Value` (skills from `settings.subagents.auto_skills`, defaulting to today's two).

Behaviour:
- `router.enabled` false: no classification and no hidden hint, effort fixed (Task 2 rule), process guard, plan mode and expert assignment still on.
- Sandbox: `extraWritable` are added to the writable roots (still never a secret dir, never Aria's own files); `jevReview` false skips the review call (the card shows no Jev line); `askToLeave` false: a request to leave the sandbox is refused without asking, and the model is told plainly that it is not allowed to leave the sandbox (owner's decision).
- Subagents: `planExecute` false: `/plan` and `/execute` are not registered; `saveSessions` true: the `pi` shim replaces a child's `--no-session` with `--session-dir <agent>/sessions/subagents`; `turnLimit` n: in a child (`subagent-mode`) the run is stopped after n turns with a final note "stopped at the turn limit (n)".

- [ ] **Step 1: Failing tests**
  - jev-router (pure `shouldRoute(config, mode)`): `router off means no hint`.
  - sandbox `policy.test.ts`: `an extra folder is writable`; `an extra folder inside a secret dir is not`; route: `askToLeave off refuses leaving without asking, saying it is not allowed`; review: `jevReview off asks without a review`.
  - subagent-mode (pure `turnLimitReached(turns, config)`): `no limit by default`; `stops at the limit`.
  - shim (Rust, `pi.rs`): `the shim keeps --no-session by default`; `with saved sessions it gives the child a session dir`.
  - `agent.rs`: `the superpowers filter lists the automatic skills from the settings`.
- [ ] **Step 2: Run** `bun test extensions`, `cargo test -p aria-core`, expect FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** all suites, PASS; `bun build` each changed `index.ts`.
- [ ] **Step 5: Commit** `feat: the router, the sandbox and the subagents follow Aria's settings`; update `extensions/README.md`.

### Task 5: The persona and the experts from templates

**Files:** `agent-files/SYSTEM.md` (becomes a template), `agent-files/agents/*.md`, `extensions/subagent-mode/SUBAGENT.md` only if a setting varies it, new `crates/aria-core/src/persona.rs`, `agent.rs` (`install_embedded` gains generated files), `extensions/jev-router/index.ts` (`STYLE_NOTE` from `config.style`).

**Interfaces:**
- Produces: `pub fn render_system(settings: &Settings) -> String` and `pub fn render_expert(source: &str, expert: &Expert) -> String` in `persona.rs`; placeholders are `{{name}}` style, rendered with plain string replacement (no template engine), and a test fails if any `{{` survives rendering.
- `install_embedded` gets the rendered bytes instead of the embedded ones for `SYSTEM.md` and `agents/*.md`; a disabled expert's file is removed **and its marker entry dropped**, so enabling it again writes it.

What varies (everything else word for word):
- `assistant.name` (the `# Aria` title and "You are Aria"), `assistant.userName` (one sentence when not empty: "The user's name is …"), `assistant.language` (`es`: "Spanish unless they write to you in another one"; `en`: "English unless …"), `assistant.style` (the "Keep answers…" bullet; one text per style id: `very-short` is today's text; `brief`, `detailed`, `technical`, `teaching` follow the setup's descriptions in `ui/i18n/en.json`).
- `git`: the **Scope** bullet only when `unaskedAsProposals`; the **Git** bullet from the four booleans (today's text when they are the defaults).
- `memory`: the "Memory (Engram)" section only when `memory.enabled`, with `memory.project` instead of `aria`.
- `subagents`: the experts list in "Parallel agents" names only the enabled ones; the "Superpowers" section says `/plan`/`/execute` only when `planExecute`, and lists `autoSkills` instead of the two fixed names.
- Experts: `readOnly` true writes `tools: read, bash, codemode` in the front matter; false removes that line.
- `STYLE_NOTE` in jev-router: one sentence per style, today's for `very-short` and for no config.

- [ ] **Step 1: Failing tests** in `persona.rs`:
  - `the defaults render the owners SYSTEM.md word for word except the Spanish line` — with the golden settings plus `language "es"`, `render_system` equals the original `~/.pi/agent/SYSTEM.md` text as copied in 2b (`agent-files/SYSTEM.md` before templating, kept as `crates/aria-core/tests/golden/SYSTEM.owner.md`).
  - `the name, user name, language and style change only their lines`
  - `memory off drops the memory section and the project is the setting`
  - `only enabled experts are listed`; `plan and execute off drop the Superpowers commands`
  - `no placeholder survives`
  - `a read-only expert gets the read-only tools line`
  - `agent.rs`: `turning an expert off and on brings its file back`; `an expert the owner edited is still kept`.
  - jev-router: `the style note follows the style`.
- [ ] **Step 2: Run**, expect FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** all suites, PASS. Live: `python3 /tmp/ask.py "hola"` with the owner's settings answers short and in Spanish.
- [ ] **Step 5: Commit** `feat: the persona and the experts are rendered from the settings`.

### Task 6: Memory (Engram) and packages

**Files:** `crates/aria-core/src/mcp.rs`, `agent.rs` (packages), `install/install.sh`, `install/install.test.sh`, `install/install-mac.test.sh`, new repo file `engram-version` (`3.3.2` and the release asset sha256 per platform).

**Interfaces:**
- Produces: `pub fn sync_memory(paths: &AriaPaths, memory: &Memory) -> Result<(), McpError>` in `mcp.rs`, called by `prepare`. Aria's entry is `{"command": "<~/.aria/bin/engram>", "args": ["mcp", "--project", <project>, "--tools=agent"], "env": {"ENGRAM_DATA_DIR": "<~/.aria/engram>"}, "toolExposure": { the seven tools: "direct" }}` (the owner's entry, ARIA-SETUP item 16, with its own database: owner's decision). It is recognised as Aria's by its content (that command and that data folder), not by an extra key Pi could reject; any other `engram` entry is the owner's and is never changed.
- Installer: `install_engram` downloads the pinned release for the platform into `~/.aria/bin/engram`, checks its sha256, and skips when that version is already there; a failure stops the install before the binary is replaced (same rule as Pi's packages).
- Packages: `integrations.packages.source == "path"` with a non-empty `path` adds that local package to Pi's `packages` (once); `gallery` adds nothing.

- [ ] **Step 1: Failing tests**
  - `mcp.rs`: `memory on writes Aria's engram entry with the project`; `memory off removes only Aria's entry`; `a hand-written engram entry is never touched`; `other servers are kept byte for byte`.
  - `agent.rs`: `a local package folder from the settings is listed once`.
  - `install.test.sh` / `install-mac.test.sh`: the dry run lists `install engram 3.3.2` before the binary is replaced; a wrong checksum is refused.
- [ ] **Step 2: Run**, expect FAIL.
- [ ] **Step 3: Implement.** Aria's Engram keeps its own database in `~/.aria/engram/` (owner's decision): it starts empty and does not mix with the owner's terminal Pi.
- [ ] **Step 4: Run** all suites, PASS. Live: a question that needs memory (`¿qué decidimos sobre el sandbox en el Mac?`) shows an Engram tool call and answers from memory.
- [ ] **Step 5: Commit** `feat: memory turns Engram on in Aria's Pi; the installer brings Engram; a local package folder from the settings`.

### Task 7: Docs, live check, review

**Files:** `docs/pending.md`, `docs/architecture.md`, `README.md`, `CHANGELOG.md` (`~/.aria/agent/aria.json`, the rendered `SYSTEM.md`/`agents`, Aria's `engram` entry in `~/.aria/mcp.json`, `~/.aria/bin/engram`, `~/.aria/engram/`, `sessions/subagents/`), `extensions/README.md`, the spec's status line.

- [ ] Live check on the owner's machine, one line each in the final message: change the style in the settings while a run is going (the run finishes, the next answer follows the new style); `/effort high` in a tab (the settings screen shows it fixed); turn memory off (no Engram calls) and on again; turn `security` off (the router no longer names it).
- [ ] Run `cargo test --workspace`, `bun test`, `bun test extensions`, `sh install/install.test.sh`, `sh install/install-mac.test.sh`, the browser checks.
- [ ] Fresh whole-branch review (no `model` parameter), one fix pass, then ask the owner about merging.

## Decisions taken by the owner (2026-10-19)

1. **"Ask before leaving the sandbox" off:** leaving is refused, and the model is told it is not allowed to leave the sandbox.
2. **Engram's database:** a separate one under `~/.aria/engram/` (`ENGRAM_DATA_DIR`), not the terminal Pi's.
3. **Execution:** inline in this session, one fresh whole-branch review at the end.
4. **Never reinstall or restart the owner's running Aria** while building this: live checks run Pi with a temporary copy of the agent directory.
