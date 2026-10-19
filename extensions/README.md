# The owner's extensions

These are the owner's Pi extensions, copied from his Pi (`~/.pi/agent/extensions/`) and adapted with the smallest diffs
that let them run inside Aria. Aria embeds this folder and writes it into its own Pi's directory (`~/.aria/agent/extensions/`)
every time a session starts; an extension the owner edited there is kept (see `crates/aria-core/src/agent.rs`).

| Extension | What it does |
|---|---|
| `jev-router` | Jev classifies each message (route, memory, web, Notion, read-only) and adds a hidden hint and the answer-style rule; reasoning effort follows the route (`/effort`); `/plan` and `/execute` |
| `sandbox` | every bash command runs in bwrap: writes only in the project, `/tmp` and caches; asks before leaving it, with Jev's review |
| `quota-fallback` | when the OpenCode Go quota is almost spent, asks to switch to the free model (countdown) |
| `autocompact` | compacts the context at a set percent (`/autocompact`) |
| `ask-user` | the `ask_user` tool: 1–4 questions with options |
| `subagent-mode` | subagents (`pi --mode json` children) get a lean prompt (`SUBAGENT.md`) |
| `subagent-watch` | children stream what they do to a socket so it can be watched |

Tests: `bun test extensions`.

## What differs from the owner's copy

- **Paths** go through `lib/agent-dir.ts`: `agentDir()` is `$PI_CODING_AGENT_DIR` (Aria sets `~/.aria/agent`), falling back to
  `~/.pi/agent`; `projectsDir()` is `$ARIA_PROJECTS_DIR` (Aria sets its working folder), falling back to `~/Projects`. Files
  changed: `jev-router/{index,usage,modes}.ts`, `autocompact/autocompact.ts`. The same files still run in the owner's own Pi.
- **The sandbox also protects Aria's credentials** (`~/.aria/auth.json`, `mcp.json`, `mcp-auth.json` and their links in
  `~/.aria/agent/`), in `sandbox/policy.ts`.
- **In Aria's dock (Pi's RPC mode)** the extensions talk to the chat instead of a terminal. The terminal paths
  (`mode === 'tui'`) are unchanged.
  - `jev-router/modes.ts`: `routedMode` routes in `tui`, `print` and `rpc`. `jev-router/route.ts` adds `jevStatus`;
    `index.ts` sends it as `setStatus('aria-jev', …)`, the chat's route line.
  - `sandbox/card.ts` (new): `canAsk` (a UI in `tui` or `rpc`) and `sandboxCard`. `sandbox/index.ts` asks with card
    titles in RPC (running outside the sandbox, writing outside the project) and passes the run's abort signal to its
    dialogs.
  - `sandbox/policy.ts`: `isAgentCode` marks Aria's own Pi files (`extensions/`, `agents/`, `SYSTEM.md`, `settings.json`,
    `.aria-files` in `~/.aria/agent`) as never writable by the model, whatever was granted. It is only active under Aria
    (both `PI_CODING_AGENT_DIR` and `ARIA_PROJECTS_DIR` set).
  - `quota-fallback/ask.ts` (new): `quotaQuestion` builds the quota card in RPC, the old text in the terminal.
  - `jev-router/modes.ts`: `/plan` asks the brainstorming questions in batches of up to 4 independent ones, not one
    per call (the owner asked for it).
  - `ask-user/index.ts`: the tool's description asks for one batch of up to 4 independent questions per call, even when
    a skill asks one at a time (the dock shows them as one card; the owner asked for it).
  - `ask-user/questions.ts`: `questionCard`, `parseCardAnswers` and `TALK_LABEL`. `ask-user/dock.ts` (new) asks with
    the card through `ui.input` with the abort signal; `index.ts` calls it in RPC.
- **Without bwrap** (macOS has none) the sandbox no longer refuses every command: each one asks first, with the same
  permission card and choices (`sandbox/route.ts`). Nothing runs unconfined without the user. On Linux nothing
  changes: without bwrap commands are still refused.
- **Aria's settings** (phase 2c) reach the extensions through one file, `aria.json` in the agent directory, read by
  `lib/config.ts`. Without it every value is the one these files always had (`DEFAULTS`), so the owner's own Pi is
  unchanged. Under Aria:
  - `jev-router`: the `/plan` model, its effort and its quota threshold (`planModel`, `planThinking`,
    `planQuotaThreshold` in `modes.ts`); the effort decision moved to `effort.ts` (`effortFor`), where Aria's fixed
    level and "top level on free models" apply. Outside Aria, `jev-router-effort.json` decides as before.
  - `quota-fallback`: the free model, the threshold, the countdown, asking first and what no answer does (`freeModel`,
    `switchThreshold`, `askTimeoutMs`; `choiceFrom` gained `onNoAnswer`).
  - `autocompact`: the share comes from the settings (`compactPercent`); outside Aria, `autocompact.json`.
  - `jev-router`: with the router off (`router.enabled`) Jev does not classify (`shouldClassify` in `modes.ts`);
    `/plan` and `/execute` are registered only with `subagents.planExecute`.
  - `sandbox`: `sandbox.extraWritable` folders that exist join the writable roots (never one holding secrets, and
    never one that holds or sits inside Aria's home);
    `jevReview` off skips Jev's review; `askToLeave` off refuses leaving the sandbox and says it is not allowed
    (`route.ts`, `refuse-not-allowed`).
  - `subagent-mode`: `subagents.turnLimit` stops a subagent's tools at that many turns (`limit.ts`). Saved subagent
    sessions are the `pi` shim's job (`crates/aria-core/src/pi.rs`).
  - `/effort` and `/autocompact` write Aria's settings (`ARIA_SETTINGS`, `writeAriaSetting` in `lib/config.ts`) instead
    of `jev-router-effort.json` and `autocompact.json`; the dock applies the change when the run settles.
- **Checks with real models:** `jev-router/eval.ts`, `sandbox/review-eval.ts` and `sandbox/accept.sh`, the owner's
  evals, run against Aria's pinned Pi through a scratch Aria home (`tools/parity/scratch-home.sh`; how to run them is in
  `tools/parity/README.md`). The cases name neutral projects instead of the owner's, and the experts are the ones Aria
  writes for its settings.
