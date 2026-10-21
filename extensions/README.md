# The owner's extensions

These are the owner's Pi extensions, copied from his Pi (`~/.pi/agent/extensions/`) and adapted with the smallest diffs
that let them run inside Aria. Aria embeds this folder and writes it into its own Pi's directory (`~/.aria/agent/extensions/`)
every time a session starts; an extension the owner edited there is kept (see `crates/aria-core/src/agent.rs`).

| Extension | What it does |
|---|---|
| `jev-router` | Jev classifies each message (route, memory, web, Notion, read-only) and adds a hidden hint and the answer-style rule; reasoning effort follows the route (`/effort`); `/plan` and `/execute` |
| `sandbox` | every bash command runs in bwrap: writes only in the project, `/tmp` and caches; asks before leaving it, with Jev's review; also `background_run` (a never-ending command for Aria's dock), on the same route as bash |
| `quota-fallback` | when the OpenCode Go quota is almost spent, asks to switch to the free model (countdown) |
| `autocompact` | compacts the context at a set percent (`/autocompact`) |
| `ask-user` | the `ask_user` tool: 1–4 questions with options |
| `subagent-mode` | subagents (`pi --mode json` children) get a lean prompt (`SUBAGENT.md`) |
| `subagent-watch` | children stream what they do to a socket so it can be watched |
| `clock` | Aria's own: each user message gets a hidden `[now]` line with the local date, time, weekday and time zone |
| `background` | Aria's own: under Aria's dock (`ARIA_JOBS_SOCK`), the tools `background_agent`, `background_list`, `background_output`, `background_stop` (and `background_run` with `PI_SANDBOX=0`); a hidden `[background]` line while something runs; an `aria-background` message when a job ends |
| `connectors` | Aria's own: every MCP tool call that writes asks first on the chat's permission card (`kind: "connector"`), by the access level of its connector in Aria's settings (`aria.json` `connectors`: Read only refuses writes, Ask asks, Allow runs them); reads run, codemode's nested calls are gated the same, `engram` never is; "don't ask again" lasts the session |
| `install` | Aria's own: the `install` tool puts in a skill, an MCP server or a Pi package the user asked for, after the user confirms it on the permission card; the `install-from-link` skill (`agent-files/skills/`) tells the model how to read a link |

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
  - `sandbox/card.ts` (new): `canAsk` (a UI in `tui` or `rpc`) and `sandboxCard` (also the `connector` card of `connectors/card.ts`). `sandbox/index.ts` asks with card
    titles in RPC (running outside the sandbox, writing outside the project) and passes the run's abort signal to its
    dialogs.
  - `sandbox/index.ts`: bash's `reason` is asked for on every command, not only with `outsideSandbox`: the dock shows it
    as the step's title (the design's "Rebuild and verify"), with the command cut to one line beside it.
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
- **The sandbox's `background_run`** (new): a command that never ends (a dev server, a watcher) starts in Aria's dock
  instead of holding up the conversation. It takes exactly bash's route (`routeOf` in `sandbox/index.ts`): a boxed
  command goes to the dock already wrapped in bwrap (`backgroundLine` in `sandbox/route.ts`), leaving the sandbox asks
  first on the same card, and the refusals are bash's own texts (`refusalText`); a session grant covers both tools. A
  subagent, a missing dock or a folder outside the project (or `/tmp`) is refused before anything is asked. The description,
  parameters and those checks are shared with `background` (`background/run.ts`); without the sandbox (`PI_SANDBOX=0`)
  `background` registers the tool itself and runs the command as it is. Outside Aria's dock it refuses with
  "Background work needs Aria's dock."
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
    a credential that is a symlink (Aria's agent dir links its credentials to `~/.aria`) is hidden at its target, since
  bwrap cannot mount on a symlink;
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
