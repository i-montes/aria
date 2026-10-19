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
- **Left out:** `jev-router/eval.ts`, `sandbox/review-eval.ts` and `sandbox/accept.sh`, which run real models against the
  owner's own Pi.

Later parts of phase 2b add: RPC as a routed mode, the chat's cards for the sandbox, the quota and `ask_user`, the router's
route line, and the subagents panel fed by Aria's own socket.
