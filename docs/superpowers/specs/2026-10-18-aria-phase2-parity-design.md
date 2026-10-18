# Aria phase 2 — parity with the owner's Pi — design

Date: 2026-10-18. Status: draft for review. Builds on `2026-10-18-aria-base-design.md` (roadmap, phase 1, now merged into `main`).

## Intent

Phase 1 gave Aria a dock that opens on `SUPER + A` and a setup that writes `~/.aria/settings.json`. Nothing runs behind it. Phase 2 makes Aria **do everything the owner's current Pi setup does**, from the dock: Aria ships its own pinned Pi, carries the owner's extensions inside the repo, turns the dock into the chat, and lets `settings.json` decide how all of it behaves.

## The rule

**Aria must work at least as well as the owner's current Pi.** "Parity" is concrete: every item in sections B–C of `ARIA-SETUP.md` (defaults and customizations) works from Aria, section D–E items (secrets, paths) are handled by the setup and the installer, and section F's checks pass against Aria. Nothing new is added in this phase.

## How it is split (agreed with the owner)

Four parts, each ending in software that works; each gets its own implementation plan.

1. **2a — The engine.** Aria runs its own pinned Pi; the dock sends a message and shows the reply, the tools and the permission questions while Pi works.
2. **2b — The owner's extensions.** The seven extensions move into the repo and run under Aria.
3. **2c — Settings drive Pi.** What the setup stored (models, effort, compaction, memory, MCP, packages, persona) becomes Pi's configuration.
4. **2d — The parity check.** `ARIA-SETUP.md` is walked item by item against Aria.

Parts 2a–2c depend on each other only loosely: 2a alone gives a plain chat; 2b and 2c each add behaviour on top.

## Goals, non-goals, anti-goals

- **Goals:** the dock is a usable chat for a full working day; every behaviour of today's Pi is reachable from it; changing a setting changes the behaviour without editing files; Aria does not depend on a Pi installed by the user or on `~/.pi`.
- **Non-goals:** autonomous goals and worktrees (phase 3), the state socket and external extensions (phase 4), Mac (phase 5), the design's memory panel, task dock, document cards and notes (new features, not parity), several chats in tabs, replacing Pi. Also left out by the owner for this phase: his personal integrations `rea`, OpenPencil and Figma (MCP servers) and the local `meeting` package; the parity check marks those items of `ARIA-SETUP.md` out of scope, and Aria never edits or removes entries it did not write in `mcp.json`.
- **Anti-goals:** Aria doing less than the Pi it replaces; the dock freezing or losing a run when it is hidden; a secret in a log or in `settings.json`; rewriting the owner's extensions beyond what running under Aria needs (small diffs keep future Pi upgrades easy); a Pi upgrade that was never tested.

## Constraints

- Everything built is in English; the conversation follows the user's language (spec of phase 1).
- The sandbox is always on for every Pi run, with the same policy as today (`bwrap`; writes only in the project, `/tmp` and download caches).
- Subagents are Pi children (`pi --mode json`) that load the same persona and extensions, as today; their sessions are not saved.
- Agents Aria launches use the chat's model (the owner's rule), and the quota, effort and compaction behaviour of today's setup is kept exactly.
- Hyprland (Omarchy) is the only platform; the engine and extension work must not close the door on Mac.

## Decided by the owner

- **Where Aria works:** Pi runs with `sandbox.projectsDir` (`~/Projects` unless the setup says otherwise) as its working folder, as the setup already implies. **The dock shows that path all the time**, so it is always clear where Aria is working.
- **Prompts while the dock is hidden:** when Aria needs an answer (a question, a permission, the quota choice) and the dock is hidden, it sends a **persistent desktop notification** that stays until the owner dismisses it or clicks it; clicking opens the dock on that prompt. The prompt waits for the answer (the quota countdown keeps running, as today); answering it, from the dock, removes the notification.
- **Out of this phase:** the owner's `rea`, OpenPencil, Figma and `meeting` (see non-goals).
- The roadmap order and the four-part split above.
- Aria ships its **own** Pi, pinned, independent of `~/.pi` (phase 1 spec).
- The design to build the dock from is `design/Aria.html` and its components; proto-1's structure and look are not used.
- Settings are the single place for configuration (`~/.aria/settings.json`), written by the onboarding.

## The shape (my calls, to be confirmed by the builder check and the owner)

### Engine (2a)

- **Pi runs as a child process in RPC mode** (`pi --mode rpc`): JSON lines over stdin/stdout, commands in, events out. Chosen over Pi's in-process SDK because Aria is Rust and RPC is language-independent, and over per-message processes because a long task must outlive the dock being hidden. The agent crashing does not take the dock down.
- A new crate, `aria-engine`, owns the child: start, supervise, restart on crash, speak the protocol (types for commands, responses and events), and hand events to the page through the existing bridge. It also exposes the extension dialogs Pi forwards (`select`, `confirm`, `input`, `editor`, `notify`, `setStatus`) as bridge messages.
- **One chat session at a time**, resumed when the dock opens (Pi's own session files are the history). The Pi process is started when first needed and kept while the dock is hidden; it stops at logout.
- **Hosting the pinned Pi.** Pi is the npm package `@earendil-works/pi-coding-agent` (Node ≥ 22.19, about 150 MB installed). The installer ensures Node, then installs the exact version recorded in a file in the repo (`pi-version`) under `~/.aria/pi/<version>/`; Aria checks it at start and says what is missing. Upgrading Pi is a deliberate change: bump the file, run the parity check, release.
- **The agent directory is `~/.aria/agent/`** (set through `PI_CODING_AGENT_DIR`), so Pi's own `settings.json` never collides with Aria's. Aria owns and regenerates it (see 2c): `settings.json` (Pi's), `auth.json`, `mcp.json` and `mcp-auth.json` (links to the files phase 1 already writes in `~/.aria/`), `SYSTEM.md`, `agents/`, `extensions/`, `sessions/`.

### The dock as chat (2a)

Built from the owner's design components, with the working folder shown at all times. Phase 2 uses: user and assistant messages with streaming, tool chips shown in execution order, the working indicator, the composer with the command, model and file pickers, the question card (`ask_user`), the sandbox permission card (with Jev's review), the quota card (with its countdown), and the subagents panel. Slash commands are sent as ordinary prompts (Pi runs extension commands immediately); the picker lists them from Pi (`get_commands`).

### Extensions (2b)

The seven extensions (`jev-router`, `sandbox`, `quota-fallback`, `autocompact`, `ask-user`, `subagent-mode`, `subagent-watch`; about 3,300 lines of TypeScript whose only dependencies are Pi's API and `typebox`) move into a top-level `extensions/` folder of the repo, are embedded in the binary, and are unpacked into `~/.aria/agent/extensions/` when Aria syncs. Their tests move with them.

What running under Aria changes, because in RPC mode `ctx.mode` is `"rpc"` (dialogs work, terminal-only UI does not):

- `jev-router` treats `rpc` as a routed mode (today only `tui` and `print`).
- `sandbox` and `quota-fallback` ask through `select` / `confirm`, which RPC forwards; the condition becomes "there is a UI" rather than "this is the terminal".
- `ask-user` cannot use its terminal form: in RPC mode it asks through the dialog methods and must keep its behaviour (1-4 questions, options, the recommended one first, an "other" answer). How is the builder's call.
- `subagent-watch`'s terminal overlay is replaced by the dock's subagents panel, fed from the same Unix socket (`PI_WATCH_SOCK`); that extension stays unloaded under RPC.
- `/effort` and `/autocompact` keep working as commands but write to `settings.json`, which is now the source of truth.

### Settings drive Pi (2c)

A "sync" step (at start and whenever a setting changes) regenerates the agent directory from `settings.json`:

| Setting | Where it lands |
|---|---|
| `provider`, `models` | Pi's `defaultProvider`/`defaultModel`; the planning model and its effort go to the extensions' config |
| `provider.quota` | the quota extension's config (thresholds, free model, ask first, countdown, on-no-answer) |
| `router` | the router extension's config; `typesafe` backend needs a TypeSafe client behind the same interface as today's OpenCode Jev (`bool` questions map to TypeSafe's `noul`) |
| `effort`, `compaction` | the effort and compaction extensions' config |
| `assistant` | `SYSTEM.md`, rendered from a template (assistant name, your name, language, answer style) |
| `sandbox` | the sandbox extension's config (projects dir, extra writable folders, ask to leave, Jev review) |
| `git` | the "way of working" rules in `SYSTEM.md` |
| `subagents` | which expert files are active and read-only, whether sessions are saved, the turn limit, `/plan` and `/execute`, the automatic skills |
| `memory` | an Engram entry in `mcp.json` (`--project <name>`), plus the Engram binary in the installer |
| `integrations` | Notion and Tavily are already in `mcp.json`; Pi packages go to Pi's `packages` |
| Claude Max (`anthropic-oauth`) | the `pi-claude-agent-sdk` package and the Anthropic sign-in `auth.json` already holds |

The extensions read one config file in the agent directory instead of constants in their code; that file is the only new interface between Aria and the extensions.

### Parity check (2d)

`ARIA-SETUP.md` becomes an acceptance matrix: each item is verified by an automated check (test, eval or script) or by a short manual step, and the matrix lives in the repo. Parity is accepted by the owner using Aria for real work, not by the checklist alone.

## Calls made after the builder check (mine; the owner can overrule)

- **Pi's children find the pinned `pi` through a shim.** `pi-parallel-agents` starts children with a `PATH` lookup of `pi`. Aria launches Pi with `~/.aria/bin` first on `PATH`, holding a `pi` wrapper that runs the pinned version. The package is not patched.
- **Structured cards ride on the dialogs.** RPC dialogs carry only `title`, `options` and `timeout`. The extensions send a JSON object as the `title` (`{"aria":1,"kind":"sandbox"|"quota"|"question"|"write", …fields}`); the dock renders the matching card and falls back to a plain dialog for any other title. The quota countdown uses the dialog's own `timeout`, which RPC forwards.
- **Rust hosts the subagent socket.** It sets `PI_WATCH_SOCK`, speaks the protocol `subagent-watch` uses today (read from that file), and names each child from the `parallel` tool events. The extension is split: the child half keeps loading in `json`/`print` children, the terminal half is dropped under RPC.
- **Settings changes never restart a run.** `~/.aria/settings.json` has one writer in practice (the dock); `/effort` and `/autocompact` also write it atomically and keep unknown keys, and the dock watches the file. A change is applied when the current run settles; things Pi only reads at start (`SYSTEM.md`, models) are applied by restarting Pi at that moment.
- **The owner's prose becomes templates.** `agents/*.md`, `SUBAGENT.md` and `SYSTEM.md` are copied into the repo as templates and templatized only where a setting varies them (assistant and user name, language, answer style, git rules, enabled experts). Everything else stays word for word.
- **Claude Max is part of parity** (item 11 of `ARIA-SETUP.md`): `anthropic-oauth` installs `pi-claude-agent-sdk` at a pinned version, and the installer and Aria's start-up check also cover Claude Code, which that package runs.
- **The TypeSafe router backend is not built in phase 2.** Today only the OpenCode Jev exists, so it is not parity. The setup still accepts the TypeSafe key; with a provider that has no Jev, the router stays off (as phase 1 already defaults) until the backend is built.
- **The `@` file picker is in scope**, as a plain, `.gitignore`-aware directory walk in Rust, relative to the chat's folder.

### Notifications for a hidden dock (2a)

Omarchy's own `omarchy-notification-send` is used (urgency `critical`, and a click action that opens the dock; `omarchy-notification-dismiss` removes it when the prompt is answered). Whether `critical` alone keeps a notification on screen indefinitely, or an expiry of `0` is also needed, is checked on the owner's shell by the builder.

## Left to the builder

Crate and module names; the exact RPC types; how `ask-user` asks in RPC mode; the config file's name and shape; how the installer ensures Node (package manager versus the owner's mise); how sync detects changes; how the dock renders streaming; tests and their layout.
