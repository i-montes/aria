# Aria base — design

Date: 2026-10-18. Status: approved by the owner (2026-10-18); phase 1 built.

## Intent

Aria is a personal desktop assistant: a dock that opens over whatever you are doing, talks to you in your language and works on your files and tasks, safely. It is a **public** project, so its first quality is legibility: anyone should be able to open the repo, understand how it is organised and change one part without reading the rest.

The owner has spent the last days building and testing a harness on top of Pi (the coding agent) — router, sandbox, quota handling, effort control, compaction, questions, experts, plan/execute, memory, Claude Max. `ARIA-SETUP.md` in this repo is the inventory of that harness. Aria is the next stage: the same harness, owned by one project, with its own desktop presence and onboarding, and later new abilities on top.

## The rule that governs everything

**Aria must work at least as well as the owner's current Pi setup** (everything in `ARIA-SETUP.md`). New abilities wait until that parity exists. A beautiful Aria that does less than today's Pi is a failure.

## Roadmap

1. **Base, dock and setup** (this document, detailed below): the repo structure, the dock that opens with a hotkey on the focused monitor, and the onboarding that writes `~/.aria/settings.json`.
2. **Parity with today's Pi:** Aria ships its own pinned Pi and the owner's extensions inside the repo; the dock becomes the chat; every item of `ARIA-SETUP.md` works from Aria and reads its configuration from `settings.json`.
3. **Autonomous goals:** a fork of `pi-goal-x` (MIT, github.com/tmonk/pi-goal-x). A goal runs until it is finished or the owner stops it — no run limit — resumes by itself after a reboot or crash, and by default works in its own git worktree inside the sandbox, never in the real checkout. The result stays on the worktree's branch for the owner to review; Aria never merges by itself.
4. **State socket and external extensions:** every event (messages, tool calls, permissions, goals, quota, compaction…) is published on a local socket; an extension is any program, in any language, that connects to it, listens, and can register tools the agent can call (example: an automatic note-taker that writes notes from conversations and offers `notes_search`). Extension tools obey the same sandbox and permission rules as everything else.
5. **Mac**, then Linux beyond Hyprland, then Windows. (macOS started on 2026-10-18, ahead of phases 3 and 4 at the owner's request: [design](2026-10-18-aria-mac-design.md).)

Phases 2-5 are listed so the base leaves room for them; they get their own designs.

## Decided by the owner

- **Language: Rust** for Aria itself. Pi (TypeScript) stays the agent engine inside Aria, and Aria ships its own copy of it: it must not depend on a Pi installed by the user or on `~/.pi/agent`.
- **Platforms:** Omarchy (Hyprland) first and only at first; Mac next; Linux and Windows later. On Mac, use whatever is native and compatible. (The macOS platform layer is designed in [2026-10-18-aria-mac-design.md](2026-10-18-aria-mac-design.md).)
- **Data lives in `~/.aria/`**, never in the repo. The repo only holds code. Aria's sandbox/workspace is under `~/.aria/` too.
- **The configuration file is `settings`** (not "preferences"): one place for every setting of the project.
- **Hotkey:** `SUPER + A` on Omarchy; on Mac a free, easy shortcut (proposal accepted: `Option + Space`).
- **Opening the dock takes keyboard focus** on every platform, so you can type at once or move through options with the arrow keys.
- **No manual steps in the user's Hyprland config** (the previous prototype, proto-1, required pasting three blocks; that is what we avoid).
- **Design:** `~/Downloads/Aria.html` (to be copied into the repo, see below) is the visual and interaction reference. Its structure and code are not to be copied from proto-1, and proto-1's design is not used.
- **Onboarding providers at the start:** OpenCode Go, Anthropic with an API key, Anthropic with OAuth sign-in.
- **Only providers with a free tier offer fallback models** (today only OpenCode Go). For the others, quota is shown and warned about, with no fallback.
- Everything built — code, names, settings keys, docs, commits — is in English; the conversation with the user follows their chosen language.

## Phase 1 — base, dock and setup

### What the owner gets

1. Aria installs as a user service that starts at login. Nothing is written to `~/.config/hypr/`.
2. The first time, a notice appears in the corner (design: "Aria is installed — Press Super A to set it up").
3. `SUPER + A` opens the dock on the focused monitor, with the design's animations, and focuses it.
4. The dock shows the welcome screen: **Quick setup** (provider, sign-in and your name; recommended defaults for the rest), **Customize** (all steps) and **Skip for now**.
5. Finishing either path writes `~/.aria/settings.json` and stores credentials in `~/.aria/auth.json`. Re-opening the setup later shows the current values and changes them.
6. `SUPER + A` or `Esc` closes the dock.

There is no chat and no agent run in phase 1. After setup the dock shows a plain "Aria is set up" state; phase 2 replaces it with the chat.

### The onboarding steps

Order and content as agreed with the owner. Defaults in parentheses are the owner's current setup; Quick setup asks steps 1, 2 and the user's name from step 4, and takes the defaults for everything else.

1. **Provider:** OpenCode Go | Anthropic (API key) | Anthropic (sign in). Only these three in phase 1, even though the prototype shows more (OpenAI, OpenRouter, Local, Other); OpenCode Go uses its API key only. Each provider is a module that declares what it supports: its models, whether it has a readable quota and how to read it, and which free models it offers. Adding a provider later means adding a module. **Quota** questions appear only if the module supports it (today OpenCode Go): threshold to switch to the free model (98%, 90% for planning), which free model, whether to ask first, countdown length (5 min) and what happens with no answer (switch).
2. **Sign-in:** paste the key, or sign in with OAuth. Stored in `~/.aria/auth.json`, mode `600`. "Anthropic (sign in)" is the Claude Pro/Max OAuth sign-in that Pi's `/login` does today (browser + localhost callback, tokens `access`/`refresh`/`expires`); it is what the owner's Claude Max route uses at run time (`pi-claude-agent-sdk` reads that credential and ignores Claude Code's own login). Phase 1 performs the sign-in and stores the tokens; using them (through the bundled Claude Code bridge) is phase 2.
3. **Router with Jev:** do you want a router that classifies every message? Explain what it brings (automatic route and effort, read-only turns, sandbox command review, expert assignment). If the provider is OpenCode Go, say Jev is already included with that account and ask nothing more; with any other provider, ask for the TypeSafe API key (TypeSafe owns Jev). If declined, everything that depends on the router is off.
   - TypeSafe facts (docs.typesafe.ai, quickstart): keys at `https://console.typesafe.ai/keys`; `POST https://api.typesafe.ai/v1/systemone` with `Authorization: Bearer <key>`; model `"jev-latest"` (responses report a versioned id such as `jev-1.13.0`); the request is a `state` plus named `questions` of type `choice`, `score` or `noul` (a 0-1 truth value, TypeSafe's equivalent of the yes/no questions Pi's classifier calls `bool`); SDKs `@typesafe-ai/sdk` and `typesafe-sdk`.
   - So the router has two backends behind one interface: through OpenCode (today: Pi's classifier with `opencode/jev-1.13-free`) and direct to TypeSafe. Phase 1 stores the key (`auth.json`, provider `typesafe`) and validates it with one minimal `systemone` call; the router itself is phase 2.
4. **Assistant:** your name; the assistant's name (Aria); conversation language: Español | English | Português | Other (type); answer style: Very short and direct, no tables | Brief, with lists when they help | Detailed, with explanations | Technical and precise, with code and paths | Teaching, step by step.
5. **Models:** daily model; planning model and its effort; whether pay-per-use models are allowed. Options come from the provider module.
6. **Effort:** auto by route (only with the router) or a fixed level; top level on free models.
7. **Compaction:** at what % of the context (40%).
8. **Memory:** Engram on or off, and its project name.
9. **Integrations:** services and Pi packages.
   - **Services (MCP), configured right there.** Only **Notion** and **Tavily** for now (the owner's call; more later). Turning one on in this step also connects it, and Continue stays closed until it is: **Tavily** asks for its API key, checks it with Tavily (`GET https://api.tavily.com/usage`, free; Tavily's MCP endpoint accepts any key, so it cannot validate one) and stores it in `~/.aria/mcp.json`; **Notion** signs in through the browser (MCP OAuth: discovery from the server's 401, a public client registered by dynamic registration as "Aria", PKCE, loopback callback on `127.0.0.1`) and stores the result in `~/.aria/mcp-auth.json`. To skip a service, turn it off. Both files are in **Pi's own formats** (`mcp.json` with `mcpServers`, `mcp-auth.json` keyed `mcp__<name>|<url>`) and mode `600`; Pi was checked to read them (`pi mcp list` with `PI_CODING_AGENT_DIR` pointing at a copy). How the bundled Pi of phase 2 is pointed at `~/.aria` (Pi keeps its own `settings.json` in its agent directory, which would collide with Aria's) is a phase-2 decision.
   - **Pi packages:** gallery, npm, git or a local folder. Packages that chosen features need are installed without asking; a third-party package shows its source and asks before installing, since packages run code. (Recorded in phase 1, installed in phase 2.)
10. **Sandbox:** projects folder (`~/Projects`); extra writable folders; whether leaving the sandbox asks; whether Jev reviews the command (router only).
11. **Git and way of working:** commit without asking; push or merge; ask before touching someone else's changes; deliver what wasn't asked for as proposals.
12. **Subagents and planning:** which experts, which are read-only; save subagent sessions; turn limit; `/plan` and `/execute` on, and which auto skills.

Steps whose features only exist from phase 2 on are still asked and saved now, so phase 2 just reads them. In particular, phase 1 **records** the Pi-package choices but does not install them; that happens in phase 2 with the bundled Pi. (Notion and Tavily are the exception: they are configured in the setup, see step 9.)

**Defaults:** where the prototype's defaults differ from the owner's current setup, the owner's setup wins (parity rule): no subagent turn limit, subagent sessions not saved, ask before committing, only `systematic-debugging` and `verification-before-completion` as auto skills. Compaction defaults to 40% (agreed as the default; the owner's own value today is 90%).

### Settings

- `~/.aria/settings.json`, one JSON document, with a `version` field so later phases can migrate it. Keys in English, grouped by the steps above (provider, router, assistant, models, effort, compaction, memory, integrations, sandbox, git, subagents).
- Written atomically (temp file + rename). If the file exists but is invalid, the dock says so and does not overwrite it.
- Keys Aria does not know are kept untouched when it writes the file.
- Secrets never go in `settings.json`; they go in `auth.json` (`600`).
- `auth.json` uses **Pi's exact format** (keyed by provider: `{type: "api_key", key}` or `{type: "oauth", access, refresh, expires}`), so phase 2's bundled Pi reads it directly. Other secrets (router key, service keys, MCP OAuth tokens) also live under `~/.aria/` in the formats Pi expects for them, never in `settings.json`.

### `~/.aria/` layout

- `settings.json`, `auth.json`, and, once Notion or Tavily is set up, `mcp.json` and `mcp-auth.json` (Pi's formats, mode 600)
- room for phase 2+: sessions, logs, the workspace where Aria works, the bundled Pi's agent directory.

The exact sub-folders beyond the two files are the builder's call, documented in the repo.

### The dock on Omarchy (learned in `spikes/dock-open/`)

- **Requirement:** Hyprland with the Lua configuration (0.56+), which is what current Omarchy uses. The classic hyprlang config (`hyprctl keyword bind`) is not supported in phase 1; Aria detects it and says so.
- GTK3 + gtk-layer-shell + WebKitGTK 4.1 (WebKitGTK 6 is not installed on the target machine). The dock UI is a web page; the same page will serve Mac later.
- **Hotkey without editing Hyprland's config:** at start, and again on every `configreloaded>>` event from Hyprland's event socket, run `hyprctl eval` with `hl.unbind("SUPER + A")` followed by `hl.bind("SUPER + A", hl.dsp.exec_cmd("<aria> toggle"), { description = "Toggle Aria" })`, plus the layer rule for the dock's namespace. Hyprland here is 0.56, configured in Lua.
- **Single instance:** a Unix socket under `$XDG_RUNTIME_DIR`; `aria toggle` starts Aria if it is not running.
- **Focused monitor:** on every open, read `j/monitors` from Hyprland's IPC socket and move the surface to the GDK monitor with the same logical position before mapping it.
- **Two WebKit gotchas:** set `WEBKIT_DISABLE_DMABUF_RENDERER=1` before GTK starts, and size the window with `set_size_request` (not `set_default_size`), or the layer never paints.
- **Shape:** a 700 px surface anchored right/top/bottom: the veil (an eased tint, `rgba(4,4,5,.94)` behind the dock column and gone ~100 px past it, with no backdrop blur: the design's blur needs compositor support and the owner approved the tint-only look). Buttons and cards use solid backgrounds that do not change on hover, focus or selection; only the border takes the accent and a 470 px dock column; only the column takes input.
- **Animations (measured from the design):** open — veil opacity 180 ms ease-out; dock opacity 200 ms and translateX 24 px → 0 in 240 ms, `cubic-bezier(0.16, 1, 0.3, 1)`; content enters translateX 16 px in 300 ms, `cubic-bezier(0.22, 1, 0.36, 1)`; accent dot glow 2.4 s loop. Close — dock opacity 120 ms / translate 140 ms, veil 140 ms after 20 ms, `cubic-bezier(0.4, 0, 1, 1)`; the surface is unmapped only after the close transition, and re-opening mid-close cancels the hide.
- **Focus:** open with on-demand keyboard (Hyprland focuses the surface when it maps, so typing and arrow keys work at once, and a click on another window takes the keyboard away); release the keyboard as soon as closing starts. Exclusive was tried and rejected: it blocked focusing any other window.
- The design's backdrop blur (12 px) is not in phase 1: it would need Hyprland's layer blur, which the owner had turned off before. Open question for later.
- **Autostart:** a systemd user unit installed by Aria's installer, not a Hyprland autostart line.
- **Install (decided by the owner):** one command, `curl -fsSL https://<owner's site>/install.sh | sh`, on a machine where Aria does not exist yet. The script (kept in the repo) checks it is on Omarchy/Hyprland with the Lua config, installs the system libraries Aria needs (GTK3, WebKitGTK 4.1, gtk-layer-shell) through pacman, downloads the prebuilt `aria` binary for the platform and verifies its checksum, puts it in `~/.local/bin/`, creates `~/.aria/`, writes and starts the systemd user unit, and leaves Aria running so the "Aria is installed — Press Super A" notice appears. Running it again updates Aria in place. An uninstall path (script flag) stops and removes the unit, the runtime bind and the binary, and asks before deleting `~/.aria/`.
- **In phase 1** the same script also accepts a locally built binary (development mode), because nothing is hosted yet. Hosting the script on the owner's site and the release pipeline that publishes binaries come later; phase 1 only needs the script to be ready for it.
- The "Aria is installed" notice shows on start whenever `~/.aria/settings.json` does not exist yet.

### Repo structure

Organised so each concern is one place a newcomer can find:

- a Rust workspace with a platform-independent core (settings model and validation, paths, provider modules, onboarding logic) and a platform layer behind a small interface (hotkey, floating panel, autostart, focused monitor), with the Hyprland implementation as its first member;
- the dock UI as its own folder (web page, design tokens, screens);
- `design/` holding the design reference (`Aria.html`) so the repo is self-contained;
- `docs/` with how the pieces fit, and how to add a provider or a platform;
- room for phase 2's bundled Pi and extensions.

The dock's interface text comes from a string catalog from day one, with English and Spanish; the UI language follows the language chosen in step 4 (system locale before setup).

Crate and folder names, the UI's build setup (plain web files or a bundler), and the exact JSON shape of `settings.json` are the builder's call, as long as the result stays easy to read and is documented.

## Goals, non-goals, anti-goals (phase 1)

- **Goals:** the hotkey opens the dock on the focused monitor with the design's look and motion; the onboarding follows the design and the 12 steps; `settings.json` and `auth.json` are written correctly and can be re-edited; installing needs no Hyprland edits.
- **Non-goals:** chat, running the agent, goals, the external extension socket, Mac, blur.
- **Anti-goals:** anything that requires editing `~/.config/hypr/`; secrets outside `auth.json`; overwriting a settings file Aria could not parse; a structure only its author understands; anything that would make phase 2 harder to reach parity with today's Pi.

## Left to the builder

Names of crates, modules and folders; the UI's tooling; the internal message format between the Rust side and the dock page; how to validate a key at sign-in (a cheap request is enough); tests and their layout.
