# Everything that makes Aria Aria today

Snapshot of the Pi setup behind Aria, as of 2026-10-18. Everything lives in `~/.pi/agent/` unless a path says otherwise. No secrets are stored here.

## A. Install and dependencies

1. **Pi 1.3.0** from the official installer at `~/.pi/agent/bin/pi`; update with `pi update`. Do not install it through mise.
2. **System packages:** bubblewrap (`bwrap`), bun, iproute2 (`ss`), Node via mise (for `rea` and `pi-acp`).
3. **engram 3.3.2** in `~/.local/bin`.
4. **Claude Code** (2.1.299 today): required by the Claude Max provider.
5. **Pi packages** (installed from `settings.json`):
   - `pi-parallel-agents`;
   - `superpowers` from git, filtered: no extensions, only the `systematic-debugging` and `verification-before-completion` skills;
   - the local repo `~/Projects/meeting`;
   - `pi-claude-agent-sdk` (v0.9.2), the `claude-bridge` provider.
6. **OpenPencil** (optional): `op`, `openpencil-desktop`, `pi-acp` registered in OpenPencil, and the skill via `op install --target codex`.

## B. Defaults

7. **Daily model:** `opencode-go/deepseek-v4.1-flash`. Never `opencode/*` (Zen, pay per use) unless the user names the model.
8. **Classifier:** `opencode/jev-1.13-free`, exact id (`jev-1.13` is paid).
9. **`/plan` model:** `opencode-go/glm-5.3` at `high` effort.
10. **Free fallback model:** `opencode-go/cat-jam-free`, from 98% of the Go quota (90% in `/plan`), always asking first. Only OpenCode Go has a free tier, so only it offers fallback models.
11. **Claude Max:** provider `claude-bridge`, settings in `claude-bridge.json` (Max plan by default).
12. **Compaction:** 40% by default; **90% today** (`autocompact.json`).
13. **Effort:** auto (no `jev-router-effort.json`).
14. **`trust.json`:** trusts `~/Projects/aria`.

## C. Customizations

15. **Persona** (global `SYSTEM.md`, in English):
    - talks to the user in Spanish;
    - answers in 2-4 sentences, no tables, no headings, no closing summary;
    - "Way of working": scope → proposals; risks starting with the most frequent case; ask before commit, stash or discarding someone else's changes; never push or merge;
    - only stop its own processes.
16. **Memory:** Engram with `--project aria` and 7 direct tools. `~/.engram/engram.db` (53 memories) is Aria's memory.
17. **MCP servers:** engram, notion (OAuth), tavily (key, `chmod 600`), rea, openpencil, figma (OAuth).
18. **jev-router:**
    - hidden hint every turn (`route=… memory=… web=… notion=… changes=…`) plus the short-answer reminder;
    - blocks writes outside `/tmp` in read-only turns;
    - assigns experts and blocks `pkill`, `killall`, `fuser -k`.
19. **Effort:** in auto the route decides, and free models run at their top level. A level picked by hand (`/thinking`, `shift+tab` or `/effort <level>`) stays until `/effort auto`.
20. **Superpowers modes:**
    - `/plan` only writes to `docs/superpowers` of repos under `~/Projects` and turns itself off;
    - `/execute` requires a plan, not a spec.
21. **Generic experts:** `frontend` and `backend` write; `security`, `reviewer` and `researcher` are read-only.
22. **Subagents:**
    - use `SUBAGENT.md`, read memory without writing it, end with `## Learnings`;
    - no turn limit, no saved sessions; watch them with `alt+a` or `/agents`.
23. **`ask_user`:** form with 1-4 questions, recommended option first. Answering lifts read-only mode for the turn.
24. **Sandbox:**
    - writes only to the project, `/tmp` and download caches; credentials are hidden;
    - leaving it asks, showing the model's description and Jev's review;
    - session grants by program + subcommand; interpreters only as an exact command;
    - `/sandbox` and `/sandbox reset`.
25. **Go quota:** one question with a 5-minute countdown, shared by the chat, `/plan` and subagents.
26. **Compaction:** `/autocompact <0-100>`.
27. **Language:** code, commands and prompts in English; the conversation in Spanish.
28. **Switches:** `PI_SANDBOX=0`, `PI_QUOTA_FALLBACK=0`, `PI_SUBAGENT_LEAN=0` turn features off; `CLAUDE_BRIDGE_DEBUG=1` logs the Claude provider.

## D. Secrets (log in again on a new machine; don't copy them)

29. **`auth.json`:** the OpenCode Go key and the **Anthropic OAuth** (Max account). Never export `ANTHROPIC_*` variables: they break the provider.
30. **`mcp.json`:** the Tavily key.
31. **`mcp-auth.json`:** Notion and Figma OAuth.

## E. Hardcoded paths to adjust

32. **`mcp.json`:** the `engram` and `rea` paths (the latter includes the mise Node version).
33. **`~/Projects`:** assumed by plan mode and the `meeting` package.
34. **`eval.ts`:** uses `checkout-widget` paths; only affects tests.

## F. Checks

35. **Tests:** `bun test` in every extension that has tests.
36. **Evals:** `bun eval.ts` (jev-router); `bun review-eval.ts` and `bash accept.sh` (sandbox).
37. **Smoke test:** `pi -p "hola"` must answer short and in Spanish; with a `claude-bridge` model in `/model` it must answer using the Max plan.
