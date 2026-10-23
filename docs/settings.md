# Settings

Everything Aria can be configured to do lives in one file: `~/.aria/settings.json` (or `$ARIA_HOME/settings.json`).
Secrets never go here. With `credentials.keyring` on (the default) they live in the system keyring (Keychain on macOS,
the Secret Service on Omarchy): `~/.aria/auth.json` and `mcp.json` hold `!aria secret …` commands in their place, and
`auth.json` and `mcp-auth.json` themselves are sealed in the keyring, with a live copy outside `~/.aria` while the dock
runs ([design](superpowers/specs/2026-10-21-keyring-design.md)). With it off, or with no keyring available, they are plain
files in `~/.aria` (Pi's formats, mode `600`).

Aria keeps keys it does not know when it saves, and refuses to overwrite a file it cannot parse.
The model lives in `crates/aria-core/src/settings.rs`; the defaults are the owner's own setup.

```json
{
  "version": 1,
  "setup": { "completed": false, "mode": null },
  "provider": { "id": null, "quota": null },
  "router": { "enabled": true, "backend": "opencode" },
  "assistant": { "userName": "", "name": "Aria", "language": "en", "style": "very-short" },
  "models": { "daily": null, "planning": null, "planningEffort": "high", "allowPayPerUse": false, "turnTimeoutMinutes": 10 },
  "effort": { "mode": "auto", "fixedLevel": null, "maxOnFree": true },
  "compaction": { "percent": 40 },
  "memory": { "enabled": true, "project": "aria" },
  "integrations": { "services": [], "packages": { "source": "gallery", "path": "", "thirdPartyTrusted": false }, "access": {} },
  "sandbox": { "projectsDir": "~/Projects", "extraWritable": [], "askToLeave": true, "jevReview": true },
  "git": { "commitWithoutAsking": false, "pushOrMergeWithoutAsking": false, "askBeforeOthersChanges": true, "unaskedAsProposals": true },
  "subagents": {
    "experts": [
      { "name": "frontend", "enabled": true, "readOnly": false },
      { "name": "backend", "enabled": true, "readOnly": false },
      { "name": "security", "enabled": true, "readOnly": true },
      { "name": "reviewer", "enabled": true, "readOnly": true },
      { "name": "researcher", "enabled": true, "readOnly": true }
    ],
    "saveSessions": false,
    "turnLimit": null,
    "planExecute": true,
    "autoSkills": ["systematic-debugging"]
  },
  "shortcut": "",
  "credentials": { "keyring": true },
  "display": { "verbose": true }
}
```

| Key | Meaning | Used from |
|---|---|---|
| `version` | Schema version, for migrations | phase 1 |
| `setup.completed`, `setup.mode` | Whether onboarding ran: `quick`, `full`, `skipped` or `null` | phase 1 |
| `provider.id` | `opencode-go`, `anthropic-api` or `anthropic-oauth` | phase 1 |
| `provider.quota` | `switchAt` / `planSwitchAt` (% of quota), `freeModel`, `askFirst`, `countdownMinutes`, `onNoAnswer` (`switch` or `stay`). `null` for providers without a readable quota or free tier | phase 2 |
| `router.enabled`, `router.backend` | Jev classifies each message; the backend is chosen from the stored keys, not from the main provider: `opencode` (OpenCode Go is the main provider or its key is stored; free Jev, even under a Claude plan), else `typesafe` (a TypeSafe key is stored: `typesafe/jev-latest`; TypeSafe gives Jev only, it is never a chat model), else the router, and Jev's sandbox review, are off. With both keys, `opencode` wins; the extensions then ask the first Jev Pi can reach (`lib/jev.ts`), so the backend decides whether the router is on | phase 2 |
| `assistant.*` | Your name, the assistant's name (`Aria`; the setup and the settings screen do not offer to change it), conversation language (`es`, `en`, `pt` or free text, named as written in the persona), answer style (`very-short`, `brief`, `detailed`, `technical`, `teaching`) | phase 1 (UI language), phase 2 |
| `models.*` | Daily and planning model, planning effort, whether pay-per-use models are allowed | phase 2 |
| `models.turnTimeoutMinutes` | A run that gets nothing from Pi (no event, reply or dialog) for this many minutes is stopped once, with a notice in the chat (messages you queued behind it are kept and run next); a running tool, a compaction or a dialog waiting for you pauses it; an unreadable settings file keeps the last limit read; `0` turns it off (default `10`) | the dock (checked every 30 s) |
| `effort.*` | `auto` (by route) or `fixed` at `fixedLevel` (`off` … `max`); `maxOnFree` runs free models at their top level | phase 2 |
| `compaction.percent` | Compact the context at this % of the window (0 turns it off) | phase 2 |
| `memory.*` | Engram on/off and its project name; before each Pi start, a project with no memory gets a first one, so Engram knows it | phase 2 |
| `integrations.services` | MCP services you turned on: `{ id, name, auth }`; for now `notion` (`oauth`) and `tavily` (`key`). They are connected during setup; their credentials live in `~/.aria/mcp.json` and `mcp-auth.json`, never here | phase 1 (connected), phase 2 (used) |
| `integrations.packages` | A local folder of extra Pi packages (`source: "local"`, `path`); the other sources and `thirdPartyTrusted` are not built | phase 2 |
| `integrations.access` | Access level per connector, keyed by its MCP server name: `read` (read only: writes are refused), `ask` (each write asks first) or `allow` (no questions). A connector not listed, or with an unknown level, is `ask`; `engram` is never gated. Written into `~/.aria/agent/aria.json` as `connectors` | phase 2 |
| `sandbox.*` | Projects folder, extra writable folders, ask before leaving the sandbox, let Jev review commands | phase 2 |
| `git.*` | Commit/push/merge without asking, ask before touching other people's changes, deliver unrequested work as proposals | phase 2 |
| `subagents.*` | Which experts are on and which are read-only, saving sessions, turn limit, whether `/plan` and `/execute` (slice) exist, automatic skills: the Superpowers skills loaded in every session, `systematic-debugging` by default (Ponytail is always on, not a setting) | phase 2 |
| `shortcut` | The dock's shortcut, modifiers then one key joined by ` + ` (`SUPER + SHIFT + A`): modifiers `SUPER`, `CTRL`, `ALT`, `SHIFT` on Linux, `Command`, `Control`, `Option`, `Shift` on macOS; the key a letter, a digit, `Space` or `F1`–`F12` (case and spaces do not matter). `""` (the default), or a value that does not parse, is the platform's default: `SUPER + A` on Omarchy, `Option + Space` on macOS. When the shortcut is taken (on Hyprland: another bind with the same modifiers and key in `hyprctl binds`; for a digit, any bind with the same modifiers that `hyprctl binds` lists with no key, since Omarchy's workspace binds by keycode show that way, so `SUPER + 1` counts as taken there), the dock tries the default, then `SUPER + SHIFT + A`, `SUPER + ALT + A` (macOS: `Control + Option + Space`, `Command + Option + Space`), and uses the first free one; with all of them taken it binds the first anyway on Hyprland, while on macOS a taken one (held by another app, or one of the system's own shortcuts such as Command + Space) is never registered, and with all of them taken the problem banner says so and the dock tries again every 10 s; while a fallback is in use it tries the chosen one (and the default) again every 10 s and switches back as soon as one is free; on Hyprland the row never lists the shortcut in use among the taken ones. A change applies within a second: the old bind goes and the new one comes (on macOS the new one is registered first, so a change with nothing free keeps the old one). The settings' Assistant section records it (press the keys; Esc cancels, Backspace goes back to the default), or takes it typed (for a combination the desktop takes before the page sees it) | the dock (Omarchy and macOS) |
| `credentials.keyring` | Keep keys in the system keyring instead of plain files (on by default). `aria keyring off` writes them back into plain files and turns this off; `aria keyring on` turns it back on, and the dock moves the keys at its next start | the dock |
| `display.verbose` | What a conversation shows: on (the default), the thinking and every tool with its output; off, one line per step with what it did and on what, without its output (`✓ Leer archivo  coupon.ts`), as the dock's accordion. `/verbose [on\|off]` and Ctrl+O in the terminal change it; the settings' Assistant section too | the terminal (the dock does not read it yet) |

Phase 1 asks and stores every key; phase 2 reads them.
