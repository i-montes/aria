# Settings

Everything Aria can be configured to do lives in one file: `~/.aria/settings.json` (or `$ARIA_HOME/settings.json`).
Secrets never go here; they go in `~/.aria/auth.json` (mode `600`, same format as Pi's `auth.json`), or for Notion and Tavily in `~/.aria/mcp.json` and `mcp-auth.json` (Pi's MCP formats, mode `600`).

Aria keeps keys it does not know when it saves, and refuses to overwrite a file it cannot parse.
The model lives in `crates/aria-core/src/settings.rs`; the defaults are the owner's own setup.

```json
{
  "version": 1,
  "setup": { "completed": false, "mode": null },
  "provider": { "id": null, "quota": null },
  "router": { "enabled": true, "backend": "opencode" },
  "assistant": { "userName": "", "name": "Aria", "language": "en", "style": "very-short" },
  "models": { "daily": null, "planning": null, "planningEffort": "high", "allowPayPerUse": false },
  "effort": { "mode": "auto", "fixedLevel": null, "maxOnFree": true },
  "compaction": { "percent": 40 },
  "memory": { "enabled": true, "project": "aria" },
  "integrations": { "services": [], "packages": { "source": "gallery", "path": "", "thirdPartyTrusted": false } },
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
    "autoSkills": ["systematic-debugging", "verification-before-completion"]
  }
}
```

| Key | Meaning | Used from |
|---|---|---|
| `version` | Schema version, for migrations | phase 1 |
| `setup.completed`, `setup.mode` | Whether onboarding ran: `quick`, `full`, `skipped` or `null` | phase 1 |
| `provider.id` | `opencode-go`, `anthropic-api` or `anthropic-oauth` | phase 1 |
| `provider.quota` | `switchAt` / `planSwitchAt` (% of quota), `freeModel`, `askFirst`, `countdownMinutes`, `onNoAnswer` (`switch` or `stay`). `null` for providers without a readable quota or free tier | phase 2 |
| `router.enabled`, `router.backend` | Jev classifies each message; backend `opencode` (through the provider) or `typesafe` (direct, needs a TypeSafe key) | phase 2 |
| `assistant.*` | Your name, the assistant's name (`Aria`; the setup and the settings screen do not offer to change it), conversation language (`es`, `en`, `pt` or free text), answer style (`very-short`, `brief`, `detailed`, `technical`, `teaching`) | phase 1 (UI language), phase 2 |
| `models.*` | Daily and planning model, planning effort, whether pay-per-use models are allowed | phase 2 |
| `effort.*` | `auto` (by route) or `fixed` at `fixedLevel` (`off` … `max`); `maxOnFree` runs free models at their top level | phase 2 |
| `compaction.percent` | Compact the context at this % of the window (0 turns it off) | phase 2 |
| `memory.*` | Engram on/off and its project name; before each Pi start, a project with no memory gets a first one, so Engram knows it | phase 2 |
| `integrations.services` | MCP services you turned on: `{ id, name, auth }`; for now `notion` (`oauth`) and `tavily` (`key`). They are connected during setup; their credentials live in `~/.aria/mcp.json` and `mcp-auth.json`, never here | phase 1 (connected), phase 2 (used) |
| `integrations.packages` | Where extra Pi packages come from, and whether third-party ones are trusted | phase 2 |
| `sandbox.*` | Projects folder, extra writable folders, ask before leaving the sandbox, let Jev review commands | phase 2 |
| `git.*` | Commit/push/merge without asking, ask before touching other people's changes, deliver unrequested work as proposals | phase 2 |
| `subagents.*` | Which experts are on and which are read-only, saving sessions, turn limit, `/plan` and `/execute`, automatic skills | phase 2 |

Phase 1 asks and stores every key; phase 2 reads them.
