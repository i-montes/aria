# Settings

Everything Aria can be configured to do lives in one file: `~/.aria/settings.json` (or `$ARIA_HOME/settings.json`). You
rarely edit it by hand: the settings screen in the dock (the gear in its header) reads and writes it, and so do the setup,
`/effort` and `/autocompact` in a conversation, and `/verbose` in the terminal. Every key below is in the file; the screen
shows most of them.

Aria keeps keys it does not know when it saves, and refuses to overwrite a file it cannot parse (the dock shows a banner
and leaves the file as it is). The model is `crates/aria-core/src/settings.rs`, and the defaults are checked against the
golden files in `crates/aria-core/tests/golden/`. A key missing from your file gets its default. For where this file sits
among the others, and which are secret, see [data and paths](reference/data-and-paths.md).

A change applies while Aria runs. The service sees the file change within about a second. What the extensions read
(`agent/aria.json`) is picked up by each session when its run settles. What Pi reads only when it starts (the persona, the
experts, the packages, `mcp.json`) restarts that session's Pi when it changes.

Secrets never go here. With `credentials.keyring` on (the default) they live in the system keyring (Keychain on macOS,
the Secret Service on Omarchy): `~/.aria/auth.json` and `mcp.json` hold `!aria secret …` commands in their place, and
`auth.json` and `mcp-auth.json` themselves are sealed in the keyring, with a live copy outside `~/.aria` while the service
runs ([design](superpowers/specs/2026-10-21-keyring-design.md)). With it off, or with no keyring available, they are plain
files in `~/.aria` (Pi's formats, mode `600`).

## The default file

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

This is the file Aria's code starts from, with English as the language; the setup starts `assistant.language` from your
system's language and lets you change it. Choosing a provider in the setup fills in the rest, from the provider's own
defaults:

| Provider (`provider.id`) | `models.daily` | `models.planning` | `provider.quota` | Router | Effort | `sandbox.jevReview` |
|---|---|---|---|---|---|---|
| `opencode-go` | `opencode-go/mimo-v2.6-flash` | `opencode-go/mimo-v2.6-pro` | `98`, `90`, free model `opencode-go/cat-jam-free`, ask first, 5 minutes, `switch` | on, `opencode` | `auto` | on |
| `anthropic-api` | `anthropic/claude-sonnet-5-5` | `anthropic/claude-opus-5-5` | `null` | off unless a Jev key is stored | `fixed`, `medium` | off |
| `anthropic-oauth` | `claude-bridge/claude-sonnet-5-5` | `claude-bridge/claude-opus-5-5` | `null` | off unless a Jev key is stored | `fixed`, `medium` | off |

These come from `crates/aria-core/src/providers.rs` and the three `settings-provider-*.json` golden files. A model is
written `provider/id`, with the provider the way Pi files it (`claude-bridge` is the Claude sign-in).

## Assistant

The first section of the settings screen.

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `assistant.userName` | string | `""` | Your name. Aria uses it to greet you and in its replies; empty leaves it out of the persona. |
| `assistant.name` | string | `"Aria"` | The assistant's name in the persona. The setup and the settings screen do not offer to change it; editing the file does. |
| `assistant.language` | string | `"en"` | The conversation language: a code (`es`, `en`, `pt`) or free text such as `French`, named as written in the persona. The dock's own interface follows it for `en` and `es`; any other language shows English. The terminal does the same. Dictation understands only `es`: any other value, `pt` included, is listened to in English. |
| `assistant.style` | string | `"very-short"` | How long answers run: `very-short`, `brief`, `detailed`, `technical` or `teaching`. Written into the persona as one rule. |
| `shortcut` | string | `""` | The dock's shortcut, [below](#the-shortcut). `""` is the platform's default. The dock only: a hub ignores it. |
| `display.verbose` | boolean | `true` | What a conversation shows. On: the thinking and every tool with its output. Off: one line per step, with what it did and on what, without its output (`✓ Read file  coupon.ts`; `Leer archivo` in Spanish), as the dock's accordion. `/verbose [on\|off]` and `Ctrl+O` change it in the terminal, and so does this screen. The terminal reads it; the dock does not yet. |

### The shortcut

The value is modifiers, then one key, joined by ` + ` (`SUPER + SHIFT + A`). The modifiers are `SUPER`, `CTRL`, `ALT`,
`SHIFT` on Linux and `Command`, `Control`, `Option`, `Shift` on macOS. The key is a letter, a digit, `Space` or `F1` to
`F12`. Case and spaces do not matter, and a few aliases work (`cmd`, `opt`). Empty, or a value that does not parse, means
the default: `SUPER + A` on Omarchy, `Option + Space` on macOS.

When the shortcut is taken, the dock tries the default, then the fallbacks, and uses the first free one: `SUPER + SHIFT + A`
then `SUPER + ALT + A` on Omarchy, `Control + Option + Space` then `Command + Option + Space` on macOS. A change applies
within a second, and the old bind goes when the new one comes.

- **Omarchy.** Aria registers it with Hyprland (the Lua configuration is required) when it starts, after every config
  reload and after a change of the setting. A shortcut is taken when `hyprctl binds` lists another bind with the same
  modifiers and key. For a digit, any bind with the same modifiers that the list shows with no key counts, because
  Omarchy's workspace binds by keycode show that way, so `SUPER + 1` is taken there. With every candidate taken, Aria binds
  the first anyway. At start it waits until Hyprland can take the bind; after that it does not retry on a timer, so a
  registration that fails after a reload or a change is not tried again until the next one.
- **macOS.** Aria registers it with Carbon, which needs no Accessibility permission. A shortcut another app holds, or one of
  the system's own (such as `Command + Space`), is never registered. With every candidate taken, the problem banner says so.
  A failed registration is tried again every 10 seconds, and so is the chosen shortcut while a fallback stands in for it, so
  it comes back as soon as it is free. A change registers the new shortcut before the old one goes, so a change with nothing
  free keeps the old one.

The settings screen's Assistant section records the shortcut (press the keys; `Esc` cancels, `Backspace` goes back to the
default) or takes it typed, for a combination the desktop takes before the page sees it.

## Providers

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `provider.id` | string or `null` | `null` | The main provider: `opencode-go`, `anthropic-api` or `anthropic-oauth` (sign in with Claude). `null` until the setup runs. |
| `provider.quota` | object or `null` | `null` | What to do as the quota runs out. `null` for providers without a quota Aria can read or a free tier to fall back to (only OpenCode Go has one today). The fields are below. |
| `provider.quota.switchAt` | integer, percent | `98` | At this share of the quota (the five-hour or the weekly window), Aria moves to the free model. The screen offers 80 to 100. |
| `provider.quota.planSwitchAt` | integer, percent | `90` | The same threshold inside `/plan`, which switches earlier so a plan is never cut halfway. The screen offers 70 to 100. |
| `provider.quota.freeModel` | string | `opencode-go/cat-jam-free` | The model to move to. |
| `provider.quota.askFirst` | boolean | `true` | Show a card with a countdown before switching. Off: switch as soon as the threshold is reached. |
| `provider.quota.countdownMinutes` | integer | `5` | How long the card waits. The screen offers 1, 3, 5 and 10. |
| `provider.quota.onNoAnswer` | string | `"switch"` | What happens when nobody answers: `switch` or `stay`. |

## Models and effort

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `models.daily` | string or `null` | `null` (the provider's own, once chosen) | The everyday model, `provider/id`. It becomes Pi's default model. |
| `models.planning` | string or `null` | `null` (the provider's own, once chosen) | The model `/plan` uses. |
| `models.planningEffort` | string | `"high"` | The thinking level `/plan` runs at. The screen offers `low`, `medium` and `high`; a free model runs at its top level instead. |
| `models.allowPayPerUse` | boolean | `false` | List the models billed per request (OpenCode Zen, `opencode/*`) in the model pickers, in the dock and in the terminal. A model already in use always stays listed. |
| `models.turnTimeoutMinutes` | integer, minutes | `10` | A run that gets nothing from Pi (no event, reply or dialog) for this long is stopped once, with a notice in the chat; messages you queued behind it are kept and run next. A running tool, a compaction or a dialog waiting for you pauses the clock. The service checks every 30 seconds and reads the setting each time; an unreadable file keeps the last limit read. `0` turns it off. The screen offers 0 to 60, in steps of 5. |
| `effort.mode` | string | `"auto"` | `auto` (by the router's route) or `fixed` (always `effort.fixedLevel`). `auto` needs the router: without Jev the screen offers only fixed levels. |
| `effort.fixedLevel` | string or `null` | `null` | The level for `fixed`: `off`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`. The screen offers `low`, `medium`, `high` and `max`. `/effort <level>` in a conversation writes `effort.mode` and this. |
| `effort.maxOnFree` | boolean | `true` | In `auto`, run free models at their top level: they cost nothing, so thinking harder is free too. |

## Router (Jev)

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `router.enabled` | boolean | `true` | Whether Jev classifies each message, which picks the route and effort, read-only mode, the review of sandbox commands and the assignment of experts. Off: every message is handled the same way. |
| `router.backend` | string | `"opencode"` | Where Jev comes from. Chosen from the keys you stored, not from the main provider: `opencode` when OpenCode Go is the main provider or its key is stored (free Jev, even under a Claude plan); else `typesafe` when a TypeSafe key is stored (`typesafe/jev-latest`; TypeSafe gives Jev only, it is never a chat model); else the router, and Jev's sandbox review, are off. With both keys, `opencode` wins. The extensions then ask the first Jev Pi can reach, so the backend decides whether the router is on. |

## Context and memory

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `compaction.percent` | integer, percent | `40` | Compact the context when it reaches this share of the model's window. The screen offers 20 to 80, in steps of 5. `0` turns it off. `/autocompact` in a conversation writes it. |
| `memory.enabled` | boolean | `true` | Engram, Aria's memory, on or off. Off removes the memory rules from the persona and Aria's Engram entry from `mcp.json`. |
| `memory.project` | string | `"aria"` | The name memories are filed under, so searches stay scoped. Before each Pi start, a project with no memory gets a first one, so Engram knows it. |

## Integrations

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `integrations.services` | list of `{ id, name, auth }` | `[]` | The services you turned on. The screen offers `notion` (`auth: "oauth"`) and `tavily` (`auth: "key"`). They are connected during setup; their credentials live in `~/.aria/mcp.json` and `mcp-auth.json`, never here. |
| `integrations.access` | object, server name to level | `{}` | The access level per connector, keyed by its MCP server name: `read` (read only: writes are refused), `ask` (each write asks first) or `allow` (no questions). A connector not listed, or with an unknown level, is `ask`; `engram` is never gated. Written into `~/.aria/agent/aria.json` as `connectors`. |
| `integrations.packages.source` | string | `"gallery"` | Where extra Pi packages come from. Only `local` (and `path`, which earlier versions wrote) does anything; `gallery`, `npm` and `git` are not built. |
| `integrations.packages.path` | string | `""` | With `source` `local`: a folder of extra Pi packages, `~` allowed. It is listed in Pi's settings once. |
| `integrations.packages.thirdPartyTrusted` | boolean | `false` | Reserved: nothing reads it yet. |

## Sandbox and git

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `sandbox.projectsDir` | string | `"~/Projects"` | Where Pi works and where the sandbox lets commands write (besides the extra folders, `/tmp` and the tool caches). `~` allowed. A session opened from a terminal works in the folder it was opened in. On a hub `aria hub init` sets it to the hub's own folder. |
| `sandbox.extraWritable` | list of strings | `[]` | More folders commands may write to. Each is an absolute path or starts with `~/`; a relative one is dropped. |
| `sandbox.askToLeave` | boolean | `true` | Ask you before a command runs outside the sandbox, which the model can only do by saying so and giving a reason. Off: such a command is refused, never asked. Where there is no sandbox (macOS, or `bwrap` missing or blocked), every command counts as outside, so with this off none runs. |
| `sandbox.jevReview` | boolean | `true` | Jev explains what a command does before you decide. Needs the router. |
| `git.commitWithoutAsking` | boolean | `false` | Aria commits at the end of each task without asking. |
| `git.pushOrMergeWithoutAsking` | boolean | `false` | Aria may push and merge without asking (it never deletes branches without asking). Off: it asks before it pushes, merges or deletes a branch. |
| `git.askBeforeOthersChanges` | boolean | `true` | If the working tree has changes Aria did not make, ask before committing, stashing or discarding them. |
| `git.unaskedAsProposals` | boolean | `true` | Work nobody asked for arrives as a proposal to approve, not as a change. |

The first three `git.*` settings write the persona's Git rule, and `unaskedAsProposals` switches the persona's "Scope" rule: they instruct the model, they are not a lock.

## Subagents

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `subagents.experts` | list of `{ name, enabled, readOnly }` | the five below | The experts: `frontend` (screens and components), `backend` (endpoints and data), `security` (reviews for risks), `reviewer` (reads the final change) and `researcher` (looks things up). A disabled expert's file leaves `agent/agents/` (unless you edited it) and the persona does not name it. A read-only expert gets the tools `read, bash, codemode` only. `security`, `reviewer` and `researcher` are read-only by default. |
| `subagents.saveSessions` | boolean | `false` | Keep each subagent's session (in `agent/sessions/subagents/`) so you can open what an expert did. |
| `subagents.turnLimit` | integer or `null` | `null` | Stop a subagent after this many turns and make it write its report. `null` lets it finish. The screen offers 5 to 100, and turns on at 30. |
| `subagents.planExecute` | boolean | `true` | Whether `/plan` and `/execute` ([slice](guide/agent.md)) exist, and the persona's planning rules. |
| `subagents.autoSkills` | list of strings | `["systematic-debugging"]` | The Superpowers skills loaded in every session. The screen offers `brainstorming`, `writing-plans`, `test-driven-development`, `systematic-debugging`, `requesting-code-review` and `verification-before-completion`. Ponytail is always on and is not a setting. |

## Setup and credentials

Not on the settings screen, except as noted.

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `version` | integer | `1` | The schema version. Aria writes it; nothing reads it yet. |
| `setup.completed` | boolean | `false` | Whether the setup ran. |
| `setup.mode` | string or `null` | `null` | How: `quick`, `full`, `skipped` or `null`. |
| `credentials.keyring` | boolean | `true` | Keep keys in the system keyring instead of plain files. `aria keyring off` writes them back into plain files and turns this off (Aria must be stopped); `aria keyring on` turns it back on, and the service moves the keys at its next start. On a hub it is `false`. |

## On a hub

A hub reads the same file, from the same place on the server. `aria hub init` sends your `settings.json` once, with
`credentials.keyring` set to `false` and `sandbox.projectsDir` set to the hub's folder; after that the hub's copy is its
own (`--settings` sends yours again). `shortcut` and the dock's settings screen mean nothing on a hub: you change its
settings by editing the file there ([the hub](guide/hub.md)). A terminal attached to a hub cannot read or change settings through its protocol; the commands `/autocompact` and `/effort`, typed in a session, still rewrite their own setting in the hub's file.

Planned, not built (milestone M2 of [the hub design](superpowers/specs/2026-10-23-hub-and-machines-design.md)): a computer
linked to a hub would take provider and models, persona and experts, language and integrations from the hub, shown
read-only on its settings screen, and keep its own shortcut, projects folder, sandbox folders, dictation, keyring and face.
The exact split is not decided. Today every computer reads only its own file.

## See also

- [Data and paths](reference/data-and-paths.md) for where `settings.json` and the files it points to live.
- [The agent](guide/agent.md) for providers, the router, experts, effort and compaction.
- [Safety](guide/safety.md) for the sandbox, connector access levels and the keyring.
- [Adding a provider](adding-a-provider.md) for how a provider module declares its defaults.
