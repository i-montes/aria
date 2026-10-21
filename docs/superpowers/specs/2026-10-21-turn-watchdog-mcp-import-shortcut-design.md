# Turn watchdog, MCP import, and the shortcut

Date: 2026-10-21. Ideas 3, 5 and 6 of `docs/ideas.md`. The owner chose not to be asked ("just do it"), so the
decisions below are the builder's unless marked **(owner)**.

## Part A — A turn that stops answering (idea 3)

### What is already there (Pi 1.3.0)

- Provider requests are retried: `retry` (3 attempts, base delay 2 s, provider retry delay up to 60 s). Pi also has
  an HTTP idle timeout setting (`httpIdleTimeoutMs`).
- MCP: an HTTP server's connection is retried twice (250 ms, 1 s) on a transient error. A server that failed
  connects again on its next use (`extensions/mcp/runtime.js` `open`/`getClient`). Pi gives extensions no way to
  reconnect a server, so Aria adds **no** MCP retry of its own **(call)**. `docs/ideas.md` and `docs/pending.md` say so.

### The watchdog **(call)**

A run that gets no event from Pi for too long is stopped, and the person is told. That covers a provider stream that
hangs without closing: the watchdog covers the model being silent, never a tool (tools have their own timeouts).

- The dock's `Chat` notes the time of the last event of the current run (every `Event`, response and UI request
  counts). Every 30 s, if a run is going, no dialog is waiting for the person, and the last event is older than the
  limit, the dock sends Pi `abort` once and posts a `ui_notice` (warning): "Aria stopped this turn: nothing came back
  for {minutes} minutes." / "Aria detuvo este turno: no llegó nada en {minutes} minutos." (the page translates the
  notice from a key: `chat.watchdog.stopped` with `minutes`).
- A running tool pauses the watchdog (tool executions are tracked from `tool_execution_start` to `tool_execution_end`
  by `toolCallId`, nested ones too): a long silent build or test (`cargo build`, `bun test`, `sleep`) sends no events
  for minutes and must not be cut. The silence clock starts again when the last tool ends. A dialog waiting for the
  person pauses the watchdog the same way.
- Setting: `models.turnTimeoutMinutes` (number, default 10; 0 turns the watchdog off), in Settings → Models next to
  the effort, with the same widgets as the other numbers.

## Part B — Import the MCP servers the person already has (idea 5)

### Where they live

| Source | Linux | macOS | Key |
|---|---|---|---|
| Claude Code | `~/.claude.json` | same | `mcpServers` (user scope) |
| Claude Desktop | `~/.config/Claude/claude_desktop_config.json` | `~/Library/Application Support/Claude/claude_desktop_config.json` | `mcpServers` |
| Cursor | `~/.cursor/mcp.json` | same | `mcpServers` |
| VS Code | `~/.config/Code/User/mcp.json` | `~/Library/Application Support/Code/User/mcp.json` | `servers` |

An entry is either `{ command, args?, env? }` (stdio) or `{ url | serverUrl, headers?, type? }` (http / sse). Anything
else is skipped. Project-scoped Claude Code servers (`projects.*.mcpServers`) are not imported **(call)**: they belong
to a project.

### The step **(owner: a step of the onboarding)**

- In the onboarding's Integrations step, below Notion and Tavily, a section "Your other MCP servers" lists what was
  found: a checkbox per server with its name, where it came from (Claude Code, Cursor…), and what it runs (the command
  name or the URL's host). Servers already in Aria's `mcp.json` under the same name, and `engram`, are not listed.
  The same list sits in Settings → Integrations. Nothing is found → the section says so in one line.
- "Import" copies the checked ones into Aria's `mcp.json`. A name already taken gets `-2`, `-3`… Secrets in `env` or
  `headers` are copied as they are (the file is mode 0600, like today); the page never receives their values, only
  the names of the variables and headers (shown as "uses SECRET_NAME").
- After an import every Pi restarts when its run settles (the existing `enginesRestart`), and the access rows (Part 1
  of the connectors work) show the new servers at Ask.
- No import happens without the person's click. Nothing is ever written to the other apps' files.

### Messages

- `PageMessage::McpScan { id }` → `DockMessage::McpFound { id, servers: [{ id, name, source, kind: "stdio"|"http",
  target, secrets: [names] }] }` (`id` = `source:name`).
- `PageMessage::McpImport { id, ids: [..] }` → `DockMessage::Saved { id }` or `Error { id, code }` (`mcp-file-broken`,
  `nothing-to-import`).

## Part C — Choose the shortcut (idea 6)

- Setting: `shortcut` (string, `""` = the platform's default: `SUPER + A` on Omarchy, `Option + Space` on macOS).
  Format: modifiers then one key, joined by ` + ` (`SUPER + SHIFT + A`). Modifiers: `SUPER`, `CTRL`, `ALT`, `SHIFT`
  on Linux; `Command`, `Control`, `Option`, `Shift` on macOS. Keys: a letter, a digit, `Space`, `F1`–`F12`.
- Settings → a new "Shortcut" row in the first section of the settings (the general one): a field that records the
  keys pressed while it has focus (Esc cancels, Backspace resets to the default), shows them as `<kbd>`s, and saves.
- **Fallbacks:** when the chosen shortcut is taken, the dock tries, in order: the chosen one, then the default, then
  `SUPER + SHIFT + A`, `SUPER + ALT + A` (macOS: `Control + Option + Space`, `Command + Option + Space`). The first
  that works is used; the page shows it (the existing `DockMessage::Platform { hotkey }`, now sent again whenever it
  changes) and the settings row says "In use: X (Y was taken)".
- **Taken, on Hyprland:** a bind with the same modifiers and key that is not Aria's own, read from `hyprctl -j binds`.
  On macOS: `RegisterEventHotKey` fails.
- A change applies at once: the dock unbinds its old shortcut and binds the new one (Hyprland), or unregisters and
  registers again (macOS). On Hyprland the bind is still re-applied after every config reload.
- The macOS part is written carefully but cannot be compiled here. `docs/pending.md` says it waits for a build on the
  Mac **(call)**. The parsing and the fallback order live in a platform-free module with tests.

## Testing

- Watchdog: a fake Pi that starts a run and then stays silent → `abort` sent once after the limit (the limit set to
  seconds in the test); a dialog open → no abort; `0` → off.
- Import: each source parsed from fixtures (stdio, http, sse, unknown shapes skipped, secrets reduced to names),
  names deduplicated, already-present names hidden, the write keeps existing servers and mode 0600.
- Shortcut: parsing (valid, invalid, case), the Hyprland bind Lua for a given shortcut, taken detection from a
  `binds` JSON fixture, the fallback order, the Platform message after a change; the settings row's key capture.
- UI: the import section (list, checkboxes, import button, nothing found), the shortcut row, the watchdog notice;
  browser checks for import and the shortcut capture.

## Docs

`docs/settings.md` (`models.turnTimeoutMinutes`, `shortcut`), `docs/architecture.md` (messages, the hotkey section),
`docs/pending.md` (macOS shortcut not built here; no MCP reconnect API), `docs/ideas.md` (3, 5, 6 done, with the MCP
note), `CHANGELOG.md` (nothing new outside the repo besides `mcp.json` entries the person imports — say so).
