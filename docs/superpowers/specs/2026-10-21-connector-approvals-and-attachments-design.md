# Connector approvals and attachments

Date: 2026-10-21. Ideas 1 and 2 of `docs/ideas.md`. The owner skipped the questions ("just do it"), so every decision
here is the builder's call unless marked **(owner)**.

## Part 1 — Approval for each connector action

### Why

A connected service (Notion today; any MCP server the person adds, such as `rea`) can change things outside the
machine, such as creating, updating or deleting pages, and today nothing asks first. The person should see what a
connector is about to send or change before it happens, and choose how much each connector may do on its own.

### Goals

1. Every call to an MCP tool that **writes** asks first: a card shows the connector, the action and a preview of what it
   will send, and the person allows it once, allows that action for the rest of the session, or says no.
2. Calls that only **read** run without asking.
3. Each connector has an access level the person sets in Settings → Integrations: **Read only** (writes are refused, never
   asked), **Ask** (default) or **Allow** (writes run without asking).
4. It holds however the call is made: a direct tool call or a call from a codemode script.

### Non-goals

- A proxy between Pi and the MCP servers, or moving connector secrets out of files (idea 4).
- Aria's own memory server (`engram`) is not a connector: it is never gated (subagents already cannot write memory).
- Per-field redaction of previews.

### Anti-goals

- A write that runs without the person's yes while its connector is on Ask.
- A card the person cannot understand (a raw JSON dump with no action name).
- So many cards for reads that the person learns to click yes without looking.

### How it works (calls)

- New Aria extension `extensions/connectors/` hooks Pi's `tool_call` event. Pi runs calls made from a codemode script
  through the same hooks, with `parentToolCallId` set (Pi 1.3.0 `core/extensions/types.d.ts`, `ExtensionToolContext.executeTool`),
  so one hook sees every MCP call.
- An MCP tool is a tool whose name is `mcp__<server>__<tool>` (Pi's `isMcpToolName`). Its server is `<server>`. Pi's
  resource tools (`list_mcp_resources`, `list_mcp_resource_templates`, `read_mcp_resource`) are reads.
- **Read or write:** from the tool's MCP annotations, which `pi.getAllTools()` returns (`annotations`):
  `readOnlyHint: true` → read. `readOnlyHint: false` or `destructiveHint: true` → write. No annotations → by name: a
  tool name whose first word is one of `search, fetch, get, list, read, query, find, retrieve, lookup, describe, view,
  show, count, check` (split on `-`, `_` or camelCase, the server prefix such as `notion-` removed) → read; anything else
  → write. Unknown means write: asking too often is the safe mistake.
- **Access levels** come from Aria's settings through `aria.json` (`connectors: { "<server>": "read" | "ask" | "allow" }`,
  missing → `ask`). `engram` is always allowed.
- **Asking:** with someone to ask (the dock or Pi's terminal, the sandbox's `canAsk`), `ctx.ui.select` with the card
  `{"aria":1,"kind":"connector", "summary", "detail", "options"}`:
  - `summary`: "<Connector> wants to <action>", where the action is the tool's name made readable
    ("notion-create-pages" → "create pages"; Pi's tool list carries no MCP title), in English (the card's chrome is translated by the page);
  - `detail`: a readable preview of the arguments: top-level keys as `key: value` lines, strings cut to 300 characters,
    nested values as compact JSON cut to 300, at most 12 lines;
  - options, in this order: "Yes, just this once" (primary), "Yes, and don't ask again for <connector> <action> this
    session", "No".
- Without anyone to ask (a subagent, `pi -p`), a write on Ask is refused with a reason the model can relay:
  "<Connector> wants to <action>, which needs the person's approval, and there is nobody here to ask."
- A write on Read only is refused at once: "<Connector> is read only (the person set it so in Aria's settings): <action> did not run."
- "No" refuses: "The person said no: <connector> <action> did not run."
- Session grants live in memory and are cleared on `session_start`, like the sandbox's.
- The dock's notification for a card nobody sees already exists. Its text for this card is the `summary`.

### What the person sees

- The chat's permission card, as for the sandbox, with a plug icon and the default title "A connector wants to act" when
  `summary` is missing. It is the same part (`ui/chat/parts/permission.js`) with a third `kind`, not a copy.
- Settings → Integrations: a row per connected MCP server (the dock already lists them as `mcp:<server>`), each with a
  three-way choice: Read only / Ask / Allow. New servers show as Ask.

## Part 2 — Attachments

### Why

The person wants to hand Aria a screenshot, a PDF or a file without saving it somewhere and typing its path.

### Goals

1. Attach from the composer: a paperclip button (the file chooser), drag and drop onto the chat, and paste (images from
   the clipboard).
2. Before sending, the attachments show as removable chips above the input.
3. Images go to the model as images when it can see them (the model's `input` includes `image`); otherwise, and for
   every other file, the file is saved and its path goes into the message for the model to read with its tools.
4. The sent message shows its attachments as chips (name and kind), live and after a restart.

### Non-goals

- Previews of images in the thread (the page never receives image data: `for_page` strips it).
- Folders, or files larger than the limits below.

### Decisions

- Limits: at most 10 attachments per message, 20 MB each, 50 MB together. The page refuses beyond them with a short
  note.
- The page reads each file as base64 and sends it with the message: `chatSend { text, behavior?, attachments: [{ name,
  mime, data }] }`.
- The dock saves every attachment under `~/.aria/attachments/<session>/<yyyy-mm-dd>/<n>-<safe name>` (mode 0600, folder
  0700). The safe name keeps letters, digits, `.`, `-`, `_`, cut to 80 characters.
- The message Pi gets is the text, then one line per file that is not sent as an image:
  `[attached: <name> (<mime>, <size>) at <path>]`. Images the model can see go in Pi's `images` field of `prompt`
  (`{ type: "image", data, mimeType }`) and are also saved, with a line `[attached image: <name> at <path>]`.
  The model's image support comes from the session's current model in `get_available_models` (`input`).
- The sandbox must let the model read `~/.aria/attachments` (reading is open except credentials; check `isSecret` and
  `secretDirs`) and never write there.
- A queued or steering message keeps its attachments.
- A message the dock cannot save (disk full) is refused like any refused prompt, and the text goes back to the composer.
- Clearing: attachments are kept; `docs/pending.md` notes that nothing removes old ones yet.
- In the thread, the user bubble shows chips for `[attached: …]` / `[attached image: …]` lines instead of the raw lines.
  History replays the same way from the message text.

## Testing

- Extension: classification (annotations, name rule, resource tools, engram), the three levels, the card JSON, grants,
  refusals with and without UI, nested calls (a `tool_call` with `parentToolCallId` is gated the same).
- Rust: the attachment saving (names, folders, modes, limits), the prompt fields (`images` only for an image-capable
  model), the refused save.
- UI: the composer chips (add, remove, limits), drop and paste handlers as far as unit tests reach, the bubble chips,
  the permission card's `connector` kind, the settings rows; a browser check in `tools/ui-sandbox` for attaching and
  for the connector card.

## Docs

`extensions/README.md` (the new extension), `docs/architecture.md` (attachments in the messages table), `docs/settings.md`
(`integrations.access`), `docs/pending.md` (what is left), `CHANGELOG.md` (`~/.aria/attachments/`).

## Settings

`integrations.access: { "<server>": "read" | "ask" | "allow" }` in `settings.json` (default empty: every connector Ask),
mirrored in Rust and JS and in the golden JSON, written into `aria.json` as `connectors`.
