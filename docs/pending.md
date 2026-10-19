# Pending: what the chat shows but cannot feed yet

The chat has every element of the design (`design/Aria.html`, `design/components/`). These parts are built and
tested as UI, with their data contract written at the top of each module in `ui/chat/parts/`, but nothing feeds
them yet. Each row says what is missing and which part of the roadmap brings it.

## Needs the owner's extensions to speak the dock's language (phase 2, part 2b)

| What the design shows | What is missing | Contract (already in the UI) |
|---|---|---|
| Permission card (leave the sandbox, write outside) with Jev's review | the `sandbox` extension must ask through `ui.select(title, ["allow","always","deny"])` with a JSON title | `{"aria":1,"kind":"sandbox"\|"write", summary, detail, path, jev}` (`ui/chat/parts/cards.js`); the answer is the option code |
| Quota card with the countdown | the `quota-fallback` extension, same mechanism, options `["free","keep"]` and the dialog's `timeout` | `{"aria":1,"kind":"quota", percent, freeModel, countdown}` |
| Question card (`ask_user`), then the answers card | `ask-user` must ask through `ui.input` with a JSON title; the answer is a JSON string | `{"aria":1,"kind":"question", questions:[{question, header, options:[{label, description, recommended}], multi}]}`; answer `[{question, answers, talk?}]` |
| Jev line before an answer ("focused task · searches memory") | `jev-router` must emit its route as a session event | `{type:"aria_jev", mode, action}` |
| "engram: saved the decisions" note after compaction | `autocompact` must notify through `ui.notify` | already shown as a notice from `ui_notice` |
| Working phases "planning", "writing the spec", "splitting into phases" | `/plan` flow events | the phase names are in `ui/chat/parts/working.js`; `state.js` maps tool names to the others |
| Quota in the header meter and in the tab tooltips (used %, resets) | the quota extension's readings | `renderMeter({used, resets})`, tab `quota`; `hasQuota` is false until then |
| Subagents view (Alt+A) and an agent's session | the Rust-hosted subagent socket (`PI_WATCH_SOCK`) and the `parallel` tool events | `chat.setAgents([{id, role, task, status, kind}])`; session steps per `ui/chat/parts/session.js` |

## Needs the goals engine (phase 3)

| What the design shows | What is missing |
|---|---|
| Task dock above the composer and the inline task log | the goals/tasks engine (fork of `pi-goal-x`) feeding `chat.setTasks(tasks, kind)` |
| Document cards (spec, plan) with approval status | the planning flow publishing documents (`renderDocument` contract) |
| Phases card | the planning flow's phases (`renderPhases` contract) |

## Needs an Engram feed (phase 2, part 2c)

| What the design shows | What is missing |
|---|---|
| Memory panel (`/memory`): list, search, filters, pin, forget with undo | a read of Engram's memories into `chat.setMemory([...])` and the actions (pin, forget) written back; the contract is in `ui/chat/parts/memory.js` |

## The Claude provider

- `/connect` installs the Claude provider (`claude-bridge`) for the sign-in route and needs Claude Code on the machine; the quota of a Claude plan is not readable yet, so the header meter shows none for it.
- Showing the sign-in's email or plan in the card needs a profile call the sign-in does not make today.
- Disconnecting a provider (removing its credential or the package) has no command yet.

## The quota bars

- OpenCode Go's windows (5h, Week, Month) are read from `GET /zen/go/v1/usage` and were checked against the real answer. A Claude plan's windows (5h, Week and any per-model one such as Fable) are read from `GET /api/oauth/usage` with the sign-in's token, an endpoint Anthropic does not document: its answer format is parsed defensively and has **not** been checked against a real account (none is connected yet).
- An API key, OpenCode Zen and an expired sign-in token show no bars. Aria does not refresh an expired Claude token (Pi does when it uses it); the bars come back after the next run.
- The card is one component (`renderQuotaCard`): the header chip opens it and each session tab shows it on hover (led by the tab's title and state, with that session's model and context). Beside the name only the 5-hour chip shows; hovering it (or focusing it with the keyboard) opens a card with every window and when each resets. With several sessions the header shows tabs, not the bar.

## Smaller gaps

- The agent-session view has no composer (the design has one): there is no way to steer a subagent yet.
- Clicking the "needs your answer" desktop notification opens the dock on the active tab, not on the tab that is waiting.
- Question card: the sliding animation between questions and the "Answers sent" / "Open questions" states were not ported (the card is removed when answered).
- Message index: chips grow with the pointer but do not fade by distance (the owner's solid-fill rule).
- Settings changes do not restart a running Pi (applied at the next start); see the phase 2 spec, part 2c.

## Known limits of the page (from the review, deferred)

- The thread is redrawn on every streamed fragment, so text cannot be selected or copied while a run streams; the other regions only redraw when they change.
- The quota card's countdown ("Times out in 300s") is static; the real timeout is Pi's.
- Choosing `model` from the command picker only types `/model `; the model list opens once a space follows (or from the chip under the input).
- Closing the active tab does not save its unsent draft; a language change drops unsent drafts.
- `rows` of the composer counts only explicit newlines; a long wrapped line does not grow it (WebKitGTK's `field-sizing` support was not checked).
- The working phase keeps the last tool's name ("Reading files") while the answer streams.
- The browser checks (`tools/ui-sandbox/check.ts`) cover the controller's riskiest flows but are not part of `bun test` (they need Chrome).

## Deliberate differences from the design

- The working folder row and the settings button (owner's decisions).
- The header has no model button or meter (owner's decision; the design has one beside the name): the model is the chip under the input and shows in each tab's tooltip. The meter's part (`parts/meter.js`) is kept for when its quota and context readings have a place.
- Buttons and cards use solid fills; only the border takes the accent (owner's taste).
