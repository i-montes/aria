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

## Smaller gaps

- The agent-session view has no composer (the design has one): there is no way to steer a subagent yet.
- Clicking the "needs your answer" desktop notification opens the dock on the active tab, not on the tab that is waiting.
- Question card: the sliding animation between questions and the "Answers sent" / "Open questions" states were not ported (the card is removed when answered).
- Message index: chips grow with the pointer but do not fade by distance (the owner's solid-fill rule).
- Settings changes do not restart a running Pi (applied at the next start); see the phase 2 spec, part 2c.

## Deliberate differences from the design

- The working folder row and the settings button (owner's decisions).
- With several tabs the meter moves from the header to the row of the working folder (the header has no room).
- Buttons and cards use solid fills; only the border takes the accent (owner's taste).
