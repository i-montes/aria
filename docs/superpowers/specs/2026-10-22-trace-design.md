# Live trace (`/debug`) — design

The owner wants to evaluate the agent: see, while it works, where a turn's time and money go. Typing `/debug true` in a
tab opens a trace panel to the left of the chat: every LLM call, tool call and subagent of the current turn as a bar on a
live timeline, with tokens and cost per step. Inspired by affectUI's Trace Waterfall (a paid component, studied in the
browser only); Aria's is its own implementation, in the page's plain JS.

Decided with the owner (2026-10-22): the panel sits on the **left** of the chat column; it shows **only the current
turn** (it stays after the turn ends, until the next one starts); subagents show **everything inside** them; `/debug`
is **per tab** and is off again after a restart. Approach A: the dock builds the trace, the page draws it.

## The command

`/debug true` and `/debug false` (also `on`/`off`; `/debug` alone toggles) are Aria commands, handled by the page like
`/compact` (`ui/chat/view.js`) and listed in the `/` menu. The page sends `PageMessage::TraceSet { session, on }`. The
dock keeps the flag on that session's chat, in memory only: closing the tab or quitting turns it off. A tab whose trace
is on and that has not run yet shows the panel with an empty state ("The next turn is traced here").

## The trace

`crates/aria/src/trace.rs`, pure (events and times in, a tree out):

```json
{
  "id": "root", "kind": "agent", "name": "Aria", "detail": "<model>", "status": "running|done|failed",
  "start": 0, "end": 52000, "input": "<the turn's prompt, first 2000 chars>",
  "children": [
    { "id": "llm-1", "kind": "llm", "name": "LLM call", "detail": "claude-opus-5-5", "status": "done",
      "start": 120, "end": 7020, "tokensIn": 18200, "tokensOut": 640, "cacheRead": 15000, "reasoning": 210,
      "cost": 0.12, "output": "<text, 2000 chars>", "error": null },
    { "id": "tool-<toolCallId>", "kind": "tool", "name": "bash", "detail": "pnpm test checkout", "status": "failed",
      "start": 43000, "end": 51400, "input": "...", "output": "...", "error": "..." ,
      "children": [ { "kind": "agent", "name": "researcher", ... } ] }
  ]
}
```

- Times are ms from the turn's start (`agent_start`). A running span has no `end`.
- Root children, in order of start: one `llm` span per assistant message (`message_start` → `message_end`, with
  `message.model`, `message.usage` — `input`, `output`, `cacheRead`, `reasoning`, `cost.total` — and `stopReason`
  `error` → failed, `aborted` → failed with "stopped"), and one `tool` span per tool call
  (`tool_execution_start` → `_end`, `isError` → failed). `detail` is the tool's target as the chat's tool cards already
  write it (command, path, query).
- Cost and tokens are each span's own; agent spans (root, subagents) show their sum over their subtree, computed by
  the page.
- `agent_start` starts a new trace (the previous one is dropped); `agent_settled` ends the root: `done`, or `failed`
  when the last message failed or the run was aborted. A compaction inside a turn is a `tool`-like span named
  "compaction" (`compaction_start` → `_end`).

### Subagents

`extensions/subagent-watch` gains what the trace needs; its lines grow optional fields, so an older dock ignores them:
`{t:"llm_start", at}`, `{t:"turn", at, model, usage, stop}` (the existing `turn`, now with the message's data),
`{t:"tool", id, name, args, at}`, `{t:"result", id, text, error, at}`, `{t:"prompt", text, at}`, `{t:"end", at}`;
`at` is the child's `Date.now()`.

`watch.rs` keeps, per child agent, a bounded list of spans (at most 200; epoch ms) built from those lines, besides the
steps it keeps today for the subagents view. `expect_tasks` gains the `parallel` call's `toolCallId`, so each named
child remembers the call that started it. The trace places a child under that tool span; a child with no matching call
(a background agent job, an older extension) goes under the root. Only children that connected after the turn started
are part of it. A child is an `agent` span (name and task as the subagents view names them) whose children are its
`llm` and `tool` spans.

### Sending it

While a tab's trace is on, the chat feeds every event of the current turn to its trace, and the dock sends
`DockMessage::Trace { session, trace }` (the whole tree) at most every 250 ms while something changed, and once more
when the turn settles. A watch change (`on_change`) counts as a change. `TraceSet { on: true }` answers with the current
tree at once (or `null` before the first turn); `on: false` drops the trace and sends `trace: null`. With the flag off
the chat builds nothing.

## The panel

`ui/chat/parts/trace.js` (+ css), pure like the other parts: props (tree, now, collapsed ids, open id) in, escaped HTML
out; the view owns the clock and the collapsed and open state.

- **Room.** While the active tab's trace is on, the page sends `PageMessage::Wide { on }`: the surface grows from 700
  to 1,060 px toward the left (macOS: the panel's frame in `dock/mac.rs`; Linux: `set_size_request` and the input
  region in `dock/gtk.rs`, which then takes input over both columns). The trace panel is ~560 px, left of the 470 px
  chat column, with a gap. Switching to a tab without a trace sends `Wide { on: false }`.
- **Header.** "Trace", the turn's prompt on one line, the state (Live, Done, Failed). Six figures: duration, cost,
  tokens, LLM calls, tool calls, errors; they climb live while the turn runs. A turn with no cost reported (a
  subscription provider) shows cost as "—".
- **Waterfall.** Left, the span tree: indent per level, an icon per kind, name, a muted detail. Right, each span's bar
  on the time axis, duration and cost after it. Colours: agents grey, LLM violet, tools teal, failed red. A running bar
  grows to a "now" line; the axis grows in round steps (10 s, 20 s, 30 s, 1 min, 2 min, 5 min, …). A "slowest" tag marks
  the slowest finished LLM or tool span.
- **Folding.** Agent spans fold by their chevron; a folded agent shows its calls as small marks on its bar and a "+N"
  count. A subagent folds by itself when it ends; a running one stays open; a fold or unfold by hand wins.
- **Details.** A click (or Enter) on a row opens a card over the panel: kind and status, duration, started at (+43 s),
  tokens (in, out, cache, reasoning) and cost, the error in red, input and output previews. Esc closes it. Arrows move
  between rows, ← and → fold and unfold.
- **Motion.** New rows slide in, bars grow from their start; `prefers-reduced-motion` shows everything at once.
- Texts in `ui/i18n` (English and Spanish), like the other parts.

## Errors and limits

- At most 500 spans per trace: beyond that the oldest finished tool and LLM spans are dropped and the header says
  "N earlier steps not shown". Previews are cut at 2,000 characters; image data is stripped (as `for_page` does).
- Nothing is written to disk. The previews show what the chat's tool cards already show.
- A child whose lines lack the new fields (an old extension) appears as an agent span with no inner spans.
- A Pi that restarts mid-turn ends the trace's root as failed; the next `agent_start` starts a new one.
- Scheduled tabs can be traced like any other.

## Testing

- `trace.rs`: a turn's events → the tree (LLM spans with usage and model, tools with ids, failed and aborted, compaction,
  a new turn resets, settle ends the root, the 500-span cut, children placed under their `parallel` call or the root,
  children from before the turn left out).
- `watch.rs`: the new fields build spans; old lines still work; `expect_tasks` keeps the call id.
- `chat.rs`: with the fake Pi, `TraceSet` on → `Trace` messages during a run, throttled, a last one on settle; off →
  `null` and nothing built.
- `extensions/subagent-watch`: the lines it sends, with `at`, ids and usage.
- `ui/chat/parts/trace.js`: render (figures, sums, colours, slowest, axis steps, running bar, folded marks and count,
  details card, empty state, reduced motion); `view.js`: `/debug` parsing and the `TraceSet`/`Wide` messages.
- `tools/ui-sandbox`: the panel in headless Chrome, wide and with a long trace.
- Platforms: the macOS frame change is checked live on the Mac; the GTK change is type-checked and its input-region
  rule unit-tested, and the run on Linux is left to the owner's Linux machine (said in the release notes if not run).

## Out of scope

Saving traces, past turns, comparing runs, exporting, a global or remembered `/debug`, tracing inside a subagent's own
subagents beyond what reaches the watch socket.
