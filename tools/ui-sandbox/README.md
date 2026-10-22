# UI sandbox

Serves the dock's page (`ui/`) with a stubbed bridge and the design's "simulate events" buttons, so each state of
the chat (permission and quota cards, questions, compaction, paged history, tool calls, queue, tabs, subagents) can be seen and
tried without Pi or the Rust side.

```sh
bun tools/ui-sandbox/serve.ts          # http://127.0.0.1:8770/   (?lang=es for Spanish, ?first-run for the setup)
```

The page talks to its host only through `window.webkit.messageHandlers.aria` and `window.aria.receive`
(see `crates/aria-core/src/bridge.rs`); the sandbox plays the host.

```sh
bun tools/ui-sandbox/serve.ts 8770 &
bun tools/ui-sandbox/check.ts          # needs Chrome; checks the controller's focus, queue, Esc and picker behaviour
```
