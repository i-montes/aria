# Aria chat — design fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (the owner chose native execution; helpers may build independent parts in parallel). Steps use checkbox (`- [ ]`) syntax.

**Goal:** The dock's chat has everything the owner's design (`design/Aria.html`, its components in `design/components/`) shows. What cannot work yet for lack of a backend is built as UI with its data contract and listed as pending.

**Architecture:** Each design component becomes a pure render module in `ui/chat/parts/` (props in, escaped HTML out, texts in both languages, its own CSS, its own tests). `view.js`/`state.js`/`app.js` compose them: a header (assistant, meter, tabs, `+`, close), the thread, the composer with its pickers, and the alternate panels (subagents, an agent's session, memory). Structured cards (permission, quota, question) arrive as JSON in the extension dialogs' `title` (`{"aria":1,"kind":…}`), so part 2b only has to make the extensions send them. A sandbox page (`tools/ui-sandbox/`) replays the design's "simulate events" so every state can be seen and tested without Pi.

**Tech Stack:** plain ES modules and CSS over `aria://app/…`, `bun test`; Rust only for the few new bridge messages.

**Spec:** `docs/superpowers/specs/2026-10-18-aria-phase2-parity-design.md` ("The dock as chat"), and the design itself (`design/Aria.html`, `design/components/*.dc.html`). The design wins over this plan wherever they differ.

## Global Constraints

- Everything built is in English in code and strings; the Spanish catalog (`ui/i18n/es.json`) gets every user-visible text.
- Fidelity: markup, sizes, colors, radii, fonts, animations and `prefers-reduced-motion` handling follow the component source in `design/components/`; demo-only logic of those files is ignored.
- Owner's taste (overrides the design's translucent fills): buttons and cards have **solid** backgrounds that never change opacity on hover, focus or selection; only the border takes the accent `#B9A8FF`.
- The working folder stays visible at all times (owner decision, not in the design).
- Every dynamic value is escaped with `esc` (`ui/onboarding/components.js`).
- A part module exports `render<Name>(props, ui, t) -> string`, `TEXTS = { en: {…}, es: {…} }` (keys `chat.<part>.…`), and its pure helpers; its interactions are `data-chat="<action>"` attributes with `data-*` parameters, listed in the module's header comment under "Emits".
- A part never touches `view.js`, `state.js`, `app.js`, `index.html` or the JSON catalogs; the integrator wires and merges.
- Anything the backend cannot do yet is listed in `docs/pending.md` with what is missing.

## Review Focus

- Escaping of every dynamic string in every new renderer.
- A card for a dialog that no longer exists (timeout, Pi exit) must disappear, never answer a dead id.
- Esc: stops a run when one is going, closes pickers first, closes the dock only when nothing else applies.
- Queued messages must survive a tab switch and show in the right tab.
- A background tab that needs an answer must be visible on the tab bar, in the overflow, and as a toast.

## Gap matrix (design component → state)

| Design | Today | This plan |
|---|---|---|
| Header: assistant name + dot, `Medidor` (model, quota, context), `Pestanas`, `+`, close | tabs + path + gear | port Medidor and tabs; header as designed; path row and gear kept |
| `Usuario` (command chips, queued + cancel) | plain bubble | port; Enter queues (Pi `follow_up`), Esc stops |
| `Aria` message (detail, source, code spans, proposed-change diff) | plain bubble | port (rendering of inline code and fenced diffs) |
| `Jev` route line | none | port; fed by the router extension (pending 2b) |
| `Pensando` tool chips (live, expanded) | simple group | port faithfully |
| `Trabajando` (phase, elapsed) | simple | port with phase labels |
| `Nota`, `Separador` | notices | port |
| `Permiso`, `Cuota`, `Pregunta`, `Respuestas` | generic dialog | port; fed by `{"aria":1,"kind":…}` dialog titles (extensions pending 2b) |
| `Indice` message minimap | none | port (pure UI) |
| `Comandos`, `Archivos`, `Modelos` pickers | simple lists | port faithfully |
| `Composer` (stop/queue/send, hints) | simple | port |
| `Tareas` (dock + inline log), `Documento`, `Fases` | none | port as UI with a data contract (pending: goals, phase 3) |
| `Subagentes` (Alt+A), `Sesion` | none | port (pending: subagent socket, 2b) |
| `Memoria` panel | none | port (pending: Engram feed) |
| Toast for a background session | none | port |
| Simulate nav | none | `tools/ui-sandbox/` |

---

### Task 1: Message parts (`ui/chat/parts/messages.js`)

**Files:** Create `ui/chat/parts/{user,assistant,jev,note,divider,working,answers}.js` + `.test.js` + `.css` each (one file per part).
**Design sources:** `user-message`, `aria-message`, `jev`, `note`, `divider`, `working`, `answer-card`.
**Interfaces — Produces:** `renderUser({text, command, queued, id}, ui, t)`, `renderAssistant({text, detail, source}, ui, t)` (inline `` `code` `` spans and a fenced ```` ```diff ```` block as the design's "Proposed change"), `renderJev({mode, action}, ui, t)`, `renderNote({label, text}, ui, t)`, `renderDivider({text}, ui, t)`, `renderWorking({phase, startedAt}, {now}, t)` (elapsed from `now`), `renderAnswers({answers}, ui, t)`. Emits: `cancel-queued` (`data-id`).

### Task 2: Tool chips and the message index

**Files:** Create `ui/chat/parts/{tools,index}.js` + tests + css.
**Design sources:** `tool-chips`, `message-index`.
**Produces:** `renderTools({calls:[{id,name,args,status,summary,result}], live}, {open:Set, openRows:Set}, t)` with the design's header text, per-kind icons and row expansion; `renderIndex({items:[{id,label}], activeId}, ui, t)` and `indexTargets(...)`. Emits: `toggle-tools`, `toggle-tool-row` (`data-id`), `index-jump` (`data-id`).

### Task 3: Cards and the meter

**Files:** Create `ui/chat/parts/{permission,quota,question,meter}.js` + tests + css.
**Design sources:** `sandbox-permission`, `quota`, `question-card`, `quota-meter`.
**Produces:** `renderPermission({id, kind, summary, detail, jev}, ui, t)` (emits `allow`, `always`, `deny`), `renderQuota({id, percent, freeModel, secondsLeft}, ui, t)` (`quota-free`, `quota-keep`; the countdown is the dialog's own timeout), `renderQuestion({id, questions:[{question, header, options:[{label, description, recommended}], multi}]}, {step, picked, other}, t)` (`q-pick`, `q-other`, `q-next`, `q-back`, `q-submit`, `q-skip`), `renderMeter({model, used, context, free, hasQuota, compactAt}, ui, t)` (`open-models`). Plus `parseCardTitle(title) -> {kind, …fields} | null` for `{"aria":1,"kind":"sandbox"|"quota"|"question"|"write",…}` titles, in `ui/chat/parts/cards.js`.

### Task 4: Tasks, documents, phases, subagents, sessions, memory

**Files:** Create `ui/chat/parts/{tasks,document,phases,subagents,session,memory}.js` + tests + css.
**Design sources:** `task-dock`, `document-card`, `phases`, `subagents`, `agent-session`, `memory-panel`.
**Produces:** one renderer each with the data contract the design's demo data implies, documented in the module header (this is the contract the backend will have to fill). Emits as the design's callbacks (`task-select`, `agent-select`, `agent-back`, `memory-back`, …).

### Task 5: Composer, pickers and tabs

**Files:** Create `ui/chat/parts/{composer,commands,files,models,tabs}.js` + tests + css.
**Design sources:** `composer`, `command-picker`, `file-picker`, `model-picker`, `session-tabs`.
**Produces:** `renderComposer({busy, hasText, rows, showHints, placeholder}, ui, t)`, `renderCommandPicker({items, query, active, open}, ui, t)`, `renderFilePicker(...)`, `renderModelPicker({models, current, query, active, open}, ui, t)` with their filters as pure helpers, and `renderSessionTabs(tabs, ui, t)` replacing `renderTabs` (tooltip with model/context/quota, `max-visible`, overflow, rename, undo bar, `hide-new`). Same emits as today where they exist.

### Task 6: Integration — thread, header, composer, panels

**Files:** Modify `ui/chat/{view.js,state.js,sessions.js}`, `ui/app.js`, `ui/index.html`, `ui/i18n.js` (merge part texts), `ui/i18n/*.json`, `ui/chat.css`; Rust: `crates/aria-core/src/bridge.rs`, `crates/aria/src/chat.rs`.
- Header as designed (assistant name from settings, dot, meter, tabs, `+`, close) with the path row and the settings gear kept.
- Thread rendering through the parts; tool chips, working phase from the running tool, notes for model change / stopped / compaction (`compaction_start`/`end`), Jev lines when an event carries a route.
- Dialog titles parsed with `parseCardTitle`: `sandbox` → permission card, `quota` → quota card, `question` → question card; other titles keep the generic dialog. Answers go out as `uiAnswer`.
- Enter while a run is going queues (`chatSend` with `behavior: 'followUp'`) and shows the queued bubble with Cancel (new bridge message `ChatClearQueue` → Pi `clear_queue`); Esc stops a run (`chatAbort`), closes a picker, and only then closes the dock.
- Context % from `get_session_stats` (`contextUsage.percent`), refreshed when a run settles (`ChatState.context`).
- Panels: Alt+A toggles the subagents view, a selected agent opens its session view, a memory panel entry; toast for a background session that needs an answer (Open selects its tab).
- Tests: state reducer additions, `view` composition tests (cards chosen by title, queue, Esc order).

### Task 7: UI sandbox and pending list

**Files:** Create `tools/ui-sandbox/{serve.ts,README.md}`, `docs/pending.md`.
- `serve.ts` serves `ui/` with a stubbed bridge and the design's "simulate" buttons (permission, quota, compaction, `/plan`, tool calls, tasks, ask, send), moved from the throwaway harness.
- Screenshot every state in headless Chrome and compare by eye against the design; fix differences.
- `docs/pending.md`: each part that still needs a backend, with what is missing and which phase brings it.
- Run all suites, rebuild, install, update `README.md`/`CHANGELOG.md`, final review, merge decision with the owner.
