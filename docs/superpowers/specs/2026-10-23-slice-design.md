# Slice and Ponytail — design

Superpowers' `/plan` (brainstorming → writing-plans) and `/execute` (subagent-driven-development) are slow and show
the work only at the end: subagents in series, chained reviews, one report. This replaces them with a method of
Aria's own, **slice**, adopted from Hola Mundo's `hola-ai` (vertical user stories, acceptance criteria, the test list
as the spec, test first, the minimum step, "don't invent, ask"), and makes **Ponytail** (the smallest complete change)
the base rule of every turn. Subagents stay: they free the chat while they work, and the work shows as phases and
tasks, live.

Decided with the owner (2026-10-23):

- `hola-ai` is not installed: its method is written into slice, an Aria skill.
- Slice is **never chosen by the model**. Jev offers it (`plan=offer`, sizable work) and the owner accepts, or the
  owner types `/plan`.
- One round of questions at the start, then a plan to approve: the **only checkpoint**. Approval starts the build.
- The build runs in a **background subagent**, one per phase; the chat stays free.
- The subagent **commits locally** (one commit for the test, one for the code); it never pushes.
- The plan lives in `docs/plans/<date>-<slug>.md`, committed. A repo without `docs/` asks the owner once whether
  `docs/` goes in the repository or in `.gitignore`.
- With an OpenCode Go key stored, **Jev classifies whatever the chat's model is**, Claude's included: Jev is free.
- **TypeSafe** (Jev's maker) is a provider in the setup, so a TypeSafe API key is another way to have Jev; it is
  never a chat model.

## 1. The two lanes

Jev already tells sizable work from the rest (`extensions/jev-router/route.ts`, `sizable`, `PLAN_WORTH`):

- **Small** (a fix, an edit, a question): the assistant does it, under Ponytail.
- **Sizable and changing things** (`plan=offer`): the assistant offers once, with `ask_user`: "Plan it in slices"
  (recommended) or "Just do it". Accepted, `start_plan` starts slice's phase A, as today. Declined, no new offer for
  that work.
- `/plan <idea>` starts phase A by hand. `/plan off` leaves it.

The skill carries `disable-model-invocation: true` (Pi 1.3.0 hides it from the model's own choice), and `SYSTEM.md`
says it is used only through those two doors.

## 2. The skill: `agent-files/skills/slice/`

### Phase A: frame (the main agent, with the owner)

- Read the code the work touches and the project's `CLAUDE.md` / `AGENTS.md`.
- **One** `ask_user` call, at most 4 independent questions, each with a recommended answer: only what the code cannot
  settle. A new project: the stack (only the stack). Ask whether an expensive-to-change decision applies: multi-tenant,
  roles and permissions, offline or sync, multi-language or multi-currency. Nothing else is asked; what was assumed
  is written down as **Assumptions**, visible, for the owner to correct.
- Write the plan, `docs/plans/<YYYY-MM-DD>-<slug>.md`:

  ```markdown
  # <title>

  ## Context
  <one or two lines: what it is and who it is for>. Stack: <…>. Expensive decisions: <the ones that apply, or none>.

  ## Assumptions
  - …

  ## Backlog
  1. As a <actor>, I want <action>, so that <benefit>.
  2. … Requires: #1

  ## Phase 1: <story 1, in a few words>
  As a <actor>, I want <action>, so that <benefit>.

  Acceptance:
  - Given <context>, when <action>, then <result>.

  Out of scope: …
  Suggestions (your call): …

  - [ ] <test 1> (<where: endpoint, service, component>)
  - [ ] <test 2> (manual check)
  ```

  Backlog rules: 10–15 one-line stories at most, each one vertical (data, backend, UI) and valuable on its own; no
  technical stories (the actor is never the developer or the system); the first is the smallest usable end to end.
  Phase rules: 2–3 criteria (more means the story splits, and the parts go back to the backlog); the tasks are the
  test list, at the highest practical level; no files, layers or patterns named in advance. The plan holds the phases
  of this cycle (by default the first story; the owner may ask for more).
- Commit the plan (`docs(plans): …`), locally.
- Show it and ask for approval (`ask_user`: approve / change something). The chat shows it as the **document card**
  (draft, then approved) and the **phases card** (§4).

Plan mode keeps its write rule with the new folder: only `docs/plans/` (and `docs/.gitignore` choice, below), `.git`
and `/tmp`. It ends when the owner approves the plan.

**`docs/` missing**: before writing, if the repo has no `docs/` folder, the same `ask_user` round carries one more
question: "Keep `docs/` in the repository (recommended) / Keep it out (`.gitignore`)". Out: `docs/` is added to the
repo's `.gitignore` and the plan is not committed.

### Phase B: build (a background subagent per phase)

- Approval launches `background_agent` with the phase as the brief and a new flag, `commits: true`: the brief is the
  plan path, the phase, the rules below and Ponytail's. The main agent ends its turn at once: the owner can talk to
  it, ask, or correct.
- A committing subagent runs **in the folder**, never in a worktree: a worktree's work comes back as one uncommitted
  `git apply` (`crates/aria/src/worktree.rs`), which would flatten the commits. The dock's brief today says "Do not
  commit." to every writer (`crates/aria/src/jobs/agent.rs`); with `commits` it says "Commit locally as the task says;
  never push, merge or rebase." It queues while another writer holds the folder.
- The subagent works the phase's tasks, each one a cycle: write the test → run it, red → **commit the test** → the
  minimum code (Ponytail's ladder) → run the tests, green → **commit**. A refactor only with everything green, never
  mixed with new behaviour, in its own commit, and only when the code calls for it (a handler that grows, tests in
  the wrong place). Conventional subjects; local commits only, never a push.
- It marks each task in the plan when it starts (`- [~]`) and ends (`- [x]`).
- It **stops and asks** (the job ends with the question as its result) when:
  - a design decision is not in the plan;
  - it would add something nobody asked for;
  - a test would have to change, be skipped or be deleted to pass;
  - two attempts did not reach green;
  - a step exceeds ~50 lines or 3 files (it proposes splitting it).

  The main agent gets the job's end (`aria-background`), asks the owner with `ask_user`, and launches the phase again
  with the answer; the plan's marks say where it was.
- Test rules: the expected value comes from the acceptance criterion, never computed with the code's own logic;
  behaviour through the public interface.
- The phase ends: a `reviewer` subagent (read-only) runs `ponytail-review` on the phase's commits; the main agent
  goes through the manual checks with the owner, proposes a line for `CLAUDE.md` when a decision applies to the whole
  project, and launches the next phase (or offers the next story of the backlog when the plan is done).

`/execute <plan>` stays, only to **resume** an approved plan (after a restart, or a question answered later): it
launches the first phase with open tasks.

## 3. Ponytail, always on

- `pi-packages` pins `git:github.com/DietrichGebert/ponytail@09c434ab73ca43761c5d8b0c5d341926c2fba710`, filtered as
  superpowers is (`crates/aria-core/src/agent.rs`): the skills `ponytail` and `ponytail-review`, and its Pi extension
  (`/ponytail lite|full|ultra`, default `full`). Its band, board and hooks are Claude Code's and do not apply.
- It is the whole small lane, the "minimum code" of phase B, and phase B's review.
- `SYSTEM.md`'s "Scope" rule is aligned with it, so the two do not say the same thing twice.
- Superpowers keeps only `systematic-debugging` (the settings' `auto_skills`); `brainstorming`, `writing-plans`,
  `subagent-driven-development` and `verification-before-completion` leave (Ponytail's closing line covers what was
  not checked).

## 4. Watching: phases and tasks, live

Every piece exists; this feeds them. One source: the plan file's checkboxes.

| What the owner sees | Piece | Fed by |
|---|---|---|
| The plan and its state | document card (`ui/chat/parts/document.js`) | phase A, at approval: `doc: 'plan'`, `draft` → `approved` |
| Phases → tasks | phases card (`ui/chat/parts/phases.js`) | phase A, from the plan |
| Progress: todo / active / done | the dock's task dock (`chat.setTasks`) and the terminal's tasks strip | the plan's marks while its phase runs |
| What the subagent does, step by step | the background pill, Alt+A | already |
| Every real change | the commits | already |

The task list reuses `tasks_update`'s shape (`{ id, title, status, agent }`, `extensions/tasks/`, in main since
0.17.0). While a slice phase runs in the background, the parent Pi watches the plan file and, on each change,
publishes the phase's tasks as the extension status `aria-tasks` (JSON), the channel the dock (`ui/chat/state.js`,
`ui_notice` → `setStatus`, as `aria-jev`) and the terminal (`crates/aria/src/term/state.rs`, as `aria-verify`)
already read; cleared when the job ends. The terminal shows it in its tasks strip; the dock feeds its task dock
(`chat.setTasks`), which nothing fed before.

## 5. Jev whatever the model, through Go or TypeSafe

Today `applyProvider` turns the router and Jev's sandbox review off for any main provider that does not include Jev
(`ui/onboarding/steps.js`; the Rust twin in `crates/aria-core/src/providers.rs`), even when the OpenCode key is
stored, and Jev through TypeSafe is "not built" (`crates/aria-core/src/agent_config.rs`).

- **TypeSafe is a provider of its own**, listed with the others in the setup's providers step: connect (its API key,
  checked by the existing probe, kept as `typesafe` in `auth.json` / the keyring), replace, remove. It is never a
  main provider and has no chat models: no "make main" action, nothing in `/model` or the models step (Pi lists
  classifier models apart already). The router step no longer asks for its key; it says which Jev is in use.
- **Jev's backend follows the stored keys**, not the main provider: OpenCode Go (main provider or key stored) →
  `opencode` (`opencode/jev-1.13-free`, free); otherwise a TypeSafe key → `typesafe` (`typesafe/jev-latest`, Pi's
  built-in classifier); neither → the router and the review are off. With both, Go wins (free).
- Aria writes the backend into `aria.json` (`router.backend`), and the extensions pick the classifier from it
  (`jev-router`, the sandbox's review); without Aria's config (the owner's own Pi) it stays `opencode`.
- Effort keeps each main provider's default (Claude's is fixed).

## 6. Out of scope

- Claude Code: slice as `~/.claude/skills/slice` and the Ponytail plugin there are a later, out-of-repo change.
- Moving the existing `docs/superpowers/` history.
- Several phases at once (one subagent at a time).
- The goals engine (phase 3).

## 7. Tests

- `persona.rs`: the new "Slice" section, Superpowers' gone, the golden `SYSTEM.owner.md`.
- `agent.rs`: the Ponytail package and its filter; superpowers down to `systematic-debugging`.
- `jev-router`: the offer only with `plan=offer`; `start_plan` and `/plan` load slice; the write rule of plan mode
  (`docs/plans/`, the `.gitignore` choice); approval launches phase B; `/execute` resumes the first open phase.
- The plan's marks → tasks (`[ ]` todo, `[~]` active, `[x]` done), and the cards' props from a plan file.
- The setup (`ui/onboarding`): the backend from the stored keys (Go, TypeSafe, both, none), TypeSafe's row without
  "make main" and in no model list; `providers.rs`'s golden files unchanged; `agent_config.rs` carries the backend;
  the extensions pick `opencode/jev-1.13-free` or `typesafe/jev-latest` from it.
- `aria-tasks` status → the dock's task dock and the terminal's strip.
- The skill, by a scenario in a scratch repo under `/tmp`: one round of questions, a backlog, one phase, two cycles
  with their commits, one stop on a test that would have to change.
- `sh tools/check-macos.sh`; the dock's parts are one webview on both platforms.
