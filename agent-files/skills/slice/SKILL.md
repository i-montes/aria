---
name: slice
description: Plan and build sizable work in vertical slices, test first. Used only when the router offers it and the user accepts, or through /plan; never chosen on your own.
disable-model-invocation: true
---

# Slice

Build software in small vertical slices that each deliver value: a short backlog of user stories, one story refined
into acceptance criteria and a test list, then the tests built one by one, each test first, each step committed. The
user decides what to build; you split it, propose the minimum and write the code. The smallest complete change wins
(Ponytail's ladder); never cut validation, error handling, security or accessibility.

## Phase A: frame

You and the user, in plan mode. Implement nothing.

1. Read the code the work touches and the project's `CLAUDE.md` or `AGENTS.md`.
2. Ask **one** `ask_user` call, at most 4 independent questions, each with your recommended answer first. Ask only
   what the code and the request cannot settle:
   - a new project: the stack (only the stack, not the architecture);
   - whether one of these expensive-to-change decisions applies (only ask, don't design it): several tenants or
     organizations, a permissions or roles model, offline or sync, several languages or currencies;
   - if the repo has no `docs/` folder: "Keep `docs/` in the repository (recommended)" or "Keep it out (`.gitignore`)".
   Everything else you would have asked goes under **Assumptions**, for the user to correct.
3. Don't invent requirements: split only what the user said. A gap you see becomes a question or an assumption,
   never a feature.
4. Write the plan to `docs/plans/<YYYY-MM-DD>-<slug>.md` in exactly this shape:

   ```markdown
   # <title>

   ## Context
   <one or two lines: what it is and who it is for>. Stack: <…>. Expensive decisions: <the ones that apply, or none>.

   ## Assumptions
   - …

   ## Backlog
   1. As a <actor>, I want <action>, so that <benefit>.
   2. As a <actor>, I want <action>, so that <benefit>. Requires: #1

   ## Phase 1: <story 1 in a few words>
   As a <actor>, I want <action>, so that <benefit>.

   Acceptance:
   - Given <context>, when <action>, then <result>.

   Out of scope: <what stays out of this story>.
   Suggestions (your call): <optional>.

   - [ ] <test name> (<where: endpoint, service, component>)
   - [ ] <test name> (manual check)
   ```

   - **Backlog:** 10–15 one-line stories at most. Each one is vertical (it crosses data, backend and UI) and valuable
     on its own; never split by layer. No technical stories: the actor is never the developer or the system; "set up
     the database" is built inside the first story that needs it. The first story is the smallest one usable end to
     end. `Requires: #N` only when a story makes no sense without another. No estimates or priorities.
   - **Phases:** one per story of this cycle; by default only the first story (the user may ask for more). 2–3
     acceptance criteria that say what the user sees or gets, never how it is built; a story that needs more is split,
     and the parts go back to the backlog. A short "Suggestions" line only when the code shows a concrete signal (for
     example, something that already repeats).
   - **Tasks** are the phase's test list: names only, at the highest practical level (endpoint, service, component).
     A criterion that cannot be tested well (a visual one) is a `(manual check)` task. No files, layers or patterns
     named in advance.
5. If the user chose to keep `docs/` out, add `docs/` to the repo's `.gitignore` and do not commit the plan; otherwise
   commit only the plan file: `docs(plans): <title>`.
6. Call `plan_ready` with the plan's path. If the user wants changes, make them, commit again and call it again.

## Phase B: build

A background subagent, one phase at a time. You were given the plan and the phase.

Before the first cycle, run `git status --porcelain`. Ignore the plan file and the files that belong to a task marked
`- [~]` (your own earlier work, left by a stop). If other files are changed, stop:
`SLICE QUESTION: There are uncommitted changes I did not make (<files>). Commit them, set them aside, or build anyway without touching them?`

Each task of the phase, in order, is one cycle:

1. Mark the task active in the plan (`- [~]`).
2. Write the test. Run it and see it **fail** (red). If it passes already, say so in the commit and go on.
3. Commit the test alone: `test(<scope>): <task>`. (A manual-check task has no test: note in the plan what the user
   must check, mark it done and go on.)
4. Write the **minimum code** that makes it pass, following Ponytail's ladder: does it need to exist, is it already in
   the codebase, the standard library, an installed dependency, one line, then the minimum. Nothing the test does not
   require.
5. Run the tests and see them **pass** (green).
6. Mark the task done in the plan (`- [x]`) and commit the code and the plan (skip the plan if it is not tracked):
   `feat(<scope>): <task>` (or `fix`).
7. Refactor only when the code calls for it (a handler that keeps growing, tests in the wrong place, something that
   repeats), only with everything green, never mixed with new behaviour, in its own commit: `refactor(<scope>): …`.

Commit rules: `git add` only the files you changed, by path (never `-A` or `.`); local commits only, never push,
merge or rebase; conventional subjects in English.

Test rules: the expected value comes from the acceptance criterion, never computed with the code's own logic; test
behaviour through the public interface, not internals.

**Stop at once**, without finishing the cycle, and end with `SLICE QUESTION: …` (the options as `- ` lines above it,
your recommended one first) when:

- a design decision is not in the plan;
- you would add something nobody asked for, however reasonable;
- a test would have to change, be skipped or be deleted to pass (never do it);
- two attempts did not reach green;
- a step would exceed about 50 lines or 3 files (propose how to split it).

When every task of the phase is marked done and the tests pass, end with `SLICE DONE`, after a short list of the
commits and anything you skipped or did not check; keep the final message short (a result is clipped to its first
8,000 characters, while the router reads the last line). If something outside your control stops you (the tests cannot
run, a tool is missing), end with `SLICE BLOCKED: <why>`.
