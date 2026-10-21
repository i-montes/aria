# Background work

Date: 2026-10-21. Status: approved by the owner in conversation; the owner then handed every remaining decision to
the builder ("you know what I want, take the rest of the decisions"). Decisions the owner made are marked **(owner)**;
the rest are the builder's calls, marked **(call)**.

## Why

One session should be enough. Today a session is one Pi with one agent loop: while it works, a message the owner
sends goes in at the run's next step, but the same busy agent handles it. A command that never ends (a dev server)
hangs the chat, and a long job (fix the tests, a refactor) keeps Aria from answering anything else.

The owner wants Aria to stay free to answer, with long work running in the background, and wants that work to be
**visible and never forgotten**, neither by Aria nor by the person using it.

## Goals

1. Commands that do not end by themselves (dev servers, watchers, tunnels) run in the background, never in the
   chat's foreground. **(owner)**
2. Long, well-defined jobs go to a background subagent while the main session keeps talking. **(owner)**
3. Aria decides by itself what goes to the background; the owner can force it either way ("in the background",
   "do it here"). Aria does not ask before delegating. **(owner)**
4. The person always sees what runs in the background, can watch it (output, the subagent's steps), open it (a URL)
   and stop it. **(owner)**
5. Aria always knows what runs in the background, learns when something ends or dies, and tells the person. **(owner)**

## Non-goals

- Steering a background subagent with new messages (the agent-session view still has no composer; see pending).
- Background work that outlives the dock. **(owner)**: when Aria quits for good, its background work stops.
- A scheduler, cron or goals engine (phase 3).
- Background jobs inside subagents (a subagent's commands run in its foreground). **(call)**

## Anti-goals (a failure even if it works)

- Something running that the person cannot see or stop.
- A finished or crashed job Aria never mentions, or Aria answering as if a running job did not exist.
- A background command that escapes the sandbox the foreground one would have been held to.
- Two subagents silently overwriting each other's edits.
- A background job cut by an update without warning.

## What the person sees

### The strip **(owner: option B)**

A row of pills right above the composer, only while the active session has background work. Each pill is one job:

- a dot: green while a process runs, accent while a subagent works, grey dashed outline while queued, red when it
  failed (a failed pill stays until dismissed or the list is opened);
- a mono title: the command's short name (`bun dev`) or the subagent's title (`fix the checkout tests`);
- a muted meta: for a process its URL's port and the time it has run (`:3000 · 12 min`); for a subagent what it is
  doing (`writing`, `running`, `queued`);
- a stop button (■) that stops that job without opening anything.

Clicking a pill opens the background list on that job. At most four pills show; a fifth and more fold into a
`+N` pill that opens the list. A job that ends normally leaves the strip (Aria says so in the chat); one that fails
stays, red, until the list is opened or it is dismissed. **(call)**

The strip shows only the active session's jobs **(owner)**. The pills reuse one pill component shared with the memory
panel's filter chips (`p-memory-chip`), extracted into its own part, not a look-alike **(call, from the owner's
rule: reuse, never copy)**.

### The background list (Alt+A)

The existing subagents view (Alt+A, `ui/chat/parts/subagents.js`) grows into the background view; it is not a new
screen **(owner)**. Its title becomes "Background" ("En segundo plano") and it shows:

1. **Running and queued**, grouped by session when more than one session has jobs (the active session first, each
   group headed by the session's tab title) **(owner: everything, grouped by session)**. Each row: dot, kind
   (`process` / `subagent`), title, meta (URL, duration, status), and actions:
   - process: **Open** (when it has a URL; opens it in the browser through the existing `openUrl`), **Output**
     (expands the row to its last output lines, refreshed while open), **Stop**;
   - subagent: **Session** (opens the subagent's steps in the existing agent-session view, `session.js`), **Stop**.
2. **Finished** (the last 20 of the dock's life plus the ones cut by the last stop): the same rows, with how they
   ended (`exited 0`, `exited 1`, `stopped`, `done`, `failed`, `cut when Aria stopped`) and, for a subagent,
   **Result** (its final answer, in the agent-session view).
3. The session's `parallel` subagents, as today, below.

The hint under the composer that says `alt+a subagents` today says `alt+a background` while the session has
background jobs, and keeps its old text when it only has `parallel` subagents. **(call)**

### In the thread

When a job ends, fails or is cut, a one-line note appears in the thread (the existing `note` part): e.g.
"background · `bun dev` stopped (exit 1)", "background · fix the checkout tests: done". Aria then answers about it
(see below). **(call)**

## What Aria does

### Deciding **(owner: autonomous)**

- A command that does not end by itself (servers, `watch`, `tail -f`, tunnels, `docker compose up` without `-d`):
  always in the background.
- A job expected to take more than a few steps that can be described up front (fix failing tests, a refactor of one
  module, research, a review): a background subagent.
- Anything short or conversational (one or two tool calls, iterative design, debugging step by step with the
  person): in the foreground, as today.
- "in the background" / "do it here" from the person always wins.

These rules live in the background tools' **descriptions**, not in the owner's persona text, so the persona stays
the owner's word for word. (Pi's `promptGuidelines` are not used: Pi only adds them to its default system prompt, and
Aria always has a custom `SYSTEM.md`.) **(call)**

While a writing subagent works in a folder (not a worktree), the `[background]` line says so, and the
`background_agent` description tells Aria not to edit files in that folder itself until it ends. The foreground agent
does not take the lock. **(call)**

### Knowing

- Every user message gets a hidden line (like the clock's `[now]`), only when something is running or queued:
  `[background] running: #3 process "bun dev" (~/Projects/shop, http://localhost:3000, 12 min); #4 subagent "fix the
  checkout tests" (writing, worktree)`. **(call)**
- A `background_list` tool returns the same, plus the finished ones.
- When a job ends, fails or is cut, Aria gets a message (Pi's custom message `aria-background`, shown in the thread as
  the note above):
  - a subagent that ended (done or failed), or a process that died by itself: **the message starts a turn** (or, if
    a run is going, it waits for that run to end: `followUp`), so Aria tells the person what happened, and for a
    subagent summarizes its result and saves its `## Learnings` to memory (subagents cannot write memory);
  - a job the person or Aria stopped: a custom message **without** `triggerTurn` (Pi saves and shows it at once when
    idle, after the current turn when a run is going); no new turn.
  - jobs cut when the dock stopped: one such silent message when the session's Pi first connects after the restart;
    the thread shows the note.
  - The extension acknowledges an event to the dock only after `sendMessage` returned, so a Pi that restarts in
    between gets it again.

### The tools (Pi extension `extensions/background/`, Aria's own) **(call)**

- `background_run({ command, title, folder })`: a command that does not end by itself. When the sandbox extension
  is on (always, under Aria) **it** registers this tool: the command goes through exactly the route `bash` uses
  (`bashRoute`: sandbox, session grants, asking with the permission card, refusing), and the exact command line the
  sandbox decided (boxed with `wrapCommand`, or plain once allowed) is handed to the dock. With `PI_SANDBOX=0` the
  background extension registers it instead and hands the command over plain, as `bash` would run it. It waits up to
  3 seconds and returns the job id, the first output lines, and the URL if one appeared. `title` is at most 6 words
  in the person's language. `folder` (default: Pi's folder) is where it runs; `~` is expanded. **(call: a tool of its
  own, not a `bash` flag, so it exists whether or not the sandbox owns `bash`)**
- `background_agent({ task, title, folder, writes, isolation, model })`:
  - `task`: the full brief (the subagent does not see the conversation);
  - `title`: at most 6 words, the person's language;
  - `folder`: the project the job works in (absolute or `~/…`; default Pi's folder). Pi runs in the projects folder,
    not in a repository, so the model names the repository here; the git root is found from it
    (`git rev-parse --show-toplevel`). It must be inside the projects folder (the same check the sandbox applies to a
    `parallel` subagent's `cwd`, `subagentProblem`), or the tool refuses;
  - `writes` (default true): false for research or review, which run without the per-folder lock and are told not to
    change files;
  - `isolation`: `auto` (default), `folder` or `worktree` (below);
  - `model`: `session` (default: the model and thinking level the session uses now, including a model the person
    picked or the quota fallback switched to; the owner's rule is that subagents inherit the chat's model) or `heavy`
    (the settings' planning model and its effort). Returns at once with the job id and whether it runs or is
    queued.
- `background_list()`, `background_output({ job, lines })` (last lines, default 60, at most 400),
  `background_stop({ job })`.

Without the dock (no `ARIA_JOBS_SOCK`: the owner's own Pi in a terminal) the tools answer that background work needs
Aria's dock, and nothing runs. In a subagent (`ctx.mode === 'json'`, the same test `subagent-mode` uses) the
extension registers nothing and the sandbox registers no `background_run`: `parallel` children inherit Pi's env, so
the variable alone cannot tell them apart. **(call)**

## How it is built

### Ownership: the dock, not Pi **(call)**

Background work belongs to the dock (`crates/aria`), not to Pi:

- a Pi restarts on credential or settings changes; a dev server must survive that (today it does, because Pi leaves
  what it started running);
- the person sees jobs across sessions, and only the dock sees every session;
- the dock already supervises children (`aria-engine`) and already hosts one socket per session (`watch.rs`).

New module `crates/aria/src/jobs.rs` (with `worktree.rs` beside it): one `Jobs` manager for the dock, shared by every
`Chat`. A job is `{ id, number, session, kind: process|agent, title, command | task, cwd, state, started_at,
ended_at, url, exit, isolation, worktree, result }`, with
`state: queued | running | exited | failed | stopped | done | cut`.

### The jobs socket

One Unix socket per session, `jobs-<session>.sock` in the run directory (`$XDG_RUNTIME_DIR/aria`, beside
`watch-<session>.sock`, same short-hash rule for long paths, mode 0600). The sandbox hides `$XDG_RUNTIME_DIR` from
boxed commands, so a boxed command cannot reach it. Pi gets `ARIA_JOBS_SOCK`; children the dock starts for a
subagent do not (no background inside subagents).

JSON lines. The extension sends requests with an `id`; the dock answers with the same `re`:

| Extension → dock | Answer |
|---|---|
| `{op:"hello"}` (on connect) | `{re, ok}`, then every undelivered event for this session |
| `{op:"start", kind:"process", command, title, cwd, env}` | `{re, ok, job}` or `{re, ok:false, error}` |
| `{op:"start", kind:"agent", task, title, cwd, writes, isolation, model}` | `{re, ok, job}` (`job.state` running or queued) |
| `{op:"list"}` | `{re, ok, jobs}` (this session's, running, queued and finished) |
| `{op:"output", job, lines}` | `{re, ok, lines:[…]}` |
| `{op:"stop", job}` | `{re, ok}` |
| `{op:"ack", event}` | none |

| Dock → extension (pushed) | |
|---|---|
| `{event:"ended", seq, job, by:"self"|"stop", result?, applied?, conflict?}` | a job ended; kept until acked |
| `{event:"cut", seq, jobs:[…]}` | jobs cut when the dock last stopped |

Events are kept per session until acknowledged, so a Pi that is restarting gets them when it connects again.

### Processes

- Spawned as `bash -c <the sandbox's command line>`, in the requested folder with the env the extension sent (Pi's
  own), stdin from `/dev/null`, stdout and stderr merged into one pipe, in **its own process group**. Every job (and
  every subagent) is spawned from **one long-lived spawner thread** of the dock, never from a socket thread: on Linux
  the leader gets `PR_SET_PDEATHSIG(SIGTERM)`, and that signal follows the thread that forked, not the process. Under
  systemd the unit's cgroup goes with the dock anyway.
- Output: the last 1000 lines (each at most 2000 chars, ANSI escapes removed) in memory. Nothing on disk.
- URL: the first `http(s)://localhost|127.0.0.1|0.0.0.0|[::1]:<port>…` in the output becomes the job's URL
  (`0.0.0.0` shown as `localhost`).
- Stop: SIGTERM to the group, SIGKILL after 3 seconds.
- At most 10 running processes per dock; an eleventh is refused with a message saying which are running.

### Subagents

- Started by the dock as `pi --mode json -p --no-session [--model <m>] [--thinking <effort>] "Task: <task>"` through
  the same `pi` shim the session uses (the session's env minus `ARIA_JOBS_SOCK`, plus `PI_WATCH_SOCK` of the
  session), so the existing `subagent-mode` (lean prompt, no memory writes, `## Learnings`) and `subagent-watch`
  (live steps) apply unchanged. Before spawning, the dock tells the session's `Watch` the job's title and task
  (`expect_tasks`), so the child shows up named, and links that watch agent to the job (the page then hides it from
  the `parallel` list and uses its steps for the job's **Session** view).
- The result is the text of the last assistant message on its stdout (`message_end`), clipped to 8000 chars; an
  API error (`stopReason: "error"`) or a non-zero exit is `failed`.
- `writes: false` adds to the task: "Do not change any file." It takes no lock.
- At most **3 background subagents run at once** (the owner's quota); more wait as `queued` and start in order.
  `parallel` subagents do not count. **(call: the number was proposed in conversation and not objected to)**
- `--model`/`--thinking`: for `model: session` the session's current model and thinking level (the `Chat` knows both
  from `get_state`); for `heavy` the settings' `models.planning` and `models.planningEffort`.

### Folder or worktree **(owner: Aria decides by the job)**

For a subagent that writes, in a git repository:

- `isolation: auto`: the dock uses a **worktree** when another writing subagent is already working in the same
  repository, and the folder otherwise; the model asks for `worktree` explicitly for a large job (its guideline says
  so).
- Outside a git repository a worktree is impossible: a second writer in the same folder waits (`queued`) until the
  first ends.

A worktree:

- is made at `<repo>/.aria-worktrees/<job id>` from `git stash create` (the person's uncommitted tracked changes,
  without touching their folder) or `HEAD` when there are none; `.aria-worktrees/` goes into the repo's
  `.git/info/exclude` (never its `.gitignore`). Untracked files of the person are not in it.
- When the subagent ends, its changes come back **as uncommitted changes in the person's folder** **(owner)**: the
  dock stages everything in the worktree, takes `git diff --binary --cached <base>`, and applies it to the person's
  folder with `git apply` (working tree only, nothing staged, nothing committed, the person's git settings untouched).
  The subagent's brief tells it not to commit.
- If the patch applies, the worktree is removed (`git worktree remove --force`). If it does not, the worktree stays,
  and the `ended` event says `conflict` with its path: Aria tells the person and asks how to merge. **(owner: "if
  they clash, it asks you")**
- A worktree subagent that was **stopped, failed or was cut** brings nothing back: its worktree stays, and the event
  (and the Finished row) names its path, so Aria can offer to apply or discard it. Nothing is removed without the
  person. In folder mode the edits made so far are already in the person's folder; the event says so. **(call)**

### The page

- `DockMessage::Jobs { jobs }` (not wrapped in a session: the page needs every session's), sent at most 4 times a
  second when something changed, and on `ready`.
- `PageMessage::JobStop { job }`, `PageMessage::JobOutput { id, job, lines }` → `DockMessage::JobOutput { id, lines }`,
  `PageMessage::JobDismiss { job }` (a failed pill leaves the strip).
- New part `ui/chat/parts/background.js` (the strip and the list rows) with its CSS and tests; the shared pill part
  extracted from `memory.css`; `subagents.js` takes the background sections; `state.js` renders the
  `aria-background` custom message as a note (live and from history).
- `tools/ui-sandbox/serve.ts` gains a "Background work" simulation and `check.ts` a check that the strip appears,
  stop sends `jobStop`, and a pill click opens the list.

### Lifetimes **(owner)**

- Hiding the dock stops nothing.
- A session's Pi restarting stops nothing; the extension reconnects and gets missed events.
- Closing a tab stops that tab's jobs (as it stops everything its Pi started today).
- An update waits: `Sessions::busy` is also true while any job runs or is queued. If an update has waited 30 minutes
  only because of background jobs, the sessions that have jobs get one notice ("An Aria update is waiting for the
  background work to finish; stop it from the background list to update now"). The update keeps waiting: it never
  cuts background work by itself. **(call: 30 minutes; never cut)**
- Right before the dock execs the new version it stops every job anyway (none should be left, but the exec keeps
  the process, so nothing would kill a leftover and the new dock would mark a live job `cut`).
- The dock quitting stops every job (process groups killed). The jobs are recorded in `~/.aria/jobs.json` (state
  only, no output) on every change; at the next start, the ones still marked running or queued become `cut`, their
  sessions get the `cut` event, and they appear under Finished.

## Settings

None new in this pass. **(call)** The limits (3 subagents, 10 processes, 1000 lines, 30 minutes) are constants in
`jobs.rs`; they can become settings later.

## Testing

- Rust (`jobs.rs`, `worktree.rs`): real child processes (`sh -c 'echo …; sleep …'`) for start, output, URL, stop,
  process-group kill, exit codes; the queue and the per-folder lock; persistence and `cut`; the socket protocol
  (hello, start, list, output, stop, events kept until ack); worktrees against temp git repos (made from a stash,
  patch applied back unstaged, conflict keeps the worktree, exclude written once).
- `chat.rs`: Pi gets `ARIA_JOBS_SOCK`; closing a tab stops its jobs; `busy` counts jobs.
- Extensions (`bun test extensions`): the client against a fake socket; `background_run` goes through the bash
  route and sends the decided command line; without the dock it refuses; the hidden `[background]` line only when
  something runs; `ended` becomes a custom message with the right `triggerTurn`/`deliverAs`.
- UI (`bun test ui`): the strip (pills, `+N`, stop, failed stays), the list (groups, actions, finished), the note in
  the thread, the hint text; the browser check above.
- `docs/parity.md` is not touched (this is new, not parity).

## Docs

`docs/architecture.md` (the module, the socket, the messages), `extensions/README.md` (the new extension and the sandbox's
`background_run`), `docs/pending.md` (what is left: steering a subagent, macOS orphan cleanup, settings for the limits),
`CHANGELOG.md` (what Aria now writes outside the repo: `~/.aria/jobs.json`, `.aria-worktrees/` and the
`.git/info/exclude` line in the person's repositories).

## Left to the builder

Internal structure of `jobs.rs`, thread layout, exact strings (English and Spanish, in the parts' `TEXTS` and
`ui/i18n/*.json`), the strip's animation (it should feel like the rest of the dock: short, no bounce), and how the
`+N` pill and the row actions are laid out within the design's spacing.
