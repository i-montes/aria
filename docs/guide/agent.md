# The agent

What runs when you talk to Aria, and how to steer it: the engine, the providers and models behind it, the router that
reads each message, the experts and subagents it can call on, how a long conversation keeps going, and the method it
uses for big work. For anyone who wants to know why Aria did what it did, or to change it.

## The engine

Every conversation is run by [Pi](https://github.com/earendil-works/pi), the agent engine Aria is built on. Aria does
not talk to a model itself: it starts Pi as a child process, sends it your messages and draws what comes back, in the
dock or in a terminal.

- **One Pi per session.** Each tab of the dock (up to eight; tabs of scheduled tasks do not count) has its own Pi. The
  dock starts the Pi of every saved tab when it opens, the tab you left active first (a scheduled task's tab stays
  asleep until its run). A tab in the background keeps working and tells you when it needs you. A session you open
  from a terminal is the same thing: the terminal is a window onto a session the service owns, so closing it stops
  nothing. See [the terminal](terminal.md). On a [hub](hub.md) the Pis run on the server.
- **A pinned version.** The Pi Aria runs is the one named in the repository's `pi-version` file, installed under
  `~/.aria/pi/<version>/`. Its own version check is off, so it changes only when a new Aria brings a new `pi-version`.
  The packages it loads (the parallel-agents package, a filtered Superpowers, Ponytail) are pinned the same way in
  `pi-packages`. Pi needs a recent enough Node, which the installer checks; the dock says so when the engine or Node
  is missing.
- **The working folder.** A dock tab works in your projects folder (`~/Projects` by default; Settings → Sandbox and
  git). A session opened from a terminal works in the folder you opened it from, and keeps it. If that folder is gone
  later, the session falls back to the projects folder and says so. Change the projects folder and the tabs that use it
  restart in the new one when their run settles.
- **Restarts.** A tab's Pi starts again when a credential or a setting that Pi reads at start changes (the persona, the
  experts, the compaction share, the connectors, the working folder), or when you type `/reload` in the dock. A busy tab
  waits until its run settles. A restart loses what lives only in the process: plan mode, the sandbox's permissions for
  the session and a switch to the free model.

## Providers and models

A provider is where the model lives. Aria supports three for chatting, and a fourth key that only gives you the
router's classifier.

| Provider | You connect it with | Daily model | Plan model | Quota bars | Free fallback |
|---|---|---|---|---|---|
| OpenCode Go | an API key | `opencode-go/mimo-v2.6-flash` | `opencode-go/mimo-v2.6-pro` | yes | yes |
| Anthropic, API key | an API key | `anthropic/claude-sonnet-5-5` | `anthropic/claude-opus-5-5` | no | no |
| Anthropic, Claude plan | a sign-in with your Pro or Max account | `claude-bridge/claude-sonnet-5-5` | `claude-bridge/claude-opus-5-5` | yes | no |

The daily and plan columns are what Aria sets when you pick that provider as the main one; you can change both. The
models you can choose from are whatever Pi offers for the providers you have connected.

**TypeSafe** is the fourth key. It gives Aria Jev (see [the router](#the-router-and-jev)) when you have no OpenCode Go
key. It is never a chat model and never appears in a model picker.

### Connecting a provider

In the dock, type `/connect`. It asks which provider:

- **OpenCode Go** and **Anthropic with an API key**: paste the key. Aria checks it with a cheap request to the provider
  before it keeps it.
- **Anthropic with your Claude plan**: choose "Sign in with Claude", finish the sign-in in your browser, and Aria goes
  on by itself. This route runs Claude Code, so it needs Claude Code installed on the machine, and the first time it
  downloads the Claude provider package (about 300 MB). Aria removes any `ANTHROPIC_*` variable from Pi's environment,
  because they break this route. The API key and the sign-in are both stored as Anthropic's credential, so
  only one of them can be connected at a time.

Every Pi restarts after you connect (each when its run settles) so it sees the new credential. The settings screen,
under Providers, lists every provider with Connect (Connect again once connected), and Make main and Remove for a
connected one that is not the main provider; it also has the TypeSafe key. In a terminal, `/signin anthropic` starts the
Claude sign-in; keys for the other providers are entered in the dock's settings. Where the keys are kept is in
[safety](safety.md).

### The daily model and the plan model

The **daily model** answers your messages. The **plan model** is used only while planning with slice (see
[Slice](#slice-plan-first-then-build)), with its own effort level (`low`, `medium` or `high`; `high` by default).
Both are in Settings → Models and effort. The settings also hold `models.daily` and `models.planning`
([reference](../settings.md)).

To change the model of one conversation, type `/model` (or press the model chip under the input in the dock). The
picker lists the models of every connected provider, grouped by provider, with a search field. A pick applies to that
tab's session; a new tab starts from the daily model. The router never changes the chat model: the conversation stays
on one model so its prompt cache is not broken.

### Effort

Effort is how hard the model thinks before it answers. Pi's levels are `off`, `minimal`, `low`, `medium`, `high`,
`xhigh` and `max`; Pi clamps a level to what the model supports.

- **Auto** (the default with OpenCode Go): the router sets the level for each message. A quick message drops to `low`,
  a message that needs thinking goes to at least `high`, and anything else keeps the session's level. A free model runs
  at its top level, since thinking costs nothing there (turn that off with "Run free models at their top level").
- **Fixed**: one level for every message. Auto needs the router, and the setup starts a Claude provider at fixed
  `medium`.

Change it in Settings → Models and effort, or type `/effort auto` or `/effort <level>` (without an argument it asks).
`/effort` writes the setting, so it applies to every session and survives restarts.

### Pay-per-use models

Most OpenCode Go models are in your subscription. OpenCode's other service, Zen, bills per request, so its models
(`opencode/*`) are left out of the pickers unless you turn on "Allow pay-per-use models" in Settings → Models and
effort (shown only with OpenCode Go as the main provider). It controls what the pickers list; it is not a spending
limit. An Anthropic API key always bills per token, and Aria sets no cap on it.

## Quota

Some providers meter your use in windows. Aria reads them and shows them where you look.

- **OpenCode Go**: a 5-hour, a weekly and a monthly window.
- **Claude plan**: a 5-hour and a weekly window, and one per model where Anthropic reports it (Fable, for example).
- **An API key** and **Zen** have nothing to read, so no bars show.

In the dock, a chip beside the assistant's name shows the 5-hour bar. Hover it, or focus it with the keyboard, and the
**quota card** opens with every window, when each resets, the model in use and how full its context is. With several
tabs the header shows tabs instead, and each tab's hover card is the quota card of that tab's own model. A bar turns
amber at 80% and red at 95%. In a terminal the footer carries the same numbers (`5h 62% 1h12`, the other windows after
it). Aria reads a quota once Pi is ready, after a run and when you change model (apart from a model change, at most every
20 seconds); a failed read shows nothing.

### The free fallback

Only OpenCode Go has a free tier, so only it has a fallback. Before each run, Aria's `quota-fallback` extension checks
the 5-hour and the weekly window. When one reaches the **switch threshold** (98% by default) and the model in use is a
paid Go model, it offers to carry on with a free one (`opencode-go/cat-jam-free` by default). The card has two buttons,
"Switch to free" and "Keep subscription", and a countdown (five minutes by default). With no answer when it ends,
Aria switches so the task goes on, or stays, as you set it. Your answer holds for that quota window and is shared by
every tab and subagent, so you are not asked again until the window is full. In a terminal the same question is a plain
choice, and it notes that the provider keeps the prompts of free models for 30 days. `/plan` has its own, lower
threshold (90% by default) and uses the same question.

The switch is not saved: your default model stays in the settings, and when the window resets Aria puts the replaced
model back and says so. All of it is in Settings → Providers (the quota section): both thresholds, the free model,
whether to ask first, the countdown and what no answer does.

## The router and Jev

The router is an extension (`extensions/jev-router`) that reads each message before the model does. It asks **Jev**, a
small classifier model, a few questions in one call, and passes the answers to the model as a hidden `[router]` line
after your message. You do not see the line, and the model is told not to mention it. Jev decides:

| Question | What changes |
|---|---|
| The route: `quick`, `focused`, `parallel` or `think` | `quick` is answered at once with few tools; `focused` is done step by step and checked; `parallel` is split between subagents; `think` is reasoned from what is known, without digging through files. In auto, the route also sets the effort. |
| Does it need memory? | The model searches Engram first (see [integrations](integrations.md)). |
| Does it need the web, or Notion? | The model is pointed at Tavily or Notion. |
| Does it change anything? | If clearly not, the turn is **read-only**: edits and writes outside `/tmp` are refused, and so are shell commands that would write. Answering a question the assistant asks you lifts it for that turn. |
| Is it sizable work? | The assistant offers once to plan it with slice. |

The dock shows the decision as a line above the answer, like `jev focused task · searches memory`. The router also
repeats the answer style you chose after each message (small models forget it), asks for the outcome and a test first on
focused code changes, and with `route=parallel` lists the experts the model can use. The same extension keeps the agent
from stopping processes that are not its session's or project's (no `pkill`, `killall`).

Jev is not a chat model, and which one you get follows the keys you have stored, not the main provider. With an
OpenCode Go key (even if Claude is your main provider) Aria uses OpenCode's free Jev. Otherwise, with a TypeSafe key,
it uses TypeSafe's. With neither, there is no router: no routes, no auto effort, no read-only turns, and the sandbox's
command review is off too. If a classifier runs out of quota or funds, Aria moves to the next one it can reach and tells
you once; if none is left, routing pauses until the first comes back (a terminal footer says `jev off` with the time).
A classifier that is slow or fails never blocks you: that message goes through without a line.

You can turn the router off in Settings → Router (Jev): every message is then handled the same way, and Auto effort is
not available.

## The experts

An expert is a subagent with a persona and a method of its own. Aria ships five, in `agent-files/agents/`:

| Expert | For | Edits files (default) |
|---|---|---|
| `frontend` | Interfaces: components, pages, styles, state, forms, accessibility, client performance. Checks that the result runs. | yes |
| `backend` | Server-side code: APIs, business logic, databases and migrations, integrations, jobs, CLIs. Verifies with tests. | yes |
| `security` | Audits code, configuration and dependencies for vulnerabilities and reports findings with evidence. | no |
| `reviewer` | Reviews code, a diff or a whole project for correctness, design, tests and performance, with prioritized findings. | no |
| `researcher` | Researches a question on the web and in documentation and reports a sourced, dated answer. | no |

Each ends with a delivery in a fixed shape (what it did, what it found, how it verified it). In Settings → Subagents you
turn each expert on or off and mark it read-only. A read-only expert gets only `read`, `bash` and `codemode` (no
`edit` or `write`), and in a read-only turn the router swaps a writing expert for a read-only one. A disabled expert's
file is removed, and the persona only names the enabled ones.

When the model splits work with `parallel`, each task without an expert is given the one Jev picks for it, or none when
nothing fits clearly. You do not have to choose experts yourself.

## Subagents

A subagent is another Pi, started by the main one for a piece of work. Two tools start them:

- `parallel` (from the parallel-agents package) runs a few independent tasks at once and waits for them. The router
  asks for it on `route=parallel`.
- `background_agent` runs a long job in the background while the conversation goes on, owned by the dock. See
  [automation](automation.md) for jobs. With a git repository, a second writer in the same folder gets its own git
  worktree (outside your repository), and its changes come back as one uncommitted patch you can review.

Subagents get a short prompt of their own instead of the persona. They can read memory but not write it, they cannot
ask you questions, they have your date and time, and each ends with a `## Learnings` section that the main agent
may save. A `parallel` subagent uses the default model, and the main agent is told never to pick another unless you
name it; a background subagent uses the model of the tab that started it, or the plan model when the main agent asks for
heavy work. Two settings (Settings → Subagents) shape them: stop a subagent after a number of turns (off by default),
and save their sessions as files instead of discarding them.

To watch them, press `Alt+A` in the dock: each subagent with what it is doing, and the background jobs above it. Select
one to open its session and see its prompt, its text and its tool calls as they stream. `Alt+A` again goes back to the
chat. On macOS `Option+A` may not fire (see [automation](automation.md)); the background pills open the same view. The
dock only: a terminal has no watch view, and Pi's own `/agents` command does nothing in the dock.

## Compaction and long conversations

When the context fills to a share of the model's window, Pi **compacts** it: it summarizes the older part and keeps the
rest, so replies stay fast and cheap.

- The share is `compaction.percent`, **40** by default. Settings → Context and memory has a slider (20–80); in a
  session, `/autocompact <0-100>` sets it and applies at once, and `/autocompact` alone says what it is now. `0` turns
  automatic compaction off. Aria turns the share into a reserve for each model's window.
- Compact by hand with `/compact` in the dock, optionally with what to keep (`/compact keep the API decisions`). The
  terminal has no `/compact` of its own.
- How full the context is shows in the quota card's Context row in the dock, and in the terminal's footer
  (`ctx 31 / 40`, amber from five points before compaction). `/context` in a terminal draws what fills it: the system
  prompt, tools, skills and the conversation. `/session` in the dock gives the session in numbers.

### One conversation that keeps going

Pi keeps everything in its session file when it compacts, so the file would keep growing and every start would slow
down. So Aria **rotates**: after a compaction, once the tab is quiet (no run, no waiting prompt, no open question), it
continues the conversation in a fresh Pi session whose first message is the summary, followed by everything Pi kept. A
tab whose file passes 8 MB without compacting is asked to compact first, which then rotates. The tab, its name, model
and effort, and your permissions for the session stay as they are. The MCP connectors reconnect, so a stateful one (a
browser, say) starts over. What you send meanwhile waits a moment and then goes through in order.

In the dock the conversation shows a divider, "Earlier in this conversation", with the summary folded under it, and a
"Show earlier" button that loads the previous 40 messages at a time, back through every earlier session of that tab.
The old files stay on disk. A terminal shows a divider at the compaction but does not page back yet. How it works:
[the design](../superpowers/specs/2026-10-21-session-rotation-design.md).

## Questions

When a decision is yours and the model cannot settle it from your request, the code or a sensible default, it asks with
`ask_user`: one card with one to four independent questions, each with two to four options, its recommended option
first, and always room to write your own answer. The model writes them in your conversation language. A card also has "Talk about this" and you can skip or dismiss it. Your
answers go back to the model as the tool's result, not as a new message, so the router does not classify them. In a
terminal the same questions are a form in the editor's place. Subagents never get the tool: a subagent picks the most
reasonable option and says so, or reports that it is blocked.

## Slice: plan first, then build

Slice is Aria's method for sizable work: a feature, a redesign, a change across several parts. It splits the work into
a short plan you approve once, then builds it in the background, test first, while you keep talking to Aria.

**Starting it.** Two ways, and only these two: type `/plan <idea>`, or accept the assistant's offer. When Jev thinks a
message is sizable and changes things, the assistant asks once, "Plan it in slices" (recommended) or "Just do it". The
model never starts slice on its own. `/plan off` leaves planning. If you turned off "Turn on `/plan` and `/execute`"
(Settings → Subagents), neither the command nor the offer exists.

**Phase A, the plan.** Aria switches to the plan model (its effort from Settings), reads the code the work touches, and
asks one round of at most four questions, each with a recommended answer and only about what the code cannot settle: the
stack for a new project, whether an expensive-to-change decision applies (several tenants, roles and permissions,
offline or sync, several languages or currencies) and, if the repository has no `docs/` folder, whether to keep `docs/`
in it. What it assumed goes under Assumptions for you to correct. Then it writes the plan to
`docs/plans/<date>-<slug>.md`:

- the **backlog**: at most 15 one-line user stories, each vertical (data, backend and UI) and useful on its own;
- the **phases**: one per story of this cycle, by default only the first, each with two or three acceptance criteria
  (what you see or get, not how it is built) and its tasks, which are the **tests**, named at the highest practical
  level, plus manual checks where a criterion cannot be tested.

It commits only the plan file (or keeps `docs/` out through `.gitignore` if you chose that), then shows it. The dock
draws a document card and a phases card; a terminal shows the plan's title, its phase and task count and the choices.
**Approve and build** or **Change something**. While planning, Aria can write only in `docs/plans/`, `.git` and
`.gitignore` of a repository under your projects folder, and `/tmp`, and it routes nothing; a message that is not about
the plan ends plan mode. In a session already past about 100,000 tokens, it asks first, since planning re-reads the
whole thing on a pricier model. It uses the free model when the quota is past the plan threshold, after the same
question as above.

**Phase B, the build.** Approving ends plan mode and starts a background subagent for the first open phase, on the
tab's model. It works in your folder, not a worktree, so its commits land where you can see them. For each task it
writes the test and watches it fail, commits the test, writes the least code that passes, runs the tests, ticks the task
in the plan and commits. Commits are local and conventional; **it never pushes, merges or rebases**. It stops and asks,
ending with a question the assistant brings to you, when a design decision is not in the plan, when it would add
something nobody asked for, when a test would have to change or be deleted to pass, after two attempts without green, or
when a step grows past about 50 lines or three files. Answer, and the assistant launches the phase again with your
answer.

While it builds, the plan's marks become a task list: the dock's task dock and the terminal's task strip show each task as
to do, active or done. Watch what the subagent does with `Alt+A`; every change is also a commit. When a phase is done,
a read-only review subagent goes over its commits with Ponytail's review. The assistant tells you the must-fix findings
(and asks whether to fix them now or move on), goes through the manual checks with you, and launches the next phase if
the plan has one. A phase that ends without finishing is reported, with a choice between launching it again and changing
the plan. `/execute [plan path]` builds the next phase that still has open tasks of the latest plan (or the one you name),
which is how you resume after a restart.

Its design is in [the spec](../superpowers/specs/2026-10-23-slice-design.md).

## Ponytail

Ponytail is a rule: the smallest complete change wins. Before writing code the agent asks whether it needs to exist at
all, then whether the codebase, the standard library or a dependency already does it, then whether one line does, and
only then writes the minimum, never skipping validation, error handling, security or accessibility. Slice's build
follows it, and by design so does a plain small fix.

Aria installs it as a Pi package (pinned in `pi-packages`), keeping its Pi extension and two of its skills: `ponytail`
and `ponytail-review`, which reviews a change against the same rule. Aria hides Ponytail's startup notice and status
line, because the dock has its own. There is no setting to turn it off. The package's own `/ponytail lite|full|ultra`
command (`full` by default) sets its level.

## Skills

A skill is a folder with a `SKILL.md` that teaches the agent a method. Pi lists the skills it can use and loads one when
the task calls for it; you can also run one by name from the `/` menu, where skills show after the commands. Aria ships:

| Skill | What it does |
|---|---|
| `slice` | The planning and build method above; only `/plan` and the offer start it. |
| `install-from-link` | Reads a link you paste and installs what it points to. |
| `verify-in-browser` | Opens a page the agent built in a headless browser and checks it before saying it works (see [integrations](integrations.md)). |
| `ponytail`, `ponytail-review` | Ponytail's rule and its review (above). |
| `systematic-debugging` | From Superpowers: find the cause before fixing. |

`systematic-debugging` is the default of "Skills that run on their own" in Settings → Subagents; you can add other
Superpowers skills there (brainstorming, writing plans, test-driven development, code review, verification before
finishing). With slice on, the persona tells the agent not to use the rest of Superpowers.

**Installing a skill, an MCP server or a Pi package.** Paste a link in the chat and say "install this". The agent reads
the page, works out what it is and calls the `install` tool, which shows a card with exactly what will be installed
(secrets masked). Nothing is installed until you confirm. A skill and a Pi package are available in the next session; a
new MCP server connects right away. The model never writes `mcp.json` or the skills folder itself. Connector access
levels are in [safety](safety.md).

## The persona, the language and the clock

Aria's persona is a file, `SYSTEM.md`, that Aria writes for Pi from your settings every time a session starts. These
settings (Settings → Assistant) shape it:

- Your **name**, so Aria greets you by it.
- The **conversation language**: Spanish, English, Portuguese or any language you type. Aria answers in it unless you
  write in another. The dock's own text is complete in English and Spanish; other languages show it in English.
- The **answer style**: `very-short` (the default), `brief`, `detailed`, `technical` or `teaching`. It goes in the persona
  and the router repeats it after each message.
- The Git rules, memory and its project name, which experts exist and whether slice is on, from their own sections.

The settings do not offer to rename Aria. If you edit `SYSTEM.md`, an expert or an extension under `~/.aria/agent/`,
Aria keeps your version and stops updating that file; it says so when a session starts.

The model has no clock of its own, so every message you send carries a hidden `[now]` line with the local date, time,
weekday and time zone. It is how "today", deadlines and the dates in memories come out right.

## Commands that belong to the agent

Type `/` in the dock to open the menu. It lists the dock's own commands, then the commands Pi's extensions register,
then skills. A terminal has its own short list ([terminal](terminal.md)) and then the session's commands, which include
the extension ones below.

| Command | What it does | Where |
|---|---|---|
| `/model` | Switch this session's model | dock, terminal |
| `/connect` | Connect OpenCode Go or Anthropic | dock |
| `/signin anthropic` | Sign in with your Claude plan | terminal |
| `/effort [auto\|level]` | Set the reasoning effort | any session |
| `/autocompact [0-100]` | Set, or show, the compaction share | any session |
| `/compact [what to keep]` | Compact the context now | dock |
| `/plan <idea>`, `/plan off` | Start slice's planning, or leave it (only while "Turn on `/plan` and `/execute`" is on) | any session |
| `/execute [plan path]` | Build the next open phase of a plan (same switch) | any session |
| `/session` | The session in numbers: messages, tools, tokens, cost, context | dock |
| `/context` | What fills the context | terminal |
| `/copy`, `/export` | Copy the last answer; export the session as an HTML page | dock |
| `/reload` | Restart this tab's Pi to read its extensions, skills and templates again | dock |
| `/verify on\|off` | The project's checks after a run that edits files | any session |
| `/sandbox`, `/sandbox reset` | Show what commands may write; forget the session's permissions | any session |

`/memory`, `/schedules` and `/debug` also appear in the dock's menu; they belong to [integrations](integrations.md),
[automation](automation.md) and [voice and face](voice-and-face.md). `/aria-rotate` is Aria's own, sent for you after a
compaction. The full list by surface is in [the commands reference](../reference/commands.md).

## Limits

- You cannot steer a subagent: its session view in the dock has no input.
- The countdown on the free-model question shows the seconds it had when it opened; it does not count down.
- A Claude plan's quota windows come from an endpoint Anthropic does not document and have not been checked against a
  real account. An expired sign-in token shows no bars until the next run.
- A provider is removed only from Settings → Providers (Remove, not for the main provider); there is no command for it,
  and removing the credential leaves the Claude provider package installed. Closing the connect card does not cancel a
  sign-in still waiting in the browser (it times out after five minutes).
- A Claude plan does not work on a [hub](hub.md#sign-in-from-a-terminal) today: the dock's `/connect` installs the Claude provider package and a hub has no dock.
- Pi packages come only from a local folder (Settings → Integrations); npm, git and a gallery are not built.
- Slice builds one phase at a time. The goals engine, where Aria pursues a goal on its own, is not built.

The full list of what is missing is in [pending](../pending.md). To add a provider, see
[adding a provider](../adding-a-provider.md); the extensions behind all of this are in [the extensions
README](../../extensions/README.md).

## See also

- [Automation](automation.md): background jobs, scheduled tasks, the turn watchdog
- [Safety](safety.md): the sandbox, connector access and where the keys live
- [Integrations](integrations.md): memory, Notion, Tavily and MCP servers
- [The dock](dock.md) and [the terminal](terminal.md): where you talk to the agent
- [Settings](../settings.md): every setting and its default
