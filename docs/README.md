# Aria documentation

Everything about Aria, by what you want to do. If you have not installed it yet, start with
[Getting started](guide/getting-started.md). The [project README](../README.md) is the short version.

## I want to…

| I want to… | Read |
|---|---|
| install Aria, or update or remove it | [Getting started](guide/getting-started.md) |
| use the dock: tabs, the composer, panels, keys | [The dock](guide/dock.md) |
| use Aria from a terminal | [Aria in a terminal](guide/terminal.md) |
| run Aria on a server and use it from another computer | [Running Aria on a server](guide/hub.md), and [from a second computer](guide/hub.md#connect-from-another-computer) |
| choose a provider or model, understand quota, effort, the router and the experts | [The agent](guide/agent.md) |
| plan big work with slice | [The agent](guide/agent.md#slice-plan-first-then-build) |
| use subagents, or watch them work | [The agent](guide/agent.md#subagents) |
| schedule something to run by itself | [Automation](guide/automation.md#scheduled-tasks) |
| run a dev server or a long job in the background | [Automation](guide/automation.md#background-work) |
| know when Aria notifies me | [Automation](guide/automation.md#notifications) |
| understand how an update waits for my work | [Automation](guide/automation.md#updates-that-wait) |
| understand the sandbox and its permission cards | [Safety](guide/safety.md#the-sandbox) |
| set what a connector may change | [Safety](guide/safety.md#connectors-what-each-may-change) |
| know where my keys and secrets are | [Safety](guide/safety.md#your-secrets) and [Data and paths](reference/data-and-paths.md#the-keyring) |
| let Aria tidy one project, with a way back | [Safety](guide/safety.md#let-aria-work-on-one-project-safely) |
| undo what Aria did to my files | [Safety](guide/safety.md#checkpoints-and-undo), and [the terminal](guide/terminal.md#undo-and-the-changes-line) for the keys (the dock has no undo) |
| know what leaves my computer, and where my text goes | [Safety](guide/safety.md#what-leaves-your-computer) |
| connect Notion or Tavily, import MCP servers, attach files | [Integrations](guide/integrations.md) |
| dictate, see Aria's face, or watch a turn step by step | [Voice, face and trace](guide/voice-and-face.md) |
| look up a command, a slash command or a shortcut | [Commands and keys](reference/commands.md) |
| find a file, a socket or a service, or know what is safe to delete | [Data and paths](reference/data-and-paths.md) |
| change a setting by hand | [Settings](settings.md) |
| build Aria, run the tests, or contribute | [Development](development.md) |
| understand how it is built | [Architecture](architecture.md) |
| find out what is missing or known to be wrong | [What is pending](pending.md) |

## Reading orders

**New user.** [Getting started](guide/getting-started.md), then [the dock](guide/dock.md), then
[the agent](guide/agent.md). Read [Safety](guide/safety.md) before you let Aria run commands on files you care about.

**Terminal user.** First get the client onto the computer: [Getting started](guide/getting-started.md) for the full
install, or, to use a hub from a computer with no dock, [Connect from another computer](guide/hub.md#connect-from-another-computer)
(a binary, a key, an ssh entry). Then [Aria in a terminal](guide/terminal.md) and [Commands and keys](reference/commands.md);
[Running Aria on a server](guide/hub.md) is the rest of the hub's story.

**Setting up a server.** [Running Aria on a server](guide/hub.md) end to end (especially "The security model"), then
[Safety](guide/safety.md#on-a-hub), [Automation](guide/automation.md#on-a-hub) and
[Data and paths](reference/data-and-paths.md).

**Contributor.** [Development](development.md), [Architecture](architecture.md), then the design document of the part
you will touch in [superpowers/specs](superpowers/specs/), and [What is pending](pending.md).

## The guides

| Guide | What it covers |
|---|---|
| [Getting started](guide/getting-started.md) | Requirements, install, the first-run setup, a first conversation, updating, uninstalling |
| [The dock](guide/dock.md) | Opening and closing, the column, session tabs, the composer, questions and approvals, settings, notifications, keys |
| [Aria in a terminal](guide/terminal.md) | The session list, the conversation, the editor, undo, context, project checks, keys, over ssh |
| [Running Aria on a server](guide/hub.md) | What a hub is, `aria hub init`, `aria connect`, sign-in, the sandbox on a server, what a remote terminal may do, security, troubleshooting |
| [The agent](guide/agent.md) | The engine, providers and models, quota, the router and Jev, experts, subagents, compaction, slice, Ponytail, skills |
| [Automation](guide/automation.md) | Scheduled tasks, background work, notifications, the turn watchdog, updates that wait, session status |
| [Safety](guide/safety.md) | The sandbox, connector access levels, secrets, the keyring, checkpoints and undo, checks after edits |
| [Integrations](guide/integrations.md) | Connectors, Notion, Tavily, importing MCP servers, attachments, memory, the browser check |
| [Voice, face and trace](guide/voice-and-face.md) | Dictation, Aria's face on Omarchy and macOS, the live trace |

## Reference

| Document | What it covers |
|---|---|
| [Commands and keys](reference/commands.md) | Every `aria` command and flag, environment variables, slash commands, keyboard shortcuts |
| [Data and paths](reference/data-and-paths.md) | Files, folders, sockets, services, the keyring and file modes, on Omarchy, macOS and a hub |
| [Settings](settings.md) | Every setting in `~/.aria/settings.json` |

## For contributors

| Document | What it covers |
|---|---|
| [Development](development.md) | Build, run a scratch copy, the test map, the repository, the project's rules, releases |
| [Architecture](architecture.md) | The crates, the dock and its page, the messages between them, the terminal and hub, background work, schedules |
| [Adding a provider](adding-a-provider.md) | Every place a new provider touches |
| [Parity](parity.md) | The checks that Aria still matches the maintainer's own Pi setup |
| [Extensions](../extensions/README.md) | The Pi extensions Aria ships and what each does |
| [Pending](pending.md) | What is missing, grouped by area, and the known gaps |
| [Ideas](ideas.md) | Things not started |
| [Design documents](superpowers/specs/) | One per feature: the decisions and the reasons. The plans they were built from are in [superpowers/plans](superpowers/plans/) |
| [RELEASES](../RELEASES.md) | What each version changed |

## Words used here

- **Service**: the process that owns your sessions and their work; the dock and the terminal are windows onto it.
- **Hub**: a service on a server with no window, reached over ssh. **Session**: one conversation, a tab in the dock.
- **Pi**: the agent engine Aria runs, one per session. An **extension** adds behaviour inside it.
- **Expert**: a persona the agent can delegate a task to. **Engram**: Aria's memory.
- **Router**, **Jev**: the router reads each message; Jev is the small classifier it asks. Jev never chats.
- **slice**: Aria's method for big work (plan, then build). **Ponytail**: a Pi package, a rule of the smallest change.
- **Connector**, **MCP**: a service such as Notion that gives the agent extra tools, and the protocol it speaks.
