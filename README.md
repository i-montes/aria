# Aria

A personal assistant that works on your own computer, with your own files, in your language.

Press **Super + A** (**Option + Space** on a Mac) and a dock opens over whatever you are doing. Ask for something and
Aria reads, edits, runs commands, plans and delegates, inside limits you can see. Or type `aria` in a terminal and get
the same sessions there. Or run Aria on a server and attach to it from any computer, so its work goes on while your
laptop is off.

> **Status.** Aria runs on **Omarchy** (Hyprland 0.56 or newer with the Lua configuration) and **macOS 12+**, and its
> service also runs on a headless Linux server. Nothing is published yet: no release binaries and no hosted installer,
> so you build it and install it from a clone ([how](#install)). What changed in each version is in
> [RELEASES.md](RELEASES.md); what is still missing, including known gaps, is in [docs/pending.md](docs/pending.md).

## Three ways to use it

| | What it is | Start with |
|---|---|---|
| **The dock** | A column on the right of the screen you open with a shortcut, with your sessions as tabs. Omarchy and macOS. | [The dock](docs/guide/dock.md) |
| **The terminal** | `aria` in a terminal: the same sessions, as a list and a conversation. Leaving it stops nothing. | [Aria in a terminal](docs/guide/terminal.md) |
| **A hub** | The service on an always-on Linux server, reached from a terminal over ssh. Scheduled tasks and background work keep going while your computer is off. | [Running Aria on a server](docs/guide/hub.md) |

All three are windows onto one thing: the **service**, which owns your sessions. Closing the dock or a terminal never
stops work in progress.

## What it does

**Talk and work**
- A chat backed by [Pi](https://github.com/earendil-works/pi), pinned to one version, working in your projects folder
  (a session you open from a terminal works in the folder you opened it from). Each session is a tab with its own
  engine, up to eight, and a tab in the background keeps working and tells you when it needs you.
- Providers: OpenCode Go, Anthropic with an API key, or your Claude plan. The quota windows show where the provider
  reports them (not for an API key).
- In the dock, while it works, you can send a message that goes in at the next step, or one that waits for the run to
  end.
- A long conversation continues in a fresh session after a compaction; the dock lets you page back through the earlier
  one, a terminal does not yet.
  → [The agent](docs/guide/agent.md), [the dock](docs/guide/dock.md)

**Plan and delegate**
- A router that reads each message and, when you have an OpenCode Go or TypeSafe key, sets the effort, hints and an
  expert for delegated work; experts for backend, frontend, research, review and security; subagents, which you can
  watch in the dock.
- **slice**, Aria's own method for big work: a round of questions, a plan of phases and tasks you approve, then a
  phase built in the background, test first, never pushed. **Ponytail**, a rule the agent follows: the smallest complete
  change.
- Skills, including one that installs a skill or a connector from a link you paste.
  → [The agent](docs/guide/agent.md)

**Work without you watching**
- Scheduled tasks you ask for in plain words ("every day at 7:30 give me my Notion tasks"), each in its own tab, with
  only the permissions you gave it.
- Dev servers and long jobs in the background, owned by the service; a second writing subagent in the same repository
  gets its own git worktree.
- Desktop notifications only when nobody is looking (none on a hub); a watchdog for a stuck run; updates that wait
  until the dock is hidden and nothing is running.
  → [Automation](docs/guide/automation.md)

**Stay safe**
- Every command runs in a sandbox (bubblewrap on Linux; macOS has none yet, and neither does a Linux machine where
  bubblewrap fails, so there every command asks first), and a card asks before anything leaves it.
- Each connector is read-only, ask-first or allowed.
- Your keys live in the system keyring where there is one (not on a hub), and Aria keeps your secrets out of what the
  model reads, within limits the guide spells out. Before each message it keeps a copy of the session's folder (where
  it can); undo, for files and conversation, is in the terminal.
  → [Safety](docs/guide/safety.md)

**Connect**
- Notion and Tavily, set up in the dock; the MCP servers your other apps already have, imported; files attached to a
  message (dock only); memory with its own Engram; a headless browser the agents use to check the pages they build
  (not installed on a hub unless you ask).
  → [Integrations](docs/guide/integrations.md)

**See and hear**
- In the dock only, not in a terminal or on a hub: dictation with Whisper on your computer, so your voice never leaves
  it; Aria's face in the Omarchy bar or beside the Mac's notch; a live trace of every step of a turn (`/debug`).
  → [Voice, face and trace](docs/guide/voice-and-face.md)

## Install

You need Node 22.19 or newer and, to build, Rust, plus the libraries listed in
[development](docs/development.md#build). Build, then install the binary you built:

```sh
cargo build --release
sh install/install.sh --local target/release/aria
```

The installer puts the `aria` binary in `~/.local/bin`, creates `~/.aria/`, installs the pinned engine, Engram (the
memory) and the agent browser (a large download), and a login service (a systemd user unit on Omarchy, a LaunchAgent on
macOS). On Omarchy it asks for `sudo` only if system libraries are missing, and it never edits `~/.config/hypr/`: Aria
registers its shortcut with Hyprland itself. Add `--dry-run` to see what it would do without changing anything. Run it
again to update; an update waits until the dock is hidden and nothing is running.
[Getting started](docs/guide/getting-started.md) has the details, the first-run setup and how to uninstall.

For a server (Linux with systemd; Aria must already be set up on your computer, because `hub init` sends your settings
and keys), build the headless binary, make a key for Aria alone and let `aria hub init` set the server up from the
checkout. It prints its plan and asks before it changes anything:

```sh
sh tools/build-headless.sh
ssh-keygen -t ed25519 -f ~/.ssh/aria-hub -N ''
aria hub init ana@hub.example --client-key ~/.ssh/aria-hub.pub
aria connect aria-hub    # after you add the ssh entry that hub init prints
```

The dedicated key and the ssh entry are part of it: read [the hub guide](docs/guide/hub.md) before you run it, in
particular what a stolen client key could do and why the server's ssh should be reachable only over a private network.

## Documentation

| | |
|---|---|
| **Guides** | |
| [Getting started](docs/guide/getting-started.md) | Install, first run, a first conversation, updating, uninstalling |
| [The dock](docs/guide/dock.md) | The window: opening it, tabs, the composer, panels, settings, keys |
| [Aria in a terminal](docs/guide/terminal.md) | Sessions, keys, commands, undo, context, the footer |
| [Running Aria on a server](docs/guide/hub.md) | Setup, connecting, the security model, limits, troubleshooting |
| [The agent](docs/guide/agent.md) | Providers, models, the router, experts, subagents, slice, skills |
| [Automation](docs/guide/automation.md) | Scheduled tasks, background work, notifications, updates |
| [Safety](docs/guide/safety.md) | The sandbox, connector access, secrets and the keyring, undo |
| [Integrations](docs/guide/integrations.md) | Notion, Tavily, MCP servers, attachments, memory, the browser check |
| [Voice, face and trace](docs/guide/voice-and-face.md) | Dictation, Aria's face, `/debug` |
| **Reference** | |
| [Commands and keys](docs/reference/commands.md) | Every `aria` command, slash command and shortcut |
| [Data and paths](docs/reference/data-and-paths.md) | Where everything lives, what is secret, what is safe to delete |
| [Settings](docs/settings.md) | Every setting, its default and its meaning |
| **For contributors** | |
| [Development](docs/development.md) | Build, run, test, layout, releases, contributing a provider or a platform |
| [Architecture](docs/architecture.md) | How the pieces fit |
| [What is pending](docs/pending.md) | What is missing and what is known to be wrong |
| [Design documents](docs/superpowers/specs/) | One per feature: the decisions and why |

The full index, with a reading order for each kind of reader, is [docs/README.md](docs/README.md).

## Repository layout

```
crates/aria-core/      what does not depend on the system: paths, settings, credentials, providers, keyring,
                       sign-ins, the messages between the dock and its page
crates/aria-engine/    Pi's RPC protocol and the supervised Pi child process
crates/aria-hyprland/  the hotkey and the focused monitor on Omarchy
crates/aria-mac/       the hotkey, the screen and the notch wing on macOS
crates/aria/           the `aria` binary: the service, the dock window, the terminal client, the hub commands
extensions/            Aria's Pi extensions (see extensions/README.md)
agent-files/           the persona, the experts and Aria's own skills
ui/                    the dock's web page: plain HTML, CSS and JS modules, embedded in the binary
install/               the installer and the service units
tools/                 release, checks, the UI sandbox, the parity checks, the hub's build and acceptance scripts
design/                the visual reference and its components
docs/                  guides, reference, architecture, pending work, and the design documents
```

## Development

```sh
cargo test --workspace              # Rust: the five crates
bun test ui                         # the dock page's logic and strings
bun test extensions                 # Aria's Pi extensions
sh tools/parity/suites.sh           # the offline suites, in one run
sh tools/check-macos.sh             # type-checks the macOS build from Linux
```

[Development](docs/development.md) has the whole test map, how to run a build without touching your own data, the
project's rules (everything in English, both platforms stay working, conventional commits decide the version) and how a
release is made. A provider is one module ([adding a provider](docs/adding-a-provider.md)); a platform needs a global
hotkey, the focused monitor and a floating panel that takes the keyboard.

## License

MIT. See [LICENSE](LICENSE).
