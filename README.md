# Aria

A personal assistant that lives on your desktop. Press **Super + A** (**Option + Space** on a Mac) and a dock opens over whatever you are doing, on the monitor you are looking at, to work with your files and tasks and to talk to you in your language.

> **Status: 0.7.** The dock, its chat and the installer work on **Omarchy** (Hyprland 0.56+ with the Lua configuration)
> and **macOS 12+** (Apple silicon or Intel). Next on [the roadmap](docs/superpowers/specs/2026-10-18-aria-base-design.md#roadmap):
> autonomous goals (phase 3). What changed in each version is in [RELEASES.md](RELEASES.md); what is still missing, in
> [docs/pending.md](docs/pending.md) (on macOS: click-through, notification clicks and a sandbox for commands).

## What it does

- **A dock on the right of the screen**, opened with `Super + A` on Omarchy or `Option + Space` on macOS (or the
  shortcut you choose in the settings, with fallbacks when one is taken), on the monitor you are looking at.
- **A chat backed by its own pinned [Pi](https://github.com/earendil-works/pi)**, working in your projects folder. Up to
  eight tabs, each with its own Pi, so a tab in the background keeps working and tells you when it needs you. A long
  conversation goes on in a fresh session after a compaction, and you can page back through the earlier one.
- **Providers**: OpenCode Go, Anthropic with an API key, or your Claude plan (`/connect` from the chat), with the quota of
  each window on hand and, on OpenCode Go, a fallback to a free model when a quota runs out.
- **Messages while it works**: Enter puts yours in at the run's next step; Alt+Enter keeps it for when the run ends.
  The `/` menu has Pi's own commands (`/compact`, `/session`, `/copy`, `/export`, `/reload`), Aria's (`/debug`, `/schedules`, `/memory`) and your skills.
- **The owner's extensions**: a sandbox for every command (bubblewrap on Omarchy; on macOS every command asks first), a
  router that picks the model and the expert, subagents you can watch (Alt+A), questions, autocompaction, and a check of
  the pages agents build in a headless browser.
- **Background work**: dev servers and long jobs run in the background, owned by the dock, as pills above the input;
  subagents get their own git worktree when the repository is taken.
- **Connectors** (Notion, Tavily, and the MCP servers your other apps already have), each read-only, ask-first or
  allowed; **attachments** by the paperclip, drop or paste; **memory** with its own Engram.
- **Keys in the system keyring** (Keychain, Secret Service), never in plain files under `~/.aria`.
- **Scheduled tasks**: ask "every day at 7:30 give me my Notion tasks" and Aria does it by itself, in the task's own tab
  with a desktop notification, with the permissions you gave it (anything else is refused, never asked); a time missed
  while the computer slept or Aria was off runs once when it is back. `/schedules` lists, pauses and deletes them.
- **A live trace to evaluate the agent**: `/debug true` in a tab opens a waterfall to the left of the chat with every LLM
  call, tool call and subagent of the turn, with its time, tokens and cost, the slowest call marked and each step's
  input, output or error a click away; `/debug false` closes it.
- **Updates that never cut a run**: the dock restarts into a new version once it is hidden and idle.

## Install

```sh
curl -fsSL https://<your site>/install.sh | sh
```

The installer detects the system. It needs **Node 22.19 or newer** (it checks, and says so if not). It puts the `aria` binary in `~/.local/bin`, creates `~/.aria/`, installs the pinned Pi engine in `~/.aria/pi/<version>` and the pinned `agent-browser` the agents check the pages they build with (`~/.aria/agent-browser/<version>`, linked as `~/.aria/bin/agent-browser`, with its Chrome in `~/.agent-browser/browsers`; if that step fails the installer only warns), and installs a login item: a **systemd user unit** on Omarchy, a **LaunchAgent** (`~/Library/LaunchAgents/com.aria.dock.plist`) on macOS. On Omarchy it adds Aria's system libraries, asking for `sudo` only if they are missing, and, when the Omarchy shell is there, puts Aria's face in its bar next to the weather (`aria bar install`, through Omarchy's own `omarchy plugin enable`). It **never edits `~/.config/hypr/`**: Aria registers `Super + A` (or the shortcut chosen in the settings) with Hyprland itself each time it starts. On macOS it needs no `sudo`, no Accessibility permission and no system configuration: `Option + Space` (or the chosen one) is registered with Carbon, and the dock runs as an accessory app (no Dock icon).

Run it again to update. An update does not cut a run: the running dock restarts into the new version once it is hidden and no session is working (`--restart-now` restarts it at once). To remove Aria: `sh install.sh --uninstall` (it asks before deleting `~/.aria`). The browser `agent-browser` downloaded stays in `~/.agent-browser/browsers`; delete it by hand if you want it gone. Add `--dry-run` to any command to see what would happen without changing anything.

### Commands

| Command | What it does |
|---|---|
| `aria toggle` / `aria show` | Opens or closes the dock (the shortcut does the same) / only opens it; starts it first if it is not running |
| `aria start` / `aria quit` | Starts the dock under its login service / stops it (its background work and engines first). In the dock, Cmd+Q asks first and quits on a second press |
| `aria bar install` / `remove` | Puts Aria's face in the Omarchy bar, next to the weather, or takes it out (the installer runs both; nothing happens without the Omarchy shell) |
| `aria keyring status` | Whether the keys are in the keyring, and which (names only) |
| `aria keyring off` / `on` | Moves every key back into plain files in `~/.aria` and turns the keyring off / turns it on again (the dock moves them at its next start) |
| `aria secret --home <root> <account>` | Prints one key from the keyring: what the `!aria secret …` commands in Pi's files run |
| `aria --version` | The version installed |

### Build it yourself

You need Rust and `bun` (for the UI tests). On Omarchy you also need the libraries `gtk3`, `webkit2gtk-4.1` and `gtk-layer-shell`; on macOS the AppKit/WebKit frameworks come with the system.

```sh
cargo build --release
sh install/install.sh --local target/release/aria
```

## Where things live

| Path | What |
|---|---|
| `~/.aria/settings.json` | Every setting of the project, in one file ([reference](docs/settings.md)) |
| the system keyring | Every key (Keychain on macOS, the Secret Service on Omarchy), under the service `aria`; `aria keyring status` lists them by name |
| `~/.aria/mcp.json` | The MCP servers (Notion, Tavily, the ones you import), in Pi's format; their keys are `!aria secret …` commands that read the keyring |
| `auth.json`, `mcp-auth.json` | Provider and MCP credentials, in Pi's formats: sealed in the keyring, with a live copy in a private temporary folder (`$XDG_RUNTIME_DIR` on Omarchy) while the dock runs. With `credentials.keyring` off (`aria keyring off`) they are plain files in `~/.aria`, mode `600` |
| `~/.aria/schedules.json` | The scheduled tasks: prompt, time, days or date, permissions, last run and result ([design](docs/superpowers/specs/2026-10-21-schedules-design.md)) |
| `~/.aria/sessions.json` | The open session tabs (ids, titles, the active one); the conversations are Pi's session files in `~/.aria/agent/sessions` |
| `~/.aria/pi/<version>/` | The pinned Pi engine (the version is in the repo's `pi-version` file) |
| `~/.aria/agent/` | Pi's own directory, regenerated by Aria: its settings, links to the credential files, the persona, sessions |
| `~/.aria/agent/extensions/`, `agents/`, `SYSTEM.md` | the owner's extensions, experts and persona, written by Aria from your settings (one you edit there is kept) |
| `~/.aria/agent/skills/` | Skills: Aria's own `install-from-link` and `verify-in-browser`, and the ones you ask Aria to install (paste a link and say "install this"; MCP servers you install that way go in `~/.aria/mcp.json`) |
| `~/.aria/agent/aria.json` | what the extensions read from your settings (models, quota, effort, compaction, router, sandbox, subagents); no secrets |
| `~/.aria/engram/`, `~/.aria/bin/engram` | Aria's memory: its own Engram database and the pinned Engram the installer brings (`engram-version`) |
| `~/.aria/agent-browser/`, `~/.aria/bin/agent-browser` | the pinned `agent-browser` (`agent-browser-version`) the agents use to check pages; its Chrome lives in `~/.agent-browser/browsers` |
| `~/.aria/agent/npm/`, `git/` | the pinned packages they use (`pi-packages`), installed by the installer |
| `~/.aria/bin/` | `pi` (a shim to the pinned version, for Pi's own subagents), `node` (a link the service uses), `engram` and `agent-browser` |
| `~/.local/bin/aria` | The binary |
| `~/.config/systemd/user/aria.service` | Starts the dock at login on Omarchy |
| `~/.config/omarchy/plugins/aria.face/` | Aria's face in the Omarchy bar (a Quickshell widget the binary carries, from `install/omarchy/`); `shell.json` gets its place, with a `shell.json.bak.<epoch>` backup first |
| `~/Library/LaunchAgents/com.aria.dock.plist` | Starts the dock at login on macOS; output in `~/.aria/logs/aria.log` |
| `~/.aria/run/control.sock` | The socket `aria toggle` talks to on macOS (on Omarchy it is `$XDG_RUNTIME_DIR/aria/control.sock`) |

The repository only holds code; your data never lives in it.

## Repository layout

```
crates/aria-core/      what does not depend on the system: paths, settings, credentials, providers,
                       key validation, Anthropic sign-in, the dock <-> page messages
crates/aria-hyprland/  the hotkey, the focused monitor and reload events, through Hyprland's own IPC
crates/aria-engine/    Pi's RPC protocol and the supervised Pi child process
crates/aria/           the `aria` binary: the dock window, the chat and the glue between the crates
agent-files/           the owner's persona and experts, written into Aria's Pi
extensions/            the owner's Pi extensions, adapted to run in Aria (see its README)
ui/                    the dock's web page: plain HTML, CSS and JS modules, embedded in the binary;
                       ui/chat/parts/ holds one module per component of the design
tools/ui-sandbox/      the page without Pi, to see and try every chat state
tools/parity/          the parity checks against the owner's Pi (the matrix is docs/parity.md)
install/               the installer and the systemd unit
design/                the visual reference (Aria.html) and its components
docs/                  how the pieces fit, the settings, how to add a provider, what is still pending (docs/pending.md)
```

Start with [docs/architecture.md](docs/architecture.md).

## Development

```sh
bun tools/ui-sandbox/serve.ts      # the dock's page with a stubbed bridge and the design's simulate buttons (permission, quota, ask, tools, tabs…)
bun tools/ui-sandbox/check.ts      # drives the chat in headless Chrome against it: queue, focus, Esc order, pickers, question card
cargo test --workspace      # Rust: core, Hyprland, engine, dock
cargo test -p aria --test live_pi -- --ignored   # the real Pi answering a prompt (needs the setup done)
bun test ui                 # the UI's logic and its strings
bun test extensions         # the owner's Pi extensions that run inside Aria (their sandbox tests are Linux-only)
sh install/install.test.sh  # the installer's Omarchy branch, without touching your system (skipped on macOS)
sh install/install-mac.test.sh  # the installer's macOS branch (skipped on Omarchy)
sh tools/check-macos.sh     # type-checks the macOS build from Linux (clippy for aarch64-apple-darwin, zig through mise; nothing runs)
sh tools/release.test.sh    # the release script, in a scratch repository
sh tools/hooks/pre-push.test.sh   # the pre-push hook (turn it on with: git config core.hooksPath tools/hooks)
ARIA_DEBUG=1 target/release/aria   # logs state changes and the page's console to the terminal
ARIA_HOME=$(mktemp -d) target/release/aria   # try a first run without touching ~/.aria
```

Versions follow SemVer and are git tags (`vX.Y.Z`); what each one changed is in [RELEASES.md](RELEASES.md), and
`sh tools/release.sh` makes the next one (it decides the version from the commits). `aria --version` says which one is installed.

## Contributing a provider or a platform

- A provider is one module: [docs/adding-a-provider.md](docs/adding-a-provider.md).
- A platform (Mac, other Linux compositors, Windows) needs the three things `aria-hyprland` and `aria-mac` do today: a global hotkey, the focused monitor, and a floating panel that takes the keyboard when it opens. The panel lives in `crates/aria/src/dock/`: `gtk.rs` on Omarchy, `mac.rs` on macOS, both behind the same `Dock` API; `ui.rs` hides the event loop. The installer branches on `uname -s` for the service manager and the artifact.
