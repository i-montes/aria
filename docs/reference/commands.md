# Commands and keys

Every `aria` command, every slash command and every keyboard shortcut, by surface: the command line, the dock and the terminal. Look things up here; the guides explain how to use them.

Platforms: Omarchy (Linux with Hyprland), macOS, and a hub (headless Linux running `aria serve`). A row says so when it is not for all three. The binary is `~/.local/bin/aria`; on macOS that folder is not on the default `PATH`, so type the full path or add the folder to your `PATH`.

- [The `aria` command line](#the-aria-command-line)
- [The installer and the developer scripts](#the-installer-and-the-developer-scripts)
- [Environment variables](#environment-variables)
- [Slash commands in the dock](#slash-commands-in-the-dock)
- [Slash commands in the terminal](#slash-commands-in-the-terminal)
- [Keyboard shortcuts](#keyboard-shortcuts)
- [See also](#see-also)

## The `aria` command line

`aria` takes one command word. `aria --help` does not exist: an unknown word prints a usage line and exits with 2. "Internal" commands are run by Aria, the installer or ssh, not typed by people.

| Command | What it does | Exit codes | For |
|---|---|---|---|
| `aria` (stdin and stdout are a terminal) | The terminal client. Starts the service first when nothing answers (detached, waits up to 3 s). See [the terminal](../guide/terminal.md) | 0 on leaving; 1 when the service cannot start, is older than this `aria`, or the link fails | all |
| `aria` (no terminal) | Runs the dock, as `aria dock`. Only an old service unit does this | as `aria dock` | Omarchy, macOS |
| `aria dock` | The service with the dock window: sessions, background work, scheduled tasks, terminals. What the login service runs | 0 when stopped or already running (it says so); 1 when the control socket cannot be bound | Omarchy, macOS |
| `aria serve` | The service with no window, for a computer with no desktop. Terminals reach it with `aria connect`. A dock and a serve on one computer exclude each other. See [the hub](../guide/hub.md) | 0 when stopped or already running; 1 when the socket fails or terminals have no way in | all (the installer's unit for it is Linux only) |
| `aria toggle` | Shows or hides the dock; the global shortcut runs it. Starts `aria dock` first when none runs (waits up to 3 s). A service with no window logs it and ignores it | 1 when the dock did not start | Omarchy, macOS |
| `aria show` | Only shows the dock; a notification's click uses it. Starts it first, as `toggle` does | 1 when the dock did not start | Omarchy, macOS |
| `aria start` | Starts the login service. On Linux it asks `systemctl --user cat` which unit is installed and starts `aria.service`, or `aria-serve.service` on a hub (it says which; `aria.service` wins when both exist). On macOS it runs `launchctl kickstart gui/<uid>/com.aria.dock`. Prints `Aria is running` | 1 when no unit is installed or the service manager refuses | Omarchy, hub, macOS |
| `aria quit` | Asks the running service to stop; its background work and engines stop first. Waits up to 2 s for the answer. Prints `Aria is not running` and exits 0 when nothing listens | 1 when a service answered badly | all |
| `aria connect [user@]host` | The terminal client on another computer's service, over `ssh -T -o ServerAliveInterval=15 -o ServerAliveCountMax=3 <host> ~/.local/bin/aria client-stdio`. No port of Aria's is open (a sign-in listens briefly on the hub's own loopback address). See [the hub](../guide/hub.md) | 1 ssh missing, hub down or older, ssh timed out (60 s), link failed; 2 no host, or a host that starts with `-`; 3 the protocols differ | all |
| `aria hub init <user@host> [flags]` | Sets a server up as a hub over ssh: probes it, prints the plan, and after a yes uploads the tree and the binary, runs the installer there, sends settings and credentials, restarts the service and, with `--client-key`, allows a restricted key. English only. Run it from the checkout. Flags below | 0 done, or declined (`nothing was changed`); 1 a step failed, the key was skipped or refused, or the hub has no `xz` and no current Node (the installer would stop); 2 bad arguments | Linux and macOS computers; the hub is Linux |
| `aria keyring [status]` | Prints whether the keyring is on (`credentials.keyring`), whether it is available, and the account names it holds (never values). No word means `status` | 0; 2 an unknown word | all |
| `aria keyring on` | Turns `credentials.keyring` on; the dock moves the keys at its next start (`aria quit && aria start`) | 0; 1 when the setting cannot be saved | all |
| `aria keyring off` | Moves every key back into plain files in `~/.aria` and turns the keyring off. Refuses while the service runs: `aria quit` first | 0; 1 when the service runs or the move fails | all |
| `aria bar install` | Puts Aria's face in the Omarchy bar, next to the weather. Does nothing, and says so, without the Omarchy shell | 0; 1 on failure | Omarchy |
| `aria bar remove` | Takes the face out of the bar | 0; 1 on failure; 2 for a missing or unknown word (as for `install`) | Omarchy |
| `aria --version`, `-V`, `version` | Prints `aria <version>` | 0 | all |
| `aria update-ready` | Internal. The installer runs it after replacing the files: the service restarts into the new binary once it is hidden and no session has a run going. The service answers `ok` on the socket; the command itself prints nothing | 1 when no service answers within 2 s (the installer then restarts the service itself) | all |
| `aria client-stdio` | Internal. The bridge `ssh <host> aria client-stdio` runs on the hub: the client's JSON lines on stdin and stdout, the hub's `clients.sock` on the other side. Reads the client's `hello`, checks the protocol, marks the connection remote, relays only JSON objects, caps a line at 8 MiB | 0 the client closed; 1 hub down or gone; 3 another protocol; 4 a line over the cap | Linux hub |
| `aria hub-import` | Internal. `hub init` runs it on the hub: reads one line from stdin (at most 16 MiB), a bundle of `settings.json`, `auth.json`, `mcp.json`, `mcp-auth.json`; writes each into `~/.aria` (mode 600, the first copy it replaces kept once as `<name>.bak`); prints the names it wrote | 0; 1 on a bad bundle or a write error | Linux hub |
| `aria secret --home <root> <account>` | Internal. Prints one keyring value, no newline. The `!aria secret …` entries in `auth.json` and `mcp.json` are this command | 0; 1 no such account or keyring error; 2 usage | all |

### `aria hub init` flags

| Flag | What it does |
|---|---|
| `<user@host>` | Required, once. Anything else that does not start with `-` is refused as a second host |
| `--binary <path>` | The Linux binary to upload. Default: the aarch64 build under `target/aarch64-unknown-linux-gnu/release/aria` in the checkout. It must match the hub's CPU (`aarch64` or `x86_64`); build one with `tools/build-headless.sh` |
| `--projects <dir>` | The hub's projects folder (`sandbox.projectsDir` in the settings it receives). Default: `<hub home>/Projects` |
| `--client-key <pubkey file>` | A bare public key line to allow on the hub as `restrict,command="…/aria client-stdio"`, so it can open Aria and nothing else. Skipped (exit 1) when the sandbox does not work on the hub, unless `--allow-no-sandbox` |
| `--allow-no-sandbox` | Install the client key even when `bwrap` is missing or blocked on the hub |
| `--settings` | Replace the hub's `settings.json` with this computer's. Without it a hub that has one keeps it |
| `--credentials` | Replace the hub's `auth.json`, `mcp.json` and `mcp-auth.json` with this computer's. Without it a hub that has them keeps them |
| `--yes`, `-y` | Do not ask `Go ahead? [y/N]`. Without a terminal and without `--yes` the answer is no |

### The control socket

`aria toggle`, `show`, `quit` and `update-ready` write one line to the service's control socket: `$XDG_RUNTIME_DIR/aria/control.sock` on Linux (then `/run/user/<uid>`, then `/tmp`), `$ARIA_HOME/run/control.sock` on macOS (`~/.aria/run`). `clients.sock`, the terminals' socket, sits beside it. [Where things live](data-and-paths.md) has the rest.

| Line | The dock does | A service with no window does |
|---|---|---|
| `toggle` | Shows or hides the window | Logs and ignores it |
| `show` | Shows the window | Logs and ignores it |
| `quit` | Stops (answers `ok` first) | Stops (answers `ok` first) |
| `update-ready` | Restarts into the new binary when hidden and idle (answers `ok` first) | Restarts when idle (answers `ok` first) |
| `hotkey` | Internal: the macOS shortcut's handler sends it in the process; it toggles | Logs and ignores it |

## The installer and the developer scripts

`install/install.sh` installs or updates Aria in place. It detects the platform; it never writes to `~/.config/hypr/`. See [getting started](../guide/getting-started.md) and [the hub](../guide/hub.md).

| Flag | What it does |
|---|---|
| (none) | Installs, or updates in place: the binary into `~/.local/bin`, the pinned Pi, Engram and agent-browser under `~/.aria`, the service unit or LaunchAgent. A running dock restarts into the new version once it is hidden and idle |
| `--local <path>` | Installs a binary you built instead of downloading one. Refused when the path is not an executable file |
| `--dry-run` | Prints what would happen (`would: …`) and changes nothing. Combines with the others |
| `--restart-now` | Restarts a running dock at once, even mid-run (a changed service file does the same) |
| `--headless` | A server with no desktop (Linux only): installs `aria-serve.service` (runs `aria serve`), the private Node when the system's is missing or too old, `bubblewrap` and `iproute2`, and turns on linger. No bar widget, no agent-browser |
| `--with-browser` | With `--headless`: also installs agent-browser and its Chrome (a large download). Without `--headless` they are always installed |
| `--uninstall` | Removes the service, the binary and (on Omarchy) the bar widget and the shortcut bind; asks before deleting `~/.aria`. With `--headless` it removes the server unit |
| `-h`, `--help` | Prints the header of the script |
| `--print-unit`, `--check-node`, `--print-pi-version`, `--print-node-version`, `--link-node`, `--install-engram`, `--install-node`, `--install-packages`, `--enable-linger`, `--install-agent-browser`, `--restart-dock`, `--can-delete` | Test hooks for `install.test.sh` and `install-mac.test.sh`; not for use |

Any other flag stops with `unknown option` (exit 1). The installer reads its pins from the repository files `pi-version`, `pi-packages`, `engram-version`, `agent-browser-version` and `node-version`.

| Script | What it does |
|---|---|
| `sh tools/build-headless.sh [triple]` | Cross-compiles `aria serve` (the `gui` feature off) with zig for a server; default `aarch64-unknown-linux-gnu`, also `x86_64-unknown-linux-gnu`. The last line of stdout is the binary's path |
| `sh tools/release.sh [X.Y.Z] [--next] [--skip-checks]` | Decides the next SemVer version from the commit subjects since the last tag, runs the suites, bumps `Cargo.toml`, writes the `RELEASES.md` section, commits and tags; `--next` only prints the version. Pushes nothing |
| `sh tools/check-macos.sh` | Type-checks the macOS build from Linux (`clippy` for `aarch64-apple-darwin` through zig); nothing links or runs |

How to build, test and release is in [development](../development.md).

## Environment variables

Only variables the code or the installer reads. Pi has its own; those Aria sets for it are listed last.

| Variable | Read by | What it does |
|---|---|---|
| `ARIA_HOME` | `aria`, the installer | Aria's data folder; default `~/.aria`. On macOS the control and clients sockets live in `$ARIA_HOME/run`; on Linux they stay in `$XDG_RUNTIME_DIR/aria` whatever `ARIA_HOME` says. A folder other than `~/.aria` gets its own keyring service (`aria:<folder>`), so a test instance never meets the real keys |
| `ARIA_SSH` | `aria connect`, `aria hub init` | The program to run instead of `ssh`; empty means `ssh` |
| `ARIA_DEBUG` | the dock (Omarchy, macOS) | When set (any value). Linux: logs the dock's state changes to stderr and the page's console to stdout. macOS: logs each message from the page, and the edit menu's installation, to stderr |
| `ARIA_DEBUG_PI` | the service | When set: Pi's stderr goes to the service's stderr instead of nowhere |
| `ARIA_FACE` | the macOS notch wing | Forces a face by name (`idle`, `thinking`, `reading`, `editing`, …) to look at it |
| `WEBKIT_DISABLE_DMABUF_RENDERER`, `WEBKIT_DMABUF_RENDERER_FORCE_SHM` | the Linux dock | When neither is set, Aria sets `WEBKIT_DMABUF_RENDERER_FORCE_SHM=1`; when you set either, it leaves them |
| `PI_SANDBOX=0` | the `sandbox` extension | Turns the command sandbox off; `background` then registers `background_run` itself |
| `PI_QUOTA_FALLBACK=0` | the `quota-fallback` extension | Turns the switch to the free model off |
| `PI_SUBAGENT_LEAN=0` | the `subagent-mode` extension | Gives subagents Pi's full prompt instead of the lean one, to compare runs |
| `ARIA_BIN_DIR` | the installer | Where the binary goes; default `~/.local/bin` |
| `ARIA_DOWNLOAD_URL`, `ARIA_PI_VERSION_URL`, `ARIA_NODE_VERSION_URL` | the installer | Where it downloads the binary and the published pins when run without a checkout (`curl … \| sh`) |
| `ARIA_PI_VERSION` | the installer | A Pi version instead of the one in `pi-version` |
| `ARIA_NODE_DIST` | the installer | The Node download mirror for a server's private Node; default `https://nodejs.org/dist`; must be `https://` (or `file://`) |
| `ARIA_ENGRAM_URL`, `ARIA_ENGRAM_SHA256` | the installer | Engram's tarball address and its checksum instead of the pinned ones |
| `ARIA_AGENT_BROWSER_VERSION` | the installer | An agent-browser version instead of the one in `agent-browser-version` |
| `XDG_RUNTIME_DIR`, `XDG_CONFIG_HOME` | `aria` | The runtime folder of the sockets (Linux); the config folder the Omarchy bar and the MCP import look in |
| `HYPRLAND_INSTANCE_SIGNATURE`, `WAYLAND_DISPLAY`, `COLORTERM` | `aria` | Finding Hyprland; copying a terminal selection with `wl-copy`; 24-bit colour in the terminal when `truecolor` or `24bit`, else 256 colours |
| `PI_CODING_AGENT_DIR`, `PI_SKIP_VERSION_CHECK`, `PONYTAIL_QUIET_STARTUP`, `PONYTAIL_HIDE_STATUS`, `ARIA_PROJECTS_DIR`, `ARIA_SETTINGS`, `PI_WATCH_SOCK`, `ARIA_JOBS_SOCK`, `ARIA_GRANTS`, `ARIA_EXE`, `ARIA_SECRET_<NAME>` | set by Aria for each Pi | Internal: Pi's folder, the working folder, the settings file, the subagent watch socket, the dock's jobs socket, session grants, the binary's path, and a secret a command asked for by name. Any `ANTHROPIC_*` variable is removed from Pi's environment |

Test and evaluation scripts read more (`ARIA_PI`, `ARIA_PI_DIST`, `ARIA_SMOKE_MODEL`, `ARIA_PARITY_*`, `ARIA_SHIM_*`, `ARIA_FACE_RENDER`); they are described in [development](../development.md).

## Slash commands in the dock

Type `/` at the start of the composer. The list narrows as you type, ranked by name, then by letters in order, then by description. `↑` `↓` move, `Enter` or `Tab` puts the command in the composer, `Esc` closes. A word that matches nothing is sent as an ordinary message. Aria's and extension commands run at once, even while Aria works; a skill or a prompt template waits in the queue like a message. See [the dock](../guide/dock.md).

Aria's own commands, built into the dock page:

| Command | What it does | Which doc explains it |
|---|---|---|
| `/memory` | Opens the memory panel | [integrations](../guide/integrations.md) |
| `/schedules` | Opens the scheduled tasks: when each runs, pause or delete | [automation](../guide/automation.md) |
| `/model [query]` | Opens the model picker, filtered by `query` | [the agent](../guide/agent.md) |
| `/connect [anthropic\|opencode-go]` | Opens the card that connects a provider | [the agent](../guide/agent.md) |
| `/compact [what to keep]` | Compacts the context (Pi's `compact`) | [the agent](../guide/agent.md) |
| `/session` | Messages, tool calls, tokens, cost and context of this session | [the dock](../guide/dock.md) |
| `/copy` | Copies the last answer | [the dock](../guide/dock.md) |
| `/export` | Exports this session as an HTML page (Pi's `export_html`) and says where | [the dock](../guide/dock.md) |
| `/reload` | Restarts this tab's Pi so it reads its extensions, skills and templates again | [the agent](../guide/agent.md) |
| `/debug [on\|off\|true\|false]` | The live trace of this tab's turns, to the left of the chat; no word toggles | [voice and face](../guide/voice-and-face.md) |

Commands from Pi's extensions that Aria ships (`extensions/`), listed by Pi (`get_commands`):

| Command | What it does | Which doc explains it |
|---|---|---|
| `/effort [auto\|off\|minimal\|low\|medium\|high\|xhigh\|max]` | Reasoning effort: `auto` follows the route; a level stays fixed across sessions. No word asks with a list | [the agent](../guide/agent.md) |
| `/autocompact [0-100]` | Compacts automatically at this percent of the context window; `0` turns it off; no word says the current value | [the agent](../guide/agent.md) |
| `/plan <idea>`, `/plan off` | Starts slice's planning: one round of questions, then a plan to approve; `off` leaves plan mode. Present unless `subagents.planExecute` is off | [the agent](../guide/agent.md) |
| `/execute [plan path]` | Builds the next open phase of a plan in the background. Same setting as `/plan` | [the agent](../guide/agent.md) |
| `/sandbox`, `/sandbox reset` | Shows what the sandbox can write and cannot see; `reset` forgets this session's permissions | [safety](../guide/safety.md) |
| `/verify [on\|off]` | The project's checks after each run that edits files; `off` stops them for this session | [the terminal](../guide/terminal.md) |
| `/agents` | Pi's own subagent viewer; it works only in Pi's terminal interface and does nothing in the dock (use `Alt+A`) | [the dock](../guide/dock.md) |
| `/aria-rotate` | Internal: Aria sends it after a compaction to continue the conversation in a new session file | [architecture](../architecture.md) |

Skills appear under their own heading, by short name (`/brainstorming`; Pi's name is `skill:brainstorming`). Aria writes `install-from-link`, `verify-in-browser` and `slice` into Pi's folder, Ponytail brings `ponytail` and `ponytail-review`, and the Superpowers skills follow the settings. Skills you install are listed too, and so are Pi prompt templates you add.

## Slash commands in the terminal

Type `/` at the start of the editor; the picker lists the commands below and the session's own (what the dock's list shows from extensions, skills and templates, by Pi's names such as `skill:brainstorming`). `Enter` runs the picked command, `Tab` or a click only completes it. See [the terminal](../guide/terminal.md).

The client's own commands:

| Command | What it does | Which doc explains it |
|---|---|---|
| `/exit` | Closes this session for real (its Pi stops, its background jobs end, its checkpoints and undo history are deleted, its tab closes in the dock) and leaves. `Ctrl+C` leaves without closing | [the terminal](../guide/terminal.md) |
| `/verbose [on\|off]` | Everything (thinking, every tool with its output) or one line per step; no word flips it. Saved as `display.verbose` | [the terminal](../guide/terminal.md) |
| `/model` | A picker of the models you can use; the choice applies to this session. Over `aria connect` it lists only the model in use | [the agent](../guide/agent.md), [the hub](../guide/hub.md#connect-to-the-hub) |
| `/sessions` | Back to the list of sessions | [the terminal](../guide/terminal.md) |
| `/undo` | Opens Rewind, as `Esc Esc` does: your messages, oldest first, each with what its turn changed; going back puts the files, the conversation or both back to before the one you pick. Over `aria connect`, and in the projects folder, only the conversation goes back | [the terminal](../guide/terminal.md), [safety](../guide/safety.md#checkpoints-and-undo) |
| `/context` | What fills the context: system, tools, skills, memory, messages, free space and the compaction reserve | [the terminal](../guide/terminal.md) |
| `/schedules` | The scheduled tasks as a panel: pause or resume, delete. A task waiting for approval shows as such; the panel does not offer approval, so it is approved in the dock | [automation](../guide/automation.md) |
| `/signin anthropic`, `/signin notion` | Signs in to Claude or Notion: shows an address to open, then takes the address the browser ends on | [integrations](../guide/integrations.md) |

Other things typed in the editor:

| Input | What it does |
|---|---|
| `@<text>` | Opens a list of file names in the session's folder (the projects folder when the session has none); the pick becomes `@<path> ` in the message |
| `!<command>` | Runs the command in the session, output joins the context. Refused over `aria connect` |
| any other `/word` | Not handled by the terminal: sent to the session as typed. Extension commands, skills (`/skill:<name>`) and prompt templates run in Pi; it gets no checkpoint |

The dock's `/memory`, `/connect`, `/compact`, `/session`, `/copy`, `/export`, `/reload` and `/debug` are not in the terminal. Over `aria connect` a new session has no folder of the terminal's, and secrets cannot be saved.

## Keyboard shortcuts

### The global shortcut

It shows or hides the dock from anywhere; a hub has none. Set it in the settings' Assistant section (press the keys; `Esc` cancels, `Backspace` goes back to the default) or type it, or write the `shortcut` setting ([settings](../settings.md)).

| | Omarchy | macOS |
|---|---|---|
| Default (an empty or unparsable setting) | `SUPER + A` | `Option + Space` |
| Fallbacks, tried in order when one is taken | `SUPER + SHIFT + A`, `SUPER + ALT + A` | `Control + Option + Space`, `Command + Option + Space` |
| Modifiers | `SUPER`, `CTRL`, `ALT`, `SHIFT` | `Command`, `Control`, `Option`, `Shift` |
| How it is registered | At run time through `hyprctl eval` (Hyprland 0.56+ with the Lua config), again after every config reload; it runs `aria toggle`. A classic config is not supported | Carbon, no permission needed; tried again every 10 s while it fails or a fallback stands in |

Format: one or more modifiers, then one key (a letter, a digit, `Space` or `F1`–`F12`), joined by ` + `; case and spaces do not matter; at least one modifier. The order tried is the chosen shortcut, the default, then the fallbacks.

### In the dock

| Keys | Context | What they do |
|---|---|---|
| `Enter` | Composer | Sends. While Aria works it is read at the next step |
| `Alt+Enter` | Composer, while Aria works | Keeps the message for when Aria finishes |
| `Shift+Enter` | Composer | New line |
| `Enter` | Composer, while dictating | Stops dictation and sends once the text is in |
| `↑` `↓` | A picker (`/`, `@`, `/model `) | Move, wrapping at the ends |
| `Enter`, `Tab` | A picker | Choose |
| `Esc` | Anywhere in the dock | Closes the first thing open: dictation (cancels), a picker, the `/connect` card, an open trace step, a panel (memory, schedules, subagents, session) back to the chat, the tab menu. With none, hides the dock. It never stops a run: use the Stop button |
| `Alt+D` | Chat | Starts dictation; again stops it and writes the text |
| `Alt+A` | Chat | The subagents view, and back. On macOS it is `Option+A`. It fires on the key labelled A on any layout (by its place or by its letter), so the character `Option+A` types does not matter; `Ctrl+Alt+A` and `Cmd+Option+A` do not fire it |
| `Cmd+Q` twice within 2 s | macOS | Quits Aria (the first press shows `Press ⌘Q again to quit Aria`). On Omarchy `Super+Q` is the compositor's |
| `Cmd+C` `V` `X` `A` | macOS | Copy, paste, cut, select all |
| `↑` `↓` | Setup and settings screens | Move focus between the controls |
| `Enter` | A field in setup or settings | Saves a key field; applies a typed shortcut; in another text field moves to the next control |
| `Enter`, `Esc` | Renaming a tab (double-click it) | Keeps the name / cancels |
| `Enter` | A card's free-answer field | Next question, or sends on the last |
| `Enter` | Secret value field, the pasted-secret name field, the `/connect` key field | Keeps the secret, keeps the pasted one by its name, saves the key |
| `↑` `↓` `Home` `End` `←` `→` `Enter` `Space` | The `/debug` trace panel | Move through the steps, fold or unfold, open a step's details |
| `Esc`, `Backspace`, `Tab` | The shortcut row while it records | Cancels / resets to the default / stops recording and moves on |

### In the terminal

| Keys | Context | What they do |
|---|---|---|
| `Ctrl+C`, `Ctrl+D` | Anywhere | Leave. The session and its work go on |
| `↑` `↓` | The session list | Move |
| `Enter` | The session list | Opens the session, or a new one in the folder you are in (last row) |
| `Esc`, `q` | The session list | Leave |
| `Enter` | Editor | Sends. While Aria works it is read at the next gap |
| `Alt+Enter`, `Shift+Enter` | Editor | New line. `Shift+Enter` needs a terminal with the extended keyboard protocol (kitty, ghostty, foot, WezTerm, recent Alacritty); `Alt+Enter` needs the terminal to send `Alt` (on macOS, `Option` set as `Meta`). A pasted line break is a new line too |
| `Ctrl+Enter` | Editor (extended keyboard protocol) | While Aria works: stops the turn and sends what is in the editor once it settles (an empty editor: nothing happens, the turn goes on). Idle: sends. Does nothing while a card is open |
| `Esc` | A turn is running | Stops the turn |
| `Esc` `Esc` within 500 ms | Idle | Opens the undo list, as `/undo` does |
| `↑` `↓` | Editor | Older and newer messages (only from an empty editor, or while browsing; a draft is never replaced) |
| `←` `→`, `Backspace` | Editor | Move the cursor, delete |
| `PageUp` `PageDown`, the mouse wheel | Conversation | Scroll half a screen / three lines. Sending returns to the bottom |
| `Ctrl+O` | Session | Everything or one line per step, as `/verbose` |
| `Ctrl+T` | Session | Folds the task strip to its first row, and back |
| `Ctrl+L` | Session | Redraws the screen |
| `↑` `↓`, `Enter`, `Tab`, `Esc` | The `/` or `@` list, the model picker | Move, choose (`Tab` only completes a command), close |
| `1`–`9` | A question | Picks that option, while nothing is typed; typing writes a free answer |
| `Enter`, `Esc` | A question | Next question or send / skip |
| `1`–`9` | A permission | The option's number (the first is "once"). The card's last line says so. The sandbox, write and connector cards are answered by number or click only |
| `y` or `Enter`, `n` | A yes-or-no question | Yes, no |
| `Enter`, `Backspace`, `Esc` | A text or secret field | Send, erase, cancel |
| `↑` `↓`, `Enter`, `1`–`9`, `Esc` | The undo list | Move, open, pick a choice by number, back or close |
| `↑` `↓`, `Enter`, `d`, `y`, `Esc` | `/schedules` | Move, pause or resume, ask to delete, confirm (any other key cancels), close |
| `Enter`, `Backspace`, `Esc` | `/signin` | Send the pasted address, erase, close |
| `Esc` | `/context` | Closes the panel |
| a click | Rows, options, lists, the task strip, `… N more lines` | The same as choosing them; a click on `… N more lines` shows the whole output |
| a drag | Anywhere | Selects text and copies it to the clipboard on release |

## See also

- [The dock](../guide/dock.md) and [the terminal](../guide/terminal.md): how the two surfaces work
- [The agent](../guide/agent.md): `/effort`, `/plan`, `/compact`, models and skills
- [Automation](../guide/automation.md): `/schedules` and background work
- [Safety](../guide/safety.md): `/sandbox`, the keyring, `/undo`
- [The hub](../guide/hub.md): `aria serve`, `aria connect`, `aria hub init`
- [Settings](../settings.md) and [where things live](data-and-paths.md)
