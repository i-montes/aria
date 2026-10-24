# Getting started

Aria is a personal assistant that lives on your desktop: a dock opens over whatever you are doing, and you ask it to work on your files and tasks in your language. This guide is for someone who has recently heard of Aria or is about to install it. It covers what you need, installing, the first run, a first conversation, updating and uninstalling.

## What you need

- **Omarchy**: Hyprland 0.56 or newer with the Lua configuration (the classic format is refused), inside a running Hyprland session, on an Arch-based system with `pacman`. To check the configuration, run `hyprctl eval "return 1"` in a terminal: Aria asks Hyprland the same thing, and only the answer `ok` counts as the Lua configuration. With the classic one the dock starts but shows "Aria needs Hyprland 0.56 or newer with the Lua configuration. Yours uses the classic format." and registers no shortcut. Aria does not convert the configuration; moving to Lua is Hyprland's own procedure.
- **macOS**: macOS 12 or newer, Apple silicon or Intel. No `sudo`, no Accessibility permission and no system configuration are needed.
- **Node 22.19 or newer**, with `npm`, on your `PATH`. Aria's engine (Pi) runs on it. The installer stops and says so when Node is missing or too old.
- **A provider account** for the first run: an OpenCode Go key (from your OpenCode Go account), an Anthropic API key, or a Claude Pro or Max plan. See [the first run](#the-first-run).

A computer with no desktop can run Aria as a hub that terminals connect to; that is a different install, described in [running Aria on a server](hub.md).

## Install

Nothing is published yet: there are no release binaries and no hosted installer (see [what is pending](../pending.md)). Today you build the binary yourself and install it from the clone you built it in.

### From a clone

You need Rust and the build libraries listed in [development](../development.md#build); `bun` is only for the tests. On Omarchy the libraries that page names come to this line (the repository does not check them):

```sh
sudo pacman -S --needed cmake clang gtk3 webkit2gtk-4.1 gtk-layer-shell alsa-lib vulkan-headers shaderc
```

On macOS you need `cmake` (Homebrew has it). Then, from the clone:

```sh
cargo build --release
sh install/install.sh --local target/release/aria
```

Run the installer from the clone: it reads `pi-version`, `pi-packages`, `engram-version` and `agent-browser-version` from the files next to it (and `node-version` too, but only on a server that needs its own Node). It installs the binary you built instead of downloading one, and running it again updates. When it ends it says "Aria is installed. Press Super+A to set it up." on Omarchy, or "Press Option + Space" on macOS.

### When a release exists

The intended path is one line that downloads the binary and the pin files and installs them. It cannot work until releases and a hosted `install.sh` are published (its address is a placeholder in the script), so do not use it today:

```sh
curl -fsSL https://<your site>/install.sh | sh
```

The installer detects the system, and running it again updates.

### What the installer puts where

Both platforms:

- The `aria` binary in `~/.local/bin`.
- Aria's data folder, `~/.aria` (mode `700`), with the pinned Pi in `~/.aria/pi/<version>` and the Pi packages Aria's extensions use in `~/.aria/agent`. The versions are pinned in the repository's `pi-version` and `pi-packages` files.
- Aria's memory, the pinned Engram, as `~/.aria/bin/engram`, checked against the checksum in `engram-version`. It keeps its own database in `~/.aria/engram`.
- A link `~/.aria/bin/node` to the Node it found (and `~/.aria/bin/claude` when Claude Code is installed), because the login service has no shell profile.
- The browser the agents use to check the pages they build: the pinned `agent-browser` in `~/.aria/agent-browser/<version>` (pinned in `agent-browser-version`), with its Chrome in `~/.agent-browser/browsers`. This is a large download. If it fails, the installer only warns and Aria works without it.
- A login item that starts the dock at login.

Omarchy only:

- The system libraries `gtk3`, `webkit2gtk-4.1`, `gtk-layer-shell`, `bubblewrap`, `iproute2`, `libsecret`, `alsa-lib` and `vulkan-icd-loader`. The installer asks for `sudo` (`pacman -S --needed`) only when some are missing.
- A systemd user unit, `~/.config/systemd/user/aria.service`, enabled and started.
- When the Omarchy shell is there, Aria's face in the bar next to the weather (`aria bar install`, which uses Omarchy's own `omarchy plugin enable`). If that fails, the installer only warns. See [Aria's face](voice-and-face.md).

macOS only:

- A LaunchAgent, `~/Library/LaunchAgents/com.aria.dock.plist`, with its output in `~/.aria/logs/aria.log`.
- The binary is signed again ad hoc as `com.aria.dock`, so macOS keeps one identity for it. macOS may ask for microphone access again after an update, because an ad hoc signature can look like a new app to it. Whether it does has not been tried live yet ([pending](../pending.md)).

What it never touches: `~/.config/hypr/`. On Omarchy the dock registers its shortcut with Hyprland itself, every time it starts, and nothing is written to disk for it. On macOS the shortcut is registered through Carbon, which needs no permission.

### Installer options

Give these to `sh install.sh`. With `curl | sh`, pass them after `sh -s --`, for example `curl -fsSL https://<your site>/install.sh | sh -s -- --dry-run`.

| Option | What it does |
|---|---|
| (none) | Installs, or updates in place. |
| `--local <path>` | Installs a binary you built yourself instead of downloading one. |
| `--dry-run` | Prints what would happen (`would: …`) and changes nothing. Works with every other option. |
| `--restart-now` | Restarts a running dock at once, even in the middle of a run. See [Updating](#updating). |
| `--uninstall` | Removes Aria. See [Uninstalling](#uninstalling). |
| `--headless` | Installs `aria serve` as a user service on a server with no desktop (Linux). With `--with-browser` it adds the agent browser too. See [the hub guide](hub.md). |
| `--help` | Prints the usage. |

## The first run

The first time the dock starts and there is no `~/.aria/settings.json` yet, a notice appears in the top-right corner for about eight seconds: "Aria is installed. Press Super + A to set it up." The notice names the shortcut actually in use. On macOS, if every shortcut is taken or the system refused the key handler, it says so and tells you to run `aria toggle` to open the dock; on Omarchy the dock binds its first shortcut even when it is taken, so the notice always names one.

| | Omarchy | macOS |
|---|---|---|
| Open or close the dock | `Super + A` | `Option + Space` |
| Fallbacks, when the first is taken | `Super + Shift + A`, then `Super + Alt + A` | `Control + Option + Space`, then `Command + Option + Space` |
| Open it without the shortcut | `aria toggle` | `~/.local/bin/aria toggle` |

On macOS `~/.local/bin` is not on the default `PATH`, so type the full path or add the folder to your `PATH` before using `aria` in a terminal.

On macOS, while Aria runs and holds `Option + Space`, other apps no longer get that key. You can pick another shortcut later in the settings: see [the dock guide](dock.md).

Press the shortcut. The dock opens on the right of the screen you are looking at, with the welcome screen.

### The welcome screen

You choose one of three paths:

- **Quick setup** (about a minute): your provider and its sign-in, then your name. Everything else takes the recommended defaults.
- **Customize** (12 steps, or 13 with OpenCode Go, which adds the quota step; the card says "up to 13 steps" until a provider is chosen): every step below.
- **Skip for now**: nothing is configured and the dock closes. The welcome screen comes back the next time you open it, and there is no chat until a setup is finished.

Each step has Back and Continue. In Customize, from the third step on, "Use recommended defaults for the rest" jumps to the review.

### Providers

The first step connects a provider. You can connect several; the one marked **Main** sets your default models and the quota meter, and every connected provider's models are in the chat's model picker. Continue stays disabled until the main provider is connected and its credential has been checked.

| Provider | What it asks for |
|---|---|
| OpenCode Go | Its API key. A subscription with a quota Aria can read, a free model to fall back to, and Jev (the router's classifier) included. |
| Anthropic · API key | Your own key from `console.anthropic.com`. |
| Anthropic · Sign in with Claude | A sign-in in your browser with your Claude Pro or Max account. Running models with it needs Claude Code installed on the computer. |
| TypeSafe (under "Only for Jev, Aria's router") | Its API key. It only provides Jev and is never the main provider or a chat model. |

A key is checked with one cheap request to the provider before it is saved, and "Check and save" tells you if it was not accepted. Keys go to the system keyring (Keychain on macOS, the Secret Service on Omarchy); [the safety guide](safety.md) says what happens without one. The Anthropic key and the Claude sign-in share one credential, so connecting one disconnects the other. When a key or sign-in works for the main provider, the setup moves on by itself.

The choices behind providers, models, quota and the router are in [the agent guide](agent.md).

### The other steps (Customize)

| Step | What it asks |
|---|---|
| Quota | OpenCode Go only: when to move to the free model (and earlier inside `/plan`), which free model, whether to ask first, the countdown and what happens with no answer. |
| Router | Whether to use Jev. Available when OpenCode Go or TypeSafe is connected. |
| Assistant | Your name, the conversation language (Español, English, Português or another you type) and the answer style. Quick setup asks only for your name and takes English or Spanish from your system's language. |
| Models | The everyday model, the planning model and its effort, pay-per-use models (OpenCode Go), and how long a run may go silent before Aria stops it. |
| Effort | Automatic by route (needs the router), or a fixed level; with automatic effort and a free model available, whether free models run at their top level. |
| Compaction | At what share of the context window Aria compacts it. |
| Memory | Engram on or off, and the project name memories are filed under. |
| Integrations | Notion and Tavily, what each connector may change without asking, your saved secrets, MCP servers to import from your other apps, and an optional local folder of Pi packages. |
| Sandbox | The projects folder, extra folders Aria may write to, whether leaving the sandbox asks, whether Jev reviews commands. |
| Git | Whether Aria commits or pushes without asking, and how it treats changes that are not its own. |
| Subagents | Which experts are on and which are read-only, `/plan` and `/execute`, and the skills that run on their own. |

Every setting, with its default, is in [the settings reference](../settings.md). Integrations are explained in [the integrations guide](integrations.md).

### The review and the end

The last step lists what Aria will use, each line with a Change link. "Start using Aria" saves the setup and shows "Aria is set up. Press Super + A whenever you want to open Aria." (with the shortcut in use, `Option + Space` on a Mac). Close takes you to the chat. From then on, opening the dock shows the chat, and the gear beside the folder path opens the same settings as a screen of sections: see [the dock guide](dock.md).

The interface is in English or Spanish and follows the conversation language you pick; another language falls back to English for the interface, while Aria still answers in yours.

## Your first conversation

Press the shortcut to open the dock. The cursor is in the message box at once, so type and press `Enter`. A new session says "Ask for something, or type / to see the commands."

At the top of the column, under the tabs, a line shows the folder Aria works in. It is your projects folder, `~/Projects` by default (`sandbox.projectsDir`), and it is created if it does not exist. Aria reads and writes there, and its commands run in a sandbox that writes only inside the project (the git repository the session works in, or that folder when there is none), `/tmp`, the tool caches and the extra folders you allowed. In the dock the project is the projects folder itself, so a session there can write every project under it, and Aria keeps no undo copy of that folder. A session opened from a terminal works in the folder where you ran `aria` instead, which is the safer way to work on one project: see [Let Aria work on one project safely](safety.md#let-aria-work-on-one-project-safely). On Omarchy the sandbox is bubblewrap; on macOS every command asks first. See [safety](safety.md).

A first request can be as plain as:

```text
Show me what is in acme/checkout-widget and tell me how to run its tests.
```

While Aria works you see its steps: a line saying what it is doing and for how long, and a group of tool calls you can open. Type `@` to point at a file in the folder and `/` to see the commands. You can send another message while it works.

Each tab at the top is a session, and you can keep up to eight. A session in the background keeps working and tells you when it needs you. Close the dock with the shortcut or `Esc`; work in progress goes on. The tabs, the composer, the cards and everything else in the dock are described in [the dock guide](dock.md).

The same sessions are in a terminal: run `aria` in a terminal. See [Aria in the terminal](terminal.md).

## Updating

Run the installer again. It replaces the files, then, if the dock was running, asks it to restart into the new version, and prints "Aria is updated. The dock restarts into the new version once it is hidden and nothing is running."

An update never cuts a run. The dock restarts only when it is hidden and no session has a run going, a question waiting for you, or background work running or queued. Background work can keep an update waiting all day, so after 30 minutes of waiting for it alone, the sessions that have that work get one notice saying how to update now: stop the work from the background list.

The restart is immediate in three cases: the unit or LaunchAgent file changed, you passed `--restart-now`, or the running dock is too old to understand the request. The tabs and their conversations come back after any restart.

On macOS, the microphone permission prompt may come back after an update; that has not been tried live yet ([pending](../pending.md)).

## Uninstalling

```sh
sh install.sh --uninstall
```

On Omarchy it stops and disables the unit, takes Aria's face out of the bar, removes the unit and the binary, and, when you run it inside a Hyprland session, removes the `Super + A` bind from the running Hyprland. On macOS it unloads the LaunchAgent and removes the plist and the binary. Then it asks "Delete `~/.aria` (your settings and credentials)? [y/N]". Anything but `y` or `yes` keeps it, and so does having no terminal to ask on. It refuses to delete a folder that does not look like Aria's data, or your home folder.

It does not touch the system keyring, and it leaves `~/.agent-browser/browsers` in place; delete those by hand if you want them gone. Add `--dry-run` to see what would be removed. On a server installed with `--headless`, give `--headless` to the uninstall too, so it removes `aria-serve.service` instead of the dock's unit.

## Limits

- No release is published, so the `curl | sh` line has nothing to download from yet; build and use `--local`. The installer's Linux download name knows only x86_64. See [what is pending](../pending.md).
- Aria is built for Omarchy and macOS. Another Linux desktop is not supported for the dock.
- On macOS, the whole panel takes clicks (a click on the dark area closes the dock instead of reaching the app below), a desktop notification does not open the dock when clicked, and commands have no sandbox, so each one asks.

## See also

- [The dock](dock.md)
- [Aria in the terminal](terminal.md)
- [Running Aria on a server](hub.md)
- [The agent](agent.md)
- [Settings reference](../settings.md)
