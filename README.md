# Aria

A personal assistant that lives on your desktop. Press **Super + A** and a dock opens over whatever you are doing, on the monitor you are looking at, to work with your files and tasks and to talk to you in your language.

> **Status: phase 1 of 5.** Today Aria installs, opens with its hotkey, and runs the first-run setup that writes your settings. The chat and the agent arrive in phase 2. See [the roadmap](docs/superpowers/specs/2026-10-18-aria-base-design.md#roadmap).

Supported today: **Omarchy** (Hyprland 0.56+ with the Lua configuration). Mac is next.

## Install

```sh
curl -fsSL https://<your site>/install.sh | sh
```

The installer adds Aria's system libraries (it asks for `sudo` only if they are missing), puts the `aria` binary in `~/.local/bin`, creates `~/.aria/` and a systemd user unit that starts the dock at login. It **never edits `~/.config/hypr/`**: Aria registers Super + A with Hyprland itself each time it starts.

Run it again to update. To remove Aria: `sh install.sh --uninstall` (it asks before deleting `~/.aria`). Add `--dry-run` to any command to see what would happen without changing anything.

### Build it yourself

You need Rust, `bun` (for the UI tests), and the libraries `gtk3`, `webkit2gtk-4.1` and `gtk-layer-shell`.

```sh
cargo build --release
sh install/install.sh --local target/release/aria
```

## Where things live

| Path | What |
|---|---|
| `~/.aria/settings.json` | Every setting of the project, in one file ([reference](docs/settings.md)) |
| `~/.aria/auth.json` | Credentials, mode `600`, in Pi's format |
| `~/.aria/mcp.json`, `mcp-auth.json` | Notion and Tavily, once you connect them in the setup (Pi's MCP formats, mode `600`) |
| `~/.local/bin/aria` | The binary |
| `~/.config/systemd/user/aria.service` | Starts the dock at login |

The repository only holds code; your data never lives in it.

## Repository layout

```
crates/aria-core/      what does not depend on the system: paths, settings, credentials, providers,
                       key validation, Anthropic sign-in, the dock <-> page messages
crates/aria-hyprland/  the hotkey, the focused monitor and reload events, through Hyprland's own IPC
crates/aria/           the `aria` binary: the dock window and the glue between the two crates
ui/                    the dock's web page: plain HTML, CSS and JS modules, embedded in the binary
install/               the installer and the systemd unit
design/                the visual reference (Aria.html) and its components
docs/                  how the pieces fit, the settings, how to add a provider
```

Start with [docs/architecture.md](docs/architecture.md).

## Development

```sh
cargo test --workspace      # Rust: core, Hyprland, dock
bun test ui                 # the UI's logic and its strings
sh install/install.test.sh  # the installer, without touching your system
ARIA_DEBUG=1 target/release/aria   # logs state changes and the page's console to the terminal
ARIA_HOME=$(mktemp -d) target/release/aria   # try a first run without touching ~/.aria
```

## Contributing a provider or a platform

- A provider is one module: [docs/adding-a-provider.md](docs/adding-a-provider.md).
- A platform (Mac, other Linux compositors) needs the three things `aria-hyprland` does today: a global hotkey, the focused monitor, and a floating panel that takes the keyboard when it opens.
