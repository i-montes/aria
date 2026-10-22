# Changes outside the repository

What Aria's installer and runtime change on a machine, in one place (file, what, how to undo it).

## 2026-10-18 — first install on the owner's Mac (`sh install/install.sh --local target/release/aria`)

The installer detected Darwin, skipped the Omarchy packages and Hyprland checks, and installed a LaunchAgent.

| File | What | Backup | Revert |
|---|---|---|---|
| `~/.local/bin/aria` | the dock binary (new on this machine) | none, new file | `sh install/install.sh --uninstall` |
| `~/Library/LaunchAgents/com.aria.dock.plist` | LaunchAgent that starts the dock at login (new) | none, new file | `sh install/install.sh --uninstall` |
| `~/.aria/` | data directory, mode 700: `pi/1.3.0/`, `agent/` (the pinned Pi packages), `bin/node`, `sessions.json`, `logs/aria.log`, `run/control.sock` (new) | none, new | `sh install/install.sh --uninstall` asks before deleting it |
| Carbon hotkey (runtime only, nothing on disk) | `Option + Space`, registered by the dock on the application event target; no Accessibility permission | not applicable | quit the dock; the registration dies with the process |

Checked: `launchctl print gui/501/com.aria.dock` shows the agent running, `~/.aria/logs/aria.log` shows "registered Option + Space", and the page wrote `~/.aria/sessions.json`. No file under `/Library` or any system configuration was touched.

## 2026-10-18 — first install on the owner's Omarchy machine (`install/install.sh --local target/release/aria`)

| File | What | Backup | Revert |
|---|---|---|---|
| `~/.local/bin/aria` | the dock binary (new) | none, new file | `sh install/install.sh --uninstall` |
| `~/.config/systemd/user/aria.service` | user unit that starts the dock at login (new) | none, new file | `sh install/install.sh --uninstall` |
| `~/.config/systemd/user/graphical-session.target.wants/aria.service` | symlink from `systemctl --user enable` (new) | none | `sh install/install.sh --uninstall` |
| `~/.aria/` | data directory, mode 700, empty until setup (new) | none | `sh install/install.sh --uninstall` asks before deleting it |
| Hyprland (runtime only, nothing on disk) | `SUPER + A` bind and the `aria-dock` layer rule, registered with `hyprctl eval` each time the dock starts | not applicable | stop the service, or `hyprctl eval 'hl.unbind("SUPER + A")'`; a Hyprland restart also drops it |

Checked on install: the 26 files in `~/.config/hypr/` had identical modification times and sizes before and after. The system libraries (`gtk3`, `webkit2gtk-4.1`, `gtk-layer-shell`) were already present, so `sudo pacman` was not needed.

## 2026-10-18 — phase 2a: the engine (`sh install/install.sh --local target/release/aria`)

| File | What | Backup | Revert |
|---|---|---|---|
| `~/.aria/pi/1.3.0/` | the pinned Pi (`npm install --prefix`, about 150 MB) (new) | none, new | `rm -r ~/.aria/pi` |
| `~/.aria/bin/node` | symlink to the Node found on the installer's `PATH` (the service has no mise on its `PATH`) (new) | none, new | `rm ~/.aria/bin/node` |
| `~/.aria/bin/pi` | shim that runs the pinned Pi, written when the chat first starts Pi (new) | none, new | `rm ~/.aria/bin/pi` |
| `~/.aria/agent/` | Pi's directory: its `settings.json`, links to `auth.json`/`mcp.json`/`mcp-auth.json`, `SYSTEM.md`, sessions (new; the links point at the files Aria already owns) | none, new | `rm -r ~/.aria/agent` |
| `~/.aria/sessions.json` | the open session tabs, mode 600 (new, written by the dock) | none, new | `rm ~/.aria/sessions.json` (the tabs start over; Pi's sessions stay) |
| `~/.local/bin/aria` | the binary, now with the chat and the session tabs (replaced) | the previous build is gone; rebuild from the earlier commit | `git checkout 0e6fde5 && cargo build --release` |

Notes: the desktop notification for a prompt that nobody can see is sent with `omarchy-notification-send -u critical -t 0 … --exec aria show`; a test notification sent the same way was dismissed with `omarchy-notification-dismiss`. `~/.pi` was not touched.

## 2026-10-19 — the Anthropic provider and `/connect`

| File | What | Backup | Revert |
|---|---|---|---|
| `~/.aria/bin/claude` | symlink to the Claude Code found on the installer's `PATH` (only when it is installed): the Claude provider runs it (new) | none, new | `rm ~/.aria/bin/claude` |
| `~/.aria/agent/npm/`, `~/.aria/agent/settings.json` (`packages`) | `pi install npm:pi-claude-agent-sdk@0.9.2` (about 300 MB), run **only** when you choose "Sign in with Claude" in `/connect`; Pi lists the package in its own settings (new) | none, new | `pi remove` with `PI_CODING_AGENT_DIR=~/.aria/agent`, or `rm -r ~/.aria/agent/npm` and delete the `packages` entry |
| `~/.aria/auth.json` | the `anthropic` credential (API key or Claude OAuth) written by `/connect`, as the setup already does | the file is rewritten atomically keeping other providers | delete the `anthropic` entry |
| `~/.local/bin/aria` | the binary with `/connect` (replaced) | rebuild from the earlier commit | `git checkout main~1 && cargo build --release` |

## 2026-10-19 — phase 2b: the owner's persona and extensions

| File | What | Backup | Revert |
|---|---|---|---|
| `~/.aria/agent/SYSTEM.md`, `agents/*.md` | the owner's persona and experts, replacing the one-line placeholder (written at every Pi start; an edited file is kept, see `.aria-files`) | the placeholder was one line, written by Aria | delete them; Aria writes them again |
| `~/.aria/agent/extensions/` | the owner's seven extensions, adapted (see `extensions/README.md`) | none, new | `rm -r ~/.aria/agent/extensions` (Aria writes them again at the next start) |
| `~/.aria/agent/.aria-files` | hashes of what Aria wrote, so an edit by the owner is kept | none, new | delete it (every file is then treated as the owner's) |
| `~/.aria/agent/npm/node_modules/pi-parallel-agents`, `~/.aria/agent/git/github.com/obra/superpowers` (at `bb92a77741419a4ab5f06e711a283343f1ada0c3`) | the packages the extensions use, installed by `install.sh` with `pi install`; listed in `~/.aria/agent/settings.json` (`packages`, superpowers filtered to two skills) | none, new | `pi remove` with `PI_CODING_AGENT_DIR=~/.aria/agent` |
| `~/.aria/agent/sessions/aria/` | every session's file in one folder; the files of the earlier per-folder layout were moved here (3 files) | none (moved, not copied) | move them back under `sessions/<per-folder name>/` |
| `$XDG_RUNTIME_DIR/aria/watch-<session>.sock` | one socket per open tab for its subagents, mode 600, removed when the tab closes | none | stopping the dock removes them |

## 2026-10-19 — macOS installer review

| File | What | Backup | Revert |
|---|---|---|---|
| `~/Library/LaunchAgents/com.aria.dock.plist` | now also has `KeepAlive` (`SuccessfulExit` false): launchd starts the dock again if it crashes | the earlier plist, rewritten by the installer | `sh install/install.sh --uninstall`, or remove the `KeepAlive` key and `launchctl bootout`/`bootstrap` |
| `~/.aria/run/w-<hash>.sock` | a subagent socket whose usual name (`watch-<session>.sock`) would pass macOS's 104-byte limit | none | removed when the tab closes |

## 2026-10-19 — phase 2c: settings drive Pi

| File | What | Backup | Revert |
|---|---|---|---|
| `~/.aria/agent/aria.json` | the extensions' settings, written from `~/.aria/settings.json` at every Pi start (no secrets) | none, new | delete it: the extensions go back to the values in their code |
| `~/.aria/agent/SYSTEM.md`, `agents/*.md` | now rendered from the settings (name, language, style, git, memory, experts); a disabled expert's file is removed | Aria's earlier copy (the owner's text) | turn the setting back, or delete the file and its line in `.aria-files` |
| `~/.aria/mcp.json` | an `engram` server written by Aria (binary `~/.aria/bin/engram`, `ENGRAM_DATA_DIR=~/.aria/engram`), removed when memory is off; an `engram` entry Aria did not write is never touched | none (the entry is new) | turn memory off in the settings |
| `~/.aria/engram/` | Aria's own Engram database (mode 700), apart from the terminal Pi's `~/.engram` | none, new | delete the folder (Aria forgets what it learned) |
| `~/.aria/bin/engram` | Engram 3.3.2, installed by `install.sh` from its GitHub release and checked against `engram-version` | none, new | delete it |
| `~/.aria/bin/pi` | the shim keeps subagent sessions in `~/.aria/agent/sessions/subagents/` when "save the subagents' sessions" is on | the plain shim | turn the setting off (the next start rewrites it) |
| `~/.aria/settings.json` | `/effort` and `/autocompact` in a session write `effort` and `compaction` here (not `jev-router-effort.json` / `autocompact.json`) | none | change them back in the settings |

## 2026-10-19 — phase 2d: the parity check

| File | What | Backup | Revert |
|---|---|---|---|
| system packages (pacman) | `install.sh` also installs `bubblewrap` and `iproute2` on Omarchy, at the next install (both were already on the owner's machine) | none | `sudo pacman -R bubblewrap iproute2`, if nothing else needs them |
| `/tmp/tmp.*` | `tools/parity/scratch-home.sh` makes a scratch copy of `~/.aria` (settings, credentials at mode 600, MCP list) for each live check and deletes it when the check ends; `~/.aria` is only read | none | none needed: removed on exit |

## 2026-10-21 — shellcheck for the installer tests (Omarchy machine)

| File | What | Backup | Revert |
|---|---|---|---|
| `~/.config/mise/config.toml` | `shellcheck = "0.11.0"` (mise's static release binary; pacman's package failed on a mirror signature) so `install/install*.test.sh` lint `install.sh` | none, one line added | `mise unuse -g shellcheck` |

## 2026-10-21 — background work

| File | What | Backup | Revert |
|---|---|---|---|
| `~/.aria/jobs.json` | the background jobs' state, a finished subagent's result (up to 8000 characters) and, until its session is told, an end note's last 20 lines of output (nothing more of the output), mode 600, written by the dock on every change; jobs still running at the next start become `cut` (new) | none, new | `rm ~/.aria/jobs.json` while the dock is stopped (the Finished list and any end note not yet told are forgotten) |
| `$XDG_RUNTIME_DIR/aria/jobs-<session>.sock` (in `~/.aria/run/` on macOS; `j-<hash>.sock` when the usual name would be too long) | one socket per open tab for its background work, mode 600, removed when the tab closes (new) | none | stopping the dock removes them |
| `<projects dir>/.aria-worktrees/<repo>-<job>` (and git's own bookkeeping for it in `<repo>/.git/worktrees/`) | a git worktree for a background subagent that writes while another writer holds the same repository (or that asked for one), outside the repository so its tools never see it; removed once its changes are back in the folder, kept when they clash or the subagent was stopped, failed or cut (new) | none, new; the person's own files are not touched | `git worktree remove --force <projects dir>/.aria-worktrees/<repo>-<job>` in the repository (after taking what you want from it), then `rmdir <projects dir>/.aria-worktrees` |

## 2026-10-21 — connector approvals and attachments

| File | What | Backup | Revert |
|---|---|---|---|
| `~/.aria/attachments/` | every file attached in the chat, saved by the dock as `<session>/<yyyy-mm-dd>/<n>-<safe name>` (folders mode 0700, files 0600) so the model can read it; nothing removes them (new) | none, new | `rm -r ~/.aria/attachments` (messages that named a file there can no longer be read again) |

## 2026-10-21 — turn watchdog, MCP import, the shortcut

The watchdog only aborts a run through Pi; nothing on disk. The import reads other apps' MCP files (`~/.claude.json`,
Claude Desktop's and VS Code's under the config directory or `~/Library/Application Support/`, `~/.cursor/mcp.json`)
and never writes them.

| File | What | Backup | Revert |
|---|---|---|---|
| `~/.aria/mcp.json` | the servers the person picks in "Your other MCP servers" are added as entries (under a name Pi accepts, `-2`, `-3`… when taken), only on the person's click; every other entry and key stays, mode 0600 | none; the entries are new and the rest of the file is kept | remove the imported entries from `mcpServers` (the next Pi start drops them) |
| Hyprland / Carbon (runtime only, nothing on disk) | the bind is now the `shortcut` setting, or the first free fallback (`SUPER + SHIFT + A`, `SUPER + ALT + A`; macOS `Control + Option + Space`, `Command + Option + Space`) when it is taken | not applicable | set `shortcut` back to `""`, or stop the dock; on Hyprland `hyprctl eval 'hl.unbind("<the shortcut>")'` |
| `rustup` target `aarch64-apple-darwin` and mise's `zig@0.14.1` (developer machines that run `sh tools/check-macos.sh`) | the Rust standard library for the Mac target and zig as its C compiler, installed on first use | none, new | `rustup target remove aarch64-apple-darwin`; `mise uninstall zig@0.14.1` |

## 2026-10-22 — Aria's face in the Omarchy bar (`aria bar install`, run by the installer)

First put in by hand as a test, then taken out with `aria bar remove` and put back by `sh install/install.sh --local`.

| File | What | Backup | Revert |
|---|---|---|---|
| `~/.config/omarchy/plugins/aria.face/` (`manifest.json`, `Face.qml`) | the bar widget the binary carries: the idle face (glances, blinks, yawns, rolls) animated per frame; its colour follows the word in `$XDG_RUNTIME_DIR/aria/face`; a click runs `aria toggle` (new) | none, new | `aria bar remove`, or `sh install/install.sh --uninstall` |
| `~/.config/omarchy/shell.json` | `{ "id": "aria.face" }` in `bar.layout.center`, right after `omarchy.weather`, put there by `omarchy plugin enable` after `omarchy-shell shell rescanPlugins` (the right section was tried first, but the bar always pins the tray before every other widget there) | `~/.config/omarchy/shell.json.bak.<epoch>`, one each time it is placed | `aria bar remove` (runs `omarchy plugin disable aria.face`), or copy a backup back (the shell reloads it on save) |
| `$XDG_RUNTIME_DIR/aria/face` | the face's state, one word (runtime only; gone at logout) | not applicable | `rm` it; the face stays idle |
