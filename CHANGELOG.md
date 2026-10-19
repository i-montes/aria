# Changes outside the repository

What Aria's installer and runtime change on a machine, in one place (file, what, how to undo it).

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
