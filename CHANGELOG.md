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
