# Aria on macOS — design

Date: 2026-10-18. Status: approved by the owner (Option + Space); phase 5 pulled forward.

## Intent

The owner works on a Mac now (macOS 26, Apple silicon). Aria's page, settings, credentials, Pi engine and chat
are already platform-independent; what is missing is the platform layer the base design reserved for phase 5.
This design adds it without changing what Aria is or how the Omarchy build behaves.

The base design's rule still governs: the same page, the same `~/.aria/`, the same Pi. Only the platform
layer differs. Phases 3 and 4 (goals, the state socket) are not required for this port and stay pending.

## What the owner gets

1. `Option + Space` opens the dock on the screen under the pointer, over whatever app is in front, and takes the
   keyboard at once; the same shortcut (or `Esc`) closes it.
2. The first run shows the corner notice and the same onboarding; finishing it writes `~/.aria/settings.json`
   and `auth.json` exactly as on Omarchy. The chat, the session tabs and `/connect` work.
3. Installing is still one command. The installer detects macOS, installs the pinned Pi and a LaunchAgent, and
   asks for no `sudo`, no Accessibility permission and no edits to system configuration.
4. Uninstalling removes the LaunchAgent and the binary and asks before deleting `~/.aria/`.

## Platform detection

There is no runtime "am I on a Mac?" in one binary: a Linux binary cannot host AppKit. The platform is chosen
when the binary is built (`cfg(target_os)`), and the installer chooses the artifact and the service manager from
`uname -s` / `uname -m`. The page is told the shortcut at start (`DockMessage::Platform`) so its notice and its
"you are set up" screen say `Option + Space` instead of `Super + A`.

## The platform layer

The small interface promised by the base design is made of three pieces:

- `crates/aria-mac` — next to `aria-hyprland`, with the same two jobs: register the global hotkey and report the
  screen the user is looking at. There is no config to reload, so no watcher.
- `crates/aria/src/dock/mac.rs` — behind the same `Dock` API as `dock/gtk.rs`: `new`, `send`, `feed`, `show`,
  `show_notice`, `set_problem`, `clear_problem`. The Linux file is untouched in behaviour and is compiled only
  on Linux.
- `install/install.sh` — branches on the platform for the service manager (systemd user unit vs LaunchAgent) and
  the artifact; the Pi install, Node link, data directory and uninstall checks are shared.

`crates/aria/src/ui.rs` hides the event-loop difference: GTK's main loop and `glib` timer on Omarchy,
`NSApplication` and an `NSTimer` on macOS.

## The Mac dock

- **Window:** an `NSPanel` with `Borderless | NonactivatingPanel`, status level, `canJoinAllSpaces`,
  `fullScreenAuxiliary`, no shadow, transparent, 700 px wide, on the right of the screen's visible frame (below
  the menu bar and beside the Dock, so both stay usable). It becomes key because the panel overrides
  `canBecomeKeyWindow`; it is an accessory app (no Dock icon).
- **Page:** `WKWebView`, transparent, loading `aria://app/index.html`. `WKURLSchemeHandler` serves the embedded
  `ui/`; `WKScriptMessageHandler` named `aria` receives the page's messages; both are the exact counterparts of
  WebKitGTK's URI scheme and script handler, so the page needs no platform fork. `window.aria.receive(...)` is
  still how the dock talks back. The app installs a minimal `Edit` menu (`undo`, `cut`, `copy`, `paste`,
  `selectAll`): macOS matches Cmd+C/V/X/A against the main menu, and without it pasting a key into the setup does
  nothing. The menu bar is not drawn for an accessory app; the key equivalents still work.
- **Open:** remember the frontmost app, place the panel, activate Aria, `makeKeyAndOrderFront`, focus the web
  view, then send `Open` and any problem. **Close:** stop taking clicks, send `Close`, keep the panel mapped for
  the page's 180 ms animation, then order it out and reactivate the previous app.
- **Notice:** mapped without activation and without taking clicks for 8.6 s, like the layer-shell notice.
- **Veil:** the page paints it; a click on it sends `closeRequest`. Click-through outside the 470 px column is
  not implemented in this pass (it needs a second window or a private API); the veil closes instead.
- **Hotkey:** Carbon `RegisterEventHotKey` with `optionKey` and `kVK_Space`. It is global without asking for
  Accessibility or Input Monitoring. The trade-off: while Aria runs, Option + Space no longer types a
  non-breaking space anywhere (the owner accepted this with the shortcut).
- **Notifications:** `osascript display notification`. A notification from an unbundled program has no click
  action, so clicking it does not open the dock yet; the chat still posts it when a prompt is hidden. See
  `docs/pending.md`.
- **Control socket:** `~/.aria/run/control.sock` (or `$ARIA_HOME/run/...`), so the LaunchAgent and a terminal
  agree. Omarchy keeps `$XDG_RUNTIME_DIR/aria/control.sock`.

## The installer

`install/install.sh` runs on both systems:

- `uname -s` = `Darwin` → macOS 12 or newer, no packages, binary `aria-<arch>-apple-darwin` (or `--local`),
  LaunchAgent in `~/Library/LaunchAgents/com.aria.dock.plist`, logs in `~/.aria/logs/aria.log`,
  `launchctl bootout`/`bootstrap` (with a short wait, because bootstrap fails while the old instance is
  still exiting).
- `uname -s` = `Linux` → the existing Omarchy path, unchanged: Hyprland check, pacman packages, systemd unit,
  `hyprctl` unbind on uninstall.
- Shared: Node 22.19+, `npm install --prefix ~/.aria/pi/<version>`, the pinned Pi packages, the `~/.aria/bin`
  links, `~/.aria` mode 700, the refusal to touch a `~/.aria` that does not look like Aria's.

`--dry-run`, `--local`, `--uninstall`, `--check-node`, `--print-pi-version`, `--link-node`, `--print-unit`
and `--can-delete` keep working. `install-mac.test.sh` checks the macOS branch without touching the system;
`install.test.sh` checks the Linux branch and skips on macOS.

## Data, settings, safety

Unchanged: `~/.aria/settings.json`, `auth.json`, `mcp.json`, `mcp-auth.json`, `sessions.json`, `agent/`,
`pi/<version>/`. Nothing is stored in the repo. The LaunchAgent runs with the user's `HOME`; `~/.aria/bin/node`
is how Pi finds Node without mise on its `PATH`.

## Not in this pass

- A `.app` bundle, code signing or notarization; the binary is unbundled. (A bundle is what notification click
  actions and the standard "open at login" UI would need.)
- Click-through outside the dock column; notification click actions; a menu-bar extra.
- A macOS implementation of the owner's `sandbox` extension (it uses `bwrap`, which is Linux-only) and the
  extensions' Linux-specific tests. The rest of the chat runs.
- Windows, Linux beyond Hyprland, phase 3 goals and phase 4's state socket.

## Risks and checks

- **Option + Space is normally a non-breaking space.** Carbon consumes it while Aria runs. Accepted by the owner.
- **Unbundled WKWebView and notifications** can behave differently from a signed app; the first real install is
  the check.
- **The panel's focus** depends on activation and `canBecomeKeyWindow`; it was checked with `isKeyWindow` and a
  150 ms follow-up. Multi-monitor is "screen under the pointer".
- **Tests:** `cargo test --workspace`, `bun test ui`, `sh install/install-mac.test.sh`, and the manual smoke
  list: Option + Space opens/closes, typing goes to the dock, the notice appears on first run, a second
  `aria` says one is running, and the LaunchAgent starts the dock after a logout/login.
