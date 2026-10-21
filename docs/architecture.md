# Architecture

Aria is a small Rust program that hosts a web page. The page is the interface; Rust owns everything that touches the system. They talk through one narrow channel.

```
 SUPER + A / Option + Space ──> Hyprland / Carbon ──> `aria toggle` ──unix socket──> the running dock (crates/aria)
                                                                      │
              aria-hyprland / aria-mac: hotkey, focused monitor       │ GTK layer-shell + WebKitGTK (Omarchy)
                                                                      │ NSPanel + WKWebView (macOS)
                                                                     ▼
                                                  ui/ (the page)  <── messages ──>  handlers.rs (settings, keys, sign-ins)
                                                                                    chat.rs (sessions)
                                                                                       │ one per tab
                                                                     aria-engine: `pi --mode rpc` child (JSONL)
                                                                                       │
                                                         aria-core: settings, auth, providers, agent dir, quota…
                                                                                       │
                                                ~/.aria/ settings.json, auth.json, sessions.json, agent/, pi/<version>/
```

## The crates

- **`aria-core`** has no GTK and no Hyprland. Everything in it can be tested without a display.
  - `paths` — where data lives (`~/.aria`, or `$ARIA_HOME`).
  - `json_file` — read without guessing, write atomically, keep keys we don't know.
  - `settings` — the `Settings` model and its defaults, load and save ([reference](settings.md)).
  - `auth` — `auth.json`, in Pi's exact format so the bundled Pi can read it later.
  - `providers` — one module per provider; the rest of Aria only reads what a module declares ([how to add one](adding-a-provider.md)).
  - `validate` — one cheap request per provider (and for Tavily) to tell a valid key from a rejected one.
  - `mcp` — the MCP services the setup configures (Notion, Tavily) and Aria's own Engram server (memory), in Pi's `mcp.json` and `mcp-auth.json` formats.
  - `mcp_import` — the MCP servers other apps already have (Claude Code's user servers in `~/.claude.json`, Claude Desktop, Cursor, VS Code): found by reading their files, never writing them; each becomes an entry in Pi's format (stdio or http; SSE is listed but not importable, Pi cannot reach it), with a name Pi accepts and `-2`, `-3`… when it is taken, and only the names of its secrets reach the page. Importing adds the picked ones to Aria's `mcp.json` (mode `600`), keeping everything else in it.
  - `oauth` — sign-ins by authorization code + PKCE with a loopback callback: `anthropic` (Claude Pro/Max), `mcp` (Notion: discovery from the 401, dynamic client registration), and the `pkce` and `loopback` pieces they share.
  - `testing` (feature `testing`) — a fake Notion-like server so the OAuth flow can be tested without the network.
  - `bridge` — the messages between the dock and the page (below).
  - `pi` — the pinned Pi (`pi-version`): where it is installed, whether Node can run it, the `pi` shim for Pi's own children.
  - `agent` — Pi's directory, `~/.aria/agent/`, prepared before every Pi start: its `settings.json` (default model, packages, the superpowers skills of the settings), links to the credential files, the owner's extensions, one sessions folder (`sessions/aria`), the Claude provider package.
  - `agent_config` — `aria.json`, the one file the extensions read Aria's settings from (`extensions/lib/config.ts`; without it they keep the values in their code). It carries each connector's access level as `connectors`, which the `connectors` extension reads: an MCP tool call that writes asks first on the chat's permission card, is refused (Read only) or runs (Allow); see [the extensions](../extensions/README.md).
  - `persona` — `SYSTEM.md` and the experts, rendered from the owner's text where a setting varies them.
  - `sessions` — the dock's tabs (`sessions.json`): ids, titles, the active one.
  - `files` — the `@` file picker's `.gitignore`-aware walk.
  - `quota` — the quota windows of the model's provider (OpenCode Go, a Claude plan).
  - `shortcut` — the `shortcut` setting, platform-free: parsing (forgiving about case and spaces, canonical label), the platform's default, and the order of the candidates (the chosen one, the default, then two fallbacks) that both hotkey adapters try.
- **`aria-hyprland`** is the Hyprland adapter: it registers the hotkey at run time, asks which monitor has the focus, and reports config reloads. On macOS, **`aria-mac`** sits next to it with the same two jobs: the `shortcut` setting through Carbon (`Option + Space` by default; no Accessibility permission, no config to reload) and the screen under the pointer (macOS has no focused monitor that survives app switches).
- **`aria-engine`** speaks Pi's RPC protocol: strict JSONL framing, commands and their responses, events, extension dialogs, and one supervised child (`Engine::close` stops it: stdin closed, killed after three seconds).
- **`aria`** is the binary. `main.rs` wires things, `ui.rs` is the platform event loop (GTK/glib or `NSApplication`/`NSTimer`), `control.rs` keeps one instance, `visibility.rs` decides when the dock is shown or closing (a pure state machine), `dock/{gtk,mac}.rs` is the window (the same `Dock` API on both platforms; on Linux it refuses every navigation outside `aria://`), `handlers.rs` answers the page's setup messages, `chat.rs` owns the sessions: one Pi per tab, started when the tab is shown, restarted when a credential or a setting changes (a busy one when its run settles; `main.rs` watches `settings.json` once a second), with desktop notifications for a prompt nobody can see, and a watchdog: every 30 s, a run that has had nothing from its Pi (no event, reply or dialog) for `models.turnTimeoutMinutes` is aborted once, with a notice in the chat (`chat.watchdog.stopped`); a running tool, an auto-compaction or a dialog waiting for the person pauses it. `attachments.rs` saves a message's attachments (below); `watch.rs` is each tab's subagent socket; `jobs.rs`, `jobs_socket.rs` and `worktree.rs` are background work (below); `lines.rs` reads a child's or a peer's lines up to a limit.

## The hotkey, without touching Hyprland's config

At start, again after every `configreloaded>>` event, and when the `shortcut` setting changes, Aria runs `hyprctl eval` with Lua that unbinds its previous shortcut, binds the chosen one (`SUPER + A` by default; see `shortcut` in [settings](settings.md)) to `aria toggle` with the description `Toggle Aria`, and adds the layer rule for the dock. Before binding it reads `j/binds` on the request socket: a bind with the same modifiers and key that is not Aria's own (told by that description) makes it fall back to the next shortcut (for a digit, any foreign bind with the same modifiers that `j/binds` lists with no key counts, since Omarchy's workspace binds by keycode show that way; with every candidate taken it binds the first anyway), and the page hears the one in use through `DockMessage::Platform`. Nothing is written to disk; a reload drops runtime binds, which is why Aria listens for it. If `$HYPRLAND_INSTANCE_SIGNATURE` is missing or stale (a service that outlives its Hyprland), the live instance is found under `$XDG_RUNTIME_DIR/hypr/`.

On macOS the shortcut is the `shortcut` setting (`Option + Space` by default), registered with Carbon (`RegisterEventHotKey`) on the application event target. It is global without the Accessibility permission. It is registered exclusive (`kEventHotKeyExclusive`), so a combination another app holds is refused (`eventHotKeyExistsErr`) and nobody can take Aria's while it runs. Carbon still accepts the system's own shortcuts (Spotlight's Command + Space, the input sources' Control + Space…) and the system then keeps the keys, so a combination listed enabled by `CopySymbolicHotKeys()` (its code and Carbon modifiers) counts as taken before it is tried. Taken, either way, makes the dock try the default, `Control + Option + Space` and `Command + Option + Space`. There is no config to reload, so it is registered at start and again when the setting changes, on the main thread: the new one is registered before the old one is unregistered, so a change with nothing free keeps the old one working. If nothing can be registered, the problem banner says so, the page hears an empty `hotkey` with every candidate taken, and the dock tries again every 10 s. The key codes (`kVK_*`, key positions), the reading of the system's shortcuts and the order live in `aria-mac/src/carbon.rs`, platform-free and tested on Linux. The trade-off of the default: while Aria runs, Option + Space no longer types a non-breaking space anywhere.

`DockMessage::Platform { hotkey, taken, os }` tells the page the shortcut in use (its notice and hints; empty when none could be registered), the ones tried before it that were taken (the settings' Shortcut row says "In use: X (Y was taken)", or "None available") and the platform (`linux`, `mac`: which modifier names the row records). The row records a press (read by key position, Super however the engine reports it) or takes a typed shortcut, for the combinations the desktop takes before the page sees them (one already bound, Aria's own). It is sent at start with the default and again whenever the registration changes.

## The dock

A GTK3 layer-shell surface, 700 px wide, anchored top, bottom and right of the focused monitor. Only the rightmost 470 px (the dock column) take input; the rest is the veil and lets clicks through. It asks for **on-demand** keyboard focus: Hyprland focuses it when it opens so you can type at once, and a click on another window takes the keyboard away.

`visibility::step(state, event)` is the whole show/close logic: pressing Super + A during the open or close animation reverses it without flicker, and a close timer from an earlier close is ignored.

Two WebKitGTK facts the first prototype paid for: set `WEBKIT_DISABLE_DMABUF_RENDERER=1` before GTK starts, and size the window with `set_size_request` (a default size never paints a layer surface with a web view in it).

On macOS the dock is an `NSPanel` (borderless, nonactivating, status level, all spaces and full-screen auxiliary) holding a `WKWebView`: 700 px wide on the right of the screen's visible frame, so the menu bar and the Dock stay usable. It becomes key because the panel overrides `canBecomeKeyWindow`; opening remembers the frontmost app and closing gives it back. The page is served through `WKURLSchemeHandler` (the counterpart of the GTK URI scheme) and the page's existing `window.webkit.messageHandlers.aria` through `WKScriptMessageHandler`, so `ui/` has no platform fork. A click on the veil closes the dock; click-through outside the column is not implemented yet. The app sets a minimal `Edit` menu so Cmd+C/V/X/A reach the web view (macOS dispatches those shortcuts through the main menu; an accessory app has no visible bar, but the key equivalents work).

## The page

`ui/` is plain HTML, CSS and ES modules, embedded in the binary and served as `aria://app/…`.

- `app.js` — the bridge, the corner notice, the problem banner, keyboard navigation.
- `onboarding/steps.js` — the onboarding as data and pure functions; **the working state is the settings object itself**.
- `onboarding/components.js` — form widgets that return HTML (every dynamic value escaped).
- `onboarding/view.js` — the screens and one delegated event handler: the first-run setup (welcome, steps, review) and, once it is done, the settings screen, which groups the same steps by topic (an overview of sections, one page per section, a bar to save or discard).
- `i18n.js` and `i18n/*.json` — every visible string, English and Spanish (the chat's parts bring their own `TEXTS`, merged by `chat/texts.js`).
- `chat/state.js` — one conversation as data: a pure reducer over Pi's events and the dock's messages.
- `chat/sessions.js` — the tabs as data.
- `chat/parts/` — one module per component of the design (messages, tool chips, cards, pickers, tabs, quota card…), each pure (props in, escaped HTML out) with its CSS and tests. A component is reused, never copied.
- `chat/thread.js`, `chat/view.js` — the thread, and the controller that composes the parts and wires them to the bridge.

`tools/ui-sandbox/` serves the page with a stubbed bridge and drives it in headless Chrome (`check.ts`).

The defaults and the provider catalog exist in Rust and in JS; both are tested against the golden JSON in `crates/aria-core/tests/golden/` so they cannot drift (`cargo test -p aria-core regenerate_golden -- --ignored` rewrites them after an intended change).

`tools/check-macos.sh` type-checks the macOS build from Linux (clippy for `aarch64-apple-darwin`, zig through mise as the C compiler; nothing links or runs), so code behind `cfg(target_os = "macos")` cannot break unseen. `tools/parity/suites.sh` runs every offline suite.

## Background work

A command that never ends (a dev server) or a long job given to a subagent runs in the background while the conversation goes on. It belongs to the dock, not to Pi: a Pi restarts on a credential or settings change, and only the dock sees every session.

- **`jobs.rs`** — one `Jobs` manager for the dock's whole life, shared by every `Chat`. A job is a process or a subagent, with a state (`queued`, `running`, `exited`, `failed`, `stopped`, `done`, `cut`). Every child is spawned from one long-lived spawner thread, as the leader of its own process group (on Linux with `PR_SET_PDEATHSIG(SIGTERM)`, which follows the thread that forked). A process is `bash -c <the sandbox's command line>` with stdin from `/dev/null` and stdout and stderr in one pipe; the dock keeps its last 1000 lines (2000 characters each, ANSI codes removed) in memory and takes the first local URL in them as the job's URL. Stop is SIGTERM to the group, SIGKILL after 3 s. The ends are events (`ended`, `cut`) kept per session until the extension acknowledges them. The limits are constants here: 10 processes, 3 running subagents (more wait, queued, in order), 20 finished jobs kept, an 8000-character result.
- **`jobs/agent.rs`** — the subagents: `pi --mode json -p --no-session [--model …] [--thinking …]` through the session's `pi` shim, with the session's env minus `ARIA_JOBS_SOCK`, so there is no background work inside background work. The session's watch is told the job's title and task before the spawn, so the child shows up named. The result is the last assistant message, and an API error or a non-zero exit is `failed`. One writer per folder: a second writer in a git repository gets a worktree (`isolation: auto`, or `worktree` when asked), and outside git it waits.
- **`worktree.rs`** — a worktree at `<projects dir>/.aria-worktrees/<repo dir name>-<job>`, made from `git stash create` (the person's uncommitted tracked changes, their folder untouched) or `HEAD`. It is outside the person's repository, so their test runners, watchers and compilers never see it, and nothing in the repository is edited besides git's own worktree bookkeeping. Every git the dock runs (outside the box, on a repository config and hooks a boxed command could write) passes `-c core.hooksPath=/dev/null -c core.fsmonitor=false -c core.untrackedCache=false -c commit.gpgSign=false -c core.sshCommand=false`, and empties every filter driver git sees (read again before the changes come back); bringing changes back is one at a time per repository. When the subagent is done, its changes come back as a binary patch applied with plain `git apply`: working tree only, nothing staged, nothing committed. A patch that does not apply, or a subagent that was stopped, failed or cut, keeps the worktree, and its path is named; nothing is removed without the person.
- **`jobs_socket.rs`** — one socket per session, `$XDG_RUNTIME_DIR/aria/jobs-<session>.sock` beside the watch socket (`j-<16 hex>.sock` when the path is long), mode `600`. Pi gets it as `ARIA_JOBS_SOCK` and the `background` extension speaks JSON lines on it: `hello` (answered with every event not acknowledged yet), `start` (a process or an agent), `list`, `output`, `stop`, `ack`. The sandbox hides `$XDG_RUNTIME_DIR` from boxed commands, so a boxed command cannot reach it; with `$XDG_RUNTIME_DIR` unset (the run dir would be under `/tmp`) no session gets a jobs socket, and the tools say they need the dock. The box also unsets `ARIA_JOBS_SOCK` and `PI_WATCH_SOCK` and masks their folders. A process gets the env Pi sent and nothing of the dock's.

Lifetimes:

- Hiding the dock stops nothing, and neither does a Pi restart: the extension connects again and gets the events it missed.
- Closing a tab stops that tab's jobs and drops its queued subagents.
- An update waits while any job runs or is queued. After 30 minutes of waiting for background work alone, the sessions with that work get one notice saying how to update now; the update never cuts the work by itself. Right before the dock execs the new version it stops every job anyway.
- Stopping the dock (`systemctl --user stop`, `launchctl`, Ctrl+C in a terminal: SIGTERM, SIGINT or SIGHUP) stops every job, running or queued, and every session's Pi, then the dock exits 0. The signal only sets a flag; the UI thread's tick does the stopping. A crash or SIGKILL stops nothing by itself: on Linux the children get SIGTERM from `PR_SET_PDEATHSIG`. `~/.aria/jobs.json` records, on every change, each job's state, a subagent's result (up to 8000 characters) and, until its session is told, an end note's last 20 lines of output; at the next start the ones still running or queued become `cut`, and their sessions are told.

The page gets every session's jobs as one `jobs` message, at most every 250 ms, and when a session loads. It shows the active session's as pills above the composer and all of them in the background list (Alt+A), with a job's output, its URL and a way to stop it.

## The messages

One JSON object per message with a `type`, defined in `crates/aria-core/src/bridge.rs`. Requests that expect an answer carry an `id` that the answer repeats.

| Page → dock | Meaning |
|---|---|
| `ready`, `opened`, `closeRequest` | the bridge is up; the open transition ended; Esc |
| `loadSettings`, `saveSettings` | read and write `settings.json` |
| `saveKey` | check a key with its provider and, only if it is accepted, store it (`opencode-go`, `anthropic-api`, `typesafe` in `auth.json`; `tavily` in `mcp.json`) |
| `startOauth`, `startMcpOauth` | start the Anthropic sign-in, or an MCP server's (`notion`) |
| `openUrl` | open an `https://` link in the browser |
| `installClaudeBridge`, `enginesRestart` | install the Claude provider; restart every Pi after a credential changed |
| `sessionsLoad`, `sessionsSave`, `sessionClose`, `chatShown` | the tabs; is the chat on screen (not the settings) |
| `session` (wrapping `chatSend`, `chatAbort`, `chatLoad`, `chatClearQueue`, `uiAnswer`, `listCommands`, `listModels`, `setModel`, `listFiles`) | everything about one tab's conversation. `chatSend` may carry `attachments: [{ name, mime, data }]` (base64; at most 10, 20 MB each, 50 MB together): the dock saves them under `~/.aria/attachments/<session>/<yyyy-mm-dd>/<n>-<safe name>` and adds a line per file to the message, `[attached: <name> (<mime>, <size>) at <path>]`, or `[attached image: <name> at <path>]` with the image itself in Pi's `images` when the session's model can see images. A save that fails, or a message with files when Pi cannot start, comes back as `command_failed` with the page's own text, so the page gives the text and the files back |
| `jobStop`, `jobDismiss` | stop a background job; take a failed one off the strip (no answer: the next `jobs` shows it) |
| `jobOutput` | a job's last lines (80 unless `lines` says otherwise) |
| `mcpScan` | find the MCP servers other apps have (answered with `mcpFound`) |
| `mcpImport` | copy the found servers with these `ids` (`source:name`) into Aria's `mcp.json`; the dock scans again, so the page never holds a config. Answered with `saved` (every Pi restarts when its run settles) or an `error`: `nothing-to-import`, `mcp-file-broken`, `mcp-write-failed` |

| Dock → page | Meaning |
|---|---|
| `open`, `close`, `notice` | animate the dock; show the first-run corner notice |
| `settings`, `saved`, `keyResult`, `oauthUrl`, `oauthResult`, `error` | answers; failures carry a string key such as `key-rejected`. `settings` also lists what is really connected (`opencode-go:api_key`, `anthropic:oauth`, `mcp:notion`…), never a secret, so the setup does not have to assume it, and `servers`, every MCP server in Aria's `mcp.json` but `engram`, so each gets its access row |
| `platform` | the shortcut in use, the ones taken before it, and `os` (above) |
| `mcpFound` | the answer to `mcpScan`: per server its `id`, `name`, `importAs` (the name Pi accepts), `source` (`claude-code`, `claude-desktop`, `cursor`, `vscode`), `kind` (`stdio`, `http`, `sse`), `target` (the command's name or the URL's host), `secrets` (env and header names only, and `input:<id>` for a VS Code prompt), `importable` and `disabled`. Servers already in Aria's `mcp.json`, and `engram`, are left out |
| `problem` | something prevents Aria from working (for example `hyprland-classic-config`) |
| `sessions` | the saved tabs |
| `session` (wrapping `chatState`, `chatEvent`, `chatDialog`, `chatDialogClosed`, `chatEngine`, `chatQuota`, and the replies `commands`, `models`, `files`, `saved`, `error`) | what one tab's Pi says |
| `jobs` | every session's background jobs, dismissed ones included (not wrapped in a session) |
| `jobOutput` | the answer to `jobOutput`: the lines (none for a subagent, which keeps no output) |

## Data and safety

- Everything Aria stores is under `~/.aria/`. `settings.json` never holds a secret; `auth.json`, `mcp.json` and `mcp-auth.json` are mode `600`, and so is `jobs.json`, which holds each job's state, a subagent's result (up to 8000 characters) and, until its session is told, an end note's last 20 lines of output. Attachments are saved in `~/.aria/attachments/` (folders `700`, files `600`) and never removed; the model can read them, and neither a command in the sandbox nor Pi's write and edit can change them (nor `settings.json` and `agent/aria.json`, whatever the session granted). The one exception is a background subagent's worktree, in the projects folder (`<projects dir>/.aria-worktrees/`).
- Files are written atomically (a unique temp file and rename), and a file that cannot be parsed is reported, never overwritten (a broken `sessions.json` is moved to `sessions.json.broken`).
- A key is validated against its provider before it is stored, and a rejected or unreachable check writes nothing.
- The installer never edits `~/.config/hypr/`.

## What is not there yet

The parity check (2d), goals (phase 3), the state socket and external extensions (phase 4), Windows. macOS (phase 5) is here for the dock, setup, chat and installer; click-through, notification clicks, a `.app` bundle and the macOS port of the sandbox extension are still pending (see [pending](pending.md)). What the chat draws but nothing feeds yet is in [pending](pending.md). See the [roadmap](superpowers/specs/2026-10-18-aria-base-design.md#roadmap).
