# Architecture

Aria is a small Rust program that hosts a web page. The page is the interface; Rust owns everything that touches the system. They talk through one narrow channel.

```
 SUPER + A ──> Hyprland ──exec──> `aria toggle` ──unix socket──> the running dock (crates/aria)
                                                                     │
                       aria-hyprland: hotkey, focused monitor        │ layer-shell window + WebKitGTK
                                                                     ▼
                                                  ui/ (the page)  <── messages ──>  handlers.rs
                                                                                       │
                                                         aria-core: settings, auth, providers, validation, OAuth
                                                                                       │
                                                                          ~/.aria/settings.json, auth.json
```

## The crates

- **`aria-core`** has no GTK and no Hyprland. Everything in it can be tested without a display.
  - `paths` — where data lives (`~/.aria`, or `$ARIA_HOME`).
  - `json_file` — read without guessing, write atomically, keep keys we don't know.
  - `settings` — the `Settings` model and its defaults, load and save ([reference](settings.md)).
  - `auth` — `auth.json`, in Pi's exact format so the bundled Pi can read it later.
  - `providers` — one module per provider; the rest of Aria only reads what a module declares ([how to add one](adding-a-provider.md)).
  - `validate` — one cheap request per provider (and for Tavily) to tell a valid key from a rejected one.
  - `mcp` — the MCP services the setup configures (Notion, Tavily), in Pi's `mcp.json` and `mcp-auth.json` formats.
  - `oauth` — sign-ins by authorization code + PKCE with a loopback callback: `anthropic` (Claude Pro/Max), `mcp` (Notion: discovery from the 401, dynamic client registration), and the `pkce` and `loopback` pieces they share.
  - `testing` (feature `testing`) — a fake Notion-like server so the OAuth flow can be tested without the network.
  - `bridge` — the messages between the dock and the page (below).
- **`aria-hyprland`** is the Hyprland adapter: it registers the hotkey at run time, asks which monitor has the focus, and reports config reloads. A Mac adapter would sit next to it with the same three jobs.
- **`aria`** is the binary. `main.rs` wires things, `control.rs` keeps one instance, `visibility.rs` decides when the dock is shown or closing (a pure state machine), `dock.rs` is the window, `handlers.rs` answers the page.

## The hotkey, without touching Hyprland's config

At start, and again after every `configreloaded>>` event, Aria runs `hyprctl eval` with Lua that unbinds and rebinds `SUPER + A` to `aria toggle`, and adds the layer rule for the dock. Nothing is written to disk; a reload drops runtime binds, which is why Aria listens for it. If `$HYPRLAND_INSTANCE_SIGNATURE` is missing or stale (a service that outlives its Hyprland), the live instance is found under `$XDG_RUNTIME_DIR/hypr/`.

## The dock

A GTK3 layer-shell surface, 700 px wide, anchored top, bottom and right of the focused monitor. Only the rightmost 470 px (the dock column) take input; the rest is the veil and lets clicks through. It asks for **on-demand** keyboard focus: Hyprland focuses it when it opens so you can type at once, and a click on another window takes the keyboard away.

`visibility::step(state, event)` is the whole show/close logic: pressing Super + A during the open or close animation reverses it without flicker, and a close timer from an earlier close is ignored.

Two WebKitGTK facts the first prototype paid for: set `WEBKIT_DISABLE_DMABUF_RENDERER=1` before GTK starts, and size the window with `set_size_request` (a default size never paints a layer surface with a web view in it).

## The page

`ui/` is plain HTML, CSS and ES modules, embedded in the binary and served as `aria://app/…`.

- `app.js` — the bridge, the corner notice, the problem banner, keyboard navigation.
- `onboarding/steps.js` — the onboarding as data and pure functions; **the working state is the settings object itself**.
- `onboarding/components.js` — form widgets that return HTML (every dynamic value escaped).
- `onboarding/view.js` — the screens and one delegated event handler.
- `i18n.js` and `i18n/*.json` — every visible string, English and Spanish.

The defaults and the provider catalog exist in Rust and in JS; both are tested against the golden JSON in `crates/aria-core/tests/golden/` so they cannot drift (`cargo test -p aria-core regenerate_golden -- --ignored` rewrites them after an intended change).

## The messages

One JSON object per message with a `type`, defined in `crates/aria-core/src/bridge.rs`. Requests that expect an answer carry an `id` that the answer repeats.

| Page → dock | Meaning |
|---|---|
| `ready`, `opened`, `closeRequest` | the bridge is up; the open transition ended; Esc |
| `loadSettings`, `saveSettings` | read and write `settings.json` |
| `saveKey` | check a key with its provider and, only if it is accepted, store it (`opencode-go`, `anthropic-api`, `typesafe` in `auth.json`; `tavily` in `mcp.json`) |
| `startOauth`, `startMcpOauth` | start the Anthropic sign-in, or an MCP server's (`notion`) |
| `openUrl` | open an `https://` link in the browser |

| Dock → page | Meaning |
|---|---|
| `open`, `close`, `notice` | animate the dock; show the first-run corner notice |
| `settings`, `saved`, `keyResult`, `oauthUrl`, `oauthResult`, `error` | answers; failures carry a string key such as `key-rejected`. `settings` also lists what is really connected (`opencode-go:api_key`, `anthropic:oauth`, `mcp:notion`…), never a secret, so the setup does not have to assume it |
| `problem` | something prevents Aria from working (for example `hyprland-classic-config`) |

## Data and safety

- Everything Aria stores is under `~/.aria/`. `settings.json` never holds a secret; `auth.json`, `mcp.json` and `mcp-auth.json` are mode `600`.
- Files are written atomically (temp file and rename), and a file that cannot be parsed is reported, never overwritten.
- A key is validated against its provider before it is stored, and a rejected or unreachable check writes nothing.
- The installer never edits `~/.config/hypr/`.

## What phase 1 does not do

No chat, no agent, no goals, no extension socket, and no Mac. See the [roadmap](superpowers/specs/2026-10-18-aria-base-design.md#roadmap).
