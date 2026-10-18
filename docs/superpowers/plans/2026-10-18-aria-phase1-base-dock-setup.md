# Aria phase 1 — base, dock and setup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Rust workspace where `SUPER + A` opens Aria's dock on the focused Hyprland monitor with the design's look and motion, the dock runs the onboarding, and finishing it writes `~/.aria/settings.json` and `~/.aria/auth.json`; installed by one script with no edits to `~/.config/hypr/`.

**Architecture:** `aria-core` holds everything platform-independent (paths, settings, auth store, provider modules, key validation, Anthropic OAuth, the dock↔page message protocol). `aria-hyprland` talks to Hyprland (hotkey registration, focused monitor, events). `aria` (the binary) is the GTK3 layer-shell dock that hosts the web UI in `ui/` and wires the two crates together. The UI is plain HTML/CSS/ES modules (no bundler), embedded in the binary and served through a custom `aria://` URI scheme.

**Tech Stack:** Rust 2021 (rustc 1.98), gtk 0.18, gdk 0.18, glib 0.18, gio 0.18, gtk-layer-shell 0.8 (`v0_6`), webkit2gtk 2.0 (`v2_40`), serde/serde_json, ureq 2 (blocking HTTP, rustls), sha2 + base64 + rand (PKCE), include_dir 0.7, tempfile (tests). UI tests: `bun test`. Installer: POSIX `sh`.

**Spec:** `docs/superpowers/specs/2026-10-18-aria-base-design.md` (read it first). Working spike with the dock mechanics: `spikes/dock-open/`. Design reference: `design/Aria.html` and its unpacked components in `design/components/*.dc.html` (the onboarding is `design/components/onboarding.dc.html`).

## Global Constraints

- Platform: Omarchy / Hyprland **0.56+ with the Lua config** only. Never write to `~/.config/hypr/`.
- All code, identifiers, settings keys, file names, docs and commit messages in English. UI strings come from the catalog (`ui/i18n/en.json`, `ui/i18n/es.json`).
- All user data under `~/.aria/` (respect `ARIA_HOME` when set, for tests). The repo holds only code.
- Secrets only in `~/.aria/auth.json`, file mode `0600`, Pi's exact format: `{ "<provider>": { "type": "api_key", "key": "…" } }` or `{ "<provider>": { "type": "oauth", "access": "…", "refresh": "…", "expires": <ms epoch> } }`.
- Every write of `settings.json` / `auth.json` is atomic (temp file in the same dir + rename) and keeps keys Aria does not know.
- An existing `settings.json` that does not parse is never overwritten; the dock shows the problem.
- Hotkey `SUPER + A`. Dock namespace `aria-dock`. Control socket `$XDG_RUNTIME_DIR/aria/control.sock`.
- Before `gtk::init`: `WEBKIT_DISABLE_DMABUF_RENDERER=1`. Size the window with `set_size_request`, never `set_default_size`.
- Dock geometry: surface 700 px wide anchored top/bottom/right; dock column 470 px on the right; only the column takes input.
- Motion (from the design): open — veil opacity 180 ms `ease-out`; dock opacity 200 ms + translateX(24px→0) 240 ms `cubic-bezier(0.16, 1, 0.3, 1)`; content translateX(16px→0) + opacity 300 ms `cubic-bezier(0.22, 1, 0.36, 1)`; accent dot glow 2400 ms loop. Close — dock opacity 120 ms / transform 140 ms, veil 140 ms after 20 ms, `cubic-bezier(0.4, 0, 1, 1)`; unmap after `CLOSE_MS = 180`.
- Accent `#B9A8FF`; font Geist (fallback `system-ui`); veil `linear-gradient(to left, rgba(4,4,5,0.78) 0%, rgba(4,4,5,0.78) 52%, rgba(4,4,5,0) 100%)`.
- Phase 1 has no chat and runs no agent; integrations and packages are recorded, not installed or connected.

## Review Focus

1. **Hyprland reloads its config while the dock runs** → the hotkey still works afterwards (re-registered on `configreloaded>>`). Test in Task 6.
2. **`SUPER + A` pressed twice fast / during the close animation** → ends open and stable, never a hidden-but-focused or mapped-but-invisible surface. Test in Task 7 (visibility state machine).
3. **Sign-in fails or is abandoned** (wrong key, 401, no network, browser closed, OAuth port 53692 busy) → the step shows an error and lets you retry; nothing is written to `auth.json`. Tests in Tasks 4 and 5.
4. **`~/.aria/` already has files** (an older settings with extra keys, an `auth.json` with other providers, a broken `settings.json`) → unknown keys and other providers survive; a broken file is reported, not replaced. Tests in Tasks 2 and 3.
5. **Running the dock or `aria toggle` when another instance is up / its socket is stale after a crash** → exactly one dock; a stale socket is replaced. Test in Task 7.

---

## File Structure

```
Cargo.toml                       workspace: crates/aria-core, crates/aria-hyprland, crates/aria
README.md                        what Aria is, install, develop, layout
crates/aria-core/src/
  lib.rs                         re-exports
  paths.rs                       AriaPaths (~/.aria layout, ARIA_HOME)
  json_file.rs                   read_json, write_json_atomic, merge_preserving
  settings.rs                    Settings model, defaults, load/save
  auth.rs                        AuthStore (Pi format)
  providers.rs                   ProviderModule catalog (3 providers), quick-setup patch
  validate.rs                    key validation requests + response classification
  oauth/anthropic.rs             PKCE, authorize URL, callback server, token exchange
  bridge.rs                      PageMessage / DockMessage (dock↔page protocol)
crates/aria-hyprland/src/
  lib.rs                         register_hotkey, focused_monitor, watch_events, detect_config
crates/aria/src/
  main.rs                        CLI: `aria` (run), `aria toggle`
  control.rs                     single instance over the control socket
  visibility.rs                  open/close state machine (pure)
  dock.rs                        GTK layer-shell window + WebView + aria:// scheme
  handlers.rs                    PageMessage → actions (settings, auth, validate, oauth)
ui/
  index.html  dock.css  tokens.css
  app.js                         boot, bridge, open/close classes, notice vs welcome
  i18n/en.json  i18n/es.json     every UI string
  i18n.js                        t(key, vars), language selection
  onboarding/steps.js            pure: step ids, quick/full flows, answers → Settings JSON
  onboarding/view.js             renders the current step (fields ported from the design)
  onboarding/steps.test.js       bun tests for steps.js
install/
  install.sh                     curl | sh installer (+ --local, --uninstall, --dry-run)
  aria.service                   systemd user unit template
  install.test.sh                dry-run assertions
docs/
  architecture.md  settings.md  adding-a-provider.md
design/                          Aria.html + components/*.dc.html (reference, already present)
```

---

### Task 1: Workspace, paths and JSON files

**Files:**
- Create: `Cargo.toml`, `crates/aria-core/Cargo.toml`, `crates/aria-core/src/{lib.rs,paths.rs,json_file.rs}`, `.gitignore` (`/target`, `ui/node_modules`)
- Test: inline `#[cfg(test)]` modules

**Interfaces:**
- Produces:
  - `pub struct AriaPaths { pub root: PathBuf }` with `AriaPaths::from_env() -> AriaPaths` (`$ARIA_HOME` or `$HOME/.aria`), `settings(&self) -> PathBuf` (`root/settings.json`), `auth(&self) -> PathBuf` (`root/auth.json`), `ensure_root(&self) -> io::Result<()>` (creates `root`, mode `0700`).
  - `pub fn read_json(path: &Path) -> Result<Option<Value>, JsonFileError>` — `Ok(None)` when missing, `Err(JsonFileError::Invalid{path, message})` when unparsable.
  - `pub fn write_json_atomic(path: &Path, value: &Value, mode: u32) -> io::Result<()>` — pretty JSON + trailing newline, temp file in the same dir, `fsync`, `rename`, chmod `mode`.
  - `pub fn merge_preserving(original: Value, ours: Value) -> Value` — deep merge of objects: keys only in `original` are kept, keys in `ours` win; non-object values are replaced.

- [ ] **Step 1:** `git init` in the repo root (it is not a repo yet), create the workspace with the three members (only `aria-core` exists in this task; leave the others commented out until their tasks).
- [ ] **Step 2: Write failing tests**
  - `paths_use_aria_home`: with `ARIA_HOME=/tmp/x`, `settings()` is `/tmp/x/settings.json`.
  - `read_missing_is_none`, `read_invalid_is_error` (file `"{ nope"` → `JsonFileError::Invalid`).
  - `write_is_atomic_and_mode`: writes `{"a":1}` with `0o600`; content parses back; `metadata().permissions().mode() & 0o777 == 0o600`; no `*.tmp` left in the dir.
  - `merge_keeps_unknown_keys`: `merge_preserving({"a":1,"x":{"keep":true,"b":1}}, {"x":{"b":2}})` == `{"a":1,"x":{"keep":true,"b":2}}`.
- [ ] **Step 3:** Run `cargo test -p aria-core` → FAIL (not defined).
- [ ] **Step 4:** Implement `paths.rs` and `json_file.rs`.
- [ ] **Step 5:** `cargo test -p aria-core` → PASS.
- [ ] **Step 6:** Commit `chore: workspace, aria paths and atomic json files`.

### Task 2: Auth store

**Files:** Create `crates/aria-core/src/auth.rs`.

**Interfaces:**
- Consumes: `read_json`, `write_json_atomic`, `merge_preserving`, `AriaPaths` (Task 1).
- Produces:
  - `pub enum Credential { ApiKey { key: String }, OAuth { access: String, refresh: String, expires: i64 } }` serialized exactly as Pi: `{"type":"api_key","key":…}` / `{"type":"oauth","access":…,"refresh":…,"expires":…}`.
  - `pub struct AuthStore { path: PathBuf }` with `new(path)`, `get(&self, provider: &str) -> Result<Option<Credential>, JsonFileError>`, `set(&self, provider: &str, cred: &Credential) -> Result<(), AuthError>`, `has(&self, provider: &str) -> bool`.

- [ ] **Step 1: Write failing tests**
  - `set_writes_pi_format_with_0600`: `set("opencode-go", ApiKey{"k1"})` → file is `{"opencode-go":{"type":"api_key","key":"k1"}}`, mode `0600`.
  - `set_keeps_other_providers_and_unknown_fields`: existing `{"opencode":{"type":"api_key","key":"z"},"x":{"type":"api_key","key":"y","extra":1}}`; after `set("anthropic", OAuth{…})` both `opencode` and `x` (with `extra`) are unchanged.
  - `oauth_round_trip`: `get("anthropic")` returns the same `expires` as written.
  - `broken_auth_is_error_not_overwritten`: file `{ nope` → `set` returns `Err`, file content unchanged.
- [ ] **Step 2:** `cargo test -p aria-core auth` → FAIL.
- [ ] **Step 3:** Implement `auth.rs` (`set` = read → merge → `write_json_atomic(…, 0o600)`).
- [ ] **Step 4:** `cargo test -p aria-core auth` → PASS.
- [ ] **Step 5:** Commit `feat(core): auth store in Pi's format`.

### Task 3: Settings model and provider modules

**Files:** Create `crates/aria-core/src/{settings.rs,providers.rs}`, `docs/settings.md`, `docs/adding-a-provider.md`.

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `pub struct Settings` (serde, `rename_all = "camelCase"`) whose JSON is exactly the shape below; `Settings::defaults(language: &str) -> Settings`; `pub fn load(paths: &AriaPaths) -> Result<Option<Settings>, JsonFileError>`; `pub fn save(paths: &AriaPaths, s: &Settings) -> Result<(), SettingsError>` (merge onto the existing raw JSON so unknown keys survive; refuses with `SettingsError::Invalid` if the existing file does not parse); `pub const SETTINGS_VERSION: u32 = 1`; `pub fn language_from_locale(lang: &str) -> String` (used for the default language from `$LANG`).
  - `pub struct ProviderModule { pub id: &'static str, pub label: &'static str, pub auth: AuthKind, pub auth_entries: &'static [&'static str], pub models: &'static [ModelInfo], pub free_models: &'static [ModelInfo], pub quota: Option<QuotaDefaults>, pub includes_jev: bool, pub daily_default: &'static str, pub planning_default: &'static str }`, `pub enum AuthKind { ApiKey, OAuth }`, `pub struct ModelInfo { pub id: &'static str, pub label: &'static str }`, `pub fn providers() -> &'static [ProviderModule]`, `pub fn provider(id: &str) -> Option<&'static ProviderModule>`, `pub fn apply_provider(s: &mut Settings, id: &str)` (sets provider id, models defaults, quota defaults or `null`, router on/backend as below, effort mode, `sandbox.jevReview`).

Settings JSON (the defaults; this is also the example in `docs/settings.md`):

```json
{
  "version": 1,
  "setup": { "completed": false, "mode": null },
  "provider": { "id": null, "quota": null },
  "router": { "enabled": true, "backend": "opencode" },
  "assistant": { "userName": "", "name": "Aria", "language": "<from locale: es|en|pt, else en>", "style": "very-short" },
  "models": { "daily": null, "planning": null, "planningEffort": "high", "allowPayPerUse": false },
  "effort": { "mode": "auto", "fixedLevel": null, "maxOnFree": true },
  "compaction": { "percent": 40 },
  "memory": { "enabled": true, "project": "aria" },
  "integrations": { "services": [], "packages": { "source": "gallery", "path": "", "thirdPartyTrusted": false } },
  "sandbox": { "projectsDir": "~/Projects", "extraWritable": [], "askToLeave": true, "jevReview": true },
  "git": { "commitWithoutAsking": false, "pushOrMergeWithoutAsking": false, "askBeforeOthersChanges": true, "unaskedAsProposals": true },
  "subagents": {
    "experts": [ {"name":"frontend","enabled":true,"readOnly":false}, {"name":"backend","enabled":true,"readOnly":false},
                 {"name":"security","enabled":true,"readOnly":true}, {"name":"reviewer","enabled":true,"readOnly":true},
                 {"name":"researcher","enabled":true,"readOnly":true} ],
    "saveSessions": false, "turnLimit": null, "planExecute": true,
    "autoSkills": ["systematic-debugging", "verification-before-completion"]
  }
}
```

Value sets: `setup.mode` `"quick"|"full"|"skipped"|null`; `provider.id` `"opencode-go"|"anthropic-api"|"anthropic-oauth"|null`; `provider.quota` `{ "switchAt": 98, "planSwitchAt": 90, "freeModel": "opencode-go/cat-jam-free", "askFirst": true, "countdownMinutes": 5, "onNoAnswer": "switch" }` (`onNoAnswer` `"switch"|"stay"`); `router.backend` `"opencode"|"typesafe"`; `assistant.style` `"very-short"|"brief"|"detailed"|"technical"|"teaching"`; `assistant.language` is a code (`es`,`en`,`pt`) or free text; `effort.mode` `"auto"|"fixed"`, `fixedLevel` one of `off minimal low medium high xhigh max`; `integrations.services` items `{ "id": "notion"|"tavily"|"figma"|"openpencil"|"rea"|"other", "name": "…", "auth": "oauth"|"key" }`.

Provider catalog (phase 1 is static; phase 2 feeds it from Pi's model registry):

| id | auth | auth.json entries written | models (daily ★, planning ◆) | free models | quota | Jev |
|---|---|---|---|---|---|---|
| `opencode-go` | ApiKey | `opencode-go`, `opencode` (same key: Jev goes through `opencode`) | `opencode-go/deepseek-v4.1-flash` ★, `opencode-go/glm-5.3` ◆, `opencode-go/kimi-k3`, `opencode-go/qwen3.8-max` | `opencode-go/cat-jam-free`, `opencode-go/longcat-2.5-preview-free` | defaults above | included |
| `anthropic-api` | ApiKey | `anthropic` | `anthropic/claude-sonnet-5-5` ★, `anthropic/claude-opus-5-5` ◆, `anthropic/claude-haiku-5-5`, `anthropic/claude-fable-5-1` | — | none | TypeSafe key |
| `anthropic-oauth` | OAuth | `anthropic` | `claude-bridge/claude-sonnet-5-5` ★, `claude-bridge/claude-opus-5-5` ◆, `claude-bridge/claude-haiku-5-5`, `claude-bridge/claude-fable-5-1` | — | none | TypeSafe key |

`apply_provider`: `opencode-go` → `router {enabled:true, backend:"opencode"}`, `effort.mode "auto"`, `sandbox.jevReview true`; the Anthropic ones → `router {enabled:false, backend:"typesafe"}` (the router step can turn it on when a TypeSafe key is given), `effort {mode:"fixed", fixedLevel:"medium"}`, `sandbox.jevReview false`.

- [ ] **Step 1: Write failing tests**
  - `defaults_match_owner_setup`: `serde_json::to_value(Settings::defaults("en"))` equals the JSON above with `"language":"en"`.
  - `language_from_locale`: `language_from_locale("es_CO.UTF-8") == "es"`, `("pt_BR.UTF-8") == "pt"`, `("de_DE.UTF-8") == "en"`, `("") == "en"` (`pub fn language_from_locale(lang: &str) -> String`).
  - `save_keeps_unknown_keys`: existing file `{"version":1,"future":{"x":1},"assistant":{"mood":"calm"}}` → after `save`, `future.x == 1` and `assistant.mood == "calm"` remain.
  - `save_refuses_broken_file`: file `{ nope` → `Err(SettingsError::Invalid)`, content unchanged.
  - `apply_opencode_go`: quota equals the defaults, `models.daily == "opencode-go/deepseek-v4.1-flash"`, `router.backend == "opencode"`.
  - `apply_anthropic_oauth`: `quota == null`, `models.planning == "claude-bridge/claude-opus-5-5"`, `router.enabled == false`, `effort.fixedLevel == Some("medium")`.
  - `only_three_providers`: `providers().len() == 3`.
- [ ] **Step 2:** `cargo test -p aria-core settings providers` → FAIL.
- [ ] **Step 3:** Implement `settings.rs` and `providers.rs`. `docs/settings.md`: the JSON above, every key in one line, which phase uses it. `docs/adding-a-provider.md`: add a `ProviderModule` entry, its validation in `validate.rs`, its i18n strings.
- [ ] **Step 4:** `cargo test -p aria-core` → PASS.
- [ ] **Step 5:** Commit `feat(core): settings model and provider modules`.

### Task 4: Key validation

**Files:** Create `crates/aria-core/src/validate.rs`.

**Interfaces:**
- Consumes: Task 3 (`provider ids`).
- Produces:
  - `pub struct ProbeRequest { pub method: &'static str, pub url: String, pub headers: Vec<(String, String)>, pub body: Option<Value> }`
  - `pub fn probe_for(kind: KeyKind, key: &str) -> ProbeRequest` with `pub enum KeyKind { OpenCodeGo, AnthropicApi, TypeSafe }`
  - `pub fn classify(status: u16) -> KeyCheck` with `pub enum KeyCheck { Valid, Rejected, Unavailable(String) }` (200–299 → Valid; 401/403 → Rejected; anything else → Unavailable("HTTP <n>"))
  - `pub fn check_key(kind: KeyKind, key: &str) -> KeyCheck` — runs the probe with ureq, timeout 10 s; network/TLS errors → `Unavailable(message)`.

Probes:
- OpenCode Go: `GET https://opencode.ai/zen/go/v1/usage`, `authorization: Bearer <key>`.
- Anthropic API: `GET https://api.anthropic.com/v1/models`, `x-api-key: <key>`, `anthropic-version: 2023-06-01`.
- TypeSafe: `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>`, `Content-Type: application/json`, body `{"model":"jev-latest","state":"ping","questions":{"ok":{"type":"noul","instructions":"The state is a short test message"}}}`.

- [ ] **Step 1: Write failing tests** `probe_opencode_go`, `probe_anthropic_api`, `probe_typesafe` (exact method, URL, headers, body above); `classify_statuses` (200→Valid, 204→Valid, 401→Rejected, 403→Rejected, 500→`Unavailable("HTTP 500")`); `check_key_unreachable` using a probe pointed at `http://127.0.0.1:9` (add `pub fn check(probe: &ProbeRequest) -> KeyCheck` so tests can pass a custom probe) → `Unavailable(_)`.
- [ ] **Step 2:** `cargo test -p aria-core validate` → FAIL.
- [ ] **Step 3:** Implement; never log the key.
- [ ] **Step 4:** `cargo test -p aria-core validate` → PASS.
- [ ] **Step 5:** Commit `feat(core): provider key validation`.

### Task 5: Anthropic OAuth sign-in

**Files:** Create `crates/aria-core/src/oauth/{mod.rs,anthropic.rs}`.

Values (copied from Pi's `pi-ai` `auth/oauth/anthropic.js`, which is what the owner's Claude Max route uses today):
- `CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"`
- `AUTHORIZE_URL = "https://claude.ai/oauth/authorize"`, `TOKEN_URL = "https://platform.claude.com/v1/oauth/token"`
- callback: listen on `127.0.0.1:53692` path `/callback` (fall back to a free port when busy); `redirect_uri = "http://localhost:<port>/callback"`
- `SCOPES = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"`
- PKCE S256; `state` = the verifier.
- authorize query: `code=true`, `client_id`, `response_type=code`, `redirect_uri`, `scope`, `code_challenge`, `code_challenge_method=S256`, `state`.
- token exchange: `POST TOKEN_URL`, JSON `{"grant_type":"authorization_code","client_id":…,"code":…,"state":…,"redirect_uri":…,"code_verifier":…}`, headers `Content-Type: application/json`, `Accept: application/json`, timeout 30 s.
- credential: `OAuth { access: access_token, refresh: refresh_token, expires: now_ms + expires_in*1000 - 300_000 }`.

**Interfaces:**
- Consumes: `Credential` (Task 2).
- Produces:
  - `pub struct Pkce { pub verifier: String, pub challenge: String }`, `pub fn pkce() -> Pkce` (32 random bytes, base64url no padding; challenge = base64url(sha256(verifier)))
  - `pub fn authorize_url(pkce: &Pkce, redirect_uri: &str) -> String`
  - `pub fn parse_callback(request_target: &str, expected_state: &str) -> Result<String, OAuthError>` (returns `code`; `OAuthError::StateMismatch`, `OAuthError::Denied(msg)` for `error=` params, `OAuthError::NoCode`)
  - `pub fn token_request_body(code: &str, state: &str, pkce: &Pkce, redirect_uri: &str) -> Value`
  - `pub fn credential_from_token(json: &Value, now_ms: i64) -> Result<Credential, OAuthError>`
  - `pub struct SignIn { pub url: String, … }`, `pub fn start() -> Result<SignIn, OAuthError>` (binds the listener, builds the URL); `impl SignIn { pub fn wait(self, timeout: Duration) -> Result<Credential, OAuthError> }` (accepts one callback, answers it with a small "You can close this tab" page, exchanges the code; `OAuthError::Timeout` after `timeout`).

- [ ] **Step 1: Write failing tests**
  - `pkce_challenge_is_s256`: for verifier `"dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"` the challenge is `"E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"` (RFC 7636 vector; expose `pub fn challenge_for(verifier: &str) -> String`).
  - `authorize_url_has_every_param`: parsing the URL gives exactly the eight query params above with the right values.
  - `parse_callback_ok`, `parse_callback_state_mismatch`, `parse_callback_denied` (`/callback?error=access_denied&state=v`).
  - `token_body_shape` (exact JSON).
  - `credential_expiry`: `{"access_token":"a","refresh_token":"r","expires_in":3600}` at `now_ms = 1_000_000` → `expires == 1_000_000 + 3_600_000 - 300_000`.
  - `busy_port_falls_back`: with `127.0.0.1:53692` already bound by the test, `start()` succeeds and its URL's `redirect_uri` uses another port.
  - `wait_times_out`: `start()?.wait(Duration::from_millis(200))` → `Err(OAuthError::Timeout)`.
- [ ] **Step 2:** `cargo test -p aria-core oauth` → FAIL.
- [ ] **Step 3:** Implement with `std::net::TcpListener` (one request, minimal HTTP parsing of the request line) and ureq for the exchange.
- [ ] **Step 4:** `cargo test -p aria-core oauth` → PASS.
- [ ] **Step 5:** Commit `feat(core): Anthropic OAuth sign-in (PKCE)`.

### Task 6: Hyprland integration

**Files:** Create `crates/aria-hyprland/{Cargo.toml,src/lib.rs}`; enable it in the workspace.

**Interfaces:**
- Produces:
  - `pub fn registration_lua(exe: &Path) -> String` — `hl.unbind("SUPER + A"); hl.bind("SUPER + A", hl.dsp.exec_cmd("<exe> toggle"), { description = "Toggle Aria" }); hl.layer_rule({ name = "aria-dock", match = { namespace = "^aria-dock$" }, blur = false, no_anim = true })` (the exe path quoted for Lua: escape `\` and `"`).
  - `pub fn register_hotkey(exe: &Path) -> Result<(), HyprError>` — runs `hyprctl eval <lua>`; output other than `ok` → `HyprError::Eval(output)`.
  - `pub enum ConfigKind { Lua, Classic }`, `pub fn detect_config() -> Result<ConfigKind, HyprError>` — `hyprctl eval "return 1"`: `ok` → `Lua`; an "unknown request"/error → `Classic`; no Hyprland (`HYPRLAND_INSTANCE_SIGNATURE` unset) → `HyprError::NotRunning`.
  - `pub struct Monitor { pub name: String, pub x: i32, pub y: i32 }`, `pub fn parse_focused(monitors_json: &str) -> Option<Monitor>`, `pub fn focused_monitor() -> Option<Monitor>` (writes `j/monitors` to `$XDG_RUNTIME_DIR/hypr/$HYPRLAND_INSTANCE_SIGNATURE/.socket.sock`).
  - `pub fn is_config_reloaded(event_line: &str) -> bool` (`starts_with("configreloaded>>")`), `pub fn watch_events(on_reload: impl Fn() + Send + 'static) -> std::thread::JoinHandle<()>` (reads `.socket2.sock` lines; reconnects after 1 s if the socket drops).

- [ ] **Step 1: Write failing tests** `lua_is_exact` (for `/home/u/.local/bin/aria`, the exact string above), `lua_escapes_quotes` (path `/tmp/a"b/aria` → `a\"b`), `parse_focused_picks_focused` (two-monitor JSON fixture copied from `hyprctl monitors -j`, DP-2 focused at x=1536 → `Monitor{name:"DP-2",x:1536,y:0}`), `parse_focused_none`, `reload_event` (`"configreloaded>>"` true, `"workspace>>2"` false).
- [ ] **Step 2:** `cargo test -p aria-hyprland` → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `cargo test -p aria-hyprland` → PASS.
- [ ] **Step 5 (manual, on the real desktop):** a tiny `examples/register.rs` that calls `register_hotkey` with `/bin/true`, then `hyprctl binds -j` shows a `SUPER`(modmask 64)+`A` bind with description `Toggle Aria`; run `hyprctl reload`, call the watcher's callback path, check the bind is back. Remove the bind afterwards with `hyprctl eval 'hl.unbind("SUPER + A")'`.
- [ ] **Step 6:** Commit `feat(hyprland): runtime hotkey, focused monitor, reload events`.

### Task 7: Dock binary — single instance, visibility, window

**Files:** Create `crates/aria/{Cargo.toml,src/main.rs,src/control.rs,src/visibility.rs,src/dock.rs}`; enable in workspace. Port the window code from `spikes/dock-open/src/main.rs`.

**Interfaces:**
- Consumes: Task 6 (`register_hotkey`, `detect_config`, `focused_monitor`, `watch_events`), Task 1 (`AriaPaths`).
- Produces:
  - CLI: `aria` runs the dock; `aria toggle` sends `toggle\n` to `$XDG_RUNTIME_DIR/aria/control.sock`, starting `aria` detached first when nobody answers (retry 60 × 50 ms).
  - `control.rs`: `pub fn bind_single_instance(path: &Path) -> Result<UnixListener, AlreadyRunning>` — if a connect succeeds → `AlreadyRunning`; else remove a stale file and bind.
  - `visibility.rs` (pure): `pub enum State { Hidden, Opening, Open, Closing }`, `pub enum Event { Toggle, Escape, CloseFinished }`, `pub enum Command { Show, StartClose, Unmap, None }`, `pub fn step(state: State, event: Event) -> (State, Command)`. `Opening` becomes `Open` on the page's `opened` message (the page reports when its open transition ends).
  - `dock.rs`: `pub struct Dock` — layer-shell window (namespace `aria-dock`, layer Overlay, anchors top/bottom/right, 700 px via `set_size_request`, RGBA visual, input region = rightmost 470 px recomputed on `size-allocate`), `WebView` with transparent background serving the embedded `ui/` through `aria://app/…` (`include_dir!` of `$CARGO_MANIFEST_DIR/../../ui`), a script message handler named `aria`, and `pub fn send(&self, msg: &DockMessage)` (runs `window.aria.receive(<json>)`). On `Show`: `set_monitor` to the GDK monitor at the focused monitor's `(x, y)`, keyboard mode **Exclusive**, `show_all`, `present`, send `DockMessage::Open{monitor}`. On `StartClose`: keyboard mode `None`, send `DockMessage::Close`, schedule `CloseFinished` after `CLOSE_MS = 180`. On `Unmap`: `hide()`.
  - Startup: `detect_config()`; `Classic` → the dock still starts but shows the problem screen (`DockMessage::Problem{code:"hyprland-classic-config"}`) on open, and the hotkey is not registered; `Lua` → `register_hotkey(current_exe)` and `watch_events(|| register_hotkey(..))`.

- [ ] **Step 1: Write failing tests**
  - visibility: `toggle_from_hidden_shows` (`Hidden,Toggle → Opening,Show`), `toggle_while_opening_closes` (`Opening,Toggle → Closing,StartClose`), `toggle_while_closing_reopens` (`Closing,Toggle → Opening,Show`), `escape_when_hidden_is_noop`, `close_finished_unmaps` (`Closing,CloseFinished → Hidden,Unmap`), `stale_close_finished_ignored` (`Opening,CloseFinished → Opening,None`).
  - control: `second_bind_is_already_running`, `stale_socket_is_replaced` (a file left at the path with nobody listening → bind succeeds).
- [ ] **Step 2:** `cargo test -p aria` → FAIL.
- [ ] **Step 3:** Implement `control.rs`, `visibility.rs`, `dock.rs`, `main.rs` (set `WEBKIT_DISABLE_DMABUF_RENDERER=1` before `gtk::init`). The UI at this point can be a placeholder `ui/index.html`; Task 9 fills it.
- [ ] **Step 4:** `cargo test -p aria` → PASS; `cargo build --release` succeeds.
- [ ] **Step 5 (manual):** run `target/release/aria`; `SUPER + A` opens on the focused monitor (try both monitors), typing goes to the dock at once, `Esc`/`SUPER + A` closes; after `hyprctl reload` the hotkey still works; a second `target/release/aria` prints that one is already running.
- [ ] **Step 6:** Commit `feat(dock): single-instance layer-shell dock with runtime hotkey`.

### Task 8: Bridge protocol and handlers

**Files:** Create `crates/aria-core/src/bridge.rs`, `crates/aria/src/handlers.rs`.

**Interfaces:**
- Consumes: Tasks 2–5, 7.
- Produces (JSON, `#[serde(tag = "type", rename_all = "camelCase")]`; every page request carries `id: u64`, every reply echoes it):
  - `PageMessage`: `Ready`, `Opened`, `CloseRequest`, `LoadSettings{id}`, `SaveSettings{id, settings: Value}`, `SaveKey{id, provider: String, key: String}` (provider is `opencode-go` | `anthropic-api` | `typesafe`), `StartOAuth{id}` (Anthropic), `OpenUrl{url}`.
  - `DockMessage`: `Open{monitor: String}`, `Close`, `Notice`, `Settings{id, settings: Option<Value>, exists: bool, problem: Option<String>}`, `Saved{id}`, `KeyResult{id, ok: bool, error: Option<String>}` (`error` is an i18n key: `key-rejected`, `network-unavailable`, `auth-file-broken`), `OAuthUrl{id, url}`, `OAuthResult{id, ok, error}` (`oauth-denied`, `oauth-timeout`, `oauth-state`, `network-unavailable`), `Problem{code}`, `Error{id, code}`.
  - `handlers.rs`: `pub fn handle(msg: PageMessage, ctx: &Ctx) -> Vec<DockMessage>` for the synchronous ones; `SaveKey` and `StartOAuth` run on a worker thread and post their result back through the GTK main loop.
  - `SaveKey` flow: `check_key` → only on `Valid` write the credential (for `opencode-go` both `opencode-go` and `opencode` entries); never write on `Rejected`/`Unavailable`.
  - `StartOAuth` flow: `oauth::anthropic::start()` → send `OAuthUrl` and open the URL with `xdg-open` → `wait(5 min)` → on success `AuthStore::set("anthropic", cred)`.
  - On first `Ready`/`Open`: if `settings.json` does not exist → the page shows the welcome; `Notice` is sent once at process start when it does not exist (the corner notice "Aria is installed — Press Super A to set it up").

- [ ] **Step 1: Write failing tests**
  - `page_message_parses` for each variant from a JSON literal (e.g. `{"type":"saveKey","id":3,"provider":"typesafe","key":"k"}`).
  - `dock_message_serializes` (e.g. `KeyResult{id:3, ok:false, error:Some("key-rejected")}` → `{"type":"keyResult","id":3,"ok":false,"error":"key-rejected"}`).
  - handlers with a temp `ARIA_HOME`: `load_settings_when_missing` (`exists:false, settings:None`), `load_settings_broken` (`problem: Some("settings-invalid")`), `save_settings_round_trip`, `rejected_key_writes_nothing` (inject a validator returning `Rejected` via `Ctx`; `auth.json` absent afterwards), `opencode_key_writes_both_entries`.
- [ ] **Step 2:** `cargo test -p aria-core bridge && cargo test -p aria handlers` → FAIL.
- [ ] **Step 3:** Implement; `Ctx` holds `AriaPaths` and a `validator: Box<dyn Fn(KeyKind, &str) -> KeyCheck + Send + Sync>` (real one = `check_key`).
- [ ] **Step 4:** tests PASS.
- [ ] **Step 5:** Commit `feat(dock): page bridge and settings/auth handlers`.

### Task 9: Dock UI — shell, notice, i18n

**Files:** Create `ui/index.html`, `ui/tokens.css`, `ui/dock.css`, `ui/app.js`, `ui/i18n.js`, `ui/i18n/en.json`, `ui/i18n/es.json`.

**Interfaces:**
- Consumes: the bridge (Task 8): `window.webkit.messageHandlers.aria.postMessage(JSON)`; the dock calls `window.aria.receive(msg)`.
- Produces: `window.aria = { receive(msg), send(msg) → Promise (resolves on the reply with the same id) }`; `t(key, vars)` from `i18n.js`; `setLanguage(code)` (`es`/`en`/`pt`, anything else → `en`; `pt` falls back to `en` strings until translated).

- [ ] **Step 0: Write failing tests** `ui/i18n.test.js` (`bun test ui`): `t('welcome.title')` is the English text after `setLanguage('en')` and the Spanish text after `setLanguage('es')`; `setLanguage('pt')` and `setLanguage('de')` fall back to English; `t('missing.key')` returns the key itself; `en.json` and `es.json` have exactly the same keys. Run → FAIL; implement `i18n.js`; run → PASS.
- [ ] **Step 1:** Port the shell from the spike's `dock.html` (veil + 470 px column, the motion values in Global Constraints, `body.open` class, two-frame open, `opened` message posted on `transitionend` of the column's transform, `CloseRequest` on Esc as a fallback). Port the corner notice and welcome screen look from `design/components/aria-on-the-desktop.dc.html` and `onboarding.dc.html` (`welcome` step: title, intro, "Quick setup — about 1 min", "Customize — 12 steps", "You can change all of this later.", "Skip for now"). Arrow keys move between the two cards, Enter picks.
- [ ] **Step 2:** All visible strings through `t()`; write `en.json` from the design's English copy and `es.json` with the Spanish translation (natural Colombian Spanish, `tú`).
- [ ] **Step 3: Verify (manual):** open with `SUPER + A` → notice first run, welcome screen, motion as in the design side by side with `design/Aria.html` ("Open Aria"); switch language by editing `assistant.language` in a test `settings.json` → Spanish strings.
- [ ] **Step 4:** Commit `feat(ui): dock shell, notice, welcome and i18n`.

### Task 10: Onboarding flow and saving settings

**Files:** Create `ui/onboarding/steps.js`, `ui/onboarding/view.js`, `ui/onboarding/steps.test.js`; update `ui/app.js`, both i18n files.

**Interfaces:**
- Consumes: `window.aria.send` (Task 9), Settings JSON shape and provider catalog (Task 3 — mirror the catalog in `steps.js` as `PROVIDERS`, same ids/models/defaults).
- Produces (`steps.js`, pure):
  - `stepIds(mode, providerId)`: quick → `['provider','auth','assistant','review']`; full → `['provider', …(providerId==='opencode-go' ? ['quota'] : []), 'auth','router','assistant','models','effort','compaction','memory','integrations','sandbox','git','agents','review']`.
  - `next(state)`, `back(state)` (back from the first step → `welcome`).
  - `toSettings(answers, base)` → a Settings object (Task 3 shape) with `setup: {completed:true, mode}`; quick mode takes `base` (defaults + `apply_provider` equivalent) for every step not asked.
  - `skipped(base)` → `setup: {completed:false, mode:"skipped"}`, nothing else changed.

Field content per step: port from `design/components/onboarding.dc.html` (`providers()`, field builders, review step), with these changes from the design: only the three providers of Task 3 (labels: "OpenCode Go", "Anthropic · API key", "Anthropic · Sign in with Claude"); OpenCode Go has no OAuth option; the router step shows "Jev is included with OpenCode Go" for `opencode-go` and the TypeSafe key field (`console.typesafe.ai/keys`) otherwise; owner defaults from Task 3 (not the design's `defaults()`); style options in this order: very short and direct, no tables | brief, with lists when they help | detailed, with explanations | technical and precise, with code and paths | teaching, step by step; language options Español | English | Português | Other (type). Sign-in step: key field + "Check and save" → `SaveKey`, shows the `KeyResult` error string, Next disabled until saved; OAuth → `StartOAuth`, shows "Waiting for your browser…" and the URL as a fallback link, then the result. Review → `SaveSettings`, then the "Aria is set up" screen. Re-opening after setup starts at `review` with the current values, any row jumps to its step.

- [ ] **Step 1: Write failing tests** (`bun test ui/onboarding`):
  - `quick_steps`, `full_steps_with_quota` (opencode-go has `quota` second), `full_steps_without_quota` (anthropic-api).
  - `back_from_first_is_welcome`.
  - `quick_settings_use_owner_defaults`: quick with `opencode-go`, userName `Ivan`, language `es` → `subagents.turnLimit === null`, `git.commitWithoutAsking === false`, `subagents.autoSkills` equals the two skills, `compaction.percent === 40`, `provider.quota.switchAt === 98`, `router.backend === 'opencode'`.
  - `anthropic_oauth_settings`: `provider.id === 'anthropic-oauth'`, `provider.quota === null`, `models.daily === 'claude-bridge/claude-sonnet-5-5'`.
  - `router_with_typesafe`: full flow, anthropic-api, router on → `router: {enabled:true, backend:'typesafe'}`.
  - `skip_records_skipped`.
- [ ] **Step 2:** `bun test ui/onboarding` → FAIL.
- [ ] **Step 3:** Implement `steps.js`, then `view.js` (arrow keys move between options, Enter selects/continues, Shift+Tab/Backspace on an empty field goes back, Esc closes the dock) and wire into `app.js`.
- [ ] **Step 4:** `bun test ui/onboarding` → PASS.
- [ ] **Step 5 (manual, real keys):** with `ARIA_HOME=$(mktemp -d)`: quick setup with OpenCode Go → `settings.json` matches the quick test's values and `auth.json` (`0600`) has `opencode-go` and `opencode`; a wrong key shows the rejection and writes nothing; Anthropic sign-in opens the browser and stores an `oauth` credential; re-open → review with current values.
- [ ] **Step 6:** Commit `feat(ui): onboarding flow writes settings`.

### Task 11: Installer and docs

**Files:** Create `install/install.sh`, `install/aria.service`, `install/install.test.sh`, `README.md`, `docs/architecture.md`.

**Interfaces:**
- `install.sh [--local <path-to-aria>] [--uninstall] [--dry-run]`; default source `ARIA_DOWNLOAD_URL` (placeholder `https://<owner site>/releases/latest/aria-x86_64-linux` + `.sha256`, both overridable by env). Steps in order: check `HYPRLAND_INSTANCE_SIGNATURE` and Lua config (`hyprctl eval 'return 1'` prints `ok`) else exit 1 with a clear message; `sudo pacman -S --needed gtk3 webkit2gtk-4.1 gtk-layer-shell`; get the binary (download + `sha256sum -c`, or copy `--local`); install to `~/.local/bin/aria` (mode 755); `mkdir -p ~/.aria` (700); write `~/.config/systemd/user/aria.service` from the template; `systemctl --user daemon-reload && systemctl --user enable --now aria.service`. Re-running replaces the binary and restarts the unit. `--uninstall`: disable/stop the unit, remove the unit file and the binary, `hyprctl eval 'hl.unbind("SUPER + A")'` when Hyprland runs, then ask `Delete ~/.aria? [y/N]`. `--dry-run` prints each action as `would: <command>` and changes nothing.
- `aria.service`: `[Unit] Description=Aria dock; PartOf=graphical-session.target; After=graphical-session.target` · `[Service] ExecStart=%h/.local/bin/aria; Restart=on-failure` · `[Install] WantedBy=graphical-session.target`.

- [ ] **Step 1: Write failing test** `install.test.sh`: `sh install/install.sh --dry-run --local /tmp/fake-aria` output contains, in order, `would: pacman`, `would: install`, `would: systemctl --user enable --now aria.service`; `--dry-run --uninstall` contains `would: systemctl --user disable --now aria.service` and `hl.unbind`; `shellcheck install/install.sh` passes.
- [ ] **Step 2:** `sh install/install.test.sh` → FAIL.
- [ ] **Step 3:** Implement the script, unit and docs. `README.md`: what Aria is, the one-line install, `--local` for development, the repo layout, the phase roadmap (link the spec). `docs/architecture.md`: the three crates, the UI, the bridge messages, `~/.aria/`, how the hotkey works without Hyprland edits.
- [ ] **Step 4:** `sh install/install.test.sh` → PASS.
- [ ] **Step 5 (manual end-to-end):** `cargo build --release && sh install/install.sh --local target/release/aria` on the real desktop → no file under `~/.config/hypr/` changed (`git -C ~/.config/hypr status` or compare mtimes), the notice appears, `SUPER + A` opens the welcome, quick setup writes `~/.aria/settings.json`; log out/in → `SUPER + A` works again; `--uninstall` removes everything and asks about `~/.aria`.
- [ ] **Step 6:** Commit `feat: installer, systemd unit and docs`; then remove `spikes/` in a separate commit `chore: drop the dock-open spike` (its lessons live in the spec and the code).
