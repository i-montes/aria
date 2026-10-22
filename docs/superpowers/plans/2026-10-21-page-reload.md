# The dock reloads its page when the web view's process dies — plan

Spec: `docs/superpowers/specs/2026-10-21-page-reload-design.md` (the authority).

## Global constraints

- Everything in English. Match the surrounding style of `crates/aria/src/dock/` (doc comments on items, short `//`
  comments that say why, `Cell`/`RefCell` fields with a doc line each).
- Commit subjects are conventional (`feat(dock): …`, `test(dock): …`) and end with the line
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- No personal data in anything committed.
- Policy values exactly: first death in a quiet period 0 ms; each further death within 60 s of the previous one 1 s,
  then doubled (1, 2, 4, 8, 16 s), capped at 30 s; a death more than 60 s after the previous one starts over at 0 ms.
- The page URL stays `aria://app/index.html` (one constant or the existing literal, not a new spelling).
- `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check` and
  `sh tools/check-macos.sh` must pass at the end of each task.
- Never kill or restart the installed dock (`aria.service`) or its processes: the controller does the live check.

## Task 1: the shared `Recovery` and `Replay`, and the Omarchy dock

Files: `crates/aria/src/dock/mod.rs`, `crates/aria/src/dock/gtk.rs`.

1. In `dock/mod.rs`, two pure types (no GTK, no AppKit), compiled on every platform, with unit tests in its `tests`
   module:
   - `pub(crate) struct Recovery` (`Default`), with `pub(crate) fn died(&mut self, now: std::time::Instant) -> std::time::Duration`
     implementing the policy above. Tests: first death → 0; a second within 60 s → 1 s; then 2, 4, 8, 16, 30, 30 s;
     a death 61 s after the previous one → 0 again; the 60 s window is measured from the previous death (a death at
     +59 s after the previous one still escalates).
   - `pub(crate) struct Replay` (`Default`), with `pub(crate) fn remember(&mut self, message: &DockMessage, json: &str)`
     (keeps the JSON of the latest `DockMessage::Platform { .. }` and the latest `DockMessage::Version { .. }`; ignores
     every other variant) and `pub(crate) fn messages(&self) -> Vec<String>` (the kept ones, `Platform` first, then
     `Version`; empty when none). Tests: nothing kept → empty; two `Platform`s → only the latest; `Version` then
     `Platform` → `[platform, version]`; an `Open` is ignored.
   Use `aria_core::bridge::DockMessage`. If a type is unused on macOS until Task 2, allow it there the way
   `stays_in_app` does (`#[cfg_attr(not(target_os = "linux"), allow(dead_code))]`).
2. In `dock/gtk.rs`:
   - New fields on `Dock`: `recovery: RefCell<Recovery>`, `replay: RefCell<Replay>`, `monitor: RefCell<String>` (the
     monitor name of the last `Command::Show`), and `reloads: Cell<u64>` (bumped on each death, so a reload timer from
     an earlier death is ignored — the same idea as `generation`).
   - `send` records the message in `replay` (`remember`) before delivering or queueing.
   - `Command::Show` stores the monitor it got from `place_on_focused_monitor` in `monitor`.
   - `webview.connect_web_process_terminated(...)` (webkit2gtk's `WebViewExt`), holding a `Weak<Dock>` like the other
     handlers: sets `ready` to false; computes the delay with `recovery.died(Instant::now())`; logs
     `aria: the page's process died (<reason:?>); reloading in <n> ms`; bumps `reloads`; and after the delay
     (`glib::timeout_add_local_once`), if `reloads` still holds that value, calls `load_uri("aria://app/index.html")`.
   - `page_ready`, when the page is ready again after a death (keep a `Cell<bool>` `reloaded` set by the handler, or
     an equivalent): first deliver `replay.messages()`, then, when `state` is not `State::Hidden`, deliver
     `DockMessage::Open { monitor }` and `DockMessage::Problem { code }` (current problem, empty when none), then the
     queued messages as today, then the notice as today. On the first load nothing changes (no replay: those messages
     are already in the queue).
   - Keep the module doc and the field docs accurate.
3. Run the constraint commands; all pass.

## Task 2: the macOS dock

Files: `crates/aria/src/dock/mac.rs` (and `crates/aria/Cargo.toml` only if an objc2-web-kit feature must be enabled).

1. The same fields and the same `send` / `Command::Show` / `page_ready` changes as Task 1 made in `gtk.rs`, using the
   shared `Recovery` and `Replay` (remove any `allow(dead_code)` Task 1 added that is no longer needed).
2. A `WKNavigationDelegate`: a new `define_class!` (`AriaNavigationDelegate`, `NSObject` subclass, `MainThreadOnly`,
   ivars holding the same `Rc<RefCell<Option<Weak<Dock>>>>` slot `ScriptHandler` uses) implementing
   `webViewWebContentProcessDidTerminate:`; it calls a `Dock` method that does what Task 1's GTK handler does (ready
   false, delay from `recovery`, log through `crate::note`, bump `reloads`, reload `aria://app/index.html` after the
   delay with the timer mechanism `mac.rs` already uses for `CLOSE_MS`). Set it with `webview.setNavigationDelegate`
   and keep it retained in a `Dock` field (`_navigation`), like `_script`.
3. Run the constraint commands (on Linux `sh tools/check-macos.sh` is the macOS proof); all pass.
