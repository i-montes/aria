# The dock reloads its page when the web view's process dies

Idea 13 of `docs/ideas.md`. The page runs in a separate web content process (WebKitGTK on Omarchy, WKWebView on
macOS). When that process dies (out of memory, a WebKit bug, killed), the panel stays mapped but paints nothing: the
shortcut seems to do nothing until Aria restarts. Both toolkits report the death; the dock reloads the page.

## Behaviour

- **Detect.** Omarchy: the web view's `web-process-terminated` signal (any reason). macOS: a `WKNavigationDelegate` on
  the web view whose `webViewWebContentProcessDidTerminate:` fires.
- **Stop delivering.** From the death until the new page says `ready`, the dock's `ready` is false, so every message
  waits in the queue as it does at start (the dead page would drop them).
- **Reload** `aria://app/index.html` (the same load as at start), after a delay from the recovery policy below.
- **Bring the new page up to date.** The page pulls its own state when it boots (settings, sessions, every tab's
  history from Pi), so a conversation comes back as after any restart. What the dock pushes only once must be pushed
  again: the last `Platform` and `Version` messages it sent are kept and replayed to the new page first; and when the
  dock is on screen (`State` not `Hidden`), it sends `Open { monitor }` (the monitor of the last open) and `Problem`
  (the current code, empty when none), as `Command::Show` does, so the page shows the open dock. Messages queued
  since the death follow, in order.
- **Log** one line per death and per reload (`aria: the page's process died (<reason>); reloading in <n> ms`), through
  the existing `crate::note` / `eprintln!` habit of each platform file.

## Recovery policy (shared, pure)

A crash at load would otherwise reload in a tight loop. `dock/mod.rs` holds a small pure type, `Recovery`, used by
both platforms:

- `Recovery::died(now) -> Duration`: the delay before reloading. The first death in a quiet period reloads at once
  (0 ms); each further death within 60 s of the previous one doubles the delay from 1 s (1, 2, 4, 8, 16 s), capped at
  30 s. A death more than 60 s after the previous one starts over at 0 ms.
- It is tested with explicit instants (no sleeping).

The replayed messages are kept by `Replay` in `dock/mod.rs`, also pure and shared: `Replay::remember(&DockMessage,
json)` keeps the JSON of the latest `Platform` and the latest `Version` (any other message is not kept), and
`Replay::messages()` returns them, `Platform` first. Tested.

## Out of scope

The main process dying (systemd and launchd restart it), a page that never sends `ready` (no watchdog on the first
load), and keeping a half-typed draft across the reload (the draft lives in the page).

## Checks

- `Recovery` and `Replay`: unit tests in `crates/aria/src/dock/mod.rs`.
- Omarchy, live: kill the dock's `WebKitWebProcess`; the log shows the death and the reload, and the dock opens with
  the chat and the right shortcut text.
- macOS: `sh tools/check-macos.sh` type-checks the delegate; the live check waits for the Mac (`docs/pending.md`).
