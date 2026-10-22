# Scheduled tasks — design

The owner tells Aria, in the chat, "every day at 7:30 give me my Notion tasks", and from then on Aria does it by itself.
Decided with the owner (2026-10-21): approach A (the dock schedules and runs); each scheduled task has its own tab plus a
desktop notification; a run missed while the computer slept or Aria was off runs once when it is back (same day only);
permissions are fixed when the task is created, and anything outside them is refused, never asked. Idea #14 of
`docs/ideas.md` (a morning brief) is one such task, nothing more.

## A scheduled task

`~/.aria/schedules.json` (mode 600), `{ "tasks": [Task] }`, owned by `aria-core/src/schedules.rs`:

```json
{
  "id": "s-<12 lowercase alphanumerics>",
  "title": "Notion tasks",
  "prompt": "Give me today's tasks from my Notion, grouped by priority",
  "when": { "at": "07:30", "days": ["mon","tue","wed","thu","fri","sat","sun"], "date": null },
  "grants": { "connectors": { "notion": "read" }, "writable": [] },
  "enabled": true,
  "created": 1792578000000,
  "lastRun": null,
  "lastResult": null
}
```

- `when.at` is local `HH:MM` (24 h). `days` is any non-empty subset of `mon … sun` (every day is all seven), or
  `date: "YYYY-MM-DD"` for one run (then `days` is ignored and the task disables itself after it runs).
- `grants.connectors` maps a connector (an `mcp.json` server name) to `read` or `allow`; a connector not listed is `read`.
  `grants.writable` lists folders outside the projects folder a command may write (default none). Nothing is ever asked
  in a scheduled tab: what the grants do not cover is refused and the answer says so.
- `lastRun` is the epoch ms of the last run started; `lastResult` is `{ "at", "ok": bool, "summary": first line ≤ 140 chars }`.
- Time is the computer's local time (`localtime_r` / `mktime`): "07:30" stays 07:30 across a time-zone change and a
  daylight-saving switch (a non-existent 02:30 runs at the time `mktime` normalises it to).

## When a task runs

One pure function decides, given a task and `now` (epoch ms) and a local clock:

- `occurrence(task, now)` = the latest local `at` on an allowed day (or the `date`), at or before `now`.
- The task is **due** when it is enabled, that occurrence exists, it is on **today's** local date, and it is later than
  both `lastRun` and `created`. It is **late** when `now − occurrence > 2 min`.

The dock evaluates every enabled task once a second (on its tick) and runs each due one. This covers every case with no
extra state: on time; after the computer slept (the tick after waking sees today's occurrence still unrun); after Aria
was off (the first tick after start). Several missed days run once (only today's counts). `lastRun` is written before
the prompt is sent, so a crash never runs a task twice.

## Creating and changing tasks by talking

A Pi extension `extensions/schedule/` registers four tools; Aria uses them when the person asks for something recurring:

- `schedule_create { title, prompt, at, days?, date?, connectors?, writable? }` — before calling it, the model states the
  schedule and the permissions in one line and waits for the person's yes (written in the tool description).
- `schedule_list {}` — every task with its next run.
- `schedule_update { id, …any field…, enabled? }`, `schedule_delete { id }`.

They reach the dock through the session's jobs socket (`ARIA_JOBS_SOCK`), as new ops `schedule_create|list|update|delete`
answered from the shared store. Inside a subagent (`ctx.mode === 'json'`) and without a dock they refuse, like the
background tools. The dock validates every field (time format, days, connector names exist in `mcp.json`, writable
paths absolute and not secret, prompt non-empty, at most 50 tasks).

## Its tab

- `sessions.json` entries gain `schedule: Option<String>` (the task id). Scheduled entries do not count against the
  limit of 8 tabs (`sanitized` keeps up to 8 regular ones plus every scheduled one whose task exists).
- The first run creates the entry (title = task title) and tells the page (`DockMessage::ScheduleTab { id, title,
  schedule }`), which adds it to its tabs without making it active. A page save never drops a scheduled entry the page
  did not send while its task exists (the dock merges them back, as it already does for `segments`).
- The page shows a scheduled tab in the strip only while it is active or has an unread result, with a clock mark; closing
  it hides it (the task stays). `/schedules` opens a panel with every task: title, when, next run, last result,
  pause/resume and delete (`schedulesList`, `scheduleUpdate`, `scheduleDelete` page messages → `DockMessage::Schedules`).
  Deleting a task removes its tab and its sessions.json entry (`sessionClose`).
- A run: the dock emits `chatEvent { type: "scheduled_run", text, at, late }` to the tab (the page shows it as the
  person's message with a clock and "scheduled 07:30", or "ran at 08:05 — it was due at 07:30" when late), then sends
  the prompt as a `ChatSend` through the chat's ordered worker. The page drops Pi's echo of it, as it already does.
- The tab's Pi is spawned with `ARIA_GRANTS` (JSON: `connectors` with every `mcp.json` server set — granted ones as given,
  the rest `read` — and `sandbox: { askToLeave: false, extraWritable: writable }`). `extensions/lib/config.ts`
  `ariaConfig()` overlays it on `aria.json`, so the connectors and sandbox extensions need no change. The same grants hold
  when the person replies in that tab.
- When the run settles the dock records `lastResult` (ok unless the run ended in an error event or was aborted; summary =
  first line of the last answer) and, unless that tab is on screen, posts a notification titled "Aria · <title>" with
  the summary. On macOS a click does not open the dock (an unbundled app has no notification action, `docs/pending.md`);
  the tab is marked unread for when the dock opens. The Pi is released as any idle tab's is.
- Two tasks due together run at once, each in its tab. A task due while its tab is still running its previous run (or the
  person is mid-run there) waits for that run to settle, then runs (late if it waited over 2 min).

## Errors

No network, no quota, a connector error: the run ends as any failed turn does; `lastResult.ok = false`, the notification
says it failed, and the next occurrence runs normally (no retry loop). A turn that hangs is stopped by the existing turn
watchdog. A broken `schedules.json` is moved aside to `schedules.json.broken` and logged (as `sessions.json` is); no task
runs until it is fixed or recreated.

## Testing

- `schedules.rs`: `occurrence`/`due`/`late` with a fixed-offset fake clock — every day, weekdays only, a one-off date,
  created after today's time, ran already today, slept across the time, three days off, disabled, late threshold;
  `next_run` for the panel; validation; store round trip and the broken-file rule.
- `sessions.rs`: scheduled entries beyond 8 kept; an orphan scheduled entry dropped; the page-save merge.
- Jobs socket: the four ops, refusals, validation errors.
- `extensions/schedule`: the tools against the fake dock; refusals in a subagent / without a dock. `config.ts`: the
  `ARIA_GRANTS` overlay.
- The page: the `scheduled_run` event as a bubble, a scheduled tab outside the 8 and hidden until it has news, the panel.
- Live: a throwaway `ARIA_HOME` with a task due one minute ahead and a fake provider: the tab appears, the prompt runs,
  the notification is posted, `lastRun`/`lastResult` are written; then a task whose time passed while Aria was off runs
  once at start, marked late.

## Out of scope

Waking a sleeping computer (`pmset`); running when Aria is not running; intervals ("every 2 hours"); results sent outside
Aria (email, Telegram); a cron syntax; editing tasks in a form (the panel pauses and deletes; changes are asked in chat).
