# Automation

What Aria does without you watching: scheduled tasks, background work, notifications, the watchdog that stops a stuck run, and updates that wait for you to be done. It is for anyone who leaves Aria running while they do something else.

All of it belongs to the service, the process that owns your sessions, not to a window or a terminal. Closing the dock or leaving a terminal stops none of it. Where a behaviour differs on Omarchy, macOS or a hub, the text says so.

## Scheduled tasks

### What it does

You ask for something recurring in plain words, in any session, and from then on Aria does it by itself:

```text
every day at 7:30 give me my Notion tasks
on weekdays at 18:00 summarise what I changed today in acme/checkout-widget
on 2026-11-02 at 9:00 remind me to renew the domain
```

Aria is told to say in one line when the task will run and what it may do, and to wait for your yes before it saves anything. You can ask it to list, change, pause or delete tasks the same way.

### What a task is

A task has:

- a **prompt**, the whole instruction as it will be sent at that time. The run does not see the conversation you made it in, so the prompt has to stand on its own.
- a **time of day** (local, 24 hours) and **days of the week**, or one **date**. Every day means all seven days.
- **grants**: what the run may do without asking (below).
- a title (a few words), shown on the task's tab and in notifications.

At most 50 tasks exist at once. The time is the clock of the computer the service runs on, in its time zone and with its daylight-saving rules: "07:30" stays 07:30 when the zone changes.

### Each task runs in its own tab

The first run creates a tab for the task, marked with a clock, and sends the prompt as if you had typed it. The message shows "Scheduled · 07:30", or "Scheduled for 07:30 · ran late" when it started more than two minutes after its time. A scheduled tab does not count against the dock's limit of eight tabs, and the dock does not take focus for it.

The tab is shown only while it is active, working, waiting for you, or has a result you have not read. Closing it hides it; the task stays. Deleting the task removes the tab. If you reply in the tab, the reply runs with the same grants as the scheduled run.

When a run ends, Aria records the first line of its answer as the task's last result (up to 140 characters; "No answer" when there is none) and whether it succeeded. A run that ends in an error, or is stopped, is recorded as failed. There is no retry: the next occurrence runs as usual.

### What a run may do: grants

A scheduled run has no one to ask, so its permissions are fixed when the task is made, and anything else is refused, never asked:

- **Connectors** (MCP services such as Notion): each is `read` or `allow` (it may change things there without asking). A connector the task does not name is `read`, never `ask`. See [safety](safety.md) for what the access levels mean.
- **Writable folders**: none by default. A folder must exist and lie **inside the projects folder** (`sandbox.projectsDir`) **or `/tmp`**; anything else is refused when the task is made or changed, with an error that names the rule (and the model is told). A `..` in the path is refused, and so is any dot folder of your home (`~/.config`, `~/.local`, `~/.ssh`, `~/.aria`, any `~/.<name>`) even if the projects folder is your home folder. Aria also still refuses the whole disk, your home folder and any folder that contains it, Aria's own folder, and a fixed list of secret and login-item folders of your home. The check is on the path as written, and a link is judged by where it sits and then by where it points; folders are kept by their real path. A task stored before this rule keeps its folders on disk, but a run does not get the ones outside the rule (a line in the log says so).
- The run cannot ask you a question, cannot ask to leave the sandbox, cannot ask for a secret and cannot get a plan approved. The answer says what was refused. Inside the projects folder the run has the sandbox's usual write access.

A task that can write, meaning any connector set to `allow` or any writable folder, is saved **paused and waiting for your approval**. The dock opens `/schedules` on it, and you press **Approve**. Changing a task so that it writes somewhere new, or rewording the prompt of a task that writes, makes it wait again. The model cannot approve, and neither can a terminal: approving is its own message, sent only by the dock page's Approve button, and the service refuses it from every client (a plain resume of a waiting task leaves it waiting). A read-only task starts at once.

### When a task runs, and what a missed run does

The service checks every task once a second. A task is due when it is enabled, its latest scheduled time is **today's**, and that time is later than both its last run and its creation. So:

- A task runs on time, whether the dock is shown or not.
- If the computer was asleep or Aria was not running at the task's time, it runs **once** as soon as Aria is back, marked late. Only today's time counts: three missed days run once, and a time from yesterday does not run.
- A task created after today's time waits for tomorrow's.
- Resuming a paused task after its time today runs it at once, late.
- A one-off task pauses itself after it runs.
- If the task's tab is still busy with its previous run, the new run waits for it to settle.
- Two tasks due together run at the same time, each in its tab.

The run is recorded as started before the prompt is sent, so a crash never runs a task twice. Aria never wakes a sleeping computer.

### Managing tasks: `/schedules`

`/schedules` lists every task: its title, when it runs, its next run and how its last run went. You can pause, resume, delete, and approve a task that waits. You cannot edit one there; ask Aria to change it.

In the dock, `/schedules` opens a panel with **Pause**, **Resume**, **Approve** and **Delete** buttons.

In a terminal, `/schedules` opens a list in the editor's place:

| Key | What it does |
|---|---|
| `Up`, `Down` | Move through the tasks |
| `Enter` | Pause a running task, or resume a paused one (nothing on a task that waits for approval) |
| `d`, then `y` | Delete the task; any other key cancels |
| `Esc` | Close the list |

A task that waits for approval shows "waiting for approval in the dock". The terminal's panel does not approve tasks, and the service does not take an approval from a terminal either.

### On a hub

Schedules belong to the service that holds them. On a [hub](hub.md#scheduled-tasks-on-the-hub) they run on the server, with the server's clock and time zone, while your laptop is off. You can ask for a task from a terminal and manage it with `/schedules` there. Only read-only tasks run: a task that may write waits for an Approve that only the dock offers, and a hub has no dock. A scheduled run cannot ask to leave the sandbox, so where the hub's sandbox does not work it cannot run shell commands. A hub sends no notification and its terminal does not list scheduled runs; to read a result, ask Aria in a session, which can list each task with the first line of its last result. Scheduling and background work both need the user's runtime directory on the server (`$XDG_RUNTIME_DIR`, `/run/user/<uid>`); without it, sessions have no jobs socket and the tools refuse.

### Where tasks are stored

`~/.aria/schedules.json` (mode `600`). A file that cannot be read is moved to `~/.aria/schedules.json.broken`, and no task runs until you fix it or ask for them again. The scheduled tab is an entry of `~/.aria/sessions.json`.

## Background work

### What it does

A command that never ends (a dev server, a watcher, a tunnel) or a long, well-defined job (fix the failing tests, review a module) runs in the background while you keep talking with Aria. It belongs to the service, not to Pi, so a Pi restart after a settings change does not stop it.

Aria decides what goes to the background; it does not ask. A command that does not end by itself always goes there. A job of more than a few steps that can be described up front goes to a background subagent. Anything short or conversational stays in the foreground. If you say "in the background" or "do it here", that wins.

### In the dock

- **The pills above the composer.** One pill per job of the active session: a dot (green for a running process, the accent colour for a subagent, a dashed outline while queued, red when failed), the job's title, a short note (for a process its port and how long it has run, for example `:3000 · 12 min`; for a subagent what it is doing) and a stop button. At most four pills show; the rest fold into a `+N` pill that opens the list. A job that ends normally leaves the strip. A failed one stays, red, until you dismiss it.
- **The Background list**, `Alt+A` (again to go back). It shows running and queued jobs, grouped by session when several have work, then the last 20 finished ones with how each ended. A process has **Open** (when it printed a local address), **Output** (its last lines, refreshed while open) and **Stop**. A subagent has **Session** (its steps) or **Result** (its final answer once done) and **Stop**.
- **Notes in the conversation.** When a job ends, fails or is cut, a one-line note appears in the thread. If it ended by itself, Aria also starts a turn to tell you what happened and, for a subagent, to summarise its result.

On macOS, `Option+A` types a character, and the dock tests the typed letter, so the shortcut may not work there; click a pill or the `+N` pill instead.

### In a terminal

The terminal has no pills, no list and no stop button for background work. Aria still knows what is running and tells you when a job ends. Ask it in the conversation ("what is running in the background?", "stop the dev server"); it can list jobs, read a process's output and stop a job.

### Subagents in the background

A background subagent is a separate Pi run with its own brief; it does not see the conversation. At most three run at once across all your sessions, and more wait in the queue. It can write files or only read (research, review). It cannot start background work of its own.

Two writers must not edit the same files:

- By default a writing subagent works in your folder. If another writing subagent is already working in the same git repository, the new one gets a **worktree** instead. Outside git, it waits until the first one ends.
- Aria can ask for a worktree directly for a large job.
- A worktree is a separate checkout at `<projects folder>/.aria-worktrees/<repository folder>-<job id>`, made from your committed and uncommitted tracked changes (untracked files are not in it). Your own repository, and your test runners and watchers, never see it.
- When the subagent ends, its changes come back to your folder as **uncommitted changes**: applied to the working tree only, nothing staged, nothing committed. The worktree is then removed.
- If the changes do not apply because they clash with yours, or the subagent was stopped, failed or was cut, the worktree stays and Aria tells you its path. Nothing is removed without you.

### How long background work lives

| What happens | What happens to the jobs |
|---|---|
| You hide the dock, or leave a terminal | Nothing; they keep running |
| A Pi restarts (a settings or key change) | Nothing; Aria hears of the jobs it missed |
| You close a tab, or `/exit` in a terminal | That session's jobs stop, and its queued subagents never start |
| You quit or stop the service | Every job stops (see [Stopping the service](#stopping-the-service)) |
| An update is installed | It waits for the jobs (see [Updates that wait](#updates-that-wait)) |
| The service crashes or is killed | On Linux the children get a stop signal; on macOS they can be left running. At the next start jobs still marked running become **cut**, and their sessions are told |

A stop is a polite signal to the job and everything it started, then a forced kill after three seconds.

### Limits

Ten processes at once, three running subagents, 1,000 kept lines of output per job (the last 20 finished jobs are kept). They are constants, not settings. A subagent cannot be steered once running: stop it and start another. See [pending](../pending.md) for the rest.

How it is built is in [the architecture](../architecture.md) and [the design](../superpowers/specs/2026-10-21-background-work-design.md). The jobs are recorded in `~/.aria/jobs.json` (mode `600`).

## Notifications

A desktop notification appears only when you cannot see what happened:

- **A question for you** (a permission, a confirmation, a question from the model) while nobody can see the session. The title is "Aria needs an answer". It goes away when the question is answered.
- **A run that finished** where nobody looks. The title names the session; the body is the first line of the answer, "Done" when there is none, or "It did not finish: ..." when it failed. A session has at most one such notification: its next one replaces it, and looking at the session removes it. A run you stopped raises none.
- **A scheduled run that finished**, titled "Aria · <task title>", with the first line of the result or "It did not finish: ...".

Nobody can see a session when the dock is hidden (or showing settings instead of the chat) and no terminal attached to that session has the focus. A run you watch end raises nothing.

| Platform | How it shows |
|---|---|
| Omarchy | `omarchy-notification-send`, critical, no expiry: it stays until you dismiss it. Clicking it opens the dock, on the active tab (not on the one that is waiting). Aria dismisses a question's notification when it is answered, and a session's "finished" one when you look. |
| macOS | `osascript` (`display notification`). A click does not open the dock: an unbundled app has no notification action, and Aria cannot take a notification back, so one stays until you clear it. The tab is marked unread for when you open the dock. |
| Hub | None. A service with no desktop has nobody to show one to. The session list in a terminal shows what each session is doing instead. |

## The turn watchdog

A run that gets nothing from its Pi (no event, reply or question) for `models.turnTimeoutMinutes` is stopped, and the conversation says "Aria stopped this turn: nothing came back for N minutes." The default is 10; `0` turns it off. Change it in Settings → Models and effort, or see [settings](../settings.md).

The service looks every 30 seconds, so a stop comes up to half a minute after the limit. It stops the run once and keeps whatever you queued behind it, which then runs. A running tool, a compaction and a question waiting for you pause the watchdog, so a long build or test is not cut. It covers a model that has gone silent, not a tool. A scheduled run the watchdog stops is recorded as failed.

## Updates that wait

An update does not cut a run. After the installer replaces the files it runs `aria update-ready`, and the service then restarts into the new version once **all** of these hold:

- the dock is hidden (always true on a hub),
- no session has a run going or a question open (or is rotating its conversation file),
- no background job is running or queued.

It checks every second. The restart keeps the same process, so systemd or launchd notice nothing, and your tabs and conversations come back as after any restart. Just before it restarts, the service stops its Pis and any job left.

If the update has waited 30 minutes for background work **alone**, each open session with such work gets one notice that the update is waiting and that you can stop the work from the background list to let it install (in a terminal, ask Aria to stop it). The update never cuts the work by itself. The notice is in English only. The dock shows nothing else while an update waits; the service's log has the line ("a new version is installed; restarting once the dock is hidden and idle"): `journalctl --user -u aria.service` on Omarchy (`aria-serve.service` on a hub), `~/.aria/logs/aria.log` on macOS.

To install at once, even mid-run, run the installer with `--restart-now`:

```sh
sh install.sh --restart-now
```

That cuts the run and stops every job. The installer also restarts at once when the service file changed, or when the running service is too old to understand `update-ready`. See [getting started](getting-started.md) for installing and updating, and [hub](hub.md) for updating a hub.

## Session status

The service keeps one status per session and tells every terminal when it changes:

| Status | Meaning |
|---|---|
| `working` | a run is going |
| `waiting` | a question is open, waiting for you |
| `done` | a run ended and nobody has looked since |
| `idle` | none of the above |

A session counts as looked at while the dock shows it or a focused terminal is attached to it; looking turns `done` into `idle` and removes its "it finished" notification. A terminal's session list shows `working`, `waiting for you` or `done` with how long ago it changed.

[Aria's face](voice-and-face.md) is a separate summary built from the same activity: it shows the most pressing state across every session (a question first, then a failure, then work, then a scheduled run), not these four words.

## Stopping the service

| How | Where |
|---|---|
| `aria quit` | Both platforms; also on a hub |
| `systemctl --user stop aria.service` | Omarchy |
| `systemctl --user stop aria-serve.service` | Hub |
| `launchctl bootout gui/<uid>/com.aria.dock` | macOS (also unloads the LaunchAgent) |
| `Cmd+Q` twice | macOS, in the dock |
| `Ctrl+C` | The service run in a terminal (`aria dock`) |

All of them do the same thing, in this order: every background job stops (queued subagents never start), then every session's Pi stops, then the service exits cleanly. A run in progress is cut; the conversation up to that point stays in its session. The service manager does not restart a clean exit.

To start it again, run `aria start` (`systemctl --user start aria.service` on Omarchy, `launchctl kickstart` on macOS), or `aria show` or `aria toggle`, which start the dock when nothing answers. After a `launchctl bootout`, `aria start` finds no loaded agent: load it with `launchctl bootstrap gui/<uid> ~/Library/LaunchAgents/com.aria.dock.plist`. On a hub, `aria start` looks for the dock's unit; use `systemctl --user start aria-serve.service`.

## Limits

- A scheduled task runs only while the service runs. A laptop that sleeps misses its times; a hub does not, but a hub runs read-only tasks only. See [pending](../pending.md) for the rest of what is not built: intervals ("every 2 hours"), results sent outside Aria, editing a task in a form.
- A scheduled run with a real model has not been tried on Omarchy yet; see [pending](../pending.md).
- Notifications on macOS cannot open the dock.

## See also

- [The agent](agent.md): models, experts, subagents and slice, which use the background.
- [The dock](dock.md): tabs, the composer and the pills.
- [Terminal](terminal.md): `/schedules`, the session list and `/exit`.
- [A hub](hub.md): running Aria on a server so all of this goes on while your computer is off.
- [Settings](../settings.md): `models.turnTimeoutMinutes`.
