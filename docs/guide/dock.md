# The dock

The dock is Aria's window: a column on the right of your screen that opens over whatever you are doing, holds your sessions as tabs, and closes again without stopping any work. This guide is for someone using it every day. It describes what you see and what you can press; the subjects that have their own guides are linked, not repeated.

## Opening and closing

Press the shortcut to open the dock and press it again to close it.

| | Omarchy | macOS |
|---|---|---|
| Default shortcut | `Super + A` | `Option + Space` |
| Fallbacks, tried in this order when one is taken | `Super + Shift + A`, `Super + Alt + A` | `Control + Option + Space`, `Command + Option + Space` |
| Which screen | The monitor that has the focus | The screen under the pointer |
| Who registers the shortcut | The dock, with Hyprland, every time it starts and after every config reload; nothing is written to `~/.config/hypr/` | The dock, with Carbon; no Accessibility permission |

The dock opens at the full usable height of that screen, on the right. Only the column takes clicks and keys. The dark veil to its left only dims the screen; on Omarchy a click on the veil goes through to the window below, while on macOS it closes the dock.

`aria toggle` does the same as the shortcut, and `aria show` only opens. Both start the dock first if it is not running, which makes them the way in when no shortcut could be registered. On macOS `~/.local/bin` is not on the default `PATH`, so type `~/.local/bin/aria toggle`.

### Changing the shortcut

Open the settings (gear), then **Assistant**, then the **Shortcut** row. Click it and press the keys you want: modifiers and one key. `Esc` cancels, and `Backspace` goes back to the platform's default. Below the row, "Or type it" takes the same thing as text, for a combination the desktop takes before the page sees it, such as `SUPER + SHIFT + A` or `Command + Shift + A`; press `Enter` or Use after typing it. Then press Save. A change applies within a second.

A shortcut needs at least one modifier, and its key is a letter, a digit, `Space` or `F1` to `F12`. The format and the way fallbacks work are in [the settings reference](../settings.md).

When your shortcut is taken (by another bind in Hyprland, by another app on macOS, or by one of macOS's own such as `Command + Space`), the dock tries the platform's default, then the fallbacks, and uses the first one that is free. The Shortcut row says "In use: X (Y was taken)".

What happens next differs. On macOS, if none is free, the row says "None available" and a banner explains; the dock tries again every 10 seconds, and while a fallback stands in it keeps trying yours. On Omarchy, when every candidate is taken the dock binds the first one anyway, and it does not retry on a timer: it registers again after a Hyprland config reload and after a change of the setting.

### What happens to the keyboard

Opening gives the keyboard to the dock, so you can type at once. On Omarchy, a click on another window takes the keyboard back and leaves the dock open; the shortcut still closes it. On macOS, opening remembers the app that was in front and closing gives it back.

`Esc` backs out one layer at a time and closes the dock when there is nothing left to back out of. In order: cancel a dictation in progress, close an open picker (`@`, `/`, models), close the connect card, close an open details card of the trace, leave a panel (subagents, background, memory, scheduled tasks) for the chat, close the tabs menu. `Esc` never stops a run: Aria keeps working with the dock closed, and the stop button in the composer stops it.

Closing the dock hides it. Sessions, runs, background work and typed-but-unsent text stay as they are. Text you typed and did not send is kept in the tab while the dock is hidden, but only in the page, so it is lost if the dock itself restarts.

On the setup and settings screens, `Esc` also closes the dock, and the arrow keys move between the controls. Opening the dock again shows the screen where you left it.

## The column

From the top: a header with your tabs, a new-session button and a close button; a row with the folder the session works in (the full path is a tooltip) and the gear for the settings; the thread; panels that replace the thread when you open them; the strips for background work and tasks; and the composer.

With one session, the header shows Aria's name and, when the provider has a quota Aria can read, a small bar for the 5-hour window. Hover it, or focus it with the keyboard, for a card with every window and when each resets. With more than one session the header shows tabs instead.

A banner at the top of the column appears when something stops Aria from working: on macOS, no shortcut could be registered or the system refused the key handler; on Omarchy, Hyprland uses the classic configuration or cannot be reached; on both, a settings file that Aria will not overwrite, or an engine (Pi) that is missing or did not start, with what to do about it.

## Session tabs

A session is a conversation with its own engine. Each is a tab, and you can have up to eight. They are the same sessions the terminal shows: a session opened from a terminal appears here, and `/exit` in a terminal closes its tab here. See [Aria in the terminal](terminal.md).

- **New**: the `+` button opens a session in your projects folder. It is disabled at eight ("Close a session to open another").
- **Title**: the first line of your first message, cut at 28 characters. Double-click a title to rename it; `Enter` or clicking away saves it, `Esc` cancels.
- **Close**: the `×` on a tab, shown only when there is more than one. A bar says "Closed … Undo" for eight seconds, and the tab's engine keeps running until that window ends. That Undo reopens the tab; it does not take back Aria's edits (see [Safety](safety.md#checkpoints-and-undo)). The last tab cannot be closed.
- **Overflow**: the header shows up to three tabs, and the active one is always among them. The rest are under a `+N` button, which turns to "a hidden session needs you" when one of them does.
- **States**: a dot shows what each session is doing: working, needs you (a card is waiting for an answer), done (it finished while you were away, until you open it) or idle.
- **Hover card**: with more than one tab, pointing at a tab shows its title and state, the model, how full its context is, and the provider's quota windows with their reset times. A free model says it uses no quota; a provider with no readable quota says that.
- **Scheduled tasks**: a task that runs by itself gets its own tab, with a clock before the title. It shows only while it is active, unread, working or waiting, and it does not count against the eight. Closing it only hides it. See [automation](automation.md).

A session that is not on screen keeps working. If it asks you something, a small card in the dock says "needs your answer" with an Open button, for eight seconds.

What survives closing the dock: everything above. What survives a restart of the dock (an update, `aria quit`, a reboot): the list of tabs, their titles and which one was active, and every conversation, which is loaded again at start so it is there before you open the dock.

## The composer

Type in the box and press `Enter` to send. `Shift+Enter` adds a line, and the box grows to six lines. Under the box, the model in use is a button that opens the model picker, then the hints for the keys that matter now.

### While Aria is working

You can keep typing. `Enter` puts your message in at the run's next step, after the tool calls going on; `Alt+Enter` keeps it for when the run ends. The message shows in the thread, dashed, with a line saying which of the two it is and a Cancel button. Cancelling one puts the others back in order. A refused message comes back into the box with its text and files.

The stop button (a square) stops the run. The thread then says "stopped by you".

A command that is Aria's own or an extension's (`/debug`, `/effort`…) runs at once even during a run. A skill and a prompt template wait in the queue like a message.

### `@` picks a file

Type `@` after a space (or at the start) and a list of files in the session's folder opens, filtered as you type. The walk lists files only (no folders), follows `.gitignore`, skips `.git`, `node_modules`, `target` and `.venv`, shows the shallowest first, and offers up to 50. In a very large folder it gives up after 50,000 entries, so the list can be incomplete. `Tab` or `Enter` inserts `@path` into your message, which becomes a chip in the thread, and `Esc` closes the list. The file is named, not attached: Aria reads it from the folder.

### `/` opens the commands

Type `/` at the start of the box for a filtered list of commands, then the skills Aria can use.

| Command | What it does |
|---|---|
| `/model` | Opens the model picker, with a search field. It lists the models of the providers you have connected, and pay-per-use models only when you allowed them. |
| `/connect [provider]` | A card in the thread to connect Anthropic (a Claude plan or an API key) or OpenCode Go. A Claude plan needs Claude Code installed, and downloads the Claude provider once. |
| `/compact [what to keep]` | Compacts the context now. |
| `/session` | Messages, tool calls, tokens, cost and context of this session. |
| `/copy` | Copies the last answer. |
| `/export` | Exports the session as an HTML page and says where. |
| `/reload` | Restarts this tab's engine, after its run, to read extensions, skills and templates again. |
| `/memory` | Opens the memory panel. See [the panels](#panels). |
| `/schedules` | Opens the scheduled tasks. See [the panels](#panels). |
| `/debug [on\|off]` | Shows or hides the live trace of this tab's turns. See [Aria's face and the trace](voice-and-face.md). |

The list also shows the commands Pi and Aria's extensions offer, such as `/effort`, `/autocompact`, `/sandbox`, `/verify`, and `/plan` and `/execute` when slice is on. [The agent guide](agent.md) explains them. `/agents` is listed by Pi but does nothing in the dock; `Alt+A` is the dock's way. Arrows move through the list, `Tab` or `Enter` picks, and with no match `Enter` sends what you typed as a message.

### Attachments

The paperclip, a file dropped anywhere on the chat, and a file pasted into the box (an image from the clipboard, for instance) all add files to the next message. A message takes up to 10 files, each up to 20 MB, and 50 MB together; what does not fit is left out with a note. The dock saves them privately under `~/.aria/attachments/` and names them in the message. A png, jpeg, gif or webp also goes to the model as an image when it can see images. Each file shows as a chip you can remove before sending. [The integrations guide](integrations.md) says more.

### Dictation

The microphone button, or `Alt+D` (`Option+D` on a Mac), records you and writes what you said into the box when you stop. The first time, Aria asks before downloading the voice model. The voice stays on your computer. `Esc` cancels, and `Enter` while recording stops and sends. See [dictation](voice-and-face.md).

### A key in your message

If you paste something that looks like a password or key, the dock holds the message back and asks: keep it safe in the keyring (the message goes with its name, and the model never sees the value), send it as it is, or edit. Keeping it needs the keyring to be on. See [safety](safety.md).

## The conversation

Your messages are on the right, in bubbles. Aria's answers are rendered as Markdown, with code, links that open in your browser and `diff` blocks drawn as a proposed change. Along the way you see:

- **A route line** (`jev` and a word such as "focused task") above an answer, when the router is on.
- **A working line**, with what Aria is doing (thinking, reading files, running a command, planning…) and a timer.
- **Tool calls** as a group with a count. It is open while Aria works on it, and folds when it finishes; click it to open it, and click a row for its result. Thinking steps appear as rows. Files that were changed show as chips with a diff preview, the first three and a `+N more`.
- **Notes** in Aria's voice for things such as a compaction ("context compacted"), a stopped turn or the result of a card you answered.
- **Earlier messages.** After a compaction the conversation goes on in a fresh engine session. A line "Earlier in this conversation" with the summary folded under it marks the join, and "Show earlier" pages back through the older messages, 40 at a time.
- **A plan**, when you use `/plan`: its document and its phases as cards, with buttons to approve or ask for changes. See [the agent guide](agent.md).

When the thread has two or more of your messages, moving the pointer to the thread's scrollbar opens a rail of chips, one per message you sent; click one to jump to it.

## Questions and approvals

Cards wait for you, and a session with one shows "needs you" on its tab.

- **Questions.** When Aria needs a decision it shows a card in place of the box: one to four questions, one step each, with up to four options, the recommended one first, an "Other…" field, and multiple choice where the question allows it. You can go back, skip, choose "Talk about this" to discuss a question, or dismiss them all. A single-choice pick moves on by itself after a moment, and sends on the last question.
- **Permission cards**, in the thread: a command that wants to leave the sandbox, a file write outside the allowed folders, or a connector (Notion, for example) that wants to create, change or delete something. A command card shows the command and Jev's review when the router reviews commands, and buttons such as "Yes, just this once", choices that last for the rest of the session, and "No". What each choice grants is in [safety](safety.md) and [integrations](integrations.md).
- **The quota card**, on OpenCode Go when the quota is almost used up: switch to the free model for the rest of the task, or keep the subscription. If you do not answer, the setting for "no answer" applies. See [the agent guide](agent.md).
- **A secret card**, when a task needs a password, key or token: paste it, and it goes to the keyring with only its name reaching the model.
- **Plain dialogs** from extensions (a choice, a yes or no, a line of text) appear above the box.

Answering is by clicking; `Enter` in a card's text field moves on or saves.

## Tasks, subagents and background work

While a slice builds (see [the agent guide](agent.md)), the plan's tasks show above the box as a list: "Tasks: 2 of 5 done", with the one being worked on. Click its header to fold it. The dock shows only the plan's marks; the task list the model keeps on its own for a multi-step job shows in the [terminal](terminal.md), not here.

Background work, such as a dev server or a subagent given a long job, shows as pills above the box, one per job of this session: a dot, a title, the minutes, and a stop button (a failed job has a dismiss button instead). Up to four show, and the rest fold into `+N`. A pill opens the list on that job.

`Alt+A` opens the subagents and background list, and again (or `Esc`) goes back to the chat. It shows what is running and what has finished, each with its status. From a row you can open a process's URL in the browser, expand its last lines of output (refreshed every second), stop it, or open a subagent's session, which shows what it read, said and edited, and its final answer once it ended. You cannot message a subagent from there. The hint "alt+a subagents" or "alt+a background" appears under the box when there is something to show. How the work runs is in [automation](automation.md) and [the agent guide](agent.md).

## Panels

These replace the thread until you press `Esc` or Back.

- **Scheduled tasks** (`/schedules`): each task with when it runs, its next run and how the last one went, with Pause or Resume and Delete. A task that asked for permission to write waits here for your Approve, and the panel opens by itself when one does. You create tasks by asking Aria in the chat. See [automation](automation.md).
- **Memory** (`/memory`): the panel is built to list, search and filter Aria's memories, with Pin and Forget. The dock does not feed it yet, so today it is empty; Aria's memory itself works. See [the pending list](../pending.md) and [integrations](integrations.md).

## Settings

The gear next to the folder path opens the settings in place of the chat. It shows a card per section with a one-line summary of what it is set to:

| Section | What is in it |
|---|---|
| Assistant | Your name, the conversation language, the answer style, the shortcut, and "Show everything" |
| Providers | Connect, replace or make main each provider, and remove one that is not the main; the quota rules for OpenCode Go |
| Models and effort | The everyday and planning models, pay-per-use, the turn time limit, the effort level |
| Router (Jev) | On or off |
| Context and memory | The compaction threshold, memory on or off and its project name |
| Integrations | Notion and Tavily, the access level of each connector, your saved secrets (delete one), importing MCP servers from your other apps, a local folder of Pi packages |
| Sandbox and git | The projects folder, extra writable folders, ask before leaving the sandbox, what Aria may do in git |
| Subagents | The experts, `/plan` and `/execute`, and the skills that run on their own |

A part that a provider does not have, such as the OpenCode Go quota rules, is left out of its section. A card says "Needs attention" when something must be finished before you can save, such as a provider that is not connected or a service you turned on but did not connect.

Open a card for its page, "All settings" goes back to the cards, and "Back to chat" leaves the settings. Nothing is written until you save: as soon as something changes, a bar at the bottom offers **Save** and **Discard**, and says "Unsaved changes" or what to finish first. After a save it says "Saved" for a moment. Back with unsaved changes asks "Save your changes before leaving?". Picking a language changes the interface language at once (English or Spanish); discarding takes it back.

The overview shows `~/.aria/settings.json` and the version of Aria. Every setting, with its default, is in [the settings reference](../settings.md). A running session reads a change when its run settles, and some changes restart its engine. If something outside the screen changes the file (a `/effort` in a session, for instance), the screen shows it unless you have unsaved changes of your own.

"Show everything" (thinking and every tool output, or one line per step) is read by the terminal for now, not by the dock, which always shows the full thread.

## Notifications

A desktop notification comes from the dock when a session needs you and you cannot see it: the dock is hidden, or the settings are open. A card waiting for an answer sends "Aria needs an answer" with a short summary of what it wants, never the raw command or a connector's arguments. A run that ends where nobody is looking sends one notification per session, with the first line of the answer or why it did not finish. A terminal showing the session counts as looking at it.

On Omarchy it is a critical notification that stays until you dismiss it or look at the session (which takes it away); clicking it opens the dock on the active tab, not always on the one that is waiting. On macOS it comes from `osascript`, clicking it does not open the dock, and Aria cannot take it back when you answer. See [automation](automation.md).

## Keys

| Key | What it does |
|---|---|
| `Super + A` / `Option + Space` (or your own) | Opens and closes the dock |
| `Esc` | Backs out one layer; closes the dock when there is none |
| `Enter` | Sends; while Aria works, puts the message in at the next step |
| `Alt+Enter` | While Aria works, keeps the message for when the run ends |
| `Shift+Enter` | A new line |
| `@` | Opens the file picker |
| `/` | Opens the command picker (at the start of the box) |
| `↑` `↓`, `Tab`, `Enter` | Move through a picker; `Tab` or `Enter` picks |
| `Alt+A` (`Option+A`) | The subagents and background list, and back |
| `Alt+D` (`Option+D`) | Starts or stops dictation |
| `Enter` / `Esc` in a tab title | Saves / cancels a rename |
| `↑` `↓` | On the setup and settings screens, move between controls |
| `⌘Q` | macOS only: the first press says "Press ⌘Q again to quit Aria", a second within two seconds quits and stops the work |

Every key and command, by surface, is in [the command reference](../reference/commands.md).

## Limits

- The dock has no undo or rewind of an instruction, and no `/context` or checks footer; the terminal has them. See [Aria in the terminal](terminal.md).
- On macOS the shortcut's fallbacks and retries are tested on Linux and type-checked there, and had not been run on a Mac when this was written.
- The memory panel is empty until it is fed. A subagent's session has no message box. On macOS the dock takes the whole panel's clicks, and a notification's click does not open it. See [what is pending](../pending.md).

## See also

- [Getting started](getting-started.md)
- [The agent](agent.md)
- [Automation](automation.md)
- [Safety](safety.md)
- [Dictation, Aria's face and the trace](voice-and-face.md)
