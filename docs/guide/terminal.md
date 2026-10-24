# Aria in a terminal

`aria`, typed in a terminal, opens the same sessions the dock shows: a list of them, and each one as a conversation you can read and write to. It is for people who live in a terminal, and for any computer with no desktop (see [Over ssh](#over-ssh)).

The terminal is a window onto the service's sessions, not a session of its own. You can close it, or leave with `Ctrl+C`, and the work goes on; run `aria` again and it is still there. The same session can be open in the dock and in a terminal at the same time, and both show it live.

## Start it

```sh
aria
```

`aria` needs a terminal on both its input and its output. Without one (a pipe, a script), plain `aria` runs the service itself, which is what an old service unit does.

If the service is not running, `aria` starts it in the background (the same `aria dock` the login service runs) and waits up to 3 seconds for it. If an older service is running that cannot take terminals, `aria` says so at once and tells you to restart it with `aria quit` and `aria`.

Leave with `Ctrl+C` or `Ctrl+D`. `/exit` is different: it closes the session, ends its background jobs and deletes its undo history.

## The session list

The first screen lists the service's open sessions, the one whose state changed last first. Scheduled runs are not listed. Each row shows:

- the session's title (`untitled` until it has one);
- the last part of its folder's path;
- what it is doing: `working` (a run is going), `waiting for you` (a question or a permission is open), `done` (a run ended and nobody has looked since). A session that is quiet shows no state word;
- how long ago that state began (a quiet session shows how long ago it went quiet): `just now`, `3 min ago`, `2 h ago`, `3 d ago`.

The last row is `+ new in <folder>`, which starts a new session in the folder you ran `aria` from. Sessions started from the dock use the projects folder of the [settings](../settings.md) instead. There can be at most 8 sessions: past that, the list says "There are already 8 sessions: close one first."

Move with `↑` and `↓` and press `Enter`, or click a row. `Esc` or `q` leaves.

The words the terminal says itself (the list, the footer, the panels) are English, or Spanish when `assistant.language` is `es`.

## A session

The screen is the conversation on top, and at the bottom, from the top down: the task strip (when the model keeps a task list), the working line and the messages waiting in the queue, a rule, the editor, a rule, and one footer line. The conversation scrolls inside the terminal's alternate screen, so your terminal's own scrollback does not hold it: use `PageUp`, `PageDown` or the mouse wheel.

### What the conversation shows

Your messages start with `›`. Aria's answers are drawn from markdown: headings, bold and italic, lists, code blocks (grey, not wrapped, not highlighted). Aria's own notices have a mark: `ⓘ` information, `▲` a warning, `✕` an error. `── context compacted ──` marks a compaction, and `· stopped by you` a turn you stopped.

How much of the work you see depends on verbose mode, which is on by default. `Ctrl+O` or `/verbose` switches it, and the choice is saved in the settings (`display.verbose`).

- **Verbose on.** The model's thinking, in grey. One line per tool call with a mark (`●` amber while it runs, `✓` green, `✗` coral), the tool, what it works on (the command's first line, the file, the pattern) and, for an edit or a write, `+14 −2`. Under it, in grey, the tool's output, cut to 12 lines. Click `… N more lines` to see the rest, and click again to fold it.
- **Verbose off.** No thinking and no output. One line per step with what it did and on what, as the dock's accordion says it: `✓ Read file  coupon.ts`, `✓ Run command  bun test coupon`, `✓ Write 12 lines  coupon.ts`.

While a turn runs, the working line shows a spinner, what Aria is doing (Thinking, Reading files, Running command, Searching memory, Searching the web, Checking Notion, Delegating, Compacting context) and how long it has run. A message you send meanwhile appears as `queued` above the editor until Aria takes it.

### The task strip

When the model keeps a task list (it does for work with more than one step), a strip shows above the editor:

```text
  tasks  ✓✓●○○  2/5
  ● Wire the coupon field        reviewer
  ○ Add the tests
```

`✓` is done, `●` is running, `○` is waiting. Done tasks only count; running and waiting ones get a row each (six at most, then `… N more`), and a running task shows its subagent's name. `Ctrl+T` or a click on the strip folds it to its first row, and again unfolds it. The strip also shows the phases of a slice (see [the agent](agent.md)).

### The footer

The footer's left side reads, for example, `opus 5.5  high    ctx 12 / 40    5h 34% 1h12   wk 61%`:

- the model, shortened, and the effort;
- `ctx 12 / 40`: how much of the context window is used, in percent, and the percent at which it compacts (`compaction.percent` in the [settings](../settings.md)). The first number turns amber 5 points before the second;
- the quota of the provider in use, when it reports one: each window with its percent (`5h`, `wk`, `mo`), and the time left to the first one's reset as `1h12`. Grey below 80 %, amber from 80 %, coral from 95 %.

The right side shows, when they apply:

- `jev off 1h12` in amber: the router's classifier cannot run (its quota or funds are spent) until that time or until it works again (see [the agent](agent.md));
- `tests ✓ 43` in green or `tests ✗` in coral: how the project's checks went after Aria's last edits (see [Project checks](#project-checks));
- the git branch, with `~3` when 3 files differ, read every 2 seconds from the session's folder (not shown outside a repository, nor over ssh);
- the session's folder name.

On a narrow terminal the whole right side is dropped before the left is cut.

## The editor

The editor sits between the rules, up to six lines tall (a longer message shows its last six). It starts with a hint: "write, @ for a file, ! runs a command, / for commands".

- **Writing.** `Enter` sends. `Alt+Enter` or `Shift+Enter` starts a new line, and so does pasting text with line breaks (a paste never sends). `←` and `→` move the cursor; `Backspace` erases. There is no `Home`, `End`, `Delete` or word-wise key.
- **History.** `↑` on an empty editor brings back the messages you sent, `↓` goes forward again. A draft is never replaced by the history. It is kept while this terminal runs, not after you leave.
- **`@` picks a file.** Type `@` and part of a path: a list of matching files opens above the editor (relative paths, shallowest first, `.gitignore` respected, up to 50; in the session's own folder, or the projects folder when the session has none; names only). `↑` and `↓` move, `Enter` or `Tab` or a click puts `@path` in the message, and `Esc` closes the list. The message then names the file for the model.
- **`!command` runs a command.** A message that starts with `!` runs the command in the session instead of going to the model. Its output shows as a step and joins the conversation, so the model knows about it. Not over ssh.
- **`/` opens the commands.** A message that starts with `/` opens a list: the terminal's own commands first, then the session's. `Enter` on one runs it at once; `Tab` or a click only writes it in the editor, so you can add its argument. Whatever the list does not recognise goes to the session as typed: `/verify off` does, because the list closes once there is a space. A word that matches no command is sent as an ordinary message when you press `Enter`, as in the dock.

While Aria works, `Enter` queues your message for its next gap (after the thinking or the tool that is running). `Ctrl+Enter` stops the turn and sends what you have typed as soon as it has stopped; with an empty editor it does nothing, and it does nothing while a card is open (use `Esc` to stop the turn). It needs a terminal with the extended keyboard protocol. `Esc` stops the turn and sends nothing.

## Questions and permissions

When Aria needs an answer, a card takes the editor's place. Every client attached to the session gets it, the first answer wins, and the others close theirs. A session with an open card shows `waiting for you` in the list.

| Card | How to answer |
|---|---|
| A question with options | `1` to `9` picks an option; a single-choice question moves on at once, a multiple-choice one toggles it and `Enter` confirms. Typing writes your own answer, and `Enter` sends it. A card can hold up to four questions, one after the other. `Esc` skips them. Or click an option |
| A permission (the sandbox, a write that needs approval, a connector) | The option's number (the first is "once") or a click. `Esc` cancels. The card's last line says so. `y`, `a` and `n` are not offered: they answer only a card whose options are `allow`, `always` and `deny`, and none of these sends those |
| The quota card (a plan's quota ran out) | The option's number or a click |
| A yes or no | `y` or `Enter` yes, `n` no, `Esc` cancels |
| A secret Aria asks for | Type the value, hidden as `•`; `Enter` saves it. The service keeps it; it is never sent to the model. Not over ssh. `Esc` cancels |

What each permission means, and what "always" covers, is in [safety](safety.md).

## Commands

These are the commands the terminal handles itself.

| Command | What it does |
|---|---|
| `/exit` | Closes this session for good: its engine stops, the background jobs it owns end, its checkpoints (and with them its undo) are deleted, and its tab closes in the dock too. The service keeps at least one session, so a fresh one takes the place of the last. It is not the way to leave: use `Ctrl+C` or `Ctrl+D`, which leave the session running |
| `/sessions` | Back to the list. The session goes on |
| `/model` | Opens a list of models to switch this session to. Only the providers you have signed in to are listed (OpenCode Zen only when `models.allowPayPerUse` is on), newest first, with `●` on the one in use. Over `aria connect` it lists only the model in use (see [Over ssh](#over-ssh)) |
| `/verbose [on\|off]` | Shows everything or one line per step. With no argument it flips |
| `/undo` | Opens Rewind, as `Esc Esc` does. Over `aria connect` it goes back in the conversation only |
| `/context` | Draws what fills the context |
| `/schedules` | Lists the scheduled tasks. `↑` `↓` select, `Enter` pauses or resumes, `d` then `y` deletes, `Esc` closes. A task waiting for approval shows as such; the panel does not offer approval, so it is approved in the dock |
| `/signin anthropic` or `/signin notion` | Signs in to your Claude plan or to Notion from the terminal. It shows an address to open in a browser, then asks for the address of the page that failed to load at `localhost`; paste it and press `Enter`. Meant for a computer with no browser, such as a hub |

Everything else you type after `/` goes to the session, which answers to the commands of the engine and its extensions, and to the skills. `/verify off` and `/verify on` (the checks below) are one of them. Type `/` to see the list.

## Undo and the changes line

When an instruction changed files, one line follows its answer: `3 files changed  +40 −12`. It is there whether or not the folder is a git repository, and it comes back when you attach to the session again.

`Esc Esc` (two presses within half a second, while Aria is idle) or `/undo` opens Rewind in the editor's place. It lists your messages, oldest first and the newest above `(current)`, each with what its turn changed. Select one with `↑` and `↓` and press `Enter`, or click it. Then choose what goes back to before that message, by `↑` `↓` and `Enter`, by its number or by a click:

1. Restore code and conversation.
2. Restore conversation.
3. Restore code.
4. Never mind.

The two code options are there only when a checkpoint was kept for that message and code changed from there on. `Esc` goes back one step, and then closes. Over `aria connect` to a [hub](hub.md) there is never a checkpoint, so only "Restore conversation" is offered.

Going back in the conversation puts the message's text back in the editor, so you can change it and send it again. Going back in the code restores the files that changed, deletes the ones created since and brings back the ones deleted; nothing outside what the checkpoint covers is touched. Undo is refused while Aria is working ("Not now: Aria is working. Undo once it is quiet.").

Checkpoints are taken one per turn, before the message that starts it (none for a `/command`), in a git repository of Aria's own under `~/.aria/checkpoints/`. Your project's `.git` is not touched, its `.gitignore` is respected, files over 5 MB are left out, and the last 50 are kept. Closing the session (`/exit`, or its tab in the dock) deletes its checkpoints, and with them the undo history. A session keeps none, and Rewind says why, in the projects folder itself, in your home folder, in a system folder, or in a folder with more than 20,000 files. In those only the conversation can go back. [Safety](safety.md) has the rest.

## Context

`/context` draws, in the editor's place, what fills the model's context: a grid of 100 cells, one per percent of the window, coloured by part, then a line each for the system prompt, the tools, the skills, the memory and the messages with their tokens and share, the free space and the compaction reserve (what is left after `compaction.percent`). The title gives the total, for example `Context  62.0k / 200.0k tokens (31%)`. `Esc` closes it.

The parts are measured at the start of each turn, so before your first message the title gives the total but the parts are empty.

## Project checks

After a turn that edited or wrote files, Aria runs the project's own checks in the background and the footer says how they went: `tests ✓ 43` (the number is read from the output when it says "N pass") or `tests ✗`. Checks run in order and stop at the first failure. When one fails, Aria is told, with the end of its output, so it can fix it; it is told once, and not again until the checks have passed.

The checks are the commands in the project's `.aria/verify.json` (`lint`, `types`, `test`), or when the file is missing, what the project shows: the `lint`, `typecheck` and `test` scripts of `package.json`, `cargo clippy` and `cargo test` for `Cargo.toml`, `ruff check` and `pytest` for `pyproject.toml`. Each has 5 minutes. On Linux they run inside the sandbox's box and only when the box works. On macOS they do not run (there is no box), whatever the project has. Subagents do not run them, and a turn you stopped is not checked.

`/verify off` stops them for this session, `/verify on` turns them back on.

## Notifications

When a run ends, or Aria needs an answer, and neither the terminal nor the dock is showing that session, you get one desktop notification from the computer the service runs on. It is titled `Aria · <session title> · <short id>` and shows the answer's first line (or "Done", or why it did not finish); a question is titled "Aria needs an answer". A session's next notification replaces the previous one, and looking at the session removes it. A service with no window (a [hub](hub.md)) sends none. On macOS, clicking a notification does not open Aria yet (see [pending](../pending.md)).

A session counts as seen when the dock shows it, or when a terminal attached to it has the focus. A terminal that does not report focus counts as focused while it is attached, so its session counts as seen.

## The mouse and selecting text

The mouse is on. A click opens a list row, picks an option or a suggestion, folds the task strip and shows or folds a call's `… N more lines`. The wheel scrolls the conversation three lines at a time, and sending a message brings you back to the bottom. A click on a command suggestion only writes the command in the editor; it never runs it.

To copy, drag over the text. The selection shows in reverse colours, and when you release the button the text goes to the clipboard at once, through the terminal (OSC 52) and, where there is one, through `wl-copy` on Wayland or `pbcopy` on macOS. It stays shown until your next key. If you would rather use your terminal's own selection, hold `Shift` while you drag (many macOS terminals use `Option`); that depends on the terminal, not on Aria.

## The look

The palette is the dock's: a near-black ground, light text, grey for what matters less, one accent (a soft violet) for the cursor and for what is chosen, and green, amber and coral only for states. With `COLORTERM=truecolor` (or `24bit`) it is drawn in 24-bit colour; otherwise it falls back to the nearest of the 256 colours.

## Keys

| Key | What it does |
|---|---|
| `Ctrl+C`, `Ctrl+D` | Leave. The session and its work go on. Works in every screen, panel and card |
| `Enter` | List: open the row. Session: send; while Aria works, queue the message for its next gap. In a suggestion list: pick (a command runs). In a card: see [Questions and permissions](#questions-and-permissions) |
| `Alt+Enter`, `Shift+Enter` | New line in the editor. `Shift+Enter` needs a terminal with the extended keyboard protocol |
| `Ctrl+Enter` | While Aria works: stop the turn and send what is in the editor once it has stopped (an empty editor: nothing happens, the turn goes on). Idle: send. Does nothing while a card is open. Needs a terminal with the extended keyboard protocol (kitty, ghostty, foot, WezTerm, recent Alacritty); in others it is an ordinary `Enter` |
| `Esc` | In a panel (Rewind, `/context`, `/schedules`, `/signin`): close or go back. In a card: cancel it. In a suggestion list: close it. While Aria works: stop the turn. Idle: pressed twice within half a second, open Rewind. On the list: leave |
| `Esc Esc` | Open Rewind (idle) |
| `Ctrl+O` | Switch verbose on or off |
| `Ctrl+T` | Fold or unfold the task strip |
| `Ctrl+L` | Clear and redraw the whole screen |
| `↑`, `↓` | List: move. Editor: history (from an empty editor). Suggestion list, Rewind, `/schedules`: move |
| `←`, `→` | Move the cursor in the editor |
| `Backspace` | Erase |
| `Tab` | In a suggestion list: complete (a command is written in the editor, not run; a file is inserted) |
| `PageUp`, `PageDown` | Scroll the conversation half a screen |
| `q` | On the list: leave |
| `1`–`9` | In a question or permission card: pick that option. In Rewind's choices: pick that choice |
| `y`, `n` | In a yes-or-no card: yes, no. A permission card is answered by number or click (see [Questions and permissions](#questions-and-permissions)) |
| Wheel | Scroll the conversation, three lines per step |
| Click | Open a row, pick an option or a suggestion, fold the task strip, show or fold `… N more lines` |
| Drag | Select and copy |

## Over ssh

`aria connect [user@]host` gives you this same client on a [hub](hub.md): Aria's service on a server, reached through ssh, with no port open for it (a sign-in listens briefly on the hub's own loopback address; see [the hub guide](hub.md#transport)). Everything above works the same, with these differences:

- The list says `+ new on <host>`. A session started there has no folder of yours: it works where the hub decides, since your folders are not on that computer.
- `!command` and secrets are refused with "not available on a connected hub". A terminal over ssh cannot run commands on the hub by hand or hand it a secret.
- The footer shows no git branch: git would read your computer's folders, not the hub's.
- `/model` lists only the model already in use, because the client picks the providers to list from the sign-ins of its own computer, and over ssh it has none. The hub accepts a model change, but this terminal has no list to choose from, so it cannot switch models there yet.
- `/undo` and `Esc Esc` go back in the conversation only: the session works in the hub's projects folder, where no code checkpoints are kept. To take back a file edit, log in to the hub with your own ssh and use git; `!git restore` is refused here.
- `/exit` closes the session on the hub, with its background jobs and its undo history. To leave and come back, use `Ctrl+C` or `Ctrl+D`.
- `@` lists file names in the session's folder: the hub's projects folder for a session you opened over `aria connect`, or the folder a session was started in on the hub itself, if you attach to one of those.
- `/signin` and `/schedules` do work, which is what makes `/signin` useful there.
- The first time you connect to a host, run `ssh host` once by hand to accept its key; `aria connect` gives up after 60 seconds. `ARIA_SSH` names another program to use instead of `ssh`.

Setting up a hub, the security model and troubleshooting are in [the hub guide](hub.md).

## macOS

The client, the keys and the checkpoints are the same on both platforms. What differs:

- The service's socket for terminals is `~/.aria/run/clients.sock` here (`$ARIA_HOME/run`); on every Linux setup, a hub included, it is `$XDG_RUNTIME_DIR/aria/clients.sock`.
- Selecting copies through `pbcopy`. The terminal's design notes say Terminal.app and iTerm2 report focus, so notifications are held back while a terminal that shows the session has the focus.
- The project checks do not run (see above).
- `Alt+Enter` needs the terminal to send `Option` as `Alt` (for instance "Use Option as Meta key" in Terminal.app); `Shift+Enter` and `Ctrl+Enter` need a terminal with the extended keyboard protocol, which not every macOS terminal has. Where they do not work, paste text with line breaks to write a multi-line message.

## Limits

- No attachments or pasted images from the terminal: `@path` names a file, but images and files are attached from the dock.
- No syntax highlighting in code blocks.
- The task strip shows a subagent's name, not its model.
- The dock's own page shows a slice's phase marks as tasks, but not the model's own task list, and it does not show the changes line, `/context` or the checks, nor read `display.verbose`. See [pending](../pending.md).
- Over ssh, a lost connection freezes the screen for up to 60 seconds while it reconnects, and the note it ends with names the local dock, not the hub.

## How it works

The service owns every session; the terminal and the dock are two clients of them. A terminal connects to `clients.sock` in the service's run folder (a socket only your user can open), speaks the same messages the dock's page does, and attaches to one session at a time to receive its events. A client that stops reading is dropped, and it reconnects once by itself ("Connected to the dock again."). The terminal's state is a pure state machine drawn with ratatui, which is why it is the same binary on Linux and macOS.

More in [the architecture notes](../architecture.md) and the designs: [the terminal](../superpowers/specs/2026-10-23-terminal-design.md) and [its second milestone](../superpowers/specs/2026-10-23-terminal-m2-design.md).

## See also

- [The dock](dock.md): the same sessions in the desktop window.
- [Running Aria on a server](hub.md): `aria connect`, ssh, and the hub's security model.
- [Safety](safety.md): the sandbox, permissions, secrets, checkpoints.
- [The agent](agent.md): models, effort, the router, subagents, slice.
- [Commands and keys](../reference/commands.md): every command and shortcut, by surface.
