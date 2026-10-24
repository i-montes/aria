# Voice, face and trace

Three things that give Aria a presence and let you see what it does: dictation (you speak, the text lands in the input), Aria's face (one animated look for what Aria is doing, in the Omarchy bar or beside the macOS notch) and the live trace (`/debug`, a waterfall of every step of a turn). This guide is for anyone who uses the dock on Omarchy or macOS.

All three are dock features. They need the window, so a terminal client and a hub do not have them (see [Where each one works](#where-each-one-works)).

## Dictation

### What it does

You press the microphone button in the composer (or `Alt+D`), speak, and press it again. Aria writes what you said in the input, in your cursor's place. It never sends it by itself: you read it, fix it if you like, and send it as any other message.

The transcription is done by Whisper (the `large-v3-turbo` model, quantized to `q5_0`) running on your computer. The audio is not sent anywhere: it is recorded in memory, transcribed locally and dropped. The only network traffic dictation causes is the one-time download of the model.

### How to use it

1. Open the dock and click into a conversation. Press the microphone button next to the send button, or `Alt+D` (on a Mac, `Option+D`; the key is read by its place, so the character `Option+D` types does not matter).
2. The first time, Aria asks "Download the voice model (574 MB)? It stays on this computer, and so does your voice." Press **Download**. A progress line shows the percentage, and the recording starts when the model is in place. **Not now** closes the question and records nothing.
3. Speak. The button turns red and shows the time recorded (`0:07`).
4. Press the button or `Alt+D` again to stop. The button shows a spinner and "Writing…" (its tooltip reads "Writing what you said…") while Whisper works.
5. The text appears in the input at the cursor (with a space before it when the previous character is not a space) and the input keeps the focus. Edit it, then press `Enter`.

While recording:

| Key | What it does |
|---|---|
| `Alt+D` or the button | Stop, transcribe, put the text in the input |
| `Enter` | Stop, wait for the text, put it in and send it |
| `Esc` | Cancel: nothing is written |

`Alt+D` works when the dock shows a conversation; it does nothing on the settings screen.

If you switch to another tab while it transcribes, the text goes into the draft of the tab where you started. Only one dictation runs at a time in the whole dock: starting one from a second tab says "Another tab is dictating."

### The language

Whisper is told to listen in Aria's conversation language, the `assistant.language` setting (see [settings](../settings.md)). Only `es` makes it listen in Spanish; every other value, including `pt` or free text, makes it listen in English. Change the language in the settings screen if you dictate in another one.

### The voice model

| | |
|---|---|
| File | `~/.aria/models/ggml-large-v3-turbo-q5_0.bin` |
| Size | 574,041,195 bytes (574 MB) |
| Source | whisper.cpp's repository on Hugging Face (the address is in `crates/aria/src/dictation/model.rs`) |

The model is not part of the install. Aria downloads it to a `.part` file, checks its size and its SHA-256 (pinned in the code) and only then renames it, so a cut or corrupt download never looks like a model. If the check fails the `.part` is deleted and you see "The downloaded voice model was not right; it was removed. Try again." If the connection fails you see "The voice model could not be downloaded. Check the connection and try again." Pressing the microphone again restarts the download. To free the 574 MB, delete the file; Aria asks again the next time you dictate.

### Memory, GPU and CPU

The model is loaded into memory when you start recording (while you speak, so it is ready when you stop) and unloaded after five minutes without dictating. While loaded it holds several hundred MB; after the unload dictation uses no memory at all. The first dictation after an unload spends a moment loading before the text appears.

- **macOS**: Whisper runs on the GPU through Metal.
- **Omarchy (Linux)**: Whisper is built with Vulkan, so it runs on the GPU when there is a driver. The installer's comment says it runs on the CPU when there is none; Aria's own code does not check that. The Omarchy installer adds `alsa-lib` (the microphone) and `vulkan-icd-loader`. On the CPU it uses half the logical cores, between four and eight; a short phrase takes seconds instead of a fraction of a second.

A recording is cut at five minutes: past that it stops by itself and transcribes what it has.

### What dictation will not write

Whisper makes up words when it hears silence. Aria guards against it: it keeps only the stretch from your first to your last voiced moment, and when that is one second or less (a click, a cough, silence) it writes nothing. It also drops whole segments Whisper itself marks as probably not speech, and the closing lines it learned from videos ("Gracias por ver el video", "Thanks for watching", "Please subscribe" and a few more) when they are a segment on their own.

### Microphone permission and errors

On **macOS** the system asks for microphone access the first time you dictate; the reason it shows is built into the `aria` binary. Aria is not signed with a stable identity, so macOS may ask again after an update (the installer signs it ad hoc, and the system treats a new build as a new app). If you said no, dictation shows "Aria may not use the microphone", and the cue Aria uses is that the recording is exact silence. Allow it in System Settings, Privacy & Security, Microphone.

On **Omarchy** there is no prompt. If the default input device is wrong, change it in your sound settings.

The notes you may see in the thread:

| Note | Meaning |
|---|---|
| No microphone was found. | There is no default input device (also what a build with no window answers) |
| Aria may not use the microphone | macOS denied it (silence), or Linux gave no sound at all |
| Another tab is dictating. | One dictation at a time across the dock |
| The voice model could not be downloaded. | No connection, or a write that failed |
| The downloaded voice model was not right; it was removed. | Size or hash did not match |
| The dictation failed. | Anything else; the dock's log has the reason (`~/.aria/logs/aria.log` on macOS, the journal of the `aria` user unit on Omarchy) |

### Limits

No text while you speak (it arrives when you stop), no custom vocabulary, no choice of model, no replies read aloud. On Linux the microphone path and the GPU backend are newer than the Mac's; [pending](../pending.md) lists what has not been tried live. Aria never records or plays sound unless you press the button.

## Aria's face

### What it does

The face is one animated look for what Aria is doing right now, so you can tell from the corner of your eye whether it works, waits for you or is done. It is computed from **every tab at once**: one face for the whole of Aria, not one per tab.

| Face | When | What you see |
|---|---|---|
| asking | A dialog waits for you: a permission, a question or a confirmation, in any tab | Round eyes; it taps on the glass, sooner the longer it waits |
| error | A run ended in an error, or its Pi stopped mid-run | A puff of smoke, `x x` eyes, a soft shake now and then |
| delegating | A tool that starts subagents is running | Two little subagents work beside it; it checks on each |
| editing | A write or edit tool is running | It types on a laptop; code fills the screen |
| working | Any other tool is running (a command, a connector) | Command output scrolls up; it leans in |
| reading | A reading tool is running (files, search, the web, memory) | Head bowed over an open book, eyes moving word by word |
| thinking | A run is going and no tool is running | It looks up into a corner, one eye narrowed, "hmm" |
| scheduled | A scheduled task is running (in its own tab) | Its alarm clock rings, then it ticks with the clock |
| done | A run ended well, for eight seconds | A jump with a full turn and confetti |
| idle | Nothing is going on | Glances, blinks, yawns, rolls side to side |
| sleeping | Idle, and nothing has happened for ten minutes | Eyes shut, it breathes, a Z floats up |

When tabs disagree, the first row of the table that applies wins: what needs you, then what went wrong, then the work going on (the most involved first), then what just ended. Tool names decide between delegating, editing, reading and working. Inside one tab, a dialog beats everything, and a scheduled run shows `scheduled` whatever tool it is using.

Two details. An **error stays** until you open the dock (looking at it counts as seeing it) or the tab runs again. A run you stop yourself shows no `done` and no `error`: it is just idle. A closed tab no longer counts.

### On Omarchy: the bar widget

The widget is a small face in the Omarchy shell's bar, 22 px tall, next to the weather. A click on it runs `aria toggle`, which opens or closes the dock. It needs the Omarchy shell: the `omarchy` command on your `PATH` and the `~/.config/omarchy` folder (or the same under `$XDG_CONFIG_HOME`).

The installer puts it there for you. To do it yourself, or to take it out:

```sh
aria bar install   # puts the widget in the bar
aria bar remove    # takes it out and deletes its files
```

`aria bar install` writes `manifest.json` and `Face.qml` in `~/.config/omarchy/plugins/aria.face/`, backs up `shell.json` as `shell.json.bak.<seconds since 1970>` next to it in `~/.config/omarchy/`, and places the widget with Omarchy's own `omarchy plugin enable`, after the weather widget (at the end of the centre section when there is no weather). Run it again when you like: it only rewrites what changed and does not add a second widget. Where there is no Omarchy shell (macOS, another desktop) it prints that it was skipped and does nothing. `aria bar remove` runs `omarchy plugin disable` and deletes the plugin folder; the uninstaller runs it for you.

The dock tells the widget what Aria is doing by writing one word to `$XDG_RUNTIME_DIR/aria/face`; the widget watches the file. To look at a face, write its word there, for example `echo reading > "$XDG_RUNTIME_DIR/aria/face"`; it stays until Aria's real face changes. The face needs the dock running: without it nothing writes the file.

### On macOS: the wing beside the notch

The face sits in a small black wing on the right of the notch, as if the notch grew a little. When no screen has a notch it is a black pill at the top centre of the main screen, as tall as the menu bar. With several screens it goes beside the notch of the first one that has one. A click on it opens or closes the dock, like the shortcut. It is above the menu bar, on every Space and beside full-screen apps, and it stops drawing while the display sleeps or the screen is locked (under 1 % CPU at idle in the live check). It moves by itself when you plug or unplug a display.

It is part of the dock: it appears when the dock starts, and there is no setting to hide it.

To see one face, start the dock yourself with `ARIA_FACE` set. Stop the running one first (`aria quit`), run it from a terminal, and bring the normal one back with `aria start`:

```sh
aria quit
ARIA_FACE=reading aria dock
```

`ARIA_FACE` takes one of the words in the table. It is read by the macOS wing only (on Omarchy, write the file as above).

### How it works

`crates/aria/src/face.rs` turns each tab's activity into a word and writes it to the face file (`~/.aria/run/face` on macOS). The bar widget (`install/omarchy/aria.face/Face.qml`) draws it with springs and timers; on macOS the same logic is ported to Rust (`face_motion.rs`) and drawn with Core Animation (`dock/face_layers.rs`, `dock/face_wing.rs`). See [the architecture](../architecture.md) and the [macOS design](../superpowers/specs/2026-10-22-mac-face-design.md).

### Limits

The wing was checked live on one notched MacBook screen; a click on it and a setup with an external display were not tried by hand. No right-click menu, no hover details. See [pending](../pending.md).

## The live trace

### What it does

The trace shows where a turn's time and money go. Type `/debug true` in a tab and a panel opens to the left of the chat: every model call, tool call and subagent of the turn, as a bar on a live timeline, with the time, the tokens and the cost of each step.

### How to use it

1. In the dock, type `/debug true` in the tab you want to look at (`on` works too; `/debug` alone switches it). The dock widens toward the left, from 700 to 1,060 px, and the panel opens with "The next turn is traced here".
2. Send a message. The panel fills while the turn runs, and stays after it ends, until the next turn starts. A turn already running when you type the command is not traced.
3. Type `/debug false` (or `off`) to close it. The dock goes back to its normal width.

The trace is **per tab**: another tab has none until you turn it on there, and the panel follows the tab you look at. It is off again after you restart the dock or close the tab.

What the panel shows:

- **The header**: the turn's prompt on one line, its state (Live, Done, Failed) and six figures that climb live: duration, cost, tokens, model calls, tool calls and errors. A provider that reports no cost (a subscription) shows the cost as a dash.
- **The waterfall**: on the left, the steps as a tree (a subagent's own calls sit under it, under the call that started it); on the right, each step's bar on a time axis, with its duration and cost. Agents are grey, model calls violet, tools teal, failures red. A running bar grows toward a "now" line. The slowest finished model or tool call carries a "slowest" tag.
- **Folding**: an agent folds by its chevron. A subagent folds by itself when it ends; a running one stays open; what you fold or unfold by hand stays as you left it.
- **The details of a step**: click a row (or press `Enter`) and a card opens over the panel with the kind, the state, the duration, when it started, the tokens (in, out, from cache, reasoning), the cost, the error in red and the step's input and output.

| Key | What it does |
|---|---|
| `Up`, `Down`, `Home`, `End` | Move between rows |
| `Right`, `Left` | Unfold or fold an agent; move into or out of it |
| `Enter` or `Space` | Open or close the details card |
| `Esc` | Close the details card |

### What it keeps

Only the current turn, in memory. Nothing is written to disk, there are no past turns, no comparing runs and no export. A new turn replaces the trace. Past 500 steps the oldest finished ones are dropped and the header says "N earlier steps not shown". Previews of input and output are cut at 2,000 characters, and image data is removed. The previews show what the tool cards in the chat already show. A scheduled task's tab can be traced like any other.

### Limits

A subagent's own subagents appear only as far as they reach the session's watch socket. A turn where the Pi restarts mid-way ends the trace as failed. The panel was checked in a headless browser; a turn with a real model had not been traced when this was written. See [pending](../pending.md) for what has not been seen live, and the [design](../superpowers/specs/2026-10-22-trace-design.md) for how it is built. Subagents and the router are explained in [the agent](agent.md).

## Where each one works

| | Omarchy dock | macOS dock | Terminal client | A hub (headless) |
|---|---|---|---|---|
| Dictation | Yes (Vulkan or CPU) | Yes (Metal) | No microphone button and no key | No: a terminal cannot start one, and a service with no window has no microphone |
| Face | Bar widget (needs the Omarchy shell) | Wing beside the notch, or pill | No | No: `aria serve` has no face |
| Trace (`/debug`) | Yes | Yes | No `/debug` command | No |

The dock does not accept the dictation or trace messages from a terminal, so even a terminal attached to your own dock cannot start either. What a terminal does show is each session's status, which a hub also sends; see [automation](automation.md) for what the statuses mean and for notifications.

## See also

- [The dock](dock.md): opening it, tabs, the composer and the commands, `/debug` among them.
- [Automation](automation.md): scheduled tasks, background work and session statuses.
- [The agent](agent.md): models, subagents and what the trace's steps are.
- [Running Aria on a server](hub.md): what a hub does without a window.
- [Settings](../settings.md): `assistant.language`, which sets the dictation language.
