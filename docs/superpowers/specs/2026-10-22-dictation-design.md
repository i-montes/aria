# Dictation — design

Idea #12 of `docs/ideas.md`, first half: dictating into the chat's input. The owner speaks, Aria writes what was said in
the input, the owner reviews and sends. macOS first; Linux follows with the same code (only its GPU backend differs).

Decided with the owner (2026-10-22), after a test on their Mac with their voice (59 s, Spanish with tool names):

- **Whisper**, the same engine on both platforms: `large-v3-turbo` quantized to **q5_0** (574 MB). On the owner's M5 Pro it
  transcribed the 59 s in 2.1 s warm (load 0.2 s), 715 MB resident, as accurate as the full model (it even got "Vercel").
  Apple's `SpeechTranscriber` was faster but no better and macOS-only.
- **Text at the end**, not live: record, stop, the text lands in the input.
- A **microphone button** in the composer and a shortcut; never sent by itself.
- The **language of Aria** (`settings.assistant.language`: `es` or `en`) is Whisper's language.
- **On the device only**: the audio never leaves the computer.

## The engine — `crates/aria/src/dictation/` (both platforms)

- `whisper-rs` 0.15 (whisper.cpp built into the binary; feature `metal` on macOS; Linux picks its backend later, CPU by
  default), with flash attention on.
- `cpal` for the microphone (CoreAudio on macOS, ALSA/PipeWire on Linux): the default input device, any rate and channel
  count, mixed to mono and resampled to 16 kHz (linear) as it arrives.
- `Dictation` (one per dock), on its own thread:
  - `start(language)`: starts recording at once; loads the model meanwhile if it is not loaded.
  - `stop()`: ends the recording, transcribes, and answers with the text (trimmed; empty when nothing was said).
  - `cancel()`: ends the recording and drops it.
  - The model is unloaded after **5 minutes** without dictating (the memory is only used around a dictation).
  - At most **5 minutes** of audio per dictation: past that it stops by itself and transcribes what it has.
- The model lives in `~/.aria/models/ggml-large-v3-turbo-q5_0.bin`. It is not part of the install: the first time the
  owner presses the microphone, the page asks whether to download it (its size said), and the dock downloads it from
  `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin` to a `.part` file, checks its
  size and SHA-256 (pinned in the code), and renames it. Progress is reported; a failed or cut download leaves nothing
  behind but the `.part`, which the next try restarts.

## The messages (shared by both docks)

Page → dock, per session (wrapped in `session`, like the chat's): `dictationStart`, `dictationStop`, `dictationCancel`,
`dictationDownload`.

Dock → page, per session: `dictationState { state, seconds?, progress?, error? }` with `state` one of `idle`,
`recording` (`seconds` so far), `transcribing`, `needsModel` (`size` in bytes), `downloading` (`progress` 0..1), `error`
(`error`: `no-microphone`, `microphone-denied`, `download-failed`, `model-broken`, `failed`); and `dictationText { text }`
when a transcription ends. Only one dictation at a time across the dock: a start from another tab while one runs is
answered with that tab's `dictationState error: busy`.

## The page

- `ui/chat/parts/composer.js` gains a microphone button beside the paperclip (`data-chat="dictate"`), with three looks:
  idle (a mic), recording (red, pulsing, with the seconds), transcribing (a spinner). **Alt+D** does what the button
  does.
- Pressing it: `needsModel` → a card in the thread asks "Download the voice model (574 MB)? It stays on this computer."
  with Download / Not now; Download shows the progress, and the dictation starts when it is there.
- While recording: **Esc** cancels (nothing is written); the button or Alt+D stops and transcribes; **Enter** stops,
  waits for the text, puts it in and sends.
- The text goes in at the cursor (a space before it if the character before is not a space), the input keeps the
  focus, and the owner sends it as any message.
- Errors are a note in the thread (`no-microphone`: "No microphone found"; `microphone-denied`: where to allow it in
  System Settings → Privacy & Security → Microphone, or the Linux equivalent; the others say what failed).
- Texts in English and Spanish.

## macOS permission

macOS asks for the microphone the first time; the reason it shows comes from `NSMicrophoneUsageDescription`, which an
unbundled binary carries in an `Info.plist` linked into its `__TEXT,__info_plist` section (`build.rs`, macOS only).
Aria is not signed with a stable identity, so macOS may ask again after an update (the binary changes); this is checked
live and, if so, written in `docs/pending.md`.

## Testing

- The resampler and the mono mix (pure): 48 kHz stereo → 16 kHz mono lengths and values; 16 kHz mono unchanged.
- The model file: the `.part` rename only after size and hash match; a wrong hash removes the `.part` and says
  `model-broken`; an existing complete file is not downloaded again (a local HTTP server in the test).
- `Dictation` with a fake recorder (samples from a WAV) and the real model when present (`#[ignore]`, run on the Mac):
  start/stop gives the text; cancel gives nothing; the 5-minute cap; unload after idle (with a shortened timer).
- The messages' names (bridge) and the page: the button's looks, Alt+D, Esc, Enter, the download card, text inserted at
  the cursor, errors as notes.
- Live on the owner's Mac: the permission prompt, a dictation in Spanish and in English, memory while loaded and after
  the unload, and whether the permission survives an update.

## Out of scope

Live text while speaking, Aria reading replies aloud, a custom vocabulary (Supabase, Vercel…), a choice of model, the
Linux GPU backend (Linux's turn).
