# Pending

This is the honest list of what Aria does not do yet, what is rough or inconsistent, and what the maintainer decided to do
differently from the design. The guides describe what works and link here for the limits; for the whole documentation
map start at [the documentation index](README.md). An entry says where the thing lives, so it can be fixed.
It is not a roadmap: ideas are in [ideas.md](ideas.md) and designs in [superpowers/specs](superpowers/specs/).

The sections follow the order of the guides: the dock, the terminal, the hub, the agent, automation, safety,
integrations, voice and face, install and releases, macOS; then parity and the deliberate differences from the design.

## The dock ([guide](guide/dock.md))

Panels and cards:

- A subagent's session view has no composer (the design has one): you cannot steer a subagent, only stop it and start
  another.
- `/agents` (Pi's own overlay, `extensions/subagent-watch/index.ts`) does nothing in the dock; `Alt+A` is the dock's way.
- The quota card's countdown is a number, not counting down; Pi's own timeout decides (`extensions/quota-fallback`).
- Question card: the sliding animation between questions and the "Answers sent" / "Open questions" states were not
  ported (the card is removed when answered).
- The page shows only the plan's marks as tasks (`aria-tasks`, `ui/chat/state.js`), not the list the model keeps with
  `tasks_update`; it has no undo, changes line, `/context` or checks footer either (the terminal has them).
- The `@` picker never lists folders: `crates/aria-core/src/files.rs` returns files only, while `ui/chat/parts/files.js` and
  `insertFile` in `ui/chat/view.js` are built for a trailing-slash folder entry.
- Clicking the "needs your answer" desktop notification opens the dock on the active tab, not on the waiting one.
- The "update is waiting" notice (`Sessions::tell_update_waits`) is shown in the conversation of each session that has
  background work, but in English only, and it reaches only the tabs whose Pi is running.
- A session's images (screenshots the model read) are not sent to the page, which never shows them: a session full of
  them made its history tens of megabytes.
- The dock wakes every saved tab's Pi when it starts (the active one first); each open tab keeps a Pi and its MCP
  servers running. Closing a tab stops its Pi and its process group; a program that leaves the group (`setsid`) is not
  followed.

The thread and the composer (known limits of the page, from the review, deferred):

- The thread is redrawn on every streamed fragment, so text cannot be selected or copied while a run streams.
- Choosing `model` from the command picker only types `/model `; the list opens once a space follows (or from the chip).
- Closing the active tab does not save its unsent draft; a language change drops unsent drafts.
- `rows` of the composer counts only explicit newlines; a long wrapped line does not grow it.
- The working phase keeps the last tool's name ("Reading files") while the answer streams.
- The browser checks (`tools/ui-sandbox/check.ts`) are not part of `bun test` (they need Chrome).

Settings screen and setup texts that disagree with the code (`ui/i18n/en.json`, `es.json`):

- `ob.router.b4` lists "Experts assigned to each part of a big task" as part of the router, but `assignExperts`
  (`extensions/jev-router/index.ts`) runs on every `parallel` call whether `router.enabled` is on or not.
- The settings screen has no row for the keyring: `credentials.keyring`, `aria keyring on|off|status` and the dock's log.
- The settings screen shows an outside change (`/effort`, `/autocompact`) at once, unless it has unsaved changes of the
  reader's own: those stay, and saving them writes over the outside change.

## The terminal ([guide](guide/terminal.md))

Milestones 1 and 2 are done ([design](superpowers/specs/2026-10-23-terminal-design.md),
[milestone 2](superpowers/specs/2026-10-23-terminal-m2-design.md)). Left:

- The task strip shows a subagent's name, not its model: the watch does not report the model.
- No attachments (the terminal sends none), no dictation, no `/debug`, and no background pills, list or stop (the job
  messages are not in a client's surface, `crates/aria/src/clients.rs`).
- Commands the terminal does not know (`/compact`, `/session`, `/copy`, `/export`, `/reload` are the dock's own) go to the
  session as typed. The terminal does not page back to earlier history (`term/` has no `chatEarlier`).
- `/model` over `aria connect` lists only the model in use: `run_remote` passes no connected providers
  (`term/mod.rs`, `connected: Vec::new()`). The hub does accept a client's `setModel`; only this client has no list to pick from.
- The editor has no Home, End, Delete or word keys, and Up and Down are only history; `Ctrl+Enter` on an empty editor
  while Aria works does nothing; the spec's one-time hint about selecting with Shift (Option) is not there.
- A checkpoint is taken once per turn (a queued message gets one only if the turn has none), not before each message
  as the spec says; none for a `/command`.
- Verify (the footer's checks) does not run on macOS at all, and on Linux only where the box works (see Safety).

## The hub ([guide](guide/hub.md))

`aria serve`, `aria connect`, `aria hub init` and the terminal's `/schedules` and `/signin` are built and checked by
`tools/hub/accept.sh` (no display, no provider); this is milestone M1 of
[the design](superpowers/specs/2026-10-23-hub-and-machines-design.md). Left:

- **Machines (M2)**: linking other computers to a hub (`aria machine add`, `aria link`), one list over all of them,
  "new here", the credentials pushed to each; attachments from a client's computer. **A web client (M3)** and **push
  notifications (M4)**. **Shared memory**: Engram is per service, so a hub has its own.
- **The public ssh port**: the hub's sshd answers on a public address until you can close it and reach the hub
  only over a private network (Tailscale). Until then the protection is key-only login and the restricted keys
  `aria hub init --client-key` writes; `hub init` warns once and never touches `sshd_config` or the firewall.
- **bwrap on Ubuntu 24.04**: unprivileged user namespaces are restricted by AppArmor there, so the sandbox can be
  installed and still fail. `hub init` probes it and prints the fix; Aria never applies it. Where the box does not work
  every command asks first, and on a hub the prompt goes to whichever client is attached, so a holder of the client key can
  approve it; that is why `hub init` does not install the client key without a working sandbox unless
  `--allow-no-sandbox` is given. Until then treat the client key as shell access to the hub user.
- **`aria update`**: there is no such command, though the hub design names one. A hub is updated by running `hub init`
  again with a new binary; the protocol-mismatch messages only say to update the older side.
- **A run that survives a dropped client** needs a model provider, so the acceptance script does not fake it: the maintainer
  checks by hand that a run started over `aria connect` goes on after the client is killed and shows again from a
  second connect (the design's "done when" for M1).
- **A scheduled task that may write can never be approved on a hub**: it waits for Approve in the dock, a hub has none,
  and the terminal's panel only says "waiting for approval in the dock". A task never asks (`noAsk`), so with no working
  sandbox it cannot run shell commands either.
- **A scheduled run's result cannot be read from a terminal**: the terminal's list hides scheduled sessions
  (`term/app.rs`), and `/schedules` shows the next run, not the last result. Only the model's `schedule_list` tool has
  the first line of it, so on a hub you ask Aria in a session.
- **A Claude plan on a hub**: `/signin anthropic` works, but the Claude provider package (`pi-claude-agent-sdk`) is
  installed only by the dock's `/connect` (`installClaudeBridge` is refused to every client), and neither `hub init` nor
  `install.sh --headless` installs it or Claude Code.
- **`/autocompact <n>` from a client** writes the hub's `settings.json`, though the design says clients cannot
  change settings.
- **The footer's git status is local**: a remote session shows no branch, because git runs on the computer of the
  terminal and the folder is on the hub (a broker for it is not built).
- **A lost connection blocks the terminal**: reconnecting over ssh waits up to 60 s for the hub's hello with the screen
  frozen, and the note shown when the connection is lost is the local dock's ("Aria's dock stopped. Run `aria` again to
  start it."), not one that names the hub or says to run `aria connect` again.
- **`hub init` audits only the key it adds**, not the existing `authorized_keys` lines as the design says, and reads
  `has_tailscale` without using it (`hub_admin.rs`). It does not create the hub's projects folder; the service makes it
  when a session starts (`working_folder` in `chat.rs`).
- **Smaller things worth doing**, from the review of the first milestone:
  - `aria client-stdio`: replies still in flight are lost when the client's stdin reaches EOF.
  - `aria serve`'s loop wakes about 62 times a second when idle (it should sleep until something happens).
  - `~/.aria/hub-src` (what `hub init` uploads) is never cleaned up; `hub_admin.rs` is large (`hub_init` can be its own
    module); one `hub init` opens a separate ssh connection for each question and step, more than a dozen on a first run (no `ControlMaster`).
  - A schedule's `replace()` can overwrite a live "awaiting" mark.
  - Esc in `/signin` leaves the dock's sign-in alive until its timeout.
  - `tools/hub/accept.sh` keeps the maintainer's environment, and a failed check hides the ones after it.
  - `hub init` sends the whole `settings.json` with only `sandbox.projectsDir` rewritten: other path-like keys
    (`integrations.packages.path`, `sandbox.extraWritable`) point at paths that do not exist on the hub. The settings
    split belongs to M2.
  - A hub still writes the face file when sessions report; nothing reads it.

## The agent ([guide](guide/agent.md))

Providers and models:

- `/connect` installs the Claude provider (`claude-bridge`) for the sign-in route and needs Claude Code on the machine.
  Closing the card does not cancel a sign-in still waiting in the browser (it times out after five minutes).
- Showing the sign-in's email or plan in the card needs a profile call the sign-in does not make today.
- Disconnecting: the settings screen has Remove on every connected provider except the main one (`credentialDelete`);
  `/connect` and the terminal have no way to disconnect.
- `providers.rs::apply_provider` always turns the router off for a provider without `includes_jev`; its twin
  `ui/onboarding/steps.js::applyProvider` follows the stored keys (a Go key kept beside a Claude main provider keeps
  the router on). Which one is right is not decided.
- "Allow pay-per-use models" is shown only with OpenCode Go's quota module.
- Nothing documents or supports adding your own expert or skill by hand; the code only keeps a file you edited.

The quota bars:

- OpenCode Go's windows (5h, Week, Month) come from `GET /zen/go/v1/usage` and were checked against the real answer. A
  Claude plan's windows (5h, Week, and any per-model one such as Fable) come from `GET /api/oauth/usage` with the
  sign-in's token, an endpoint Anthropic does not document: its answer is parsed defensively and has **not** been checked
  against a real account (none is connected yet).
- An API key, OpenCode Zen and an expired sign-in token show no bars. Aria does not refresh an expired Claude token (Pi
  does when it uses it); the bars come back after the next run.
- Only the 5-hour and weekly windows trigger the free-model fallback; the monthly one is only shown.
- The quota card is one component (`renderQuotaCard`) opened from the header chip and each tab's hover; with several
  sessions the header shows tabs, not the bar.

Router, language and compaction:

- The router eval is 93/94: one expert case ("performance opportunities in the bundle") sits on the line between
  `reviewer` and `frontend` (0.48 against 0.52) and flips between runs; the experts are the maintainer's, word for word.
- The router's "no expert" value is `ninguno` (`extensions/jev-router/route.ts`), whatever `assistant.language` is; it is an
  internal token, never shown.
- A slice job that ends while the main agent is in the middle of a turn gets no `[slice]` hint and no tasks-feed finish:
  Pi queues the `aria-background` message as a follow-up and `before_agent_start` does not fire for it. The design
  accepts it; the later fix is to emit the hint from a `message_end` hook for `aria-background`.
- A session rotation reloads the extensions in-process: the grants are kept, but the MCP servers reconnect, so a stateful
  MCP session (a browser, a login held by the server) starts over.
- A segment whose tokens exceed the model's window cannot compact (the summary call itself does not fit), so the
  compaction Aria asks for over 8 MB fails and is tried again once per Pi start; the file keeps growing meanwhile.
- While Pi compacts, the page's messages wait for the compaction's end, and the composer says nothing about why.

Settings reach Pi (`~/.aria/agent/aria.json`, the rendered persona and experts, Aria's Engram entry):

- A settings change restarts the running Pis only when it touches what Pi reads at start (persona, experts, its
  settings with the compaction reserves, `mcp.json`, the shim, the working folder), each when its run settles. A new
  compaction percent from the settings screen is written as Pi's reserves before that restart; `/autocompact` in a
  session applies it at once.
- Pi packages come only from a local folder (`integrations.packages.path`); npm, git, a gallery and trusting third-party
  packages are not built, and the setup no longer offers them. `thirdPartyTrusted` stays in the file and nothing reads it.
- An empty `assistant.language` renders the persona as Spanish (`persona.rs`, `language_name`), though the default is `en`.
- `assistant.language` accepts values such as `pt` or free text, but the interface is English or Spanish only and
  dictation understands only `es` (everything else is English).

## Automation ([guide](guide/automation.md))

Scheduled tasks:

- A task runs only while Aria runs; a sleeping computer is never woken (`pmset`), so a missed time runs when it is back,
  once, and only on the same day.
- No intervals ("every 2 hours") and no results sent outside Aria (email, Telegram).
- Tasks are created and reworded by asking Aria; the `/schedules` panel only pauses, resumes, approves and deletes them.
- On macOS a click on the notification does not open the dock (an unbundled app has no notification action); the tab is
  marked unread for when it opens.
- A successful scheduled run was checked live only with the fake Pi (a real one needs a key); the late start, the tab,
  Pi's grants and a refused run's record were checked live on macOS. Not yet tried on Omarchy.

Background work (`crates/aria/src/jobs.rs`, the `background` extension, the strip and the Alt+A list; see the
[architecture](architecture.md#background-work)):

- A background subagent cannot be steered: its session view shows its steps but has no composer.
- macOS has no `PR_SET_PDEATHSIG`: if the dock crashes (rather than quits), the processes it started keep running and
  nothing stops them; the next start marks them `cut` without killing them. On Linux the signal reaches only the job's
  own leader, so a child it forked outlives a crash unless systemd's cgroup takes it (the installed service does).
- The limits (3 subagents at once, 10 processes, 1000 lines of output, 30 minutes before the update notice) are
  constants in `jobs.rs`, not settings.
- A worktree has the person's committed and uncommitted tracked changes, not their untracked files.
- When a worktree's changes cannot come back because git itself failed (not a clash), it is reported as a clash, the
  worktree is kept, and git's message is only in the log.
- If the dock dies between making a worktree and recording it, the worktree stays in `<projects dir>/.aria-worktrees/`
  without a job naming it (`git worktree list` shows it); an empty `.aria-worktrees/` is left after the last one goes.
  If the projects folder is itself inside a git repository, `.aria-worktrees/` lands inside it.
- A subagent stopped while it is still starting is shown as failed, not stopped.
- A broken `sessions.json` (moved aside at start) makes the dock forget every session's pending end notes.
- If the dock never answers a `jobOutput`, that row's output stops refreshing until it is folded and opened again.
- `background_output` for a subagent says its state, not what its watch says it is doing now (the jobs socket does not
  see the watch); two background subagents in one session with the same title and task share one watch agent.

The turn watchdog (`models.turnTimeoutMinutes`):

- No MCP reconnect in Aria: Pi has no way for Aria to ask it to reconnect a server, so a connector that failed to start
  waits for its next use (or a Pi restart).
- A tool that never reports its end keeps the watchdog paused for the rest of that run (a running tool pauses it on
  purpose); the watchdog only covers the model being silent.

Updates that wait:

- Updating Aria waits for the dock: the installer tells it a new version is installed (`aria update-ready`), and it
  restarts into the new binary (same process, so systemd and launchd keep it) once it is hidden and no session has a run
  going or a question open. A changed service file, `--restart-now`, or a dock older than this restarts at once,
  cutting a run.

## Safety ([guide](guide/safety.md))

The sandbox:

- The box uses `bwrap` (Linux). Where it does not work (macOS, or a Linux host with no `bwrap` or one the system blocks)
  every command asks first with the permission card (`extensions/sandbox/route.ts`). A Seatbelt (`sandbox-exec`) profile
  with the same limits is the macOS sandbox still to design; some of its tests assume Linux paths.
- With `sandbox.askToLeave` off and no working box, nothing runs; with `PI_SANDBOX=0` the whole extension returns early,
  so commands also lose the `$ARIA_SECRET_<NAME>` variables and the closing of Aria's own files against write and edit.
- The sandbox keeps a repository's `.git/config` and `hooks` read-only, but the rest of `.git` stays writable, and git
  follows a `commondir` file in any git dir: a boxed command can point a repository at a config of its own, which the
  person's own git then reads. Repositories nested under a folder that is not one are not protected at all, and that includes a repository inside a
  folder granted by `sandbox.extraWritable` or a scheduled task (only the granted folder's own `.git/config`, `.git/hooks`,
  `.pi` and `.aria` are read-only, and only when they exist as the box starts). The dock's own
  git commands neutralize hooks, fsmonitor and filter drivers, but a boxed background process that swaps `commondir`
  between the dock's read of the filter drivers and its `git add -A` could slip one in. To do: protect `commondir` and
  `gitdir` in `extensions/sandbox/policy.ts`, and give the dock's git an empty attributes source.
- Pi's write and edit can never touch `aria.json`, the settings file or the attachments folder, whatever was granted; a
  bash command the person let run outside the sandbox still can.
- A scheduled task's writable grants are judged when its Pi starts (`runnable_writable`), but the sandbox extension binds each
  granted root by path on every command, so a boxed run could rename an ancestor of a granted root and replace it with a
  link; only the extension's secret and Aria-home filter guards that. The start-time check does not re-run `too_wide`.
- The checks after edits (`extensions/verify`) run nothing on macOS, not even `.aria/verify.json`, and on Linux nothing
  where the box does not work (`extensions/verify/checks.ts`, `index.ts`).
- The dock takes checkpoints for every session but has no way to show or use them; undo and the changes line are
  terminal-only, and a client on a hub may use undo too.

Secrets and the keyring:

- A command in the sandbox can still run `aria secret` (or `security`, `secret-tool`) and read a key: the keyring keeps
  keys out of plain files, not out of reach of the person's own processes
  ([design](superpowers/specs/2026-10-21-keyring-design.md)).
- `secret_import` (`crates/aria/src/jobs_socket.rs`) is a tool the model can call with no card; only the persona
  (`agent-files/SYSTEM.md`) tells it to offer first. It reads only files inside the session's project or the projects
  folder, judged on the file it opened (a regular file of at most 1 MiB; never a home dot folder or Aria's own folder). It
  trusts the `cwd` the extension sends, which only the sandbox protects: outside the box (macOS, or after an approved
  command) a process could forge it on the jobs socket. A project that is the home folder exposes its non-hidden files
  (macOS `~/Library`).
- `aria keyring off` writes back the `key:`, `mcp:` and `file:` accounts only; the person's own secrets (`user:<name>`)
  stay in the keyring, and a later save refuses with `keyring-off`.
- Claude Code's own credentials (the Claude sign-in through `claude-bridge`) stay where Claude Code keeps them.
- On Omarchy the keyring needs a running Secret Service (gnome-keyring); without one the dock keeps plain files and says
  why in its log. Not yet tried on the Omarchy machine. No Touch ID: the keyring opens with the session (the maintainer's
  choice, 2026-10-21).

## Integrations ([guide](guide/integrations.md))

Memory:

- The memory panel (`/memory`: list, search, filters, pin, forget with undo) is built but unfed: nothing calls
  `chat.setMemory` (only the UI sandbox does), so it always shows "No memories here yet". It needs a read of Engram's
  memories and the actions (pin, forget) written back; the contract is in `ui/chat/parts/memory.js`.
- Aria's memory starts with a single memory of its own (Aria saves it so Engram knows the project): its Engram database
  is its own (the maintainer's decision), so the memories in `~/.engram` are not in it. Importing them would be a new feature.

Connectors, sign-ins and the MCP import:

- Only Notion and Tavily are offered in the setup and settings; `rea`, OpenPencil and Figma, which the README and
  `ARIA-SETUP.md` name, are out of scope (see [parity](parity.md)). The settings' list of turned-on services
  (`integrations.services`) only drives the setup's screens: turning one off does not remove its entry from `mcp.json`.
- Once a Notion token record exists the dock has no Reconnect button; the terminal's `/signin notion` is the only way,
  and `/signin` accepts only `anthropic` and `notion`, so a Tavily key on a hub has no way in other than
  `hub init --credentials`.
- A connector card's preview shows the arguments as they are, not redacted; a secret in them is on the card.
- An imported server whose env var is empty in the other app reaches Pi without that variable. What Pi cannot read as the
  other app did is imported as it is and only warned about in the row ("check it after importing"): a variable in a
  command, argument, folder or URL, Claude Code's `${NAME:-default}`, an editor's `${workspaceFolder}`, a `~` argument,
  VS Code's `envFile`. The person fixes those in `~/.aria/mcp.json` by hand.
- The editor variables (`${userHome}`, `${workspaceFolder}`…) are translated whatever app the server came from, and
  Claude Desktop is assumed to expand `${X}` in env like Claude Code.
- In the import list, the checks are lost when the setup goes Back and returns to Integrations. After a "nothing to
  import" refusal the note shows on the Import button; when the new scan leaves nothing, there is no button and no note.
- The test `an_import_that_cannot_write_says_so` fails when the tests run as root (a read-only folder does not stop root).

Attachments (the paperclip, drop and paste; `~/.aria/attachments/`):

- Nothing removes old attachments: every file sent stays until the person deletes it.
- The thread shows an attached image as a chip (name and kind), never as a preview: the page never receives image data.
- The macOS dock has no navigation guard yet (GTK refuses anything outside `aria://`); the page stops file drops on both.
- A message with files that the dock refuses right after a run settled can come back without its files (the page keeps at
  most three messages' data).
- Paste and drop were checked in headless Chrome only, not on the real dock (WebKitGTK).
- A queued message that is re-queued keeps its path lines, but an image that went to the model as an image does not go
  again as one; two queued messages with the same text and different files can swap their chips until Pi's next queue
  update.

## Voice and face ([guide](guide/voice-and-face.md))

- Dictation on Linux builds Whisper with its Vulkan backend and falls back to the CPU when there is no driver (the installer's note; Aria's own code does not check); the
  microphone path there (cpal on ALSA/PipeWire) has not been run live yet.
- A denied macOS microphone is told by exact silence (cpal gives zeros, not an error). Not tried live in the dock yet: the
  first permission prompt from Aria itself, and whether macOS asks again after an update (Aria is not signed with a
  stable identity).
- No live text while speaking, no custom vocabulary (Supabase, Vercel…), no reading replies aloud, no dictation in the
  terminal.
- The face on macOS was checked live on one notched MacBook screen (the wing at the notch's right edge, 38 x 32 pt,
  under 1 % CPU); a click on the wing and a setup with an external display were not tried by hand. No setting hides the
  wing; no right-click menu or hover details. Omarchy's bar keeps its QML widget; `face_motion.rs` is not used there.
- The live trace (`/debug`): the wide surface was not seen live (on Omarchy the GTK side, `set_size_request` and the input
  region over both columns, was written on the Mac and not compiled there yet; on macOS the panel's frame change was
  type-checked but not opened by hand). A turn with a real model was not traced yet (the scratch home has no keys since they
  moved to the keyring). Only the current turn is kept: no saved traces, no comparing runs, no export. A subagent's own
  subagents appear only as far as they reach the session's watch socket.

## Install and releases ([guide](guide/getting-started.md))

- The install downloads from the GitHub releases of `i-montes/aria` (`install/install.sh`; the files of a release are made
  by `tools/release-assets.sh` or `.github/workflows/release.yml`), and until a release is made with those assets the
  one-line install has nothing to download and `--local` is the install that works. A release must carry `install.sh`,
  `pi-version`, `pi-packages`, `engram-version`, `agent-browser-version` and `node-version` next to the binary: the
  piped installer reads them from there.
- `.github/workflows/release.yml` has never run. Unchecked: the Ubuntu package names for the build (`glslc` and
  `spirv-headers` are the doubtful ones), whisper.cpp's Vulkan build there, and the cross-compile of the Intel Mac
  binary on an Apple silicon runner (Intel runners were not relied on). Its `publish` job publishes the Linux assets
  even when a macOS build failed. Its header says what to look at first.
- macOS binaries exist only for a release the workflow built them for (or whose maker ran `tools/release-assets.sh` on a
  Mac), so the installer's download of one may fail with a message that says to build from a clone. The Mac binaries are
  signed ad hoc by the installer after the download and are not notarised; none of this has been tried from a release on a
  Mac.
- The checksum is downloaded from the same place as the binary (the release), so it shows the download is whole, not that
  the release is authentic: whoever can replace the binary there can replace its `.sha256`, and a piped install also
  takes the pins' checksums (`engram-version`, `node-version`) from that release. Nothing is signed and no provenance is
  attached.
- The Linux release binary is built on Ubuntu 24.04 (glibc 2.39, the Ubuntu soname of GTK, WebKitGTK and the rest), so
  an older distribution does not run it; Omarchy's glibc and libstdc++ are newer and stay compatible with it. That it
  starts on Omarchy has not been tried from a release.
- The hub's binaries (`aria-x86_64-linux-headless`, `aria-aarch64-linux-headless`) are built by the workflow's `hub` job,
  natively on `ubuntu-24.04` and on `ubuntu-24.04-arm`, and by `ARIA_HEADLESS=1 sh tools/release-assets.sh`. That job has
  never run either (`ubuntu-24.04-arm` is unchecked), and no release carries them yet, so the one-line install of a hub
  works only after the next release is published. Until then a hub is installed from a checkout (`aria hub init
  --binary`). The dock's Linux binary still exists for x86_64 only.
- The installer's two questions, the guided screen and the AppArmor profile are tested with a terminal and a sandbox
  stood in by files and stubs (`install/install.test.sh`); none has been tried by a person at a real terminal, and the
  profile has not been loaded on a real Ubuntu 24.04 server. The profile is Ubuntu's own shape for one program
  (`abi <abi/4.0>`, `userns,`), so it needs that ABI file and nothing older.
- `aria hub init` run by the installer asks for the server's `sudo` password through the ssh session's terminal; a server
  whose `sudo` needs a password and whose ssh cannot give a terminal fails at the packages step and says what to run.
- The installer moves the private Node into place non-atomically.
- The uninstaller leaves the keyring items (the person's own secrets included), `~/.agent-browser`,
  `shell.json.bak.*` and system packages; it does not offer `aria keyring off` first.
- There is no CONTRIBUTING file and no CI that tests a change (the one workflow publishes a release and has never run):
  nothing says how an outside contributor sends a change.
- `tools/parity/suites.sh` does not run `aria-mac`'s tests or `tools/hooks/pre-push.test.sh` (`cargo test --workspace`
  runs the first).
- On Linux the control and clients sockets ignore `ARIA_HOME` (only macOS puts them under it), so a scratch run next to an
  installed dock meets the same socket; `aria dock` there exits "already running". See [development](development.md).

## macOS ([guide](guide/getting-started.md))

The dock, setup, chat, session tabs and installer run on macOS (`Option + Space`). Left:

- Click-through: the whole 700 px panel takes clicks; a click on the veil closes the dock instead of passing through to
  the app below.
- Desktop notifications come from `osascript`; clicking one does not open the dock (an unbundled app has no notification
  action). The chat still posts one when a prompt is hidden.
- There is no sandbox (every command asks) and no checks after edits.
- Checked on the Mac (2026-10-21): the dock runs, `cargo test --workspace` passes, and so does
  `install/install-mac.test.sh`. Still open:
  - A missing page file is served as an empty 200 (`dock/mac.rs`, logged as "missing page asset"); GTK answers NotFound.
  - The `aria-mac` hotkey test checks its own constants; the screen tests check the geometry only.
  - The installer gives up waiting for the old agent after 5 s, and `bootstrap` then fails after the binary was replaced.
  - The page reload after the web content process dies (`webViewWebContentProcessDidTerminate`, `dock/mac.rs`) is
    type-checked only: killing the WebContent process on a Mac is the check still to run.
- The chosen shortcut: `aria-mac` registers the `shortcut` setting with Carbon, falls back to `Option + Space`,
  `Control + Option + Space`, `Command + Option + Space` when `RegisterEventHotKey` refuses one, retries every 10 s, and
  registers again (new first, then the old one unregistered) when the setting changes. The key codes and the order are
  tested on Linux (`aria-mac/src/carbon.rs`) and the Carbon calls type-check (`tools/check-macos.sh`), but nothing of it
  has run on a Mac yet. Registration is exclusive, and the system's enabled shortcuts (`CopySymbolicHotKeys`) count as
  taken. To try there: Command + Space (Spotlight: it should fall back, the row saying "In use: …"), Control + Space (input
  sources), a combination another app holds, a change in the settings (the new one works at once, the old one no
  longer opens the dock), every candidate taken (the banner, "None available", a retry once one is freed), a fallback in
  use (freeing the chosen one: within 10 s the dock takes it back), and a letter on a non-US layout. A failed
  `InstallEventHandler` (`hotkey-handler-failed`, retried every 10 s) cannot be provoked on purpose. A later retry that
  fails for another reason keeps the first problem's banner.
- On Hyprland a combination already bound (or Aria's own) never reaches the page, so the Shortcut row cannot record it by
  pressing it; it can be typed in the field under the row. There is no retry timer there: a freed shortcut comes back only
  after a reload or a settings change.
- `crates/aria/src/dock/mac.rs` repeats the page queue, the open/close generations and the notice timing of
  `dock/gtk.rs`; they can be shared now that the Mac build is at hand.
- No `.app` bundle, signing or notarization, so the standard "open at login" UI and notification actions wait.
- The panel opens on the screen under the pointer and joins all spaces; multi-monitor and full-screen behavior needs
  manual testing on the maintainer's layout.
- `~/.local/bin` is not on the default macOS `PATH`, so the CLI is `~/.local/bin/aria toggle` (the hotkey does not need it).

## Parity with the maintainer's Pi

[parity.md](parity.md) walks every item of `ARIA-SETUP.md` with its check and its last result (2026-10-19 to 23). Left:

- The manual rows wait for the maintainer: Claude Max in the dock, Notion and Tavily, a subagent's `## Learnings`, the
  `ask_user` answer lifting read-only mode, and the `PI_*` switches through the dock's environment.
- Parity is accepted by the maintainer using Aria for real work, not by the matrix.

## Deliberate differences from the design

- Esc never stops a run (the maintainer's choice, 2026-10-19): it closes a picker or a panel, and otherwise closes the dock
  while the run goes on. The stop button stops it; the composer no longer shows "esc stop".
- The working folder row and the settings button (the maintainer's decisions).
- The header has no model button or meter (the maintainer's decision; the design has one beside the name): the model is the chip
  under the input; the quota chip beside the name and each tab's hover open the same quota card.
- Buttons and cards use solid fills; only the border takes the accent (the maintainer's taste). The message index's chips grow
  with the pointer but do not fade by distance.
