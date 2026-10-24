# Safety

How Aria keeps what the agent does inside limits you can see, and what it never does: the sandbox every command runs in, the cards that ask before something leaves it, the access level of each connector, your secrets and the keyring, the checkpoints behind undo, and the checks after edits. It is for anyone who lets Aria run commands and edit files on their own computer or on a server.

In short: Aria asks before it does something outside the project; it keeps your keys and passwords where the model cannot read them; and, in a terminal session in a project folder, it keeps a copy of the folder before each turn so you can go back. It is a set of guard rails, not a promise that nothing can go wrong. [What this protects against](#what-this-protects-against-and-what-it-does-not) says where the rails end, and [What leaves your computer](#what-leaves-your-computer) says where your text goes.

## The sandbox

On Linux every command the agent runs (the `bash` tool, a command started in the background, and the project checks after edits) goes through bubblewrap (`bwrap`). Inside that box:

- **It can write** to the project, to `/tmp`, and to package download caches that already exist (npm, bun, pip, uv, pnpm, Go, Cargo's registry and git folders, and a few more). Never to the `bin` or install trees of those tools: programs the host runs later are not a cache. You can add folders in Settings → Sandbox ("Extra folders Aria can write to").
- **The project** is the git repository the session works in (its root), or the session's folder when there is no repository. Your home folder and `/` are never a project. A session in the projects folder itself, when that folder is not a repository, can write the whole projects folder. That is where every session opened from a terminal on a [hub](hub.md) starts.
- **It cannot see** your credentials: `~/.ssh`, `~/.gnupg`, the cloud and container CLIs' folders (AWS, GitHub CLI, Docker, Kubernetes, Google Cloud), `.netrc`, `.git-credentials`, `.npmrc`, Claude Code's credentials, and Aria's own (`auth.json`, `mcp.json`, `mcp-auth.json`). It cannot see `$XDG_RUNTIME_DIR` either, which is where the desktop's sockets and the live copy of Aria's credentials sit, nor the Docker, Podman and system bus sockets, nor the X11 socket (`DISPLAY` is unset). `/sandbox` prints the exact list on your machine.
- **It can read** the rest of the disk. The box mounts everything read-only except what it hides.
- **It shares the network and the processes** with the rest of the system, so a dev server started in the background keeps running and a command can download things.
- **Some paths stay read-only even inside the project**: the repository's `.git/config` and `.git/hooks`, the project's `.pi/` and its `.aria/` (which holds the commands of [the checks after edits](#checks-after-edits)). That keeps a boxed command from planting a hook or changing what runs later.

Pi's own file tools (`read`, `write`, `edit`, `grep`, `find`, `ls`) run in Pi's process, not in the box, so the sandbox extension holds them to the same rules. They cannot read or change credentials, `write` and `edit` ask before writing outside the project, and a subagent started in a folder outside the project or given a credentials file as context is refused.

Some of Aria's own files are closed to the model's `write` and `edit` tools whatever you allow, "no sandbox" for the session included: its extensions, its persona and experts, `settings.json`, `aria.json` and the attachments folder. A model that could edit them could switch the sandbox off for every later session.

### When something is blocked: the permission card

When a command fails because of the box, the model may ask to run that one command outside it, and it says in a few words what the command does. You see a card:

- **A command outside the sandbox** shows the model's description as the title, the first lines of the command, and Jev's review of it.
- **A write outside the project** (`write` or `edit`) shows the path.
- **Without a working sandbox** every command shows this card, with a note that no sandbox is available. See [below](#where-the-box-does-not-work).

You have four choices.

| Choice on the card | What it does |
|---|---|
| Yes, just this once | Runs this command (or writes this file) and asks again next time |
| Yes, and don't ask again for `git status` this session | Allows those programs (with their subcommand, such as `git status` or `docker compose`) for the rest of the session. For a write it is the folder the file is in |
| Yes, no sandbox for the rest of the session | Nothing asks any more in this session |
| No | The command does not run; the model is told you said no |

A few details about the second choice:

- It names programs, not command lines. `git status` allowed does not allow `git push`.
- A command that can run anything (an interpreter, a package runner, a build tool, `ssh`, `sudo`, and similar) is never allowed as a program. The card offers "Yes, and allow this exact command this session" instead, and only that exact text is covered later.
- A command that hides another command (`$(…)`, backticks, `eval`, `sh -c`, `xargs`, `find -exec`) is asked about every time, unless you allowed that exact text.
- What you allow lasts for the session. A new session (`/new`, resume, a restart) starts with nothing allowed. Aria's own continuation in a fresh session file after a compaction keeps it. `/sandbox reset` forgets it at once.

Even with "no sandbox" for the session, the file tools still cannot read or write credentials. A command you let run outside the box can, because it is not in the box.

In the terminal the card is the same text; press the number of the choice.

### Jev's review

Next to a command outside the sandbox, Jev (the router's small classifier; see [the agent](agent.md)) gives a second opinion. It reads the command and the model's description and answers whether the description is accurate and complete, and whether the command does any of four things: deletes or overwrites data, sends data out of the machine, installs or removes software, or changes the system outside the project. The card shows one line, for example "matches the description" or "does NOT match the description", followed by the flags it raised.

The review is a short classifier call with a four-second limit. When no classifier can be reached, or the review is turned off, the card says "no review (unavailable)". Treat it as a prompt to read the command, not as a verdict. It is on by default; turn it off in Settings → Sandbox ("Jev reviews the command"). With neither an OpenCode Go nor a TypeSafe key stored, the router and this review are off ([settings](../settings.md)).

### Nobody to ask

Some runs have no one to answer a card: a subagent, a scheduled task running by itself, and a non-interactive Pi run. There, a command that needs to leave the box is refused, and the model is told so, with no card. The same holds for commands if you turn off "Ask before leaving the sandbox" in Settings → Sandbox (`sandbox.askToLeave`): leaving is then refused everywhere, not asked. A `write` or `edit` outside the project, an install and a connector write still ask where someone can answer.

### Where the box does not work

macOS has no bubblewrap yet, and a Linux machine may have none, or have one the system blocks (Ubuntu 24.04 restricts the user namespaces `bwrap` needs unless a profile allows it). Aria does not trust `bwrap --version`. It runs the real thing once per process: `bwrap --unshare-user --ro-bind / / true` must exit 0.

When it does not, **every command asks first**, with the same card and the same choices, and the card says "No sandbox is available on this machine: this command runs with full access." Nothing runs unconfined without you. On Linux the checks after edits do not run at all without a working box, and on macOS they do not run either.

A Seatbelt-based sandbox for macOS is not built; see [what is pending](../pending.md). On a server where the box does not work, see [the hub](hub.md#the-sandbox-on-a-server): the prompt then goes to whoever is attached.

### `/sandbox` and `PI_SANDBOX=0`

`/sandbox` prints the project, the writable folders, the hidden ones, what you have allowed outside the box this session, and whether `bwrap` works here. `/sandbox reset` clears the session's permissions.

`PI_SANDBOX=0` in the environment of Aria's service turns the sandbox extension off: no box, no cards, no review, no `/sandbox`. It is an escape hatch for testing, not a setting, and it takes more with it than the box. The closing of Aria's own files and the `$ARIA_SECRET_<NAME>` variables for commands (see [secrets](#your-secrets)) belong to the same extension, so they go too. Under systemd you would set it with `systemctl --user set-environment PI_SANDBOX=0` and restart the service, as [the parity notes](../parity.md) record.

### The process guard

The router also refuses commands that kill processes by pattern: `pkill`, `killall`, `fuser -k`, `kill $(pgrep …)`, and `… | xargs kill`. A `kill` with literal process ids is allowed only for processes of the current session or whose working folder is inside the project. The refusal tells the model how to find and stop its own process. It exists because an agent once stopped a dev server of another project. It applies to the conversation's own commands, not to subagents. It matches the command's text, so it stops the usual ways of killing by pattern and is not a barrier against a determined command.

## Connectors: what each may change

A connector is an MCP service such as Notion. Every call to one of its tools that **writes** is checked, whether the model calls the tool directly or from inside a script. Reads run without asking. Whether a call writes comes from the tool's own MCP annotations (`readOnlyHint`, `destructiveHint`); with none, from its name (a name starting with `search`, `fetch`, `get`, `list`, `read`, `query`, `find`, `retrieve`, `lookup`, `describe`, `view`, `show`, `count` or `check` reads); anything else, and any tool Aria cannot identify, counts as a write. Asking too often is the safe mistake.

Each connector has an access level, which you set in Settings → Integrations under "What each connector may change":

| Level | What a write does |
|---|---|
| Read only | Refused at once; the model is told you set it so |
| Ask (the default for every connector) | A permission card |
| Allow | Runs without asking |

A connector you have not set is on Ask. Aria's memory (Engram) is not a connector and is never checked.

On Ask the card reads "`<Connector>` wants to `<action>`", with the arguments below as `key: value` lines (long values cut). The action is the tool's name made readable ("create pages"). Choices: "Yes, just this once", "Yes, and don't ask again for `<Connector> <action>` this session", and "No". The session choice is per connector and action, and lasts as long as the session's other permissions do. The preview shows the arguments as they are; a secret inside them would be on the card ([pending](../pending.md)).

With nobody to ask, a write on Ask is refused. A [scheduled task](automation.md) is fixed when you create it: each connector it may use is `read` or `allow`, and one it does not list is read only. Nothing is asked in the task's tab.

The access levels are written into `~/.aria/agent/aria.json` from your settings (`integrations.access`; see [settings](../settings.md)). The extensions read it, and no secret goes in it. For connecting Notion, Tavily and other servers, see [integrations](integrations.md).

**Installing asks too.** The `install` tool (a skill, an MCP server or a Pi package you asked for, often from a link) puts nothing in until you confirm on a card that shows exactly what goes in, with secrets masked. The model cannot edit `mcp.json` itself, so it goes through this tool.

## Your secrets

Passwords, API keys and tokens you hand to Aria go to the keyring. The model only ever learns their names. Four things make that work.

**The model asks with a card.** When a task needs a secret, the model checks `secrets_list` (names and purposes only). If it is not there, it calls `request_secret` with a short name (`github-token`) and what it is for. You get a card with a hidden field. What you type goes straight to Aria's service, which keeps it in the keyring under `user:<name>`; the model is answered with the name only. A subagent and a scheduled task cannot ask, and a terminal connected to a hub cannot answer the card (the save is refused).

**A key you paste in the chat is caught.** In the dock's composer, before a message goes out, Aria looks for key shapes: OpenAI and Anthropic keys, GitHub, GitLab and Slack tokens, AWS access keys, Google API keys, JWTs, and assignments with a secret-sounding name such as `DB_PASSWORD=…`. If it finds one it holds the message and offers "Keep it safe" (the value goes to the keyring and the message goes out with `[secret saved as <name>: use $ARIA_SECRET_…]` in its place), "Send as is", or "Edit". The terminal has no such check.

**A key in a file is masked, and Aria offers to keep it.** Every tool output is masked before the model reads it: the values of your stored secrets (replaced by `[secret <name>]`), anything shaped like a key, and the value in an assignment whose name ends in `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD`, `PWD`, `CREDENTIAL(S)` or `AUTH`. The model gets a note saying what was masked and where, and the persona tells it to offer `secret_import`: Aria's service reads the file you name, takes the value of the variable (or the one key-shaped value if you give none) and keeps it in the keyring. The file still holds the key; Aria tells you so, and removing it is up to you.

**Commands use it as a variable.** A command that contains `$ARIA_SECRET_<NAME>` (`github-token` becomes `ARIA_SECRET_GITHUB_TOKEN`) gets that variable in its environment when it starts, read from the keyring at that moment. The model writes the variable's name, never the value.

Settings → Integrations lists your secrets by name, purpose and variable, with Delete (it asks twice), which removes the value from the keyring.

Know the limits of the masking. It matches the stored value as written and the shapes above; a command that transforms a value before printing it (encodes it, splits it) is not caught. A command given a secret can send it anywhere the network reaches. The point is that the key is not in the conversation, not that a command you let run cannot misuse it.

There is also a gap in the hiding of credentials. `secret_import` reads the file the model names, wherever it is, and keeps one value from it in the keyring, with no card and no sandbox path rule (only the persona tells the model to offer it first). A model that something it read has misled can point it at a file the sandbox hides (for example `~/.aws/credentials` or `~/.config/gh/hosts.yml`, with the name of the variable that holds the value) and then send the value out with a command that names `$ARIA_SECRET_<NAME>`. Where the keyring is off, as on a hub, `secret_import` keeps nothing and fails.

Without a keyring nothing can be kept: the card answers that the keyring is off, and on a hub there is none (see [the hub](#on-a-hub)).

## The keyring

Aria keeps keys in the system keyring, not in plain files under `~/.aria`: the Keychain on macOS (through Apple's `security` tool, so updates of Aria do not raise a permission prompt) and the Secret Service on Linux (through `secret-tool`; gnome-keyring on Omarchy). It is on by default (`credentials.keyring`).

| Where | What is there |
|---|---|
| The keyring, service `aria` (`aria:<root>` for a home other than `~/.aria`, so a test instance never meets your keys) | `key:<provider>`: provider API keys. `mcp:<server>:header:<name>` and `mcp:<server>:env:<name>`: secret values of connectors. `file:auth.json` and `file:mcp-auth.json`: those two files, sealed whole. `user:<name>`: your own secrets |
| `~/.aria/auth.json`, `~/.aria/mcp.json` | A `!aria secret --home <root> <account>` command in place of each key. Pi runs it when it needs the key |
| A private folder outside `~/.aria` while the service runs: `$XDG_RUNTIME_DIR/aria-secrets-<id>/` on Linux (in memory), the per-user temporary folder on macOS | The live copies of `auth.json` and `mcp-auth.json`. Pi reads and rewrites these; the service seals every change back within a second and deletes them when it quits |
| `~/.aria/secrets.json` (mode `600`) | The names of your own secrets, what each is for and when it was saved. Never a value |

The first start with the keyring on moves your existing keys in. Each value is read back and compared before its file changes, and a step that fails leaves the files as they were. Claude Code's own credentials (the Claude sign-in) stay where Claude Code keeps them.

### Commands

| Command | What it does |
|---|---|
| `aria keyring status` | Prints whether the keyring is on, whether it is available, and the accounts it holds (names only) |
| `aria keyring off` | Writes every key back into plain files (mode `600`) in `~/.aria`, turns `credentials.keyring` off and removes those keys from the keyring. The secrets you keep for commands (`user:<name>`) are left where they are. Quit Aria first: `aria quit` |
| `aria keyring on` | Turns it on; the service moves the keys at its next start (`aria quit && aria start`) |
| `aria secret --home <root> <account>` | Prints one value. It is what the `!aria secret …` commands in the files run; you do not need to type it |

### When there is no keyring

Where no keyring answers (a server, or a Linux desktop with no Secret Service running), or with the setting off, Aria runs as before: plain files in `~/.aria`, mode `600`, and its log says why. A home whose keys are already in the keyring but which cannot reach it starts without its credentials, and says so in its log, rather than falling back to empty files. A hub runs with the keyring off by design; see [the hub](hub.md).

The keyring keeps keys out of plain files and backups. It does not keep them from your own processes: a command inside the sandbox can still run `aria secret` or the keyring tools ([pending](../pending.md)). The card and the variables above are about the model, not about malware.

## Checkpoints and undo

Before the message that starts a turn, Aria's service saves the session's folder in a private git repository of its own, `~/.aria/checkpoints/<session>/`. It never touches the project's own `.git`. A message you queue behind a run that is going does not get one when you send it; Pi starting it takes one only if the turn that was going has ended by then. When the turn ends, the service compares the folder with that snapshot, so the terminal can print one line after the answer ("3 files changed +40 −12") and offer to go back.

What a checkpoint leaves out:

- everything the project's `.gitignore` ignores (also the ones above the folder, so a `node_modules` ignored at a monorepo's root is not kept);
- the project's `.git`, and any file over 5 MB;
- the folder itself when it is your home folder, the projects folder (a session there keeps no code checkpoints; this is the case for every session started from the dock's `+`, and for every session opened over `aria connect` to a hub), a system folder (`/etc`, `/usr`, `/var`, `/opt`, `/System`, `/Library` and the like), a mount point, or a folder with more than 20,000 files (it then stops keeping checkpoints for good). A folder under `/tmp` is fine.

A command such as `/plan` or `/model` takes no checkpoint, since it changes no files. Undo can go back through the last 50 checkpoints of a session, and Aria deletes a session's repository when you close the session.

**Undo** is in the terminal: press `Esc` twice, or type `/undo`, and pick a message. You choose to restore the code and the conversation, the conversation only, the code only, or nothing. Restoring code puts back files that changed since the checkpoint, deletes files made since (only ones created after it), and brings back files deleted since. Nothing else is touched: ignored files, big files and the project's git history stay as they are. Restoring the conversation continues it from before that message, with the message's text back in the editor. Where no code is kept (the projects folder, a system folder, a folder with too many files), only the conversation can go back and the screen says why. Undo is refused while a run is going. [The terminal guide](terminal.md) has the keys.

The dock takes the same snapshots, but it does not show the changes line or an undo. A session started from the dock's `+` works in the projects folder, where no snapshot is kept, so there is nothing to go back to in the default way of using the dock.

Undo is about files in the folder. It cannot take back a `git commit` or push, a connector write, a message sent, something installed outside the folder, or a command you let run outside the sandbox.

### Let Aria work on one project safely

Use this when you want Aria to reorganise or tidy a project and be able to go back:

1. **Commit first** (or stash). Undo restores files, not git history, and it skips what `.gitignore` ignores and files over 5 MB. A tidy-up that deletes `node_modules` or build output cannot be undone with it.
2. **Open the session from inside the project.** In a terminal, `cd` to the project's top folder (the one that holds `.git`), run `aria`, and choose the last row, `+ new in <folder>`. The session works in that folder: its sandbox writes only the project (its git repository, or the folder when there is none), `/tmp`, the tool caches and the extra folders you allowed, and Aria keeps checkpoints of the folder you ran it from. A session opened from a subfolder is checkpointed only there, while its commands may still write the whole repository.
3. **Do not use the dock's `+`** for this. It opens a session in the projects folder (`~/Projects` by default), where the sandbox can write the whole folder, every project in it, and no checkpoint is kept. Setting `sandbox.projectsDir` to the project does not change that: the projects folder never gets checkpoints.
4. **Expect no cards inside the project.** With a working sandbox, moving, renaming and deleting files in the project run without asking, because the project is writable. Cards are for what leaves it ([the card](#when-something-is-blocked-the-permission-card)).
5. **Go back with `Esc Esc`** (or `/undo`) once Aria is idle, pick the message before the change and restore the code, the conversation, or both.

## Checks after edits

When a run that edited files ends, the `verify` extension runs the project's checks, in the box, at most five minutes each. The commands come from `.aria/verify.json` (`{ "lint": "…", "types": "…", "test": "…" }`) when the project has one, otherwise from what the project shows: the `lint`, `typecheck` and `test` scripts in `package.json` (with `bun run` when there is a `bun.lock`, else `npm run`), `cargo clippy` and `cargo test` for a `Cargo.toml`, `ruff check` and `pytest` for a `pyproject.toml`. They run in order and stop at the first failure.

A pass shows in the terminal's footer. A failure shows there too, and the model is told once, with the command and the end of its output, and asked to fix it. A second failure in a row is shown, not fed back again. Subagents do not check, and nothing is checked after you stop a run.

`/verify off` turns it off for the session and `/verify on` back on. The model cannot write `.aria/`, so it cannot change which commands run. On macOS, and on Linux where the box does not work, the checks do not run, since there would be no box and nobody to ask.

## On a hub

A terminal that connects to a hub over ssh gets a narrower set of powers than a terminal on the same computer as the service. It cannot save a secret and it cannot run a `!command`; both are refused. It cannot choose a session's folder, only reaches sessions the hub already has, and of Pi's own commands may send only `get_session_stats`. It cannot read or change settings or keys through the protocol, though commands typed in a session, such as `/autocompact` and `/effort`, still write their own setting.

Permission cards still reach it, since otherwise nobody could answer one on a hub. So whoever holds the client key can say yes to a command outside the sandbox, a write outside the project, an install or a connector write, and can approve a scheduled task that waits for approval. **Treat a stolen client key as a shell as the hub's user.** With a working sandbox the model has to raise the card and the key approves it; without one, every command is such a card. What an attacker can then read and reach, and how to protect the key, is in [what a stolen client key can do](hub.md#what-a-stolen-client-key-can-do). That is why `aria hub init` does not install the key unless the sandbox works on the server or you pass `--allow-no-sandbox`. The hub runs with the keyring off, so its keys are in mode `600` files: a command inside the sandbox cannot see them, a command you let out of it can.

## What leaves your computer

Aria runs on your computer, but the model does not. This is every network call Aria itself makes, from the code, with what it carries and when. The programs it starts (Pi, the MCP servers, Engram, the browser) have traffic of their own that Aria does not control; the table names what it can.

| What | Goes to | When |
|---|---|---|
| Your messages, the files and command output the model reads, the images you attach | The provider of the model you chose: OpenCode Go, Anthropic with an API key, or Anthropic through Claude Code for a Claude plan. Pi makes these calls; their addresses are not in Aria's code. For OpenCode Go's free models the provider keeps prompts for 30 days (what Aria's confirmation says) | Every turn |
| The router's questions: your message (up to its first 8,000 characters) and the end of the previous answer (2,000); in a read-only turn a command that may write (6,000 characters and its folder); a delegated task's text (6,000); in plan mode, whether your message goes on with the plan (6,000 characters of it and 2,000 of the previous answer) | The first Jev classifier Pi can reach: OpenCode's free Jev when an OpenCode Go key is stored, even if Claude is your main provider; with only a TypeSafe key, TypeSafe's (Aria's key check for it uses `api.typesafe.ai`; the classifier calls themselves go through Pi). A classifier that is out of quota or funds is skipped, and the same question goes to the next one Pi can reach | Each turn, while the router is on (it needs an OpenCode Go or a TypeSafe key; [the agent](agent.md#the-router-and-jev)) |
| Jev's review of a command: the command (up to 6,000 characters) and the model's description of it (500) | The same classifier | Only when a card asks to run a command outside the sandbox, and "Jev reviews the command" is on (the default) |
| A quota reading, with your OpenCode Go key or your Claude sign-in token as the credential | `opencode.ai` (the key), `api.anthropic.com` (the token). Nothing is read for an Anthropic API key | For the bars: when a session loads, after a run and when you change the model; apart from a model change, at most once every 20 seconds per session. Separately, with an OpenCode Go key stored (whatever your main provider), the `quota-fallback` extension reads the Go windows with that key before each run (an answer is reused for 60 seconds), and `/plan` does too |
| A key check: the key you typed, in one request | `opencode.ai`, `api.anthropic.com`, `api.tavily.com` or `api.typesafe.ai`, by the key's kind; a Claude sign-in is checked at `api.anthropic.com` | When you save a key (the setup's "Check and save", the `/connect` card), and when the setup finds a credential already stored and checks it before going on |
| A sign-in: Claude's, then Notion's | `claude.ai` and `platform.claude.com` for Claude; Notion's sign-in service, found from `mcp.notion.com`, for Notion | When you sign in; while it waits Aria listens on the loopback address of the computer the service runs on |
| What the model asks of Notion or Tavily: search queries, pages it reads or writes, with your key or sign-in | `mcp.notion.com`, `mcp.tavily.com` | When the model uses them; a write follows the connector's access level |
| What the model sends to an MCP server you added or imported | That server's own address, or the command that is the server | When the model uses it |
| A skill, a package or a server you ask it to install | The link you gave, or the package's host (npm, a git host). A skill's link is fetched before the card, which then shows what it holds; a package is downloaded after you confirm; a server is only added to the list, and what it contacts is its own | When you ask for an install; nothing is kept without the card's yes |
| Pages the browser check opens | Whatever address the agent gives it | When an agent checks a page it built |
| The voice model, one file of about 574 MB | `huggingface.co` (whisper.cpp's repository), and wherever it redirects the download. Nothing of yours goes with the request | Once, after you agree on your first dictation |
| The installer's downloads | Pi from the npm registry; Pi's packages (`pi-packages`) from npm and `github.com`; Engram from `github.com`; on a server with no recent Node, Node from `nodejs.org`; the agent browser from npm and its Chrome from the host `agent-browser` chooses; the Claude provider package (about 300 MB) from npm when you connect a Claude plan in the dock; the system packages it asks `apt-get` or `pacman` for. A hosted `curl \| sh` install would also download the binary and the pin files from its release address (none exists yet) | When you install or update, on a hub from the server itself; the pinned versions and checksums are in the repository's pin files |
| Your settings and your keys, as values | The server you name, over ssh | When you run `aria hub init`; the terminal's conversation with a hub also travels over ssh ([the hub](hub.md)) |
| Whatever a command sends | Anywhere the network reaches: the sandbox shares the network ([the sandbox](#the-sandbox)) | When a command you or the model run does it |

What never leaves:

- **Your voice.** Dictation records in memory, transcribes on your computer and drops the audio ([voice](voice-and-face.md#dictation)). The text you dictate is a message like any other once you send it.
- **Your keys' values, to anyone but their own service.** The model never sees a stored key, and the keyring is not uploaded. A key goes to the provider it belongs to as the login of a call, and `hub init` copies it to your own server. [Your secrets](#your-secrets) says where the masking stops.
- **Your files, unless the model reads them.** Nothing in Aria uploads a folder. What the model reads in a turn goes to its provider in that turn, and a boxed command can send what it reads ([what this protects against](#what-this-protects-against-and-what-it-does-not)).

Pi's own check for a newer version is switched off in every Pi Aria starts (`PI_SKIP_VERSION_CHECK`).

## What this protects against, and what it does not

Aria is built for an agent that is helpful and sometimes wrong, or one that something it read (a web page, a file, a connector's reply) has misled. The guards keep a wrong step small and visible: it cannot change files outside the project, read your credentials, write to a connector, kill your other processes or learn your keys without you seeing a card or a refusal, and you can go back to before a message. The exceptions are in this guide: a card you answer yes to, `secret_import`, and a scheduled task you approved.

It does not make the agent safe to leave alone with a machine you cannot afford to lose. The sandbox shares the network and can read most of the disk, so a boxed command can still read your other projects and send them somewhere. Anything you allow outside the box, or on Allow for a connector, runs with full access. The specs do not claim the sandbox holds against a determined attacker, and the pending list has known gaps in it, such as the repository's `.git` outside `config` and `hooks`, and the keyring's reach. Treat it as a seat belt. On macOS, or wherever the box does not work, the only guard is you answering each card. Disk encryption, the machine's own accounts and who can log in are not Aria's business.

## Limits

- No sandbox on macOS yet; every command asks. See [pending](../pending.md).
- The sandbox does not limit the network or the processes, and the rest of the disk is readable.
- A command you allow outside the box is not protected by it, including against changes to `aria.json`, the settings file or the attachments folder.
- Old attachments are never removed.
- Masking works on values as they are written.
- `secret_import` can keep a value from any file the service can read, with no card (see [your secrets](#your-secrets)).
- On a hub, whoever holds the client key can answer every card ([what a stolen key can do](hub.md#what-a-stolen-client-key-can-do)).
- Settings under `git.*` (ask before committing, pushing or touching someone else's changes) are instructions in the persona. The sandbox does not enforce them.
- The dock draws no undo, no changes line and no pass or fail line for the checks; the terminal does.
- The full list is in [pending](../pending.md).

## See also

- [The agent](agent.md): the router, Jev, experts and subagents.
- [Integrations](integrations.md): connecting Notion, Tavily and other MCP servers, attachments, memory.
- [The hub](hub.md): running Aria on a server, and what a client key can do.
- [Scheduled tasks and background work](automation.md): the fixed permissions of a task.
- [Where everything lives](../reference/data-and-paths.md): the files named here and which are secret.
