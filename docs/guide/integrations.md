# Integrations

What Aria connects to: connectors such as Notion and Tavily, MCP servers you already use in other apps, files you attach to a message, its memory, and the browser its agents use to check the pages they build. This guide is for anyone setting these up or wondering where a file or a key went.

## Connectors

A connector is an MCP server: a service that gives the agent extra tools. Aria keeps them in `~/.aria/mcp.json`, in the format Pi (the engine) reads, so a server you add by hand works as it would in Pi:

```json
{
  "mcpServers": {
    "notion": { "url": "https://mcp.notion.com/mcp" },
    "docs": { "url": "https://docs.example.com/mcp", "headers": { "Authorization": "Bearer ..." } },
    "files": { "command": "npx", "args": ["-y", "@acme/files-mcp"], "env": { "ACME_TOKEN": "..." } }
  }
}
```

An entry is either remote (`url`, optional `headers`) or local (`command`, `args`, `env`). The file is mode `600` and Aria writes it atomically. A file that does not parse is never replaced: Aria reports "Aria could not write ~/.aria/mcp.json or mcp-auth.json" and leaves it as it is, so fix it by hand first. OAuth state (the Notion sign-in) lives next to it in `~/.aria/mcp-auth.json`. With the system keyring on, the headers of remote servers and the secret-looking `env` values of `mcp.json` become commands that read them from the keyring, and `mcp-auth.json` is sealed whole in the keyring (the service keeps a live copy outside `~/.aria` while it runs); see [safety](safety.md) for how that works and what the model never sees.

### What a connector may do

Each connector has an access level, set in **Settings → Integrations** under "What each connector may change", one row per server in `mcp.json`:

| Level | A tool that changes something |
|---|---|
| Read only | refused, without asking |
| Ask (the default) | shows a card first; you allow it once, allow it for the session, or say no |
| Allow | runs without asking |

Tools that only read always run. A server not listed is at Ask, and `engram` (the memory) has no row because it is never gated. [Safety](safety.md) explains how Aria tells a read from a write and what the card shows.

### Notion

Notion signs in through your browser. You do not paste a key.

1. Open the setup's **Integrations** step (the full "Customize" setup, not the quick one) or **Settings → Integrations**, and turn on Notion.
2. Press **Sign in with Notion**. Aria opens your browser, where you choose what Aria can see in your workspace.
3. When the browser says "Signed in. You can close this tab and go back to Aria.", the row shows "Connected".

The row says "Waiting for your browser…" meanwhile, with **Open the sign-in page again** and **Cancel**. The sign-in gives up after five minutes ("Sign-in took too long"). Nothing is written unless the whole sign-in succeeds.

Under the hood Aria does what the MCP authorization spec asks. It finds Notion's sign-in from the server's own answer, registers itself as a client there (dynamic client registration, so there is no Aria-owned client id or secret), sends you to the browser with PKCE, and receives the code on a loopback address on your computer. The tokens are stored in `mcp-auth.json` and Notion is listed in `mcp.json`.

**Reconnecting.** The dock shows "Connected" as long as a token record exists, and it has no reconnect button. The stored record holds a refresh token, and refreshing it is Pi's job, not Aria's (Pi is not in this repository, so this guide cannot say how well it does it). If Notion stops answering because the sign-in is no longer valid, run `/signin notion` in the terminal client (see below): it starts a new sign-in at any time and replaces the stored record.

**On a hub.** A hub has no browser, and the loopback address the browser returns to is on the hub, not on the computer where you sign in. Use the terminal client connected to the hub:

```text
/signin notion
```

The terminal shows an address. Open it in any browser, sign in, and the browser then fails to load a page at `localhost`. Copy that page's address (or only what follows its `?`) from the address bar, paste it into the terminal and press `Enter`. The hub takes the code from it. What you paste is never drawn and never reaches the conversation. See [the hub](hub.md) for the rest of connecting to a hub, including how `aria hub init --credentials` sends the connectors you already set up on your own computer.

### Tavily

Tavily is web search and page extraction for the agent. It uses an API key.

1. In the same Integrations step or settings section, turn on Tavily.
2. Paste the key (you can get one at app.tavily.com) and press **Check and save**.

Aria checks the key against Tavily's usage endpoint, which answers 401 (or 403) for a bad key and costs no credits. Only a key that passes is saved: in `mcp.json` as the `Authorization` header of the `tavily` entry, or in the keyring with the keyring on. A rejected key writes nothing. If Aria cannot reach Tavily it says so ("Could not reach the service") and also writes nothing.

### Importing the servers your other apps have

If you already set up MCP servers in Claude Code, Claude Desktop, Cursor or VS Code, Aria can copy them. The list is in the same Integrations screen, under "Your other MCP servers": Aria scans when the screen opens and shows a checkbox per server, with where it came from, what it runs (the command's name or the address's host) and the names of the secrets it uses (never their values).

| App | File read | Key |
|---|---|---|
| Claude Code | `~/.claude.json` (user scope only; per-project servers are not imported) | `mcpServers` |
| Claude Desktop | `~/.config/Claude/claude_desktop_config.json` on Linux (`$XDG_CONFIG_HOME` is honoured), `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS | `mcpServers` |
| Cursor | `~/.cursor/mcp.json` | `mcpServers` |
| VS Code | `~/.config/Code/User/mcp.json` on Linux, `~/Library/Application Support/Code/User/mcp.json` on macOS | `servers` |

Those files are only ever read, never written. Nothing is imported until you check servers and press **Import**; the checked ones are then copied into `~/.aria/mcp.json`, nothing else in it changes, and every session's Pi restarts when its run settles so it sees them. Some details:

- **stdio and http servers are importable.** An SSE server is listed as "Not supported (SSE)" and never imported, because Pi does not support the SSE transport.
- **Names.** A name Pi would refuse (anything other than letters, digits, `_` and `-`) is shown as `my docs → my-docs`. A server Aria already has under that name, and anything called `engram`, is not offered. Two servers that end up with the same name (the same name in two apps) both import: the second gets `-2`, then `-3`…
- **Disabled servers** stay disabled: "disabled there" in the list, imported with `enabled: false`.
- **Secrets** in `env` or `headers` are copied as they are into `mcp.json` (mode `600`, or the keyring when it is on). The page only ever receives their names.
- **Variables.** Pi expands `${NAME}` only in `env` and `headers`. Aria translates what it can on the way in (`${env:NAME}` becomes `${NAME}`, `${userHome}` becomes your home folder, a value starting with `!` is kept as text so nothing runs). What it cannot translate is imported anyway and the row says "check it after importing":

| Warning in the row | What it means |
|---|---|
| fills in NAME in its command, arguments, folder or address | the other app expands a variable there; Pi does not, so the text stays as it is |
| gives NAME a fallback value | Claude Code's `${NAME:-default}`; Aria does not apply the default |
| uses `${workspaceFolder}` (or another editor variable) | Aria has nothing to put there |
| an argument starts with `~` | Pi reads it as your home folder; the other app passed it on unchanged |
| its env file is not imported | VS Code's `envFile`; add those variables yourself |
| asks for `input:<id>` in VS Code | VS Code prompted for the value; set it after importing |

Fix these in `~/.aria/mcp.json` by hand. If Aria's `mcp.json` does not parse, nothing is imported and the screen says so.

### Installing a connector or skill from a link

You can also ask Aria in the chat: paste a link and say "install this". Aria's `install-from-link` skill tells the agent to read the page, work out what it is, and call its `install` tool. It handles:

- **A skill** (a `SKILL.md`, a folder or a repository holding skills): copied into `~/.aria/agent/skills/`; it appears from the next session.
- **An MCP server**, remote (a URL) or local (`npx`, `uvx`, a binary): written into `~/.aria/mcp.json`, and for a new server it connects right away. If the server signs in through the browser, Aria starts that sign-in after installing. When a README offers both, the agent prefers the remote one. If the server needs a key you have not given, the agent asks you for it first.
- **A Pi package**, from what `pi install` takes (`npm:` or `git:` sources).

You confirm each install on a card that shows exactly what goes in, with secrets masked, so the agent never asks "should I install it?" first. The agent never writes `mcp.json`, Pi's settings or the skills folder itself; the sandbox keeps it out of them on purpose, and the `install` tool is the only way in. Subagents have no `install` tool, and where nobody can confirm (no dock or terminal attached), nothing is installed. A package can bring extensions that run inside Aria with your permissions, and the card says so.

## Attachments

Hand Aria a screenshot, a PDF or any file without saving it somewhere and typing its path.

**In the dock** you can attach three ways: the paperclip next to the input ("Attach files"), dragging files onto the chat, or pasting an image into the input. Attached files show as removable chips above the input and, after sending, as chips in your message (name and kind, not a preview).

- **Limits:** at most 10 files per message, 20 MB each, 50 MB together. The dock shows a note when a file is left out, and the dock refuses beyond them too.
- **Where they go:** every file is saved under `~/.aria/attachments/<session>/<yyyy-mm-dd>/<n>-<name>` (mode `600`, folders `700`). The saved name keeps letters, digits, `.`, `-` and `_`, and is cut at 80 characters.
- **What the model gets:** your text followed by one line per file, such as `[attached: report.pdf (application/pdf, 1.3 MB) at <path>]`, so the model opens it with its tools. A PNG, JPEG, GIF or WebP also goes to the model as an image when the session's model can see images, with a line `[attached image: shot.png at <path>]`. With a model that cannot see images it is a path line like any other file.
- **Who can touch them:** the sandbox lets the model read that folder and never write in it.
- **Queued messages** keep their files.

Nothing removes old attachments: they stay under `~/.aria/attachments/` until you delete them. Paste and drop were checked in a headless browser, and not yet on the real dock on Linux (WebKitGTK), so if file managers or the clipboard hand files over differently there, use the paperclip. See [pending](../pending.md) for the rest.

**The terminal and a hub.** The terminal client sends text only: it has no way to attach a file. On a hub there is no dock page, so no paperclip either. Type a path in the message instead; the path has to exist on the computer where the service runs, which on a hub is the hub. See [the terminal](terminal.md).

## Memory

Aria remembers decisions, preferences and solved problems between conversations. The memory is Engram, a small separate program that Aria runs as one of its connectors, and it is on by default.

**It is Aria's own.** The installer puts the pinned Engram (`engram-version` in the repository) in `~/.aria/bin/engram`, and Aria's memories live in a database of their own under `~/.aria/engram`. An Engram you may already use on the computer is a different database and is never read or changed: if your `mcp.json` has an `engram` entry that is not Aria's, Aria leaves it alone. Importing the memories of another Engram is not built.

**What it stores.** The agent saves one memory per fact, with a short searchable title: a preference or a correction you gave, a decision and its reason, a problem it solved, a lasting fact about a person, a project or a date. It does not save small talk, files it can read again, or passwords, keys and tokens; for a secret you keep in Aria it saves the name and what it is for, never the value. When it finishes a significant task, or you say goodbye, it saves a summary.

**How Aria uses it.** At the start of a conversation the agent reads the recent context. Before answering, it searches memory when the router marked the message `memory=yes` (or, without a router, when the message touches something discussed before), then treats what it finds as context, not as an order: what you say now wins, and it updates the memory. Subagents can read memory and cannot write it. Search matches words, not meaning; the settings screen says so ("until you add embeddings").

**Settings.** **Settings → Context and memory** has the switch ("Remember with Engram") and a project name (`aria` by default), the name memories are filed under. Turning memory off removes Aria's `engram` entry from `mcp.json` and the agent forgets between conversations; it does not delete the database. Before each Pi start, a project with no memory gets a first one ("Aria's memory"), because Engram only knows a project once something is saved in it. A hub has its own Engram, as a hub is its own service; memory is not shared between a hub and your computer.

**The memory panel.** `/memory` in the dock opens a panel designed to list, search, pin and forget memories. Nothing feeds it yet: it always says "No memories here yet", whatever Aria has saved. Until it is built, ask Aria what it remembers. See [pending](../pending.md).

## The browser check

The agents build web pages, and without a browser they would say "done" without having looked. Aria installs `agent-browser`, a browser driver that opens a headless Chrome and gives the model the page as a list of its elements, together with the `verify-in-browser` skill. The skill's rule is: after building or changing anything a browser shows, open it and check it before saying it works. If the agent cannot (no browser, nothing it may run, a sign-in or captcha in the way), it says so instead of claiming it.

What it checks, with real measurements: which controls exist and how they are labelled, whether the page fits a phone-width viewport without horizontal scrolling, computed colours and sizes for contrast, whether a flow works after clicking and filling in, errors in the console, and a screenshot for what the element list cannot show. A page behind a sign-in or a captcha is yours; the agent stops there.

**Where it is installed.** The installer puts the pinned `agent-browser` (`agent-browser-version`) under `~/.aria/agent-browser/<version>`, links it as `~/.aria/bin/agent-browser` (on Pi's path), and runs `agent-browser install`, which downloads a Chrome for Testing into `~/.agent-browser/browsers`. That download is large. A failure of either step is a warning, not a failed install: Aria works without the browser check, and `~/.aria/bin/agent-browser install` retries the download. The skill is written into `~/.aria/agent/skills/verify-in-browser/`.

**On a server.** A headless install (`install.sh --headless`, see [the hub](hub.md)) skips the browser, because Chrome is a large download and not available everywhere. Pass `--with-browser` to include it:

```sh
sh install.sh --headless --with-browser
```

Without it, an agent on the hub that is asked to check a page says it could not.

**Platforms.** Inside the sandbox each command is a fresh box, so one check is one command, with the session closed at the end. On macOS there is no sandbox yet and every command asks first.

## What is deliberately not there

The setup offers Notion and Tavily; those are the only two connectors Aria signs in to for you. Other services you add as MCP servers (by hand, by import or by link) work like any connector and get an access row. The owner's Figma, OpenPencil and `rea` integrations from the original setup are out of scope: the setup does not offer them and the terminal's `/signin` takes only `anthropic` and `notion` (see [the parity matrix](../parity.md)). Beyond that:

- Turning a service off in the settings does not remove its entry from `mcp.json`; edit the file to remove one.
- Aria has no MCP reconnect of its own: a connector that failed to start waits for its next use or a Pi restart.
- Pi packages come only from a local folder (`integrations.packages.path`); npm, git, a gallery and trusting third-party packages are not offered in the setup, though an agent can install a package for you by link.
- The memory panel is unfed (above), and the dock cannot attach files on a hub or in the terminal.

The full list is in [pending](../pending.md).

## Keys and commands

| Where | What it does |
|---|---|
| Settings → Integrations | turn Notion and Tavily on, connect them, set each connector's access level, import servers from other apps, list and delete your saved secrets |
| Settings → Context and memory | memory on or off, the memory project name |
| Paperclip, drag and drop, paste an image | attach files to the next message (dock) |
| `/signin notion` | terminal client: sign in to Notion with a pasted redirect address, locally or on a hub |
| `/memory` | dock: open the memory panel (empty until it is fed) |
| "install this" and a link | the agent installs a skill, an MCP server or a Pi package, after your confirmation |

## See also

- [Safety](safety.md): connector access levels, the sandbox, secrets and the keyring
- [The hub](hub.md): signing in to Notion and keeping connectors on a server
- [The agent](agent.md): the router, experts and skills that use these connectors
- [The dock](dock.md): the composer and the settings screen
- [The terminal](terminal.md): `/signin` and the commands you have without a dock
