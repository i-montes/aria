---
name: install-from-link
description: Install a skill, an MCP server or a Pi package the user asks for, usually by pasting a link ("instala esto", "agrega este MCP", "install this skill"). Use whenever the user wants something installed or connected to you, with or without a link.
---

# Install from a link

Everything goes in through the `install` tool. The user confirms each install on a card, so do not ask "should I install it?" first. Never write `mcp.json`, Pi's `settings.json` or the `skills/` folder yourself, and never run `pi mcp add` or `pi install` through bash. The sandbox blocks them, and the tool is the only way in.

## 1. Read the link

Open the page: a GitHub URL with `curl` (`https://raw.githubusercontent.com/<owner>/<repo>/HEAD/README.md` for a repository's README), anything else with the web tools you have. Work out which of these it is:

| What you find | `kind` | What to pass |
|---|---|---|
| A `SKILL.md`, a folder with one, or a repository of skills | `skill` | `source`: the link as the user gave it (a repository, `/tree/…` folder or `/blob/…/SKILL.md`), or a raw `SKILL.md` URL |
| An MCP server reached by URL ("remote", "hosted", "streamable HTTP") | `mcp` | `name`, `url`, and `headers` for a key |
| An MCP server that runs locally (`npx`, `uvx`, `docker`, a binary) | `mcp` | `name`, `command` (one word), `args`, and `env` for keys |
| A Pi package (its README says `pi install …`) | `package` | `source`: exactly what `pi install` takes (`npm:<name>@<version>`, `git:github.com/<owner>/<repo>@<ref>`) |

If the page is about something else (an app, a library, a blog post), say what it is and that there is nothing to install. Do not guess.

## 2. Fill in what the install needs

- **Skills:** when a repository holds several skills and the user named one, pass its name in `skills`. Otherwise every skill in it is installed, and the card lists them.
- **MCP name:** short and lowercase, after the service (`github`, `linear`, `rea`).
- **MCP config:** copy the server's documented configuration (the `mcpServers` JSON block most READMEs show for Claude Desktop or Cursor). Its `command`, `args` and `env` map one to one. A remote server's URL usually ends in `/mcp`. Pi does not support SSE, so when the README only shows a `/sse` URL, try the same host with `/mcp`.
- **Keys:** if the server needs an API key or token the user has not given you, ask for it with `ask_user` (they can type it as their own answer) before calling `install`. Never invent a value or leave a placeholder. A remote server takes its key in `headers` (usually `{"Authorization": "Bearer <key>"}`), a local one in `env`.
- **OAuth:** a remote server that signs in through the browser (the README says OAuth or "sign in", and has no key) gets `signIn: true`.
- **When the README offers both** a remote URL and a local command, use the remote one. It needs nothing installed and runs no code on this machine.

## 3. Call `install`, then report

Call `install` once per thing. After it, tell the user in one or two lines what was installed and when it is ready:

- a skill: in the next session (to use it now, read the `SKILL.md` path the tool gives you);
- a new MCP server: now, with its tools named `mcp__<name>__<tool>`; a replaced one, in the next session;
- a Pi package: in the next session.

If the user cancelled the card, say nothing was installed and stop. If the tool reports an error, say what failed in one line and what would fix it.
