# Agent files

The owner's persona and experts, from his Pi (`~/.pi/agent/`). Aria embeds them and writes them into its own Pi
directory (`~/.aria/agent/`) every time a session starts, filled in from Aria's settings
(`crates/aria-core/src/persona.rs`):

- `SYSTEM.md` — the persona, as a template: his text word for word, with `{{…}}` placeholders and
  `<!-- if X -->` … `<!-- end X -->` blocks only where a setting varies it (assistant and user name, language, answer
  style, git rules, memory and its project, enabled experts, `/plan` and `/execute`, the automatic skills). With the
  owner's settings it renders his own file exactly (`crates/aria-core/tests/golden/SYSTEM.owner.md`).
- `agents/*.md` — the experts subagents can be (frontend, backend, security, reviewer, researcher), word for word. A
  disabled expert is not written; "read-only" sets the `tools: read, bash, codemode` line of its front matter.

- `skills/install-from-link/SKILL.md` — Aria's own skill: how to read a link the user pastes and install what it
  points to (a skill, an MCP server or a Pi package) with the `install` extension's tool.

A file the owner edited in `~/.aria/agent/` is kept (Aria records the hash of what it wrote in `.aria-files`).
