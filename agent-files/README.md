# Agent files

The owner's persona and experts, copied word for word from his Pi (`~/.pi/agent/`). Aria embeds them and writes them into
its own Pi directory (`~/.aria/agent/`) every time a session starts:

- `SYSTEM.md` — the persona: who Aria is, the answer style (short, no tables), how it uses tools, memory and subagents.
- `agents/*.md` — the experts subagents can be (frontend, backend, security, reviewer, researcher).

A file the owner edited in `~/.aria/agent/` is kept (Aria records the hash of what it wrote in `.aria-files`). Part 2c of
phase 2 turns the parts that depend on the settings (names, language, answer style, git rules, enabled experts) into
templates.
