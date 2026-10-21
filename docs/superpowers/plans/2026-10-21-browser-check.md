# Checking a page in the browser — plan

Spec: `docs/superpowers/specs/2026-10-21-browser-check-design.md` (the authority).

## Global constraints

- Everything in English: code, names, messages, docs, commits. Match the surrounding style (comment density, the
  installer's `say`/`run`/`die` helpers, the tests' `pass`/`fail` lines).
- Commit subjects are conventional (`feat(install): …`, `feat(skills): …`, `test(…)`, `docs: …`) and end with the line
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- No personal data in anything committed: no real home paths, employer or task names; fixtures use `/home/ana`-style
  names.
- The pinned version is exactly `0.38.2`.
- Never start a browser outside a test that cleans it up; a test that starts one must leave no `agent-browser` or
  Chrome process behind.
- Do not touch `agent-files/agents/*.md` or `agent-files/SYSTEM.md` (the owner's files, word for word).

## Task 1: the installer brings `agent-browser` and its browser

Files: `agent-browser-version` (new), `install/install.sh`, `install/install.test.sh`, `README.md`.

1. New repo file `agent-browser-version` holding the single line `0.38.2`.
2. In `install/install.sh`, a function `install_agent_browser`, placed after `install_engram` and called in
   `do_install` right after `install_engram` (so before `install_binary`). Its comment says what it is for (the
   agents' browser check, the `verify-in-browser` skill) and that a failure only warns.
   - The version: `$SCRIPT_DIR/../agent-browser-version` when there; else in a dry run
     `say "would: download ${DOWNLOAD_URL%/*}/agent-browser-version"` and treat the version as `PINNED`; else download
     it from `${DOWNLOAD_URL%/*}/agent-browser-version` (a failed download warns and returns). An env override
     `ARIA_AGENT_BROWSER_VERSION` wins over both (like `ARIA_PI_VERSION`). A value that is not a version number
     (`*[!0-9.]*` or empty) warns and returns.
   - Prefix `$DATA_DIR/agent-browser/$version`. If `$prefix/node_modules/.bin/agent-browser` is executable, say
     `agent-browser $version is already installed`; otherwise
     `run npm install --prefix "$prefix" --no-audit --no-fund "agent-browser@$version"` (a failure warns and returns).
   - `run ln -sf "$prefix/node_modules/.bin/agent-browser" "$DATA_DIR/bin/agent-browser"` (after
     `run install -d -m 700 "$DATA_DIR/bin"`).
   - The browser: `run "$DATA_DIR/bin/agent-browser" install` (a failure warns: the check will not work until it is
     installed, and how to install it by hand: `~/.aria/bin/agent-browser install`).
   - Warnings go through a `warn()` helper next to `say`/`die`: `printf 'aria installer: warning: %s\n' "$*" >&2`.
     The function never calls `die`.
3. `install/install.test.sh`, following the Engram block's style:
   - The dry run (with the repo's version file) contains, in this order, after the Pi packages and before
     `would: install -Dm755`: `would: npm install --prefix <ARIA_HOME>/agent-browser/0.38.2 --no-audit --no-fund agent-browser@0.38.2`,
     `would: ln -sf <ARIA_HOME>/agent-browser/0.38.2/node_modules/.bin/agent-browser <ARIA_HOME>/bin/agent-browser`,
     `would: <ARIA_HOME>/bin/agent-browser install`. The version in the test is read from `agent-browser-version`,
     not written in the test.
   - A failing step leaves the install going: with a fake `npm` first on PATH that exits 1 and a test hook
     `--install-agent-browser` (add it to the installer's mode switch next to `--install-engram`, running only
     `install_agent_browser`), the installer exits 0 and prints a `warning:` line.
   - `ARIA_AGENT_BROWSER_VERSION=bad` warns and installs nothing (no `$ARIA_HOME/agent-browser` directory).
4. `README.md`: in the installer paragraph and the `~/.aria` table, one row/sentence for
   `~/.aria/agent-browser/`, `~/.aria/bin/agent-browser`: the pinned `agent-browser` (`agent-browser-version`) the agents
   use to check pages, and its Chrome in `~/.agent-browser/browsers`.
5. Run `sh install/install.test.sh` and `shellcheck install/install.sh install/install.test.sh`; both clean.

## Task 2: the skill, and proof it works in the sandbox

Files: `agent-files/skills/verify-in-browser/SKILL.md` (new), `crates/aria-core/src/agent.rs` (one test),
`extensions/sandbox/browser.test.ts` (new), `agent-files/README.md`, `README.md` (the skills row).

1. `agent-files/skills/verify-in-browser/SKILL.md` with exactly this content:

````markdown
---
name: verify-in-browser
description: Open a page you built or changed in a headless browser and check it before saying it works — layout on a phone, the controls that exist, contrast, a flow that must work, errors. Use whenever you build or change anything a browser shows (a page, a component, styles, a form) and before you claim it works or looks right, and whenever the user asks you to look at, test or check a web page.
---

# Check it in the browser

You cannot see what you built until you open it. After building or changing anything a browser shows, open it with
`agent-browser` and check it against what was asked **before** you say it works or looks right. If you cannot open it
(no `agent-browser`, no server you may start, a sign-in or a captcha in the way), say so plainly: "I could not check it
in the browser because …". Never claim what you did not see.

## One check is one command

Each `bash` call runs in a fresh sandbox, and the browser does not survive between calls: a second call finds an empty
browser and leaves the first one running. So put the whole check in **one** `bash` call, with `batch`, and always
close the session at the end:

```sh
s=check-$$; agent-browser --session "$s" batch --bail \
  "open http://localhost:5173" \
  "set viewport 390 844" \
  "snapshot -i" \
  "eval document.documentElement.scrollWidth" \
  "screenshot /tmp/check-phone.png"
st=$?; agent-browser --session "$s" close >/dev/null; exit $st
```

Each step prints its result in order. `--bail` stops at the first step that fails. If you need to look again, run
another whole check; never split one check across calls, and never leave out the `close`.

## What to open

- A server the project already runs, or one you start yourself in the background for the check
  (`http://localhost:<port>`). Stop it afterwards if you started it.
- A plain HTML file: `file:///absolute/path/index.html`.
- A page behind a sign-in or a captcha belongs to the user: say so, do not try to get past it.

## Measure, do not guess

| Question | Steps |
|---|---|
| What is on the page, which controls exist, are they labelled | `snapshot -i` (interactive elements with roles, names and `@e` refs) |
| Does it fit a phone | `set viewport 390 844`, then `eval document.documentElement.scrollWidth` must be 390 or less |
| Is there one primary action, is a text present | `snapshot -i`, or `eval` counting elements (`document.querySelectorAll('…').length`) |
| Contrast, sizes, colours | `eval` with `getComputedStyle(el).color` / `.backgroundColor` / `.fontSize`, and work out the WCAG ratio |
| Does a flow work | `click @e3`, `fill @e2 "text"`, `wait 500`, then `snapshot -i` again to see the result (refs come from the latest snapshot) |
| Errors | `errors` and `console` at the end of the batch |
| How it looks (overlap, spacing, colour) | `screenshot /tmp/<name>.png`, then read the image if you can see images; otherwise say you only checked the structure |

## Report

Say what you checked and what you saw, in a short list: each requirement, `ok` or what is wrong. Fix what is wrong
and check again before you say it is done.
````

2. `crates/aria-core/src/agent.rs`: next to `the_install_skill_and_extension_are_written_into_pis_folders`, a test
   `the_browser_skill_is_written_into_pis_skills` asserting `skills/verify-in-browser/SKILL.md` in the agent dir after
   `prepare` equals `include_str!` of the repo file.
3. `extensions/sandbox/browser.test.ts` (bun test), skipped unless `/usr/bin/bwrap` exists and `agent-browser` is on
   PATH (`Bun.which`). It writes a small page (`<title>Check</title><h1>Hello</h1><button>Copy</button>`) into a temp
   project folder (with a `.git` dir so it is a project), builds the command with `wrapCommand` from `./policy` for that
   folder, using the skill's shape (unique session, `batch --bail`, `open file://…`, `set viewport 390 844`,
   `snapshot -i`, `eval document.documentElement.scrollWidth`, then `close`), runs it with `bash -c`
   (timeout 90 s), and asserts: exit 0; the output has `button "Copy"` and a line `390`; afterwards no process whose
   command line holds the session name is left (`pgrep -f <session>` finds nothing). It always closes the session in a
   `finally`, and removes the temp folder.
4. `agent-files/README.md`: one bullet for `skills/verify-in-browser/SKILL.md` — Aria's own skill: check a page in a
   headless browser (`agent-browser`) before saying it works; one `bash` call per check, because the sandbox does not
   keep the browser between calls. `README.md`: the skills row names `verify-in-browser` next to `install-from-link`.
5. Run `cargo test -p aria-core the_browser_skill`, `bun test extensions/sandbox/browser.test.ts`, and
   `bun test extensions`; all pass.
