# Checking a page in the browser

Aria's agents build web pages and then say "done" without having seen them. This gives them eyes: a pinned
`agent-browser` (open source, Linux and macOS) that drives a headless Chrome and hands the model the page as an
accessibility tree with a reference for each element, plus a skill that makes "look before you say it works" the rule.

## What the agent gets

- **`agent-browser` on its PATH.** The installer puts the pinned version in `~/.aria/agent-browser/<version>` with npm
  (the way it installs Pi) and links `~/.aria/bin/agent-browser` to it; Pi's PATH already starts with `~/.aria/bin`.
  The version lives in the repo file `agent-browser-version` (one line, `0.38.2`: the newest release at least two weeks
  old on 2026-10-21). Its browser comes from `agent-browser install` (Chrome for Testing in `~/.agent-browser/browsers`).
  A failure of either step is a warning, never a failed install: the check is something the agent can do, not something
  Aria needs to start.
- **The skill `verify-in-browser`** (`agent-files/skills/verify-in-browser/SKILL.md`), Aria's own, written into Pi's
  skills folder like `install-from-link`. Pi lists it by its description, so it applies to the main agent and to every
  expert; the experts' files stay the owner's, word for word.

## The rule the skill sets

After building or changing anything a browser shows (a page, a component, styles, a form), the agent opens it and
checks it before saying it works or looks right. If it cannot (no `agent-browser`, no server it may start, a sign-in
or a captcha in the way), it says so plainly instead of claiming it.

## One check, one command

Inside Aria's sandbox each `bash` call is a fresh bwrap box with its own `$XDG_RUNTIME_DIR`, where `agent-browser` keeps
the socket to its browser. A second call cannot find the first call's browser: it opens an empty one, and the first is
left running (found 2026-10-21). On macOS every command asks the user first. So one check is **one `bash` call**:

```sh
s=check-$$; agent-browser --session "$s" batch --bail "open <url>" "…" ; st=$?; agent-browser --session "$s" close >/dev/null; exit $st
```

`batch` runs every step in one execution (fewer turns and tokens, which is also what lets a small model get through),
`--bail` stops at the first failure, a session named after the shell's pid never meets another check, and `close` runs
whatever happened, so no browser outlives the call.

## What a check looks at

Measure, never guess: the skill names the steps and the measurements that answer the usual questions.

- `snapshot -i`: the interactive elements, roles and names (one primary action, labels, buttons that exist).
- `set viewport 390 844` and `eval` of `document.documentElement.scrollWidth`: no horizontal overflow on a phone.
- `eval` for what the tree does not say: computed colours for contrast, a count of elements, text that must appear.
- `click`, `fill`, `wait`, then `snapshot` again: a flow does what it says ("Copied" appears).
- `screenshot /tmp/<name>.png`: for what the tree cannot see (overlap, spacing, colour); the agent reads the file when
  its model takes images, and says it only checked the structure when it does not.
- `errors` / `console`: the page loads without errors.

A local server the agent started itself is fine (`http://localhost:<port>`); `file://` works for a plain file. Pages
behind a sign-in or a captcha are the user's: the agent says so.

## Out of scope

A headed browser the user watches, saved sign-ins (`--profile`, `state`), remote browser providers, and Ego Light
(closed source, no Linux build).

## Checks

- The installer's dry run installs the pinned `agent-browser` and its browser, after Pi and before the binary; a failed
  `agent-browser` step leaves the install going (`install/install.test.sh`).
- The skill is written into Pi's skills folder (`crates/aria-core/src/agent.rs`).
- The skill's command shape works in the real sandbox: one boxed call opens a page, reads its tree and measures it, and
  leaves no browser running (`extensions/sandbox`, skipped without `agent-browser` or bwrap).
