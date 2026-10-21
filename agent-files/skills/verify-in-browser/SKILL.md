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
s=check-$$; agent-browser --session "$s" --idle-timeout 2m batch --bail \
  'open http://localhost:5173' \
  'set viewport 390 844' \
  'snapshot -i' \
  'eval document.documentElement.scrollWidth' \
  'screenshot /tmp/check-phone.png'
st=$?; agent-browser --session "$s" close >/dev/null; exit $st
```

`batch` splits each step like a shell line, so quote in two layers: each step in single quotes for `bash`, an argument
with spaces or quotes in double quotes inside it, and strings in JavaScript as backticks, e.g.
`` 'eval "document.querySelector(`#name`).value"' `` or `'fill @e2 "hello world"'`.

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
| Does it fit a phone | `'set viewport 390 844'`, then `'eval document.documentElement.scrollWidth'` must be 390 or less |
| Is there one primary action, is a text present | `snapshot -i`, or `eval` counting elements (`` 'eval "document.querySelectorAll(`button`).length"' ``) |
| Contrast, sizes, colours | `eval` with `getComputedStyle`, e.g. `` 'eval "getComputedStyle(document.querySelector(`h1`)).color"' `` / `.backgroundColor` / `.fontSize`, and work out the WCAG ratio |
| Does a flow work | `'click @e3'`, `'fill @e2 "text"'`, `'wait 500'`, then `snapshot -i` again to see the result (refs come from the latest snapshot) |
| Errors | `errors` and `console` at the end of the batch |
| How it looks (overlap, spacing, colour) | `screenshot /tmp/<name>.png`, then read the image if you can see images; otherwise say you only checked the structure |

## Report

Say what you checked and what you saw, in a short list: each requirement, `ok` or what is wrong. Fix what is wrong
and check again before you say it is done.
