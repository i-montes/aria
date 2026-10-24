`aria-dock.png` is the dock's chat column (the real `ui/` page) at 2x: two session tabs, a short conversation with tool calls, the changed-file chips, the task strip and the composer.
It is rendered by the UI sandbox (`tools/ui-sandbox/`) with its stubbed bridge and fake content only (project `checkout-widget`, user `ana`); no Pi, no real session or machine data.
Regenerate it after a UI change with `sh docs/assets/regenerate.sh` (needs `bun`, the pinned `~/.aria/bin/agent-browser`, `magick` and `optipng`); `dock-scene.js` is the scene it plays.
There is no terminal-client image yet: the client needs a running dock, and nothing renders its view to a file without a test harness.
