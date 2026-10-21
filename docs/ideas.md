# Ideas

Ideas to implement in Aria. Not a plan: each one gets its own design before anything is built. Ordered by the
owner, first to last.

| # | Idea | Why it fits | Effort |
|---|---|---|---|
| 1 | **Done (2026-10-21).** **Per-action approval for connectors**: each one read-only or read-and-send, with a preview and a question before anything that sends or changes something | Notion is connected and can write today without asking; the sandbox's permission card can be reused | Medium |
| 2 | **Done (2026-10-21).** **Attach files and images** (paperclip, drag and drop, paste): saved in `~/.aria/attachments/`; images also go to the model when it can see them, every file is named in the message by its path | Missing from the chat; used every day | Medium |
| 3 | **A per-turn timeout for the model, and MCP connections retried with backoff** | Cheap robustness | Low |
| 4 | **Secrets in the system keyring** (Secret Service on Omarchy, Keychain on the Mac) instead of `auth.json` at mode 600 | Keeps credentials out of files a command outside the sandbox could read | Medium |
| 5 | **Import the MCP servers the person already has** (Claude Code, Cursor, VS Code): one more step of the onboarding, which finds them and lets the person pick which to bring, beside Notion and Tavily in Integrations (and the same in Settings) | Saves configuring them by hand | Low–medium |
| 6 | **Choose the shortcut in the settings, with fallbacks when it is taken** | `SUPER + A` / `Option + Space` are fixed today | Low |
| 7 | **One continuous conversation with session rotation**: when it grows too large, go on in a fresh session seeded with a summary, with paged history | Goes with autocompact and with "one session is enough"; to be designed together with background work | Medium |
| 8 | **Role profiles** (saved accounts and models per role) | Fits the daily / heavy models | Low–medium |
| 9 | **Packages** (Arch / AUR, a self-contained runtime) | Distribution, once there are releases | Medium |
| 10 | **Remote access**: a phone PWA, or a Telegram / WhatsApp bridge | The page is already separate from Rust; needs a gateway with a token | High |
| 11 | **Use the computer**: clicks, typing, screenshots, with a per-session grant, a blocklist, confirmations for risky actions and a panic stop that live outside the agent | The largest step; on Omarchy through Wayland tools (e.g. `ydotool`/`wtype`, `grim`) | Very high |
| 12 | **Voice**: dictation into the input, and replies read aloud | Through the owner's own `memo` (local Spanish dictation for Wayland, faster-whisper) rather than a new engine. Only on request: Aria never records or plays sound without asking | High |
