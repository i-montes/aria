# Ideas

Ideas to implement in Aria. Not a plan: each one gets its own design before anything is built. Ordered by the
owner, first to last.

| # | Idea | Why it fits | Effort |
|---|---|---|---|
| 1 | **Done (2026-10-21).** **Per-action approval for connectors**: each one read-only or read-and-send, with a preview and a question before anything that sends or changes something | Notion is connected and can write today without asking; the sandbox's permission card can be reused | Medium |
| 2 | **Done (2026-10-21).** **Attach files and images** (paperclip, drag and drop, paste): saved in `~/.aria/attachments/`; images also go to the model when it can see them, every file is named in the message by its path | Missing from the chat; used every day | Medium |
| 3 | **Done (2026-10-21): the dock stops a turn that stops answering (a running tool or compaction pauses it); MCP reconnects stay Pi's.** **A per-turn timeout for the model, and MCP connections retried with backoff** | Cheap robustness | Low |
| 4 | **Done (2026-10-21).** **Secrets in the system keyring** (Secret Service on Omarchy, Keychain on the Mac) instead of `auth.json` at mode 600 | Keeps credentials out of files a command outside the sandbox could read | Medium |
| 5 | **Done (2026-10-21).** **Import the MCP servers the person already has** (Claude Code, Cursor, VS Code): one more step of the onboarding, which finds them and lets the person pick which to bring, beside Notion and Tavily in Integrations (and the same in Settings) | Saves configuring them by hand | Low–medium |
| 6 | **Done (2026-10-21).** **Choose the shortcut in the settings, with fallbacks when it is taken** | `SUPER + A` / `Option + Space` are fixed today | Low |
| 7 | **Done (2026-10-21, 0.7.0).** **One continuous conversation with session rotation**: when it grows too large, go on in a fresh session seeded with a summary, with paged history | Goes with autocompact and with "one session is enough"; to be designed together with background work | Medium |
| 8 | **Role profiles** (saved accounts and models per role) | Fits the daily / heavy models | Low–medium |
| 9 | **Packages** (Arch / AUR, a self-contained runtime) | Distribution, once there are releases | Medium |
| 10 | **Remote access**: a phone PWA, or a Telegram / WhatsApp bridge | The page is already separate from Rust; needs a gateway with a token | High |
| 11 | **Use the computer**: clicks, typing, screenshots, with a per-session grant, a blocklist, confirmations for risky actions and a panic stop that live outside the agent | The largest step; on Omarchy through Wayland tools (e.g. `ydotool`/`wtype`, `grim`) | Very high |
| 12 | **Voice**: dictation into the input, and replies read aloud | **Dictation done on macOS (2026-10-22)**: Whisper large-v3-turbo q5_0 built into Aria (whisper.cpp), the same on Linux once its GPU backend is set; replies read aloud still to do. Only on request: Aria never records or plays sound without asking | High |
| 13 | **Done (2026-10-21).** **The dock reloads its page when the web view's process dies**: today a crashed web content process (memory, a WebKit bug) leaves the panel transparent, so the shortcut seems to do nothing; macOS (`webViewWebContentProcessDidTerminate`) and WebKitGTK (`web-process-terminated`) both report it | Found while looking into why the dock sometimes does not open; the conversation comes back because the new page asks every tab's Pi for it again | Low |
| 14 | **Possible now with scheduled tasks (2026-10-21): ask for it as a daily task.** **A morning brief of every connected app**: each morning, a short summary of what matters most across all the apps connected to Aria (today Notion, Tavily and the imported MCP servers; any app connected later joins by itself, with nothing to configure per app) | The owner asked for it; the connectors already reach the apps, and a scheduled run fits the goals engine (phase 3) | Medium |
| 15 | **Copy on one machine, paste on another**: what the owner copies on the Mac is ready to paste on Omarchy and the other way round (text first; images and files later), through Aria on both, end-to-end encrypted, never kept longer than needed | The owner works on both machines; Aria already runs on each with a dock and a hotkey, and a sync layer (see the Supabase discussion) could carry it | Medium |
