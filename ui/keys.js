// The chat's Alt shortcuts. They are read by the key's place (`event.code`), never by the character it types:
// Option+A types "å" on a Mac, so `event.key` would not match there. Pure; app.js adds the "chat is showing" guard.

/** Alt+D (Option+D on a Mac): dictate, or stop and write it. */
export const isDictateKey = (event) => event.altKey && event.code === 'KeyD'

/** Alt+A (Option+A on a Mac): the subagents and background view, and back. */
export const isAgentsKey = (event) => event.altKey && event.code === 'KeyA'
