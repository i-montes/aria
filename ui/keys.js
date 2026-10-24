// The chat's Alt shortcuts. Each fires on the key labelled with its letter, on any layout: by the key's place
// (`event.code`), because Option+A types "å" on a Mac so the typed character would not match there, or by the typed
// letter, because on AZERTY, Dvorak or Colemak the key labelled A is not in the QWERTY place. Alt alone: Ctrl+Alt+A
// and Cmd+Option+A are other shortcuts. Pure; app.js adds the "chat is showing" guard.

const altLetter = (event, code, letter) =>
  !!event.altKey &&
  !event.ctrlKey &&
  !event.metaKey &&
  (event.code === code || (typeof event.key === 'string' && event.key.length === 1 && event.key.toLowerCase() === letter))

/** Alt+D (Option+D on a Mac): dictate, or stop and write it. */
export const isDictateKey = (event) => altLetter(event, 'KeyD', 'd')

/** Alt+A (Option+A on a Mac): the subagents and background view, and back. */
export const isAgentsKey = (event) => altLetter(event, 'KeyA', 'a')
