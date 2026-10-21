// Cmd+Q on a Mac asks before Aria quits: the first press shows "Press ⌘Q again to quit Aria", a second one within
// QUIT_WINDOW_MS quits (the dock stops its background work and its Pis first). Pure; app.js keeps the state.

export const QUIT_WINDOW_MS = 2000

/** Cmd+Q and nothing else, on a Mac only: on Omarchy the same keys are Super+Q, which belongs to the compositor. */
export const isQuitKey = (event, mac) =>
  mac && event.metaKey && !event.altKey && !event.ctrlKey && !event.shiftKey && event.key.toLowerCase() === 'q'

/** A press of Cmd+Q at `now`, the first one having been at `armedAt` (null when none is waiting). */
export function quitPress(armedAt, now) {
  if (armedAt !== null && now - armedAt <= QUIT_WINDOW_MS) return { quit: true, armedAt: null }
  return { quit: false, armedAt: now }
}
