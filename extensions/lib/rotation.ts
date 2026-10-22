// A rotation (extensions/rotate) replaces the session inside the same Pi process: Pi tears the extensions down and
// loads them again (a fresh module, so fresh state), then fires `session_start` with reason `new`. For the person it is
// the same conversation, so what they granted for "this session" must survive it. The rotate command marks the switch
// on globalThis while it runs (one mark per process, as the background extension keeps its jobs client); an extension
// hands its state over at `session_shutdown` and takes it back at the `session_start` that follows, while the mark is
// set. Anything not taken back is dropped when the mark is cleared.
const ROTATING = Symbol.for('aria.rotation.rotating')
const HANDED = Symbol.for('aria.rotation.handed')
const shared = globalThis as unknown as { [ROTATING]?: boolean; [HANDED]?: Map<string, unknown> }

/** Set by the rotate command around `ctx.newSession`; cleared (with what was not taken back) once it returns. */
export function markRotating(on: boolean) {
  if (on) {
    shared[ROTATING] = true
    shared[HANDED] = new Map()
  } else {
    delete shared[ROTATING]
    delete shared[HANDED]
  }
}

export const isRotating = () => shared[ROTATING] === true

/** At `session_shutdown`: keeps `value` for the next instance, only during a rotation. */
export function handOver(key: string, value: unknown) {
  if (isRotating()) shared[HANDED]?.set(key, value)
}

/** At `session_start`: what the instance before handed over, when this start is the rotation's (reason `new` with the
 * mark set); undefined otherwise (startup, /new, /resume, fork), so the caller starts afresh. */
export function takeOver<T>(key: string, event: any): T | undefined {
  if (event?.reason !== 'new' || !isRotating()) return undefined
  const value = shared[HANDED]?.get(key) as T | undefined
  shared[HANDED]?.delete(key)
  return value
}
