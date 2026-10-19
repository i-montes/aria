// Where a bash command runs. Without bwrap (macOS has none) there is no box to run it in, so every command is treated
// as one that leaves the sandbox: the user is asked first, and nothing runs unconfined without them.

export type BashRoute = 'plain' | 'ask' | 'refuse-nobody' | 'boxed'

export function bashRoute(o: { covered: boolean; outside: boolean; hasBwrap: boolean; canAsk: boolean }): BashRoute {
  if (o.covered) return 'plain'
  if (o.outside || !o.hasBwrap) return o.canAsk ? 'ask' : 'refuse-nobody'
  return 'boxed'
}
