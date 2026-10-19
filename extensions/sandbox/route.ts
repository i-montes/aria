// Where a bash command runs. macOS has no bwrap, so there is no box to run a command in: every command is treated as
// one that leaves the sandbox, and the user is asked first. On Linux a missing bwrap is a broken install, and commands
// are refused as they always were (only one the user explicitly allows outside the sandbox runs).

export type BashRoute = 'plain' | 'ask' | 'refuse-nobody' | 'refuse-no-bwrap' | 'boxed'

export function bashRoute(o: { covered: boolean; outside: boolean; hasBwrap: boolean; canAsk: boolean; platform: string }): BashRoute {
  if (o.covered) return 'plain'
  const noBox = !o.hasBwrap && o.platform === 'darwin'
  if (o.outside || noBox) return o.canAsk ? 'ask' : 'refuse-nobody'
  return o.hasBwrap ? 'boxed' : 'refuse-no-bwrap'
}
