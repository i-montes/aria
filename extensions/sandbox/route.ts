// Where a bash command runs. Without a box that works (macOS has no bwrap; on Linux bwrap may be missing or blocked by
// the system, see probe.ts) there is nothing to run a command in: every command is treated as one that leaves the
// sandbox, and the user is asked first. `hasBwrap` means "the box works here".
import { wrapCommand } from './policy.ts'

export type BashRoute = 'plain' | 'ask' | 'refuse-nobody' | 'refuse-not-allowed' | 'boxed'

/** What the card and the status say when there is no box. */
export const NO_BOX_NOTE = 'No sandbox is available on this machine: this command runs with full access.'

/** `mayLeave` false (Aria's settings: "ask before leaving the sandbox" off): leaving is refused, never asked. */
export function bashRoute(o: { covered: boolean; outside: boolean; hasBwrap: boolean; canAsk: boolean; mayLeave?: boolean }): BashRoute {
  if (o.covered) return 'plain'
  if (o.outside || !o.hasBwrap) {
    if (o.mayLeave === false) return 'refuse-not-allowed'
    return o.canAsk ? 'ask' : 'refuse-nobody'
  }
  return 'boxed'
}

/** What bash (and background_run) returns for a route that refuses; null for a route that runs. */
export function refusalText(route: BashRoute, hasBwrap: boolean): string | null {
  if (route === 'refuse-nobody') {
    return hasBwrap
      ? "Running outside the sandbox needs the user's permission, and there is nobody here to ask. Find another way inside the project or explain what is needed."
      : "There is no sandbox on this system, so every command needs the user's permission, and there is nobody here to ask."
  }
  if (route === 'refuse-not-allowed') return "Leaving the sandbox is not allowed (the user turned it off in Aria's settings): the command did not run. Do it inside the project, or tell the user what is needed."
  return null
}

/** The user answered no to running a command outside the sandbox. */
export const SAID_NO = 'The user said no: the command did not run.'

/** The line a background command runs as (in `folder`): the bwrap line when boxed, the command itself when it runs
 * outside (already allowed, or `ask` once the user said yes: `allowed`), null when the route refuses or nobody allowed
 * an `ask`. */
export function backgroundLine(route: BashRoute, command: string, folder: string, allowed = false): string | null {
  if (route === 'boxed') return wrapCommand(command, folder)
  if (route === 'plain') return command
  if (route === 'ask') return allowed ? command : null
  return null
}
