// Where a bash command runs. macOS has no bwrap, so there is no box to run a command in: every command is treated as
// one that leaves the sandbox, and the user is asked first. On Linux a missing bwrap is a broken install, and commands
// are refused as they always were (only one the user explicitly allows outside the sandbox runs).
import { wrapCommand } from './policy.ts'

export type BashRoute = 'plain' | 'ask' | 'refuse-nobody' | 'refuse-not-allowed' | 'refuse-no-bwrap' | 'boxed'

/** `mayLeave` false (Aria's settings: "ask before leaving the sandbox" off): leaving is refused, never asked. */
export function bashRoute(o: { covered: boolean; outside: boolean; hasBwrap: boolean; canAsk: boolean; platform: string; mayLeave?: boolean }): BashRoute {
  if (o.covered) return 'plain'
  const noBox = !o.hasBwrap && o.platform === 'darwin'
  if (o.outside || noBox) {
    if (o.mayLeave === false) return 'refuse-not-allowed'
    return o.canAsk ? 'ask' : 'refuse-nobody'
  }
  return o.hasBwrap ? 'boxed' : 'refuse-no-bwrap'
}

/** What bash (and background_run) returns for a route that refuses; null for a route that runs. */
export function refusalText(route: BashRoute, hasBwrap: boolean): string | null {
  if (route === 'refuse-nobody') {
    return hasBwrap
      ? "Running outside the sandbox needs the user's permission, and there is nobody here to ask. Find another way inside the project or explain what is needed."
      : "There is no sandbox on this system, so every command needs the user's permission, and there is nobody here to ask."
  }
  if (route === 'refuse-not-allowed') return "Leaving the sandbox is not allowed (the user turned it off in Aria's settings): the command did not run. Do it inside the project, or tell the user what is needed."
  if (route === 'refuse-no-bwrap') return 'The sandbox cannot run: bwrap is missing (sudo pacman -S bubblewrap). No commands run without it.'
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
