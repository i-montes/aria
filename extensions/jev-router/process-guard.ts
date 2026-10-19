// Stops the agent from killing processes that are not its own.
//
// On 2026-10-13 the agent ran `kill $(pgrep -f "vite --port 5173")` to restart its dev server and
// also killed the user's dev server of another project (and its own shell). Commands that pick
// victims by pattern or by port (`pkill`, `killall`, `fuser -k`, `kill $(pgrep …)`, `… | xargs
// kill`) are refused outright. A `kill` with literal PIDs passes only when every PID belongs to
// this Pi session (a descendant of its process) or runs inside the project directory.
import { readFileSync, readlinkSync, realpathSync } from 'node:fs'

export interface GuardVerdict {
  block: boolean
  reason?: string
}

const HOW_TO =
  'To stop a process of yours: find its PID (for example `ss -ltnp "sport = :5173"` for a port), check with `readlink /proc/<PID>/cwd` that it belongs to this project or that you started it, and stop only that PID with `kill <PID>`. If it is not yours, ask the user.'

const BY_PATTERN: [RegExp, string][] = [
  [/(^|[\s;&|(`])(pkill|killall)\b/, '`pkill`/`killall` kill everything that matches the pattern'],
  [/(^|[\s;&|(`])fuser\b[^;&|\n]*\s-[a-zA-Z]*k/, '`fuser -k` kills any process using that port or file'],
  [/\bkill\b[^;&|\n]*\$\(/, '`kill $(…)` picks the processes with a search'],
  [/\bkill\b[^;&|\n]*`/, '`kill` with a search in backticks'],
  [/\|\s*xargs\b[^;&|\n]*\bkill\b/, '`… | xargs kill` kills whatever the search returns'],
]

/** The PIDs (and anything that is not a literal PID) a `kill` command targets. */
export function killTargets(command: string): { pids: number[]; nonLiteral: boolean }[] {
  const out: { pids: number[]; nonLiteral: boolean }[] = []
  for (const segment of command.split(/&&|\|\||;|\n|\|/)) {
    const words = segment.trim().split(/\s+/)
    const at = words.findIndex((w) => w === 'kill' || w.endsWith('/kill'))
    if (at < 0) continue
    const args = words.slice(at + 1).filter((w) => w && !w.startsWith('-'))
    const pids = args.filter((w) => /^\d+$/.test(w)).map(Number)
    // `kill %1` (a job of this same shell) is ours by definition.
    const nonLiteral = args.some((w) => !/^\d+$/.test(w) && !/^%\d+$/.test(w))
    out.push({ pids, nonLiteral })
  }
  return out
}

export interface ProcessInfo {
  ppid: (pid: number) => number | null
  cwd: (pid: number) => string | null
}

export const procInfo: ProcessInfo = {
  ppid(pid) {
    try {
      const stat = readFileSync(`/proc/${pid}/stat`, 'utf8')
      // The command name may contain spaces and parentheses; fields resume after the last ')'.
      const rest = stat.slice(stat.lastIndexOf(')') + 2).split(' ')
      return Number(rest[1])
    } catch {
      return null
    }
  },
  cwd(pid) {
    try {
      return readlinkSync(`/proc/${pid}/cwd`)
    } catch {
      return null
    }
  },
}

function isDescendant(pid: number, root: number, info: ProcessInfo): boolean {
  let p: number | null = pid
  for (let i = 0; i < 64 && p && p > 1; i++) {
    if (p === root) return true
    p = info.ppid(p)
  }
  return false
}

function realOr(path: string): string {
  try {
    return realpathSync(path)
  } catch {
    return path
  }
}

function inside(dir: string, path: string): boolean {
  const d = realOr(dir).replace(/\/+$/, '')
  const p = realOr(path)
  return p === d || p.startsWith(`${d}/`)
}

/**
 * Whether `command` may run. `root` is this Pi process (its descendants are the session's own
 * processes); `project` is the session's working directory.
 */
export function checkKill(command: string, root: number, project: string, info: ProcessInfo = procInfo): GuardVerdict {
  for (const [re, why] of BY_PATTERN) {
    if (re.test(command)) {
      return { block: true, reason: `Blocked by the router: ${why}, and that can kill processes of the user's other projects. ${HOW_TO}` }
    }
  }
  for (const { pids, nonLiteral } of killTargets(command)) {
    if (nonLiteral) {
      return { block: true, reason: `Blocked by the router: \`kill\` only with literal, checked PIDs. ${HOW_TO}` }
    }
    for (const pid of pids) {
      const cwd = info.cwd(pid)
      if (cwd === null) continue // already gone, or not visible: killing it changes nothing
      if (isDescendant(pid, root, info) || inside(project, cwd)) continue
      return {
        block: true,
        reason: `Blocked by the router: process ${pid} is not from this session or this project (its folder is ${cwd}). Don't stop it; if it really has to be stopped, ask the user.`,
      }
    }
  }
  return { block: false }
}
