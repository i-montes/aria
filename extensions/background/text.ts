// What Aria reads about its background work: the hidden `[background] running: …` line added to each user message,
// the `background_list` text, and the message Aria gets when a job ends (or jobs were cut when the dock stopped).
// The job objects are the dock's (`crates/aria/src/jobs.rs`, camelCase).

export type Job = {
  id: string
  number: number
  session?: string
  kind: 'process' | 'agent'
  title: string
  command?: string | null
  task?: string | null
  folder: string
  state: 'queued' | 'running' | 'exited' | 'failed' | 'stopped' | 'done' | 'cut'
  startedAt?: number | null
  endedAt?: number | null
  url?: string | null
  exit?: number | null
  isolation?: 'folder' | 'worktree' | null
  worktree?: string | null
  writes?: boolean
  result?: string | null
  conflict?: boolean
  dismissed?: boolean
}

const active = (j: Job) => j.state === 'running' || j.state === 'queued'
const kindName = (j: Job) => (j.kind === 'agent' ? 'subagent' : 'process')
const name = (j: Job) => `#${j.number} ${kindName(j)} "${j.title}"`
const minutes = (j: Job, now: number) => `${Math.max(0, Math.floor((now - (j.startedAt ?? now)) / 60_000))} min`

/** One running or queued job: `#4 subagent "fix the tests" (~/Projects/shop, writing, worktree, 3 min)`. */
function activeEntry(j: Job, now: number): string {
  const parts = [j.folder]
  if (j.kind === 'process') {
    if (j.url) parts.push(j.url)
  } else if (j.writes === false) {
    parts.push('read-only')
  } else if (j.state === 'running' && j.isolation === 'folder') {
    parts.push('writing in the folder: do not edit it')
  } else {
    parts.push('writing')
    if (j.isolation === 'worktree') parts.push('worktree')
  }
  parts.push(j.state === 'queued' ? 'queued' : minutes(j, now))
  return `${name(j)} (${parts.join(', ')})`
}

/** A subagent that wrote straight into its folder: its edits are already there. */
const wroteInFolder = (j: Job) => j.kind === 'agent' && j.writes !== false && j.isolation === 'folder'
const editsSoFar = (j: Job) => `The edits it made so far are in ${j.folder} (not committed).`

/** Where a finished subagent's changes are. */
function changes(j: Job): string | null {
  if (wroteInFolder(j)) return j.state === 'done' ? 'changes in the folder, not committed' : 'edits so far in the folder, not committed'
  if (j.kind !== 'agent' || j.isolation !== 'worktree') return null
  if (j.conflict && j.worktree) return `conflict: kept in ${j.worktree}`
  if (j.worktree) return `worktree kept in ${j.worktree}`
  return j.state === 'done' ? 'applied' : null
}

function finishedEntry(j: Job): string {
  const parts: string[] = [j.state]
  if (j.exit !== null && j.exit !== undefined) parts.push(`exit ${j.exit}`)
  const c = changes(j)
  if (c) parts.push(c)
  parts.push(j.folder)
  return `${name(j)} (${parts.join(', ')})`
}

/** The hidden line for a user message, or null when nothing is running or queued. */
export function nowLine(jobs: Job[], now = Date.now()): string | null {
  const running = (jobs ?? []).filter(active)
  if (!running.length) return null
  return `[background] running: ${running.map((j) => activeEntry(j, now)).join('; ')}`
}

/** `background_list`: what runs (the line) and what finished. */
export function listText(jobs: Job[], now = Date.now()): string {
  const all = jobs ?? []
  if (!all.length) return 'No background work in this session.'
  const out: string[] = []
  const line = nowLine(all, now)
  out.push(line ? line.replace(/^\[background\] running: /, 'Running: ') : 'Nothing running.')
  const done = all.filter((j) => !active(j))
  if (done.length) out.push(`Finished: ${done.map(finishedEntry).join('; ')}`)
  return out.join('\n')
}

const lastLines = (lines: unknown): string => (Array.isArray(lines) && lines.length ? `\nLast lines:\n${lines.join('\n')}` : '')

/** The message Aria gets for an `ended` or `cut` event, and whether it starts a turn. */
export function endedMessage(e: any): { text: string; triggerTurn: boolean } {
  if (e?.event === 'cut') {
    const jobs: Job[] = Array.isArray(e.jobs) ? e.jobs : []
    const entries = jobs.map((j) =>
      j.worktree
        ? `${name(j)} (its worktree is kept in ${j.worktree})`
        : wroteInFolder(j)
          ? `${name(j)} (the edits it made so far are in ${j.folder}, not committed)`
          : name(j),
    )
    return {
      text: `Aria's dock stopped, so this background work was cut and is no longer running: ${entries.join('; ') || 'none'}.`,
      triggerTurn: false,
    }
  }
  const j: Job = e?.job ?? {}
  const by = e?.by === 'stop' ? 'stop' : 'self'
  const label = `${name(j)} in ${j.folder}`
  if (j.kind !== 'agent') {
    if (by === 'stop') return { text: `Background ${label} was stopped.`, triggerTurn: false }
    const how = j.state === 'failed' ? 'failed' : `ended by itself${j.exit !== null && j.exit !== undefined ? ` (exit ${j.exit})` : ''}`
    return { text: `Background ${label} ${how}.${lastLines(e?.lastLines)}\nTell the person what happened.`, triggerTurn: true }
  }
  const kept = j.worktree ? ` Its worktree is kept in ${j.worktree}.` : wroteInFolder(j) ? ` ${editsSoFar(j)}` : ''
  if (by === 'stop') return { text: `Background ${label} was stopped.${kept}`, triggerTurn: false }
  if (j.state !== 'done') {
    const result = j.result ? `\nWhat it said:\n${j.result}` : ''
    return { text: `Background ${label} failed.${kept}${result}${lastLines(e?.lastLines)}\nTell the person what happened.`, triggerTurn: true }
  }
  let where = ''
  if (wroteInFolder(j)) where = ` Its changes are in ${j.folder} (not committed).`
  else if (j.isolation === 'worktree') {
    where = j.conflict
      ? ` Its changes could not be applied to ${j.folder} cleanly; they are kept in the worktree ${j.worktree}. Ask the person how to merge them.`
      : j.worktree
        ? ` Its changes are in the worktree ${j.worktree}.`
        : ` Its changes were applied to ${j.folder} (not committed).`
  }
  const result = j.result ? `\nIts report:\n${j.result}` : ''
  return { text: `Background ${label} is done.${where}${result}\nTell the person what it did, and save its Learnings to memory.`, triggerTurn: true }
}
