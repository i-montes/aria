// The task list of a slice phase while its job builds: watches the plan file the brief names and, on each change, hands the
// phase's tasks (the plan's marks) to `publish`. Kept apart from index.ts (which needs Pi) so it is tested with a fake
// file system and clock. One feed watches one job at a time.
import { readFileSync, watch as fsWatch } from 'node:fs'
import type { Task } from '../tasks/tasks.ts'
import { parsePlan, phaseTasks } from './slice.ts'

export const DEBOUNCE_MS = 300
// A file an editor is replacing is missing for a moment: how many times the watch is tried again.
const REARM_TRIES = 10

type Closable = { close(): void }
export type Io = {
  read: (path: string) => string
  /** `onChange(event)` gets `change`, `rename` or, when the watcher fails, `rename` too (it has to be armed again). */
  watch: (path: string, onChange: (event: string) => void) => Closable
  setTimer: (run: () => void, ms: number) => unknown
  clearTimer: (timer: unknown) => void
}
export type Publish = (tasks: Task[]) => void

export const realIo: Io = {
  read: (path) => readFileSync(path, 'utf8'),
  watch: (path, onChange) => {
    const watcher = fsWatch(path, (event) => onChange(event))
    watcher.on('error', () => onChange('rename'))
    return watcher
  },
  setTimer: (run, ms) => setTimeout(run, ms),
  clearTimer: (timer) => clearTimeout(timer as ReturnType<typeof setTimeout>),
}

export function createTaskFeed(io: Io = realIo) {
  let current: { id: string; number: number; plan: string; phase: number; publish: Publish } | null = null
  let watcher: Closable | null = null
  let timer: unknown = null
  let rearm = false
  let tries = 0

  const unwatch = () => {
    if (timer !== null) io.clearTimer(timer)
    timer = null
    try {
      watcher?.close()
    } catch {
      // already closed
    }
    watcher = null
    rearm = false
  }

  /** Publishes the phase's tasks now; false when the plan cannot be read, true when the phase still has work. */
  const push = (): { read: boolean; open: boolean } => {
    if (!current) return { read: false, open: false }
    let tasks: Task[]
    try {
      tasks = phaseTasks(parsePlan(io.read(current.plan)), current.phase, current.number)
    } catch {
      return { read: false, open: true }
    }
    try {
      current.publish(tasks)
    } catch {
      // no UI to tell
    }
    return { read: true, open: tasks.some((t) => t.status !== 'done') }
  }

  const arm = (): boolean => {
    if (!current) return false
    try {
      watcher = io.watch(current.plan, onChange)
      return true
    } catch {
      watcher = null
      return false
    }
  }

  const tick = () => {
    timer = null
    if (!current) return
    if (rearm) {
      try {
        watcher?.close()
      } catch {
        // already closed
      }
      watcher = null
      if (arm()) rearm = false
      else if (++tries < REARM_TRIES) {
        timer = io.setTimer(tick, DEBOUNCE_MS)
        return
      }
    }
    // The phase has nothing left to build (or the plan lost it): the list stays as it is, the watching ends.
    if (!push().open) unwatch()
  }

  function onChange(event: string) {
    if (event === 'rename') {
      rearm = true
      tries = 0
    }
    if (timer !== null) io.clearTimer(timer)
    timer = io.setTimer(tick, DEBOUNCE_MS)
  }

  return {
    /** Starts watching for a build job (replacing any other) and publishes the list as it is. */
    start(job: { id: string; number: number }, at: { plan: string; phase: number }, publish: Publish) {
      unwatch()
      current = { ...job, ...at, publish }
      tries = 0
      const now = push()
      if (now.read && now.open) arm()
    },
    /** The job ended: one last publish with the final marks, then the watching stops (the list is left as it is). */
    finish(id: string) {
      if (current?.id !== id) return
      unwatch()
      push()
    },
    /** Nothing is watched and nothing is published after this (the plan is done, or the session shuts down). */
    stop() {
      unwatch()
      current = null
    },
  }
}
