// Run: bun test extensions/jev-router/tasks-feed.test.ts
import { describe, expect, test } from 'bun:test'
import type { Task } from '../tasks/tasks.ts'
import { createTaskFeed } from './tasks-feed.ts'

const PLAN = (marks: string[]) => `# Sign-up

## Phase 1: Sign up
${marks.map((m, i) => `- [${m}] task ${i + 1}`).join('\n')}

## Phase 2: Create a task
- [ ] creates a task
`

// A clock and a file system the test drives by hand.
function rig(initial: string) {
  const files = new Map<string, string>([['/home/ana/plan.md', initial]])
  const timers = new Map<number, { at: number; run: () => void }>()
  let now = 0
  let next = 1
  const watchers: { path: string; fire: (event: string) => void; closed: boolean }[] = []
  const published: (Task[] | null)[] = []
  const feed = createTaskFeed({
    read: (path) => {
      const text = files.get(path)
      if (text === undefined) throw new Error('ENOENT')
      return text
    },
    watch: (path, onChange) => {
      if (!files.has(path)) throw new Error('ENOENT')
      const w = { path, fire: onChange, closed: false }
      watchers.push(w)
      return { close: () => void (w.closed = true) }
    },
    setTimer: (run, ms) => {
      timers.set(next, { at: now + ms, run })
      return next++
    },
    clearTimer: (id) => void timers.delete(id as number),
  })
  const publish = (tasks: Task[]) => void published.push(tasks)
  const advance = (ms: number) => {
    now += ms
    for (const [id, t] of [...timers]) if (t.at <= now && timers.delete(id)) t.run()
  }
  const open = () => watchers.filter((w) => !w.closed)
  const job = { id: 'j4', number: 4 }
  const at = { plan: '/home/ana/plan.md', phase: 1 }
  const statuses = () => published.map((p) => p?.map((t) => t.status[0]).join(''))
  return { files, timers, watchers, published, feed, publish, advance, open, job, at, statuses }
}

describe('the task feed', () => {
  test('publishes the list at once, then once per burst of changes (debounced)', () => {
    const r = rig(PLAN([' ', ' ']))
    r.feed.start(r.job, r.at, r.publish)
    expect(r.statuses()).toEqual(['tt'])
    r.files.set('/home/ana/plan.md', PLAN(['~', ' ']))
    r.open()[0].fire('change')
    r.advance(100)
    r.open()[0].fire('change')
    r.advance(299)
    expect(r.statuses()).toEqual(['tt'])
    r.advance(1)
    expect(r.statuses()).toEqual(['tt', 'at'])
    expect(r.published[1]?.[0]).toEqual({ id: 'p1-t1', title: 'task 1', status: 'active', agent: '#4' })
  })

  test('a replaced file (rename) is watched again', () => {
    const r = rig(PLAN([' ', ' ']))
    r.feed.start(r.job, r.at, r.publish)
    r.files.set('/home/ana/plan.md', PLAN(['x', '~']))
    r.open()[0].fire('rename')
    r.advance(300)
    expect(r.statuses()).toEqual(['tt', 'da'])
    expect(r.watchers).toHaveLength(2)
    expect(r.watchers[0].closed).toBe(true)
    expect(r.open()).toHaveLength(1)
  })

  test('a file that is missing for a moment is tried again', () => {
    const r = rig(PLAN([' ', ' ']))
    r.feed.start(r.job, r.at, r.publish)
    const text = r.files.get('/home/ana/plan.md')!
    r.files.delete('/home/ana/plan.md')
    r.open()[0].fire('rename')
    r.advance(300)
    expect(r.published).toHaveLength(1)
    r.files.set('/home/ana/plan.md', text.replace('- [ ] task 1', '- [x] task 1'))
    r.advance(300)
    expect(r.statuses()).toEqual(['tt', 'dt'])
    expect(r.open()).toHaveLength(1)
  })

  test('stops by itself when the phase has no open task left, and the list stays', () => {
    const r = rig(PLAN([' ', ' ']))
    r.feed.start(r.job, r.at, r.publish)
    r.files.set('/home/ana/plan.md', PLAN(['x', 'x']))
    r.open()[0].fire('change')
    r.advance(300)
    expect(r.statuses()).toEqual(['tt', 'dd'])
    expect(r.open()).toHaveLength(0)
    expect(r.timers.size).toBe(0)
  })

  test("the job's end stops the watcher and publishes the final marks", () => {
    const r = rig(PLAN([' ', ' ']))
    r.feed.start(r.job, r.at, r.publish)
    r.files.set('/home/ana/plan.md', PLAN(['x', '~']))
    r.open()[0].fire('change')
    r.feed.finish('j4')
    expect(r.statuses()).toEqual(['tt', 'da'])
    expect(r.open()).toHaveLength(0)
    expect(r.timers.size).toBe(0)
    r.advance(1000)
    expect(r.published).toHaveLength(2)
  })

  test("another job's end changes nothing", () => {
    const r = rig(PLAN([' ', ' ']))
    r.feed.start(r.job, r.at, r.publish)
    r.feed.finish('j9')
    expect(r.published).toHaveLength(1)
    expect(r.open()).toHaveLength(1)
  })

  test('a new build job replaces the watcher: one at a time', () => {
    const r = rig(PLAN([' ', ' ']))
    r.feed.start(r.job, r.at, r.publish)
    r.feed.start({ id: 'j5', number: 5 }, { plan: '/home/ana/plan.md', phase: 2 }, r.publish)
    expect(r.open()).toHaveLength(1)
    expect(r.watchers[0].closed).toBe(true)
    expect(r.published.at(-1)).toEqual([{ id: 'p2-t1', title: 'creates a task', status: 'todo', agent: '#5' }])
    r.feed.finish('j4')
    expect(r.published).toHaveLength(2)
  })

  test('stop (the session shuts down) leaves no timer or watcher, and nothing is published after it', () => {
    const r = rig(PLAN([' ', ' ']))
    r.feed.start(r.job, r.at, r.publish)
    r.open()[0].fire('change')
    expect(r.timers.size).toBe(1)
    r.feed.stop()
    expect(r.timers.size).toBe(0)
    expect(r.open()).toHaveLength(0)
    r.feed.finish('j4')
    expect(r.published).toHaveLength(1)
  })

  test('a plan that cannot be read at the start is not an error and not published', () => {
    const r = rig('')
    r.files.delete('/home/ana/plan.md')
    r.feed.start(r.job, r.at, r.publish)
    expect(r.published).toHaveLength(0)
    expect(r.open()).toHaveLength(0)
  })
})
