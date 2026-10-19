// Run: bun test ui/chat/parts/tasks.test.js
import { describe, expect, test } from 'bun:test'
import { TEXTS, normalizeStatus, renderTasks, taskStats } from './tasks.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const EVIL = `<script>alert("x")</script> & 'q'`
const tasks = [
  { label: 'Define the contract', status: 'done' },
  { label: 'Consumer with retries', status: 'active', note: 'src/queue/consumer.ts' },
  { label: 'Tests', status: 'todo' },
]
const agents = [
  { role: 'researcher', label: 'Compare queues', status: 'done' },
  { role: 'backend', label: 'Producer and consumer', status: 'active' },
  { role: 'reviewer', label: 'Review', status: 'todo' },
]

describe('stats', () => {
  test('counts done tasks and gives the active one a partial share', () => {
    expect(taskStats(tasks, 'tasks', 0.5)).toMatchObject({ total: 3, done: 1, percent: 50, complete: false })
    expect(taskStats(tasks, 'tasks', 0).percent).toBe(33)
  })
  test('agents count half a step per working agent', () => {
    expect(taskStats(agents, 'agents', 0).percent).toBe(50)
  })
  test('everything done is complete at 100', () => {
    expect(taskStats(tasks.map((x) => ({ ...x, status: 'done' })), 'tasks')).toMatchObject({ complete: true, percent: 100 })
  })
  test('empty and unknown input does not throw', () => {
    expect(taskStats([], 'tasks')).toMatchObject({ total: 0, percent: 0, complete: false })
    expect(taskStats(undefined, 'agents')).toMatchObject({ total: 0 })
    expect(normalizeStatus('weird')).toBe('todo')
    expect(normalizeStatus('active')).toBe('active')
  })
})

describe('the dock', () => {
  test('shows the count, the progress bar and a toggle', () => {
    const html = renderTasks({ tasks }, {}, t)
    expect(html).toContain('class="p-tasks is-open"')
    expect(html).toContain('data-chat="tasks-toggle"')
    expect(html).toContain('aria-expanded="true"')
    expect(html).toContain('1/3')
    expect(html).toContain('width:33%')
    expect(html).toContain('src/queue/consumer.ts')
    expect(html).toContain('aria-current="step"')
    expect(html).toContain('chat.tasks.kind.tasks')
  })
  test('the note shows only on the active task', () => {
    const html = renderTasks({ tasks: [{ label: 'a', status: 'done', note: 'hidden-note' }] }, {}, t)
    expect(html).not.toContain('hidden-note')
  })
  test('closed hides the list and the chevron flips', () => {
    const open = renderTasks({ tasks }, { open: true }, t)
    const closed = renderTasks({ tasks }, { open: false }, t)
    expect(closed).not.toContain('is-open')
    expect(closed).toContain('aria-expanded="false"')
    expect(open).toContain('chev-flip')
    expect(closed).not.toContain('chev-flip')
  })
  test('the inline log variant flips the layout and the chevron rule', () => {
    const html = renderTasks({ tasks, inline: true }, { open: true }, t)
    expect(html).toContain('is-inline')
    expect(html).not.toContain('chev-flip')
    expect(renderTasks({ tasks, inline: true }, { open: false }, t)).toContain('chev-flip')
  })
  test('a finished list reads Done and the bar turns green', () => {
    const html = renderTasks({ tasks: tasks.map((x) => ({ ...x, status: 'done' })) }, {}, t)
    expect(html).toContain('chat.tasks.done')
    expect(html).toContain('is-complete')
    expect(html).not.toContain('is-live')
  })
  test('a working list animates its bar; the pulse flag adds the pulse class', () => {
    expect(renderTasks({ tasks }, {}, t)).toContain('is-live')
    expect(renderTasks({ tasks }, { pulse: true }, t)).toContain('is-pulse')
  })
  test('plain tasks are not clickable', () => {
    const html = renderTasks({ tasks }, {}, t)
    expect(html).not.toContain('task-select')
  })
  test('nothing to show renders nothing; show=false keeps it hidden', () => {
    expect(renderTasks({ tasks: [] }, {}, t)).toBe('')
    expect(renderTasks({}, {}, t)).toBe('')
    expect(renderTasks(undefined, undefined, t)).toBe('')
    expect(renderTasks({ tasks, show: false }, {}, t)).toContain('is-hidden')
  })
  test('animations can be turned off', () => {
    expect(renderTasks({ tasks }, { animate: false }, t)).toContain('is-static')
  })
})

describe('agents', () => {
  test('rows are buttons that emit task-select with the id (role by default)', () => {
    const html = renderTasks({ tasks: agents, kind: 'agents' }, {}, t)
    expect(html).toContain('data-chat="task-select" data-id="backend"')
    expect(html).toContain('chat.tasks.status.working')
    expect(html).toContain('chat.tasks.status.queued')
    expect(html).toContain('chat.tasks.status.done')
    expect(html).toContain('chat.tasks.kind.agents')
  })
  test('an explicit id wins over the role', () => {
    const html = renderTasks({ tasks: [{ id: 'a-1', role: 'x', label: 'y', status: 'todo' }], kind: 'agents' }, {}, t)
    expect(html).toContain('data-id="a-1"')
  })
  test('selectable=false makes them plain rows', () => {
    const html = renderTasks({ tasks: agents, kind: 'agents' }, { selectable: false }, t)
    expect(html).not.toContain('task-select')
  })
  test('an unknown kind falls back to tasks', () => {
    expect(renderTasks({ tasks, kind: 'nope' }, {}, t)).toContain('chat.tasks.kind.tasks')
  })
})

describe('escaping and odd data', () => {
  test('every text field is escaped, in both kinds', () => {
    for (const kind of ['tasks', 'agents']) {
      const html = renderTasks({ kind, tasks: [{ id: EVIL, role: EVIL, label: EVIL, note: EVIL, status: 'active' }] }, {}, t)
      expect(html).not.toContain('<script')
      expect(html).not.toContain('alert("x")')
      expect(html).toContain('&lt;script&gt;')
      expect(html).toContain('&amp;')
      expect(html).toContain('&#39;q&#39;')
    }
  })
  test('an unknown status renders as todo, a missing label does not throw', () => {
    const html = renderTasks({ tasks: [{ status: 'zzz' }, null, {}] }, {}, t)
    expect(html).toContain('status-todo')
    expect(html).not.toContain('status-zzz')
  })
  test('partial is clamped', () => {
    expect(taskStats(tasks, 'tasks', 9).percent).toBe(67)
    expect(taskStats(tasks, 'tasks', -3).percent).toBe(33)
  })
})

test('texts: both languages have the same keys, prefixed', () => {
  expect(Object.keys(TEXTS.en).sort()).toEqual(Object.keys(TEXTS.es).sort())
  for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.tasks.')).toBe(true)
})
