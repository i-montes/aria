// Run: bun test ui/chat/parts/subagents.test.js
import { describe, expect, test } from 'bun:test'
import { TEXTS, agentsSummary, jobAgent, parallelAgents, renderSubagents } from './subagents.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const EVIL = `<script>alert("x")</script> & 'q'`
const agents = [
  { role: 'researcher', task: 'best practices', status: 'done', kind: 'done' },
  { role: 'frontend', task: 'form', status: 'writing', kind: 'active' },
  { role: 'backend', task: 'endpoint', status: 'testing', kind: 'active' },
  { role: 'reviewer', task: 'waits', status: 'queued', kind: 'queued' },
]

describe('summary', () => {
  test('counts running, done and queued like the design', () => {
    expect(agentsSummary(agents, t)).toBe('chat.agents.summary running=2,done=1,queued=1')
  })
  test('empty, missing and unknown kinds', () => {
    expect(agentsSummary([], t)).toBe('chat.agents.summary running=0,done=0,queued=0')
    expect(agentsSummary(undefined, t)).toBe('chat.agents.summary running=0,done=0,queued=0')
    expect(agentsSummary([{ kind: 'weird' }, null], t)).toBe('chat.agents.summary running=0,done=0,queued=1')
  })
})

describe('the list', () => {
  test('each agent is a button that emits agent-select with its role', () => {
    const html = renderSubagents({ agents }, {}, t)
    expect(html.match(/data-chat="agent-select"/g)).toHaveLength(4)
    expect(html).toContain('data-id="frontend"')
    expect(html).toContain('chat.agents.open role=frontend')
    expect(html).toContain('kind-active')
    expect(html).toContain('kind-done')
    expect(html).toContain('kind-queued')
  })
  test('an explicit id wins over the role', () => {
    expect(renderSubagents({ agents: [{ id: 'x1', role: 'r', kind: 'active' }] }, {}, t)).toContain('data-id="x1"')
  })
  test('header carries the title and the summary; the footnote has a default', () => {
    const html = renderSubagents({ agents }, {}, t)
    expect(html).toContain('chat.agents.title')
    expect(html).toContain('chat.agents.summary')
    expect(html).toContain('chat.agents.footnote')
    expect(renderSubagents({ agents, footnote: 'pick one' }, {}, t)).toContain('pick one')
  })
  test('an unknown kind is styled as queued; empty list says so', () => {
    expect(renderSubagents({ agents: [{ role: 'r', kind: 'zzz' }] }, {}, t)).toContain('kind-queued')
    const empty = renderSubagents({ agents: [] }, {}, t)
    expect(empty).toContain('chat.agents.empty')
    expect(empty).not.toContain('agent-select')
    expect(renderSubagents(undefined, undefined, t)).toContain('chat.agents.empty')
  })
  test('every text field is escaped', () => {
    const html = renderSubagents({ agents: [{ id: EVIL, role: EVIL, task: EVIL, status: EVIL, kind: 'active' }], footnote: EVIL }, {}, t)
    expect(html).not.toContain('<script')
    expect(html).not.toContain('alert("x")')
    expect(html).toContain('&lt;script&gt;')
    expect(html).toContain('&amp;')
    expect(html).toContain('&#39;q&#39;')
  })
})

test('texts: both languages have the same keys, prefixed', () => {
  expect(Object.keys(TEXTS.en).sort()).toEqual(Object.keys(TEXTS.es).sort())
  for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.agents.')).toBe(true)
})

describe('the status word from the socket', () => {
  test('is translated in the list', () => {
    const es = (key, vars) => ({ ...TEXTS.en, ...TEXTS.es })[key]?.replace(/\{(\w+)\}/g, (_, n) => vars?.[n] ?? '') ?? key
    expect(renderSubagents({ agents: [{ id: 'a', role: 'backend', task: 'x', status: 'writing', kind: 'active' }] }, {}, es)).toContain('escribiendo')
  })
})

describe('with background work', () => {
  const jobs = [
    { id: 'j1', session: 's1', kind: 'agent', title: 'tests', task: 'fix the checkout tests', state: 'running' },
    { id: 'j2', session: 's1', kind: 'process', title: 'bun dev', task: null, state: 'running' },
  ]
  const watch = [
    { id: 'a1', role: 'tests', task: 'fix the checkout tests', status: 'writing', kind: 'active' },
    { id: 'a2', role: 'frontend', task: 'form', status: 'writing', kind: 'active' },
    { id: 'a3', role: 'x', task: '', status: 'queued', kind: 'queued' },
  ]
  test('the subagents list leaves out the agent a background job runs, and keeps parallel ones', () => {
    expect(parallelAgents(watch, jobs).map((a) => a.id)).toEqual(['a2', 'a3'])
    expect(jobAgent(jobs[0], watch)?.id).toBe('a1')
    expect(jobAgent(jobs[1], watch)).toBeUndefined()
    const html = renderSubagents({ agents: watch, jobs, background: '<section id="bg"></section>' }, {}, t)
    expect(html).not.toContain('data-id="a1"')
    expect(html).toContain('data-id="a2"')
    expect(html).toContain('data-id="a3"')
    expect(html).toContain('chat.bg.title')
    expect(html.indexOf('id="bg"')).toBeLessThan(html.indexOf('data-id="a2"'))
  })
  test('with jobs and no subagents the view shows the jobs, not the empty text', () => {
    const html = renderSubagents({ agents: [watch[0]], jobs, background: '<section id="bg"></section>' }, {}, t)
    expect(html).toContain('id="bg"')
    expect(html).not.toContain('chat.agents.empty')
    expect(html).not.toContain('agent-select')
    expect(html).not.toContain('chat.agents.summary')
    expect(html).toContain('chat.bg.title')
    expect(renderSubagents({ agents: [] }, {}, t)).toContain('chat.agents.title')
  })
})
