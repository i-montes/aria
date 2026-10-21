import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { TEXTS, hasLiveWork, isFailed, jobMeta, renderJobRows, renderStrip, resultAgent } from './background.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const NOW = 1_800_000_000_000
const job = (over = {}) => ({
  id: 'j1', number: 1, session: 's1', kind: 'process', title: 'bun dev', command: 'bun dev', task: null, folder: '~/proj', state: 'running',
  startedAt: NOW - 12 * 60_000, endedAt: null, url: 'http://localhost:3000', exit: null, isolation: null, worktree: null, writes: false,
  result: null, conflict: false, dismissed: false, ...over,
})
const pills = (html) => html.match(/data-chat="bg-open"/g)?.length ?? 0

describe('renderStrip', () => {
  test('only the active session, only running, queued and failed-not-dismissed', () => {
    const jobs = [
      job({ id: 'a' }),
      job({ id: 'b', session: 's2' }),
      job({ id: 'c', state: 'queued' }),
      job({ id: 'd', state: 'failed', endedAt: NOW }),
      job({ id: 'e', state: 'failed', dismissed: true }),
      job({ id: 'f', state: 'exited', exit: 0 }),
      job({ id: 'g', kind: 'agent', state: 'done' }),
      job({ id: 'h', state: 'stopped' }),
      job({ id: 'i', state: 'cut' }),
    ]
    const html = renderStrip({ jobs, session: 's1', now: NOW }, {}, t)
    expect(pills(html)).toBe(3)
    for (const id of ['a', 'c', 'd']) expect(html).toContain(`data-chat="bg-open" data-id="${id}"`)
    for (const id of ['b', 'e', 'f', 'g', 'h', 'i']) expect(html).not.toContain(`data-id="${id}"`)
    expect(html).toContain('tone-failed')
    expect(html).toContain('tone-queued')
    expect(html).toContain('tone-process')
  })
  test('a process that exited non-zero stays as a failed pill with a dismiss, and its dot is red', () => {
    const exited = job({ id: 'x', state: 'exited', exit: 2, endedAt: NOW })
    expect(isFailed(exited)).toBe(true)
    expect(isFailed(job({ state: 'exited', exit: 0 }))).toBe(false)
    expect(isFailed(job({ state: 'stopped', exit: null }))).toBe(false)
    const html = renderStrip({ jobs: [exited], session: 's1', now: NOW }, {}, t)
    expect(html).toContain('data-chat="bg-open" data-id="x"')
    expect(html).toContain('tone-failed')
    expect(html).toContain('data-chat="bg-dismiss" data-id="x"')
    expect(html).not.toContain('data-chat="bg-stop"')
    expect(html).toContain('chat.bg.end.exited exit=2')
    expect(renderStrip({ jobs: [{ ...exited, dismissed: true }], session: 's1', now: NOW }, {}, t)).toBe('')
    expect(renderJobRows({ jobs: [exited], now: NOW }, {}, t)).toContain('tone-failed')
  })
  test('five jobs: four pills and +1', () => {
    const jobs = [1, 2, 3, 4, 5].map((n) => job({ id: `j${n}`, number: n }))
    const html = renderStrip({ jobs, session: 's1', now: NOW }, {}, t)
    expect(pills(html)).toBe(4)
    expect(html).not.toContain('data-id="j5"')
    expect(html).toContain('data-chat="bg-list"')
    expect(html).toContain('chat.bg.more n=1')
    const four = renderStrip({ jobs: jobs.slice(0, 4), session: 's1', now: NOW }, {}, t)
    expect(pills(four)).toBe(4)
    expect(four).not.toContain('bg-list')
  })
  test('a process shows its port and minutes; an agent its status word; queued says queued', () => {
    expect(jobMeta(job(), NOW, t)).toBe(':3000 · chat.bg.minutes n=12')
    expect(jobMeta(job({ url: null }), NOW, t)).toBe('chat.bg.minutes n=12')
    expect(jobMeta(job({ url: 'https://example.com/x' }), NOW, t)).toBe(':443 · chat.bg.minutes n=12')
    expect(jobMeta(job({ startedAt: NOW - 20_000 }), NOW, t)).toBe(':3000 · chat.bg.minutes n=0')
    const es = (key) => ({ 'chat.agents.status.writing': 'escribiendo', 'chat.agents.status.running': 'ejecutando', 'chat.agents.status.testing': undefined })[key] ?? key
    expect(jobMeta(job({ kind: 'agent', writes: true, url: null }), NOW, es)).toBe('escribiendo')
    expect(jobMeta(job({ kind: 'agent', writes: false, url: null }), NOW, es)).toBe('ejecutando')
    expect(jobMeta(job({ kind: 'agent', status: 'testing', url: null }), NOW, es)).toBe('testing')
    expect(jobMeta(job({ state: 'queued' }), NOW, t)).toBe('chat.bg.queued')
    expect(jobMeta(job({ kind: 'agent', state: 'queued' }), NOW, t)).toBe('chat.bg.queued')
    expect(jobMeta(job({ state: 'failed' }), NOW, t)).toBe('failed')
  })
  test('stop, open and dismiss carry the job id', () => {
    const html = renderStrip({ jobs: [job({ id: 'j7' }), job({ id: 'j8', state: 'failed' })], session: 's1', now: NOW }, {}, t)
    expect(html).toContain('data-chat="bg-open" data-id="j7"')
    expect(html).toContain('data-chat="bg-stop" data-id="j7"')
    expect(html).toContain('data-chat="bg-dismiss" data-id="j8"')
    expect(html).not.toContain('data-chat="bg-stop" data-id="j8"')
    expect(html).not.toContain('data-chat="bg-dismiss" data-id="j7"')
    expect(html).toContain('aria-label="chat.bg.stop: bun dev"')
  })
  test('nothing to show renders nothing', () => {
    expect(renderStrip({ jobs: [], session: 's1', now: NOW }, {}, t)).toBe('')
    expect(renderStrip({ jobs: [job({ session: 's2' })], session: 's1', now: NOW }, {}, t)).toBe('')
    expect(renderStrip({}, {}, t)).toBe('')
  })
  test('escapes the title', () => {
    const html = renderStrip({ jobs: [job({ title: '<img src=x onerror=alert(1)>' })], session: 's1', now: NOW }, {}, t)
    expect(html).not.toContain('<img')
  })
})

describe('texts', () => {
  test('every key used exists in both languages with the same keys, all prefixed', () => {
    const source = readFileSync(new URL('./background.js', import.meta.url), 'utf8')
    const used = [...source.matchAll(/\bt\('(chat\.bg\.[a-zA-Z.]+)'/g)].map((m) => m[1])
    expect(used.length).toBeGreaterThan(3)
    for (const lang of ['en', 'es']) for (const key of used) expect(TEXTS[lang][key], `${lang}: ${key}`).toBeTruthy()
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.bg.')).toBe(true)
    for (const outcome of ['exited', 'failed', 'stopped', 'done', 'conflict', 'cut']) expect(TEXTS.en[`chat.bg.note.${outcome}`]).toContain('{title}')
    expect(TEXTS.en['chat.bg.note.exited']).toBe('{title} ended (exit {exit})')
  })
})

describe('renderJobRows', () => {
  const sessions = [{ id: 's1', title: 'Shop' }, { id: 's2', title: 'Docs <b>' }]
  const rows = (props) => renderJobRows({ sessions, active: 's1', now: NOW, ...props }, {}, t)
  const ids = (html) => [...html.matchAll(/data-job="([^"]+)"/g)].map((m) => m[1])

  test('rows are grouped by session only when several have jobs, the active one first', () => {
    const one = rows({ jobs: [job({ id: 'a' }), job({ id: 'b', state: 'queued' })] })
    expect(one).toContain('chat.bg.running')
    expect(one).not.toContain('Shop')
    expect(ids(one)).toEqual(['a', 'b'])
    const two = rows({ jobs: [job({ id: 'x', session: 's2' }), job({ id: 'a' })], active: 's1' })
    expect(two).toContain('Shop')
    expect(two).toContain('Docs &lt;b&gt;')
    expect(two).not.toContain('Docs <b>')
    expect(two.indexOf('Shop')).toBeLessThan(two.indexOf('Docs &lt;b&gt;'))
    expect(ids(two)).toEqual(['a', 'x'])
    // only running or queued work counts: a finished job of another session does not split Running into groups
    const finishedElsewhere = rows({ jobs: [job({ id: 'a' }), job({ id: 'f', session: 's2', state: 'exited', exit: 0, endedAt: NOW })] })
    expect(finishedElsewhere).not.toContain('Shop')
    expect(ids(finishedElsewhere)).toEqual(['a', 'f'])
    expect(rows({ jobs: [] })).toBe('')
  })

  test('a process row has Open only with a URL, Output and Stop; an agent row Session and Stop', () => {
    const p = rows({ jobs: [job({ id: 'p1' })] })
    expect(p).toContain('class="p-agents-row')
    expect(p).toContain('chat.bg.kind.process')
    expect(p).toContain('data-chat="bg-url" data-id="p1"')
    expect(p).toContain('data-chat="bg-output" data-id="p1"')
    expect(p).toContain('aria-expanded="false"')
    expect(p).toContain('data-chat="bg-stop" data-id="p1"')
    expect(p).not.toContain('bg-session')
    expect(p).toContain(':3000 · chat.bg.minutes n=12')
    const noUrl = rows({ jobs: [job({ id: 'p2', url: null })] })
    expect(noUrl).not.toContain('bg-url')
    expect(noUrl).toContain('data-chat="bg-output" data-id="p2"')
    const a = rows({ jobs: [job({ id: 'g1', kind: 'agent', title: 'fix tests', task: 'fix the tests', url: null, writes: true })] })
    expect(a).toContain('chat.bg.kind.agent')
    expect(a).toContain('data-chat="bg-session" data-id="g1"')
    expect(a).toContain('chat.bg.session')
    expect(a).toContain('data-chat="bg-stop" data-id="g1"')
    expect(a).not.toContain('bg-output')
    expect(a).not.toContain('bg-url')
    for (const html of [p, a]) expect(html).toContain('class="p-bg-act')
  })

  test('finished rows say how they ended and name a kept worktree', () => {
    const jobs = [
      job({ id: 'e0', state: 'exited', exit: 0, endedAt: NOW - 5000 }),
      job({ id: 'e1', state: 'exited', exit: 1, endedAt: NOW - 1000 }),
      job({ id: 'st', state: 'stopped', endedAt: NOW - 9000 }),
      job({ id: 'fa', state: 'failed', endedAt: NOW - 8000, dismissed: true }),
      job({ id: 'cu', state: 'cut', endedAt: NOW - 7000 }),
      job({ id: 'dn', kind: 'agent', state: 'done', endedAt: NOW - 2000, worktree: '/projects/.aria-worktrees/repo-j9', conflict: true, result: 'ok' }),
    ]
    const html = rows({ jobs })
    expect(html).toContain('chat.bg.finished')
    expect(html).not.toContain('chat.bg.running')
    expect(ids(html)).toEqual(['e1', 'dn', 'e0', 'cu', 'fa', 'st'])
    expect(html).toContain('chat.bg.end.exited exit=0')
    expect(html).toContain('chat.bg.end.exited exit=1')
    for (const end of ['stopped', 'failed', 'cut', 'done']) expect(html).toContain(`chat.bg.end.${end}`)
    expect(html).toContain('chat.bg.worktree path=/projects/.aria-worktrees/repo-j9')
    expect(html).not.toContain('bg-stop')
    expect(html).not.toContain('bg-url')
    expect(html).toContain('data-chat="bg-session" data-id="dn"')
    expect(html).toContain('chat.bg.result')
  })

  test('an expanded process row shows its output lines escaped', () => {
    const html = rows({ jobs: [job({ id: 'p1' })], output: { p1: ['ready on :3000', '<script>x</script> & y'] } })
    expect(html).toContain('aria-expanded="true"')
    expect(html).toContain('ready on :3000')
    expect(html).toContain('&lt;script&gt;x&lt;/script&gt; &amp; y')
    expect(html).not.toContain('<script>')
    expect(rows({ jobs: [job({ id: 'p1' })], output: { p1: [] } })).toContain('chat.bg.noOutput')
    expect(rows({ jobs: [job({ id: 'p1' })] })).not.toContain('p-bg-output')
  })

  test('the row the list was opened on is outlined', () => {
    const html = rows({ jobs: [job({ id: 'a' }), job({ id: 'b' })], open: 'b' })
    expect(html.match(/is-target/g)).toHaveLength(1)
    expect(html).toMatch(/data-job="b"[^>]*class="[^"]*is-target|is-target[^"]*"[^>]*data-job="b"/)
  })

  test('every value is escaped', () => {
    const evil = '<img src=x onerror=alert(1)>'
    const html = rows({ jobs: [job({ id: evil, title: evil, url: 'http://x/"><b>' }), job({ id: 'z', state: 'done', kind: 'agent', worktree: evil, endedAt: NOW })] })
    expect(html).not.toContain('<img')
    expect(html).not.toContain('"><b>')
  })
})

describe('live work and results', () => {
  test('the hint counts running, queued and failed-not-dismissed jobs of the session only', () => {
    expect(hasLiveWork([job({ state: 'exited', exit: 0 }), job({ state: 'failed', dismissed: true })], 's1')).toBe(false)
    expect(hasLiveWork([job({ session: 's2' })], 's1')).toBe(false)
    expect(hasLiveWork([job({ state: 'queued' })], 's1')).toBe(true)
    expect(hasLiveWork([job({ state: 'failed' })], 's1')).toBe(true)
    expect(hasLiveWork(undefined, 's1')).toBe(false)
  })
  test('a finished agent without a watch shows its result as the session', () => {
    const agent = resultAgent(job({ kind: 'agent', title: 'fix tests', state: 'done', result: '## Learnings\nall green' }), t)
    expect(agent).toMatchObject({ role: 'subagent', task: 'fix tests', kind: 'done', steps: [{ kind: 'msg', text: '## Learnings\nall green' }] })
    expect(agent.status).toBe('chat.bg.end.done')
    expect(resultAgent(job({ kind: 'agent', state: 'failed', result: null }), t).steps).toEqual([])
  })
})
