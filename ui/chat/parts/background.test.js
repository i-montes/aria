import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { TEXTS, jobMeta, renderStrip } from './background.js'

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
