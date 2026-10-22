// Run: bun test ui/chat/parts/schedules.test.js
import { describe, expect, test } from 'bun:test'
import { TEXTS, renderSchedules, whenLabel } from './schedules.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(' ')}` : key)
const task = (extra = {}) => ({
  id: 's-abcdefghijkl',
  title: 'Notion tasks',
  prompt: 'My tasks',
  when: { at: '07:30', days: ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'], date: null },
  grants: { connectors: { notion: 'read' }, writable: [] },
  enabled: true,
  next: new Date(2026, 9, 22, 7, 30).getTime(),
  lastResult: { at: 1, ok: true, summary: '3 tasks for today' },
  ...extra,
})

describe('the scheduled tasks panel', () => {
  test('a row per task with its time, next run and last result', () => {
    const html = renderSchedules({ tasks: [task()], now: new Date(2026, 9, 21, 9, 0).getTime() }, {}, t)
    expect(html).toContain('Notion tasks')
    expect(html).toContain('chat.schedules.everyDay at=07:30')
    expect(html).toContain('3 tasks for today')
    expect(html).toContain('data-chat="schedule-toggle"')
    expect(html).toContain('data-schedule="s-abcdefghijkl"')
    expect(html).toContain('data-enabled="false"')
    expect(html).toContain('data-chat="schedule-delete"')
  })

  test('a paused task offers to resume and shows no next run', () => {
    const html = renderSchedules({ tasks: [task({ enabled: false, next: null })], now: 0 }, {}, t)
    expect(html).toContain('data-enabled="true"')
    expect(html).toContain('chat.schedules.resume')
    expect(html).toContain('chat.schedules.paused')
  })

  test('a task waiting for the person says so and offers to approve it', () => {
    const html = renderSchedules({ tasks: [task({ enabled: false, awaiting: true, next: null, grants: { connectors: { notion: 'allow' }, writable: [] } })], now: 0 }, {}, t)
    expect(html).toContain('chat.schedules.awaiting')
    expect(html).toContain('chat.schedules.approve')
    expect(html).toContain('data-enabled="true"')
    expect(html).toContain('notion')
  })

  test('a failed run says so', () => {
    expect(renderSchedules({ tasks: [task({ lastResult: { at: 1, ok: false, summary: 'no network' } })], now: 0 }, {}, t)).toContain('chat.schedules.failed')
  })

  test('with no task it says how to make one', () => {
    expect(renderSchedules({ tasks: [], now: 0 }, {}, t)).toContain('chat.schedules.empty')
  })

  test('titles and summaries are escaped', () => {
    const html = renderSchedules({ tasks: [task({ title: '<b>x</b>', lastResult: { at: 1, ok: true, summary: '<script>' } })], now: 0 }, {}, t)
    expect(html).not.toContain('<b>x</b>')
    expect(html).not.toContain('<script>')
  })

  test('when reads as the person said it', () => {
    expect(whenLabel({ at: '18:00', days: ['mon', 'tue', 'wed', 'thu', 'fri'] }, t)).toBe('chat.schedules.weekdays at=18:00')
    expect(whenLabel({ at: '09:00', days: [], date: '2026-11-02' }, t)).toBe('chat.schedules.once at=09:00 date=2026-11-02')
    expect(whenLabel({ at: '09:00', days: ['mon', 'fri'] }, t)).toBe('chat.schedules.days at=09:00 days=chat.schedules.day.mon, chat.schedules.day.fri')
  })

  test('every key used exists in both languages', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
  })
})
