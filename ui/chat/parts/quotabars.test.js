// Run: bun test ui/chat/parts/quotabars.test.js
import { describe, expect, test } from 'bun:test'
import { TEXTS, formatReset, renderQuotaBars } from './quotabars.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const NOW = Date.parse('2026-10-19T00:00:00Z')
const go = [
  { id: '5h', label: '5h', percent: 42, resetsAt: '2026-10-19T03:20:00Z' },
  { id: 'week', label: 'Week', percent: 17.4, resetsAt: '2026-10-24T00:00:00Z' },
  { id: 'month', label: 'Month', percent: 6, resetsAt: null },
]

describe('the bars', () => {
  test('one item per window, each with its label and rounded percent', () => {
    const html = renderQuotaBars({ windows: go, now: NOW }, {}, t)
    expect(html.match(/class="p-qb-item/g)).toHaveLength(3)
    expect(html).toContain('>5h<')
    expect(html).toContain('chat.quotabars.week')
    expect(html).toContain('chat.quotabars.month')
    expect(html).toContain('42%')
    expect(html).toContain('17%')
    expect(html).toContain('width: 42%')
  })

  test('a per-model window (Fable) shows under the model\'s name', () => {
    const html = renderQuotaBars({ windows: [...go, { id: 'fable', label: 'Fable', percent: 88 }], now: NOW }, {}, t)
    expect(html.match(/class="p-qb-item/g)).toHaveLength(4)
    expect(html).toContain('>Fable<')
  })

  test('no windows draws nothing, and junk entries are skipped', () => {
    expect(renderQuotaBars({ windows: [] }, {}, t)).toBe('')
    expect(renderQuotaBars({}, {}, t)).toBe('')
    expect(renderQuotaBars({ windows: [null, { id: 'x' }, { id: '5h', label: '5h', percent: 'lots' }] }, {}, t)).toBe('')
  })

  test('the colour follows how full it is', () => {
    const level = (percent) => renderQuotaBars({ windows: [{ id: '5h', label: '5h', percent }] }, {}, t).match(/p-qb-item is-(\w+)/)[1]
    expect(level(10)).toBe('ok')
    expect(level(79)).toBe('ok')
    expect(level(80)).toBe('warn')
    expect(level(94)).toBe('warn')
    expect(level(95)).toBe('crit')
    expect(level(150)).toBe('crit')
  })

  test('the bar and the number are clamped to 0..100', () => {
    const html = renderQuotaBars({ windows: [{ id: '5h', label: '5h', percent: 180 }, { id: 'week', label: 'Week', percent: -5 }] }, {}, t)
    expect(html).toContain('width: 100%')
    expect(html).toContain('width: 0%')
    expect(html).not.toContain('180')
  })

  test('the tooltip says how much is used and when it resets', () => {
    const html = renderQuotaBars({ windows: go, now: NOW }, {}, t)
    expect(html).toContain('chat.quotabars.used label=5h,percent=42')
    expect(html).toContain('chat.quotabars.resets when=3h 20m')
    expect(renderQuotaBars({ windows: [go[2]], now: NOW }, {}, t)).not.toContain('chat.quotabars.resets')
  })

  test('labels and ids are escaped', () => {
    const html = renderQuotaBars({ windows: [{ id: '<b>', label: '<script>alert(1)</script>', percent: 5 }] }, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).toContain('&lt;script&gt;')
  })
})

describe('formatReset', () => {
  test('minutes, hours with minutes, days with hours', () => {
    const at = (ms) => new Date(NOW + ms).toISOString()
    expect(formatReset(at(25 * 60_000), NOW)).toBe('25m')
    expect(formatReset(at(3 * 3_600_000 + 20 * 60_000), NOW)).toBe('3h 20m')
    expect(formatReset(at(5 * 86_400_000 + 4 * 3_600_000), NOW)).toBe('5d 4h')
    expect(formatReset(at(2 * 3_600_000), NOW)).toBe('2h')
  })

  test('already past, empty or unreadable gives nothing', () => {
    expect(formatReset(new Date(NOW - 1000).toISOString(), NOW)).toBe('')
    expect(formatReset(null, NOW)).toBe('')
    expect(formatReset('not a date', NOW)).toBe('')
  })
})

describe('texts', () => {
  test('both languages have the same keys', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
  })
})
