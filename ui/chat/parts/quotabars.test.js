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

describe('the compact bar', () => {
  test('beside the name only the 5-hour window shows, as a label, a bar and a percent', () => {
    const html = renderQuotaBars({ windows: go, now: NOW }, {}, t)
    expect(html.match(/class="p-qb-inline/g)).toHaveLength(1)
    const inline = html.slice(html.indexOf('p-qb-inline'), html.indexOf('p-qb-pop'))
    expect(inline).toContain('>5h<')
    expect(inline).toContain('42%')
    expect(inline).toContain('width: 42%')
    expect(inline).not.toContain('17%')
    expect(inline).not.toContain('chat.quotabars.week')
  })

  test('without a 5-hour window the first one shown stands in', () => {
    const html = renderQuotaBars({ windows: [go[1], go[2]], now: NOW }, {}, t)
    const inline = html.slice(html.indexOf('p-qb-inline'), html.indexOf('p-qb-pop'))
    expect(inline).toContain('chat.quotabars.week')
    expect(inline).toContain('17%')
  })

  test('no windows draws nothing, and junk entries are skipped', () => {
    expect(renderQuotaBars({ windows: [] }, {}, t)).toBe('')
    expect(renderQuotaBars({}, {}, t)).toBe('')
    expect(renderQuotaBars({ windows: [null, { id: 'x' }, { id: '5h', label: '5h', percent: 'lots' }] }, {}, t)).toBe('')
  })

  test('the colour of the compact bar follows how full the 5-hour window is', () => {
    const level = (percent) => renderQuotaBars({ windows: [{ id: '5h', label: '5h', percent }] }, {}, t).match(/p-qb-inline is-(\w+)/)[1]
    expect(level(10)).toBe('ok')
    expect(level(79)).toBe('ok')
    expect(level(80)).toBe('warn')
    expect(level(94)).toBe('warn')
    expect(level(95)).toBe('crit')
    expect(level(150)).toBe('crit')
  })

  test('it can be reached with the keyboard and names itself', () => {
    const html = renderQuotaBars({ windows: go }, {}, t)
    expect(html).toContain('tabindex="0"')
    expect(html).toContain('aria-label="chat.quotabars.title"')
  })
})

describe('the card on hover', () => {
  const card = (html) => html.slice(html.indexOf('class="p-qb-pop"'))
  const windowsIn = (html) => (card(html).match(/class="p-qb-(hero|row) is-/g) ?? []).length

  test('the 5-hour window leads the card as a hero, big number, the time it resets in, and a thick bar', () => {
    const html = renderQuotaBars({ windows: go, now: NOW }, {}, t)
    const hero = html.slice(html.indexOf('class="p-qb-hero'), html.indexOf('class="p-qb-list"'))
    expect(hero).toContain('chat.quotabars.window.5h')
    expect(hero).toContain('>42<')
    expect(hero).toContain('chat.quotabars.resets when=3h 20m')
    expect(hero).toContain('width: 42%')
  })

  test('the other windows follow as compact rows: label, bar, percent and the time left on one line', () => {
    const html = renderQuotaBars({ windows: go, now: NOW }, {}, t)
    expect(windowsIn(html)).toBe(3)
    const list = html.slice(html.indexOf('class="p-qb-list"'))
    expect(list.match(/class="p-qb-row is-/g)).toHaveLength(2)
    expect(list.indexOf('chat.quotabars.week')).toBeLessThan(list.indexOf('chat.quotabars.month'))
    expect(list).toContain('17%')
    expect(list).toContain('6%')
    expect(list).toContain('5d')
    const week = list.slice(list.indexOf('p-qb-row is-'), list.indexOf('p-qb-row is-', list.indexOf('p-qb-row is-') + 5))
    for (const part of ['p-qb-row-label', 'p-qb-row-track', 'p-qb-row-num', 'p-qb-row-reset']) expect(week).toContain(part)
  })

  test('the header names the provider when it is known', () => {
    const html = renderQuotaBars({ windows: go, provider: 'OpenCode Go', now: NOW }, {}, t)
    expect(card(html)).toContain('OpenCode Go')
    expect(card(renderQuotaBars({ windows: go, now: NOW }, {}, t))).not.toContain('p-qb-pop-provider')
  })

  test('a per-model window (Fable) is one more row under the model\'s name', () => {
    const html = renderQuotaBars({ windows: [...go, { id: 'fable', label: 'Fable', percent: 88 }], now: NOW }, {}, t)
    expect(windowsIn(html)).toBe(4)
    expect(card(html)).toContain('>Fable<')
  })

  test('a single window is only the hero; with no 5-hour window the first stands as the hero', () => {
    expect(card(renderQuotaBars({ windows: [go[0]], now: NOW }, {}, t))).not.toContain('class="p-qb-list"')
    const html = renderQuotaBars({ windows: [go[1], go[2]], now: NOW }, {}, t)
    expect(html.slice(html.indexOf('class="p-qb-hero'))).toContain('chat.quotabars.window.week')
    expect(windowsIn(html)).toBe(2)
  })

  test('a window that never says when it resets shows no reset text, and a past one neither', () => {
    expect(renderQuotaBars({ windows: [go[2]], now: NOW }, {}, t)).not.toContain('chat.quotabars.resets')
    expect(renderQuotaBars({ windows: [{ id: '5h', label: '5h', percent: 3, resetsAt: '2020-01-01T00:00:00Z' }], now: NOW }, {}, t)).not.toContain('chat.quotabars.resets')
  })

  test('every window has its own level, and bar and number are clamped to 0..100', () => {
    const html = renderQuotaBars({ windows: [{ id: '5h', label: '5h', percent: 180 }, { id: 'week', label: 'Week', percent: -5 }, { id: 'month', label: 'Month', percent: 85 }] }, {}, t)
    expect(card(html)).toContain('p-qb-hero is-crit')
    expect(card(html)).toContain('p-qb-row is-ok')
    expect(card(html)).toContain('p-qb-row is-warn')
    expect(html).not.toContain('180')
    expect(html).not.toContain('-5')
    expect(card(html)).toContain('width: 100%')
    expect(card(html)).toContain('width: 0%')
  })

  test('labels, ids and the provider are escaped', () => {
    const html = renderQuotaBars({ windows: [{ id: '5h', label: '5h', percent: 5 }, { id: '<b>', label: '<script>alert(1)</script>', percent: 5 }], provider: '<i>x</i>' }, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).not.toContain('<i>x')
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

  test('every key the renderer uses exists', () => {
    const html = renderQuotaBars({ windows: [...go, { id: 'fable', label: 'Fable', percent: 5, resetsAt: '2026-10-20T00:00:00Z' }], now: NOW }, {}, (k) => `«${k}»`)
    for (const key of new Set([...html.matchAll(/«(chat\.quotabars\.[a-zA-Z.]+)»/g)].map((m) => m[1]))) expect(TEXTS.en[key], key).toBeDefined()
  })
})
