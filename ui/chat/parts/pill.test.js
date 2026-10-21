import { describe, expect, test } from 'bun:test'
import { TEXTS, renderPill } from './pill.js'

const t = (key) => key
const hostile = '<script>alert("x")</script> & \'q\''

describe('renderPill', () => {
  test('a pill has the memory chip look: dot, label, meta, pressed', () => {
    const html = renderPill({ label: 'Decision', meta: 3, dot: true, tone: 'decision', pressed: true, action: 'memory-filter', value: 'decision' }, {}, t)
    expect(html.startsWith('<button type="button" class="p-pill tone-decision"')).toBe(true)
    expect(html).toContain('aria-pressed="true"')
    expect(html).toContain('data-chat="memory-filter"')
    expect(html).toContain('data-value="decision"')
    expect(html).toContain('<span class="p-pill-dot" aria-hidden="true"></span>')
    expect(html.indexOf('p-pill-dot')).toBeLessThan(html.indexOf('Decision'))
    expect(html.indexOf('Decision')).toBeLessThan(html.indexOf('<span class="p-pill-meta">3</span>'))
    const plain = renderPill({ label: 'All', meta: 9, pressed: false, action: 'memory-filter', value: 'all' }, {}, t)
    expect(plain).toContain('aria-pressed="false"')
    expect(plain).not.toContain('p-pill-dot')
  })
  test('no pressed state means no aria-pressed; no meta means no meta span', () => {
    const html = renderPill({ label: 'x', action: 'a' }, {}, t)
    expect(html).not.toContain('aria-pressed')
    expect(html).not.toContain('p-pill-meta')
  })
  test('with extra controls the pill is a group: its main button and the extra beside it (no button in a button)', () => {
    const html = renderPill({ label: 'bun dev', meta: ':3000', dot: true, tone: 'process', action: 'bg-open', id: 'j3', extra: '<button type="button" data-chat="bg-stop">■</button>' }, {}, t)
    expect(html.startsWith('<span class="p-pill has-extra tone-process"')).toBe(true)
    expect(html).toContain('<button type="button" class="p-pill-main" data-chat="bg-open" data-id="j3"')
    expect(html.indexOf('p-pill-main')).toBeLessThan(html.indexOf('data-chat="bg-stop"'))
    expect(html.indexOf('</button>')).toBeLessThan(html.indexOf('data-chat="bg-stop"'))
  })
  test('a static pill is a span with the same look and no action', () => {
    const html = renderPill({ label: 'a.pdf', meta: '1 KB', tone: 'file', title: '/x/a.pdf', static: true }, {}, t)
    expect(html.startsWith('<span class="p-pill is-static tone-file" title="/x/a.pdf">')).toBe(true)
    expect(html).not.toContain('<button')
    expect(html).toContain('<span class="p-pill-meta">1 KB</span>')
  })
  test('escapes every value', () => {
    const html = renderPill({ label: hostile, meta: hostile, tone: hostile, action: hostile, id: hostile, value: hostile }, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).not.toContain('"x"')
  })
})

describe('texts', () => {
  test('the pill has no texts of its own', () => {
    expect(TEXTS).toEqual({ en: {}, es: {} })
  })
})
