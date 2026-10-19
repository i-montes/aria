// Run: bun test ui/chat/parts/phases.test.js
import { describe, expect, test } from 'bun:test'
import { TEXTS, phasesSummary, renderPhases } from './phases.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const EVIL = `<script>alert("x")</script> & 'q'`
const phases = [
  { name: 'Contract', tasks: ['A', 'B'] },
  { name: 'Consumer', tasks: ['C'] },
]

describe('phases', () => {
  test('lists numbered phases with their task counts and bullets', () => {
    const html = renderPhases({ phases }, {}, t)
    expect(html).toContain('Contract')
    expect(html).toContain('Consumer')
    expect(html.match(/class="p-phases-num"[^>]*>(\d)</g)).toHaveLength(2)
    expect(html.match(/<li class="p-phases-task"/g)).toHaveLength(3)
    expect(html).toContain('chat.phases.tasksN n=2')
    expect(html).toContain('chat.phases.tasks1 n=1')
  })
  test('summary pluralizes phases and tasks', () => {
    expect(phasesSummary(phases, t)).toBe('chat.phases.phasesN n=2 · chat.phases.tasksN n=3')
    expect(phasesSummary([{ name: 'x', tasks: ['a'] }], t)).toBe('chat.phases.phases1 n=1 · chat.phases.tasks1 n=1')
    expect(phasesSummary([], t)).toBe('chat.phases.phasesN n=0 · chat.phases.tasksN n=0')
  })
  test('empty and odd input renders a header and an empty line, never throws', () => {
    for (const props of [{}, { phases: [] }, undefined, { phases: [null, {}, { name: 'x' }, { name: 'y', tasks: 'nope' }] }]) {
      const html = renderPhases(props, undefined, t)
      expect(html).toContain('class="p-phases')
    }
    expect(renderPhases({ phases: [] }, {}, t)).toContain('chat.phases.empty')
  })
  test('static mode turns animations off', () => {
    expect(renderPhases({ phases }, { animate: false }, t)).toContain('is-static')
  })
  test('every text field is escaped', () => {
    const html = renderPhases({ phases: [{ name: EVIL, tasks: [EVIL] }] }, {}, t)
    expect(html).not.toContain('<script')
    expect(html).not.toContain('alert("x")')
    expect(html).toContain('&lt;script&gt;')
    expect(html).toContain('&amp;')
    expect(html).toContain('&#39;q&#39;')
  })
})

test('texts: both languages have the same keys, prefixed', () => {
  expect(Object.keys(TEXTS.en).sort()).toEqual(Object.keys(TEXTS.es).sort())
  for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.phases.')).toBe(true)
})
