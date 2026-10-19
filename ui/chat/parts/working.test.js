import { describe, expect, test } from 'bun:test'
import { PHASES, TEXTS, formatElapsed, phaseLabel, renderWorking } from './working.js'

const t = (key) => key
const hostile = '<script>alert("x")</script> & \'q\''

describe('formatElapsed', () => {
  test('under a minute: seconds with one decimal', () => {
    expect(formatElapsed(0)).toBe('0.0s')
    expect(formatElapsed(1234)).toBe('1.2s')
    expect(formatElapsed(59900)).toBe('59.9s')
  })
  test('a minute or more: minutes and seconds', () => {
    expect(formatElapsed(60000)).toBe('1m 0.0s')
    expect(formatElapsed(125400)).toBe('2m 5.4s')
  })
  test('never negative or NaN', () => {
    expect(formatElapsed(-500)).toBe('0.0s')
    expect(formatElapsed(NaN)).toBe('0.0s')
  })
})

describe('phaseLabel', () => {
  test('a known phase uses its text key', () => {
    expect(phaseLabel('searching memory', '', t)).toBe('chat.working.phase.searching_memory')
    expect(phaseLabel('compacting context', '', t)).toBe('chat.working.phase.compacting_context')
  })
  test('a custom label wins and is capitalised', () => {
    expect(phaseLabel('working', ' running ls ', t)).toBe('Running ls')
  })
  test('an unknown phase is shown capitalised', () => {
    expect(phaseLabel('doing x', '', t)).toBe('Doing x')
  })
  test('every phase has a text in both languages', () => {
    for (const p of PHASES) {
      const key = `chat.working.phase.${p.replace(/ /g, '_')}`
      expect(TEXTS.en[key]).toBeTruthy()
      expect(TEXTS.es[key]).toBeTruthy()
    }
  })
})

describe('renderWorking', () => {
  test('escapes a custom label and an unknown phase', () => {
    for (const props of [{ label: hostile, startedAt: 0 }, { phase: hostile, startedAt: 0 }]) {
      const html = renderWorking(props, { now: 1000 }, t)
      expect(html).not.toContain('<script>')
      expect(html).toContain('&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &#39;q&#39;')
    }
  })
  test('elapsed time comes from now and startedAt', () => {
    const html = renderWorking({ phase: 'thinking', startedAt: 10000 }, { now: 13400 }, t)
    expect(html).toContain('3.4s')
    expect(html).toContain('role="timer"')
    expect(html).toContain('data-since="10000"')
  })
  test('draws nine cells with staggered delays', () => {
    const html = renderWorking({ phase: 'working', startedAt: 0 }, { now: 0 }, t)
    expect(html.match(/p-working-cell/g)).toHaveLength(9)
    expect(html).toContain('animation-delay: 0ms')
    expect(html).toContain('animation-delay: 180ms')
  })
  test('a missing startedAt shows zero', () => {
    expect(renderWorking({ phase: 'working' }, { now: 5000 }, t)).toContain('0.0s')
  })
})

describe('texts', () => {
  test('both languages have the same keys, all prefixed', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.working.')).toBe(true)
  })
})
