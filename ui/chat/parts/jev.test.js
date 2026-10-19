import { describe, expect, test } from 'bun:test'
import { TEXTS, modeKey, renderJev } from './jev.js'

const t = (key) => key
const hostile = '<script>alert("x")</script> & \'q\''

describe('modeKey', () => {
  test('known modes map to a text key, others to null', () => {
    expect(modeKey('focused task')).toBe('chat.jev.mode.focused_task')
    expect(modeKey('Needs thinking')).toBe('chat.jev.mode.needs_thinking')
    expect(modeKey('quick')).toBe('chat.jev.mode.quick')
    expect(modeKey('something else')).toBe(null)
  })
})

describe('renderJev', () => {
  test('escapes mode and action', () => {
    const html = renderJev({ mode: hostile, action: hostile }, {}, t)
    expect(html).not.toContain('<script>')
    expect((html.match(/&lt;script&gt;alert\(&quot;x&quot;\)&lt;\/script&gt; &amp; &#39;q&#39;/g) || []).length).toBe(2)
  })
  test('a known mode is translated', () => {
    expect(renderJev({ mode: 'quick' }, {}, t)).toContain('chat.jev.mode.quick')
  })
  test('without an action there is no separator', () => {
    const html = renderJev({ mode: 'quick' }, {}, t)
    expect(html).not.toContain('·')
    expect(renderJev({ mode: 'quick', action: 'searches memory' }, {}, t)).toContain('·')
  })
  test('the accent must be a hex color, else the default is used', () => {
    expect(renderJev({ mode: 'quick', accent: '#7DD3FC' }, {}, t)).toContain('style="color: #7DD3FC"')
    const bad = renderJev({ mode: 'quick', accent: 'red;x:"><script>' }, {}, t)
    expect(bad).not.toContain('<script>')
    expect(bad).not.toContain('red')
    expect(bad).toContain('style="color: #B9A8FF"')
  })
})

describe('texts', () => {
  test('both languages have the same keys, all prefixed', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.jev.')).toBe(true)
  })
})
