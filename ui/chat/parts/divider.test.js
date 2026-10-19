import { describe, expect, test } from 'bun:test'
import { TEXTS, renderDivider } from './divider.js'

const t = (key) => key
const hostile = '<script>alert("x")</script> & \'q\''

describe('renderDivider', () => {
  test('escapes the text', () => {
    const html = renderDivider({ text: hostile }, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).toContain('&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &#39;q&#39;')
  })
  test('is a separator with two rules around the text', () => {
    const html = renderDivider({ text: 'context at 40% · compacted' }, {}, t)
    expect(html).toContain('role="separator"')
    expect(html.match(/p-divider-rule/g)).toHaveLength(2)
    expect(html).toContain('context at 40% · compacted')
  })
})

describe('texts', () => {
  test('both languages have the same keys, all prefixed', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.divider.')).toBe(true)
  })
})
