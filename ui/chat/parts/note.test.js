import { describe, expect, test } from 'bun:test'
import { TEXTS, renderNote } from './note.js'

const t = (key) => key
const hostile = '<script>alert("x")</script> & \'q\''

describe('renderNote', () => {
  test('escapes label and text', () => {
    const html = renderNote({ label: hostile, text: hostile }, {}, t)
    expect(html).not.toContain('<script>')
    expect((html.match(/&lt;script&gt;alert\(&quot;x&quot;\)&lt;\/script&gt; &amp; &#39;q&#39;/g) || []).length).toBe(2)
  })
  test('shows the label before the text', () => {
    const html = renderNote({ label: 'engram', text: 'saved: you prefer pnpm' }, {}, t)
    expect(html.indexOf('engram')).toBeLessThan(html.indexOf('saved: you prefer pnpm'))
    expect(html).toContain('p-note-label')
  })
  test('a note without a label has no label span', () => {
    expect(renderNote({ text: 'plain' }, {}, t)).not.toContain('p-note-label')
  })
})

describe('texts', () => {
  test('both languages have the same keys, all prefixed', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.note.')).toBe(true)
  })
})
