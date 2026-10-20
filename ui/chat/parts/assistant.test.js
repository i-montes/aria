import { describe, expect, test } from 'bun:test'
import { TEXTS, parseDiff, renderAssistant, renderInline } from './assistant.js'

const t = (key) => key
const hostile = '<script>alert("x")</script> & \'q\''

describe('renderInline', () => {
  test('backtick spans become code and everything is escaped', () => {
    expect(renderInline('Commit `a41c9e2` removed `<b>`')).toBe(
      'Commit <code class="p-assistant-code">a41c9e2</code> removed <code class="p-assistant-code">&lt;b&gt;</code>',
    )
  })
  test('an unclosed backtick stays text', () => {
    expect(renderInline('a ` b')).toBe('a ` b')
  })
})

describe('parseDiff', () => {
  test('classifies add, del, context, hunk and file headers', () => {
    const lines = parseDiff('--- a/f\n+++ b/f\n@@ -1 +1 @@\n keep\n-old\n+new')
    expect(lines.map((l) => l.kind)).toEqual(['meta', 'meta', 'meta', 'ctx', 'del', 'add'])
    expect(lines[3].text).toBe('keep')
  })
  test('after the first change a line starting with -- is a deletion', () => {
    const lines = parseDiff('-a\n--- comment')
    expect(lines.map((l) => l.kind)).toEqual(['del', 'del'])
  })
})

describe('renderAssistant', () => {
  test('escapes text, detail and source', () => {
    const html = renderAssistant({ text: hostile, detail: hostile, source: hostile }, {}, t)
    expect(html).not.toContain('<script>')
    expect((html.match(/&lt;script&gt;alert\(&quot;x&quot;\)&lt;\/script&gt; &amp; &#39;q&#39;/g) || []).length).toBe(3)
  })
  test('escapes code spans and diff lines', () => {
    const html = renderAssistant({ text: '`<script>`\n```diff\n-<script>"\n+a & b\n```' }, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).toContain('<code class="p-md-code">&lt;script&gt;</code>')
    expect(html).toContain('&lt;script&gt;&quot;')
    expect(html).toContain('a &amp; b')
  })
  test('detail and source are optional', () => {
    const html = renderAssistant({ text: 'hi' }, {}, t)
    expect(html).not.toContain('p-assistant-detail')
    expect(html).not.toContain('p-assistant-source')
  })
  test('shows detail and the source line', () => {
    const html = renderAssistant({ text: 'hi', detail: 'more', source: 'engram' }, {}, t)
    expect(html).toContain('p-assistant-detail')
    expect(html).toContain('chat.assistant.source')
    expect(html).toContain('engram')
  })
  test('the diff group is labelled and marks added and removed lines', () => {
    const html = renderAssistant({ text: '```diff\n-a\n+b\n```' }, {}, t)
    expect(html).toContain('role="group"')
    expect(html).toContain('aria-label="chat.assistant.change"')
    expect(html).toContain('is-del')
    expect(html).toContain('is-add')
  })
})

describe('texts', () => {
  test('both languages have the same keys, all prefixed', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.assistant.')).toBe(true)
  })
})

describe('markdown in the answer', () => {
  test('lists, bold and links are drawn, and a diff block keeps the design\'s diff group', () => {
    const html = renderAssistant({ text: 'Done:\n- **one**\n- [two](https://x.dev)\n\n```diff\n+a\n```' }, {}, (key) => key)
    expect(html).toContain('<ul class="p-md-list"><li><strong>one</strong></li>')
    expect(html).toContain('data-chat="open-link" data-url="https://x.dev"')
    expect(html).toContain('<div class="p-assistant-line is-add">')
  })
})
