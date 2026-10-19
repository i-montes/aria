import { describe, expect, test } from 'bun:test'
import { TEXTS, parseBlocks, parseDiff, renderAssistant, renderInline } from './assistant.js'

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

describe('parseBlocks', () => {
  test('splits text and fenced diffs in order', () => {
    const blocks = parseBlocks('before\n```diff\n-a\n+b\n```\nafter')
    expect(blocks.map((b) => b.type)).toEqual(['text', 'diff', 'text'])
    expect(blocks[0].text).toBe('before')
    expect(blocks[1].lines).toEqual([
      { kind: 'del', text: 'a' },
      { kind: 'add', text: 'b' },
    ])
    expect(blocks[2].text).toBe('after')
  })
  test('an unterminated diff (a streaming message) is still a diff', () => {
    const blocks = parseBlocks('x\n```diff\n+one')
    expect(blocks.map((b) => b.type)).toEqual(['text', 'diff'])
    expect(blocks[1].lines).toEqual([{ kind: 'add', text: 'one' }])
  })
  test('other fences stay text', () => {
    expect(parseBlocks('```js\nlet a\n```')).toEqual([{ type: 'text', text: '```js\nlet a\n```' }])
  })
  test('no blocks for empty text', () => {
    expect(parseBlocks('')).toEqual([])
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
    expect(html).toContain('<code class="p-assistant-code">&lt;script&gt;</code>')
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
