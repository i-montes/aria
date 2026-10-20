// Run: bun test ui
import { describe, expect, test } from 'bun:test'
import { renderInline, renderMarkdown } from './markdown.js'

describe('inline markdown', () => {
  test('bold, italic, strike and code', () => {
    expect(renderInline('**a** *b* _c_ ~~d~~ `e`')).toBe('<strong>a</strong> <em>b</em> <em>c</em> <del>d</del> <code class="p-md-code">e</code>')
  })

  test('nothing inside code is markdown, and everything is escaped', () => {
    expect(renderInline('`**x** <b>`')).toBe('<code class="p-md-code">**x** &lt;b&gt;</code>')
    expect(renderInline('<img src=x onerror=alert(1)> & "q"')).toBe('&lt;img src=x onerror=alert(1)&gt; &amp; &quot;q&quot;')
  })

  test('snake_case words and lone asterisks stay as they are', () => {
    expect(renderInline('read_only_mode and 2 * 3 * 4')).toBe('read_only_mode and 2 * 3 * 4')
  })

  test('links open in the browser, only with a safe scheme', () => {
    expect(renderInline('[docs](https://pi.dev/docs)')).toBe('<a class="p-md-link" href="https://pi.dev/docs" data-chat="open-link" data-url="https://pi.dev/docs">docs</a>')
    expect(renderInline('see https://example.com/a_b.')).toContain('data-url="https://example.com/a_b"')
    expect(renderInline('see https://example.com/a_b.')).toMatch(/<\/a>\.$/)
    expect(renderInline('[x](javascript:alert(1))')).not.toContain('<a')
  })

  test('a link label keeps its own markdown', () => {
    expect(renderInline('[**the** `api`](https://x.dev)')).toContain('><strong>the</strong> <code class="p-md-code">api</code></a>')
  })

  test('mentions become chips only when asked, without trailing punctuation', () => {
    expect(renderInline('look at @src/main.rs, please', { mentions: true })).toBe('look at <span class="p-user-chip">@src/main.rs</span>, please')
    expect(renderInline('mail me at a@b.com', { mentions: true })).toBe('mail me at a@b.com')
    expect(renderInline('@a', {})).toBe('@a')
  })
})

describe('block markdown', () => {
  test('paragraphs keep their line breaks', () => {
    expect(renderMarkdown('one\ntwo\n\nthree')).toBe('<p class="p-md-p">one<br>two</p><p class="p-md-p">three</p>')
  })

  test('headings, rules and quotes', () => {
    expect(renderMarkdown('## Plan')).toBe('<p class="p-md-h p-md-h2">Plan</p>')
    expect(renderMarkdown('a\n\n---\n\nb')).toContain('<hr class="p-md-hr">')
    expect(renderMarkdown('> quoted\n> **line**')).toBe('<blockquote class="p-md-quote"><p class="p-md-p">quoted<br><strong>line</strong></p></blockquote>')
  })

  test('lists, ordered, nested and with their continuation', () => {
    expect(renderMarkdown('- a\n- b')).toBe('<ul class="p-md-list"><li>a</li><li>b</li></ul>')
    expect(renderMarkdown('3. c\n4. d')).toBe('<ol class="p-md-list" start="3"><li>c</li><li>d</li></ol>')
    expect(renderMarkdown('- a\n  - a1\n  - a2\n- b')).toBe('<ul class="p-md-list"><li><p class="p-md-p">a</p><ul class="p-md-list"><li>a1</li><li>a2</li></ul></li><li>b</li></ul>')
    expect(renderMarkdown('Intro:\n- x\n- y\n\nAfter')).toBe('<p class="p-md-p">Intro:</p><ul class="p-md-list"><li>x</li><li>y</li></ul><p class="p-md-p">After</p>')
  })

  test('fenced code is escaped; a block the caller draws is left to it; an unclosed fence runs to the end', () => {
    expect(renderMarkdown('```js\nif (a < b) {}\n```')).toBe('<pre class="p-md-pre"><code>if (a &lt; b) {}</code></pre>')
    expect(renderMarkdown('```diff\n+x\n```', { code: (lang, text) => (lang === 'diff' ? `[diff:${text}]` : null) })).toBe('[diff:+x]')
    expect(renderMarkdown('```\nstreaming')).toBe('<pre class="p-md-pre"><code>streaming</code></pre>')
  })

  test('tables, with alignment and inline markdown in cells', () => {
    const html = renderMarkdown('| Name | Size |\n|:--|--:|\n| `a` | 2 |\n| b |')
    expect(html).toContain('<th style="text-align:right">Size</th>')
    expect(html).toContain('<td><code class="p-md-code">a</code></td><td style="text-align:right">2</td>')
    expect(html).toContain('<td>b</td><td style="text-align:right"></td>')
  })
})

describe('what older WebKit needs', () => {
  test('a pipe escaped inside a cell stays in the cell', () => {
    expect(renderMarkdown('| a | b |\n|---|---|\n| x \\| y | z |')).toContain('<td>x | y</td><td>z</td>')
  })

  test('the module uses no lookbehind (macOS 12 WebKit cannot parse it)', async () => {
    const source = await Bun.file(new URL('./markdown.js', import.meta.url)).text()
    expect(source).not.toMatch(/\(\?<[!=]/)
  })
})
