// Run: bun test ui/chat/parts/composer.test.js
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { TEXTS, renderComposer } from './composer.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const hostile = '<script>alert("x")</script> & \'q\''
const has = (html, id) => html.includes(`id="${id}"`)

describe('renderComposer', () => {
  test('idle: a textarea and only the send button', () => {
    const html = renderComposer({ busy: false, hasText: false }, {}, t)
    expect(has(html, 'chat-input')).toBe(true)
    expect(html).toContain('<textarea')
    expect(has(html, 'chat-send')).toBe(true)
    expect(html).toMatch(/id="chat-send"[^>]*data-chat="send"|data-chat="send"[^>]*id="chat-send"/)
    expect(has(html, 'chat-stop')).toBe(false)
    expect(has(html, 'chat-queue')).toBe(false)
  })

  test('typing while idle changes nothing about the buttons', () => {
    const html = renderComposer({ busy: false, hasText: true }, {}, t)
    expect(has(html, 'chat-send')).toBe(true)
    expect(has(html, 'chat-stop')).toBe(false)
  })

  test('busy without text: only stop, in its solid light form', () => {
    const html = renderComposer({ busy: true, hasText: false }, {}, t)
    expect(has(html, 'chat-stop')).toBe(true)
    expect(html).toMatch(/id="chat-stop"[^>]*data-chat="stop"|data-chat="stop"[^>]*id="chat-stop"/)
    expect(html).toContain('p-composer-stop')
    expect(html).not.toContain('has-text')
    expect(has(html, 'chat-send')).toBe(false)
    expect(has(html, 'chat-queue')).toBe(false)
  })

  test('busy with text: stop turns outlined and the queue button appears; send stays away', () => {
    const html = renderComposer({ busy: true, hasText: true }, {}, t)
    expect(html).toContain('p-composer-stop has-text')
    expect(has(html, 'chat-queue')).toBe(true)
    expect(html).toMatch(/id="chat-queue"[^>]*data-chat="send"|data-chat="send"[^>]*id="chat-queue"/)
    expect(has(html, 'chat-send')).toBe(false)
  })

  test('the placeholder follows the state and is escaped', () => {
    expect(renderComposer({}, {}, t)).toContain('placeholder="chat.composer.placeholder"')
    expect(renderComposer({ busy: true }, {}, t)).toContain('placeholder="chat.composer.queuePlaceholder"')
    const html = renderComposer({ placeholder: hostile }, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).toContain('placeholder="&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &#39;q&#39;"')
  })

  test('rows are clamped between one and six', () => {
    expect(renderComposer({ rows: 3 }, {}, t)).toContain('rows="3"')
    expect(renderComposer({ rows: 99 }, {}, t)).toContain('rows="6"')
    expect(renderComposer({ rows: 0 }, {}, t)).toContain('rows="1"')
    expect(renderComposer({ rows: '"><script>' }, {}, t)).toContain('rows="1"')
    expect(renderComposer({}, {}, t)).toContain('rows="1"')
  })

  test('the hints line shows by default and can be hidden', () => {
    expect(renderComposer({}, {}, t)).toContain('p-composer-hints')
    expect(renderComposer({ showHints: false }, {}, t)).not.toContain('p-composer-hints')
  })

  test('idle: "/ for commands" and "alt+a subagents"; while Aria works only "enter queues your message" takes their place', () => {
    const idle = renderComposer({ agents: true }, {}, t)
    expect(idle).toContain('<kbd class="p-composer-kbd">/</kbd> chat.composer.hint.commands')
    expect(idle).toContain('<span class="p-composer-key">alt+a</span> chat.composer.hint.agents')
    expect(idle).not.toContain('chat.composer.hint.queue')
    const busy = renderComposer({ busy: true, agents: true }, {}, t)
    expect(busy).toContain('<span class="p-composer-key">enter</span> chat.composer.hint.queue')
    expect(busy).not.toContain('chat.composer.hint.commands')
    expect(busy).not.toContain('alt+a')
  })

  test('without subagents there is no "alt+a subagents"', () => {
    const html = renderComposer({}, {}, t)
    expect(html).toContain('chat.composer.hint.commands')
    expect(html).not.toContain('alt+a')
    expect(html).not.toContain('chat.composer.hint.agents')
  })

  test('the model sits under the input, first on the hints row, and opens the model picker', () => {
    const html = renderComposer({ model: 'DeepSeek V4.1 Flash', modelTitle: 'opencode-go/deepseek-v4.1-flash' }, {}, t)
    const hints = html.slice(html.indexOf('p-composer-hints'))
    expect(hints.indexOf('p-composer-model')).toBeGreaterThan(-1)
    expect(renderComposer({ busy: true, model: 'M' }, {}, t).indexOf('p-composer-model')).toBeLessThan(renderComposer({ busy: true, model: 'M' }, {}, t).indexOf('chat.composer.hint.queue'))
    expect(html).toContain('data-chat="open-models"')
    expect(html).toContain('DeepSeek V4.1 Flash')
    expect(html).toContain('title="opencode-go/deepseek-v4.1-flash"')
    expect(html.indexOf('p-composer-box')).toBeLessThan(html.indexOf('p-composer-model'))
  })

  test('no model known shows no chip, and the model name is escaped', () => {
    expect(renderComposer({}, {}, t)).not.toContain('p-composer-model')
    const html = renderComposer({ model: '<script>x</script>', modelTitle: 'a"b' }, {}, t)
    expect(html).not.toContain('<script>x')
    expect(html).toContain('&lt;script&gt;x&lt;/script&gt;')
    expect(html).toContain('title="a&quot;b"')
  })

  test('the model stays with the hints while busy', () => {
    expect(renderComposer({ busy: true, model: 'GLM 5.3' }, {}, t)).toContain('GLM 5.3')
  })

  test('while busy the hints say enter queues (Esc closes the dock, it does not stop the run)', () => {
    const html = renderComposer({ busy: true }, {}, t)
    expect(html).not.toContain('chat.composer.hint.stop') // Esc closes the dock; the stop button stops a run
    expect(html).toContain('chat.composer.hint.queue')
    expect(html).not.toContain('shift+enter')
  })

  test('the buttons are labelled', () => {
    expect(renderComposer({}, {}, t)).toContain('aria-label="chat.composer.send"')
    const busy = renderComposer({ busy: true, hasText: true }, {}, t)
    expect(busy).toContain('aria-label="chat.composer.stop"')
    expect(busy).toContain('aria-label="chat.composer.queue"')
    expect(busy).toContain('title="chat.composer.queueTitle"')
  })
})

describe('texts', () => {
  test('every key used exists in both languages with the same keys', () => {
    const source = readFileSync(new URL('./composer.js', import.meta.url), 'utf8')
    const used = [...source.matchAll(/\bt\('(chat\.[a-zA-Z.]+)'/g)].map((m) => m[1])
    expect(used.length).toBeGreaterThan(8)
    for (const lang of ['en', 'es']) for (const key of used) expect(TEXTS[lang][key], `${lang}: ${key}`).toBeTruthy()
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.composer.')).toBe(true)
  })
})
