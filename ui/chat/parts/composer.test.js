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
    const html = renderComposer({}, {}, t)
    expect(html).toContain('p-composer-hints')
    expect(html).toContain('<kbd class="p-composer-kbd">/</kbd>')
    expect(html).toContain('chat.composer.hint.commands')
    expect(html).toContain('alt+a')
    expect(html).toContain('shift+enter')
    expect(html).toContain('chat.composer.hint.send')
    expect(html).toContain('chat.composer.hint.newline')
    expect(renderComposer({ showHints: false }, {}, t)).not.toContain('p-composer-hints')
  })

  test('while busy the hints say esc stops and enter queues', () => {
    const html = renderComposer({ busy: true }, {}, t)
    expect(html).toContain('esc')
    expect(html).toContain('chat.composer.hint.stop')
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
