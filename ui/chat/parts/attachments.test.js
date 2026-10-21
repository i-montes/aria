// Run: bun test ui/chat/parts/attachments.test.js
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { LIMITS, TEXTS, admit, humanSize, plainText, renderAttachments, splitAttached } from './attachments.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const hostile = '<script>alert("x")</script> & \'q\''
const MB = 1024 * 1024

describe('humanSize', () => {
  test('bytes, kilobytes and megabytes like the dock writes them', () => {
    expect(humanSize(7)).toBe('7 B')
    expect(humanSize(12 * 1024)).toBe('12 KB')
    expect(humanSize(1536)).toBe('2 KB')
    expect(humanSize(1.3 * MB)).toBe('1.3 MB')
    expect(humanSize(-1)).toBe('0 B')
  })
})

describe('splitAttached', () => {
  const text = 'look at these\n\n[attached: report (final).pdf (application/pdf, 1.3 MB) at /h/.aria/attachments/s/2026-10-21/1-report_final_.pdf]\n[attached image: shot.png at /h/.aria/attachments/s/2026-10-21/2-shot.png]'
  test('the trailing attachment lines become files and leave the text', () => {
    expect(splitAttached(text)).toEqual({
      text: 'look at these',
      files: [
        { name: 'report (final).pdf', mime: 'application/pdf', size: '1.3 MB', path: '/h/.aria/attachments/s/2026-10-21/1-report_final_.pdf', image: false },
        { name: 'shot.png', mime: '', size: '', path: '/h/.aria/attachments/s/2026-10-21/2-shot.png', image: true },
      ],
    })
  })
  test('a message of attachments only has no text', () => {
    expect(splitAttached('[attached: a.txt (text/plain, 7 B) at /x/1-a.txt]')).toEqual({ text: '', files: [{ name: 'a.txt', mime: 'text/plain', size: '7 B', path: '/x/1-a.txt', image: false }] })
  })
  test('a file line for an image is an image too', () => {
    expect(splitAttached('[attached: s.png (image/png, 7 B) at /x/1-s.png]').files[0].image).toBe(true)
  })
  test('text without lines, or a line in the middle of the text, stays as it is', () => {
    expect(splitAttached('hello')).toEqual({ text: 'hello', files: [] })
    const middle = '[attached: a (b, c) at d]\nand more'
    expect(splitAttached(middle)).toEqual({ text: middle, files: [] })
    expect(splitAttached(undefined)).toEqual({ text: '', files: [] })
  })
  test('plainText is the text without its lines', () => {
    expect(plainText(text)).toBe('look at these')
    expect(plainText('hi')).toBe('hi')
  })
})

describe('admit', () => {
  const f = (name, size) => ({ name, size })
  test('files within the limits are all taken', () => {
    expect(admit([], [f('a', 10), f('b', 20)])).toEqual({ take: [f('a', 10), f('b', 20)], notes: [] })
  })
  test('more than ten files together refuses the new ones', () => {
    const have = Array.from({ length: 9 }, (_, i) => f(`h${i}`, 1))
    expect(admit(have, [f('a', 1), f('b', 1)])).toEqual({ take: [], notes: [{ key: 'chat.attach.tooMany', vars: { max: LIMITS.files } }] })
    expect(admit(have, [f('a', 1)]).take.length).toBe(1)
  })
  test('a file over 20 MB is left out, the others come in', () => {
    expect(admit([], [f('big.iso', 21 * MB), f('ok', 1)])).toEqual({ take: [f('ok', 1)], notes: [{ key: 'chat.attach.tooBig', vars: { name: 'big.iso', max: '20 MB' } }] })
  })
  test('more than 50 MB together refuses the new ones', () => {
    const have = [f('a', 19 * MB), f('b', 19 * MB)]
    expect(admit(have, [f('c', 19 * MB)])).toEqual({ take: [], notes: [{ key: 'chat.attach.tooMuch', vars: { max: '50 MB' } }] })
  })
})

describe('renderAttachments', () => {
  test('chips with names, sizes and remove', () => {
    const html = renderAttachments({ files: [{ id: 'a1', name: 'notes.txt', size: 12 * 1024 }, { id: 'a2', name: 'shot.png', size: 7 }], removable: true }, {}, t)
    expect(html).toContain('p-attach')
    expect(html.match(/class="p-pill /g).length).toBe(2)
    expect(html).toContain('notes.txt')
    expect(html).toContain('<span class="p-pill-meta">12 KB</span>')
    expect(html).toContain('<span class="p-pill-meta">7 B</span>')
    expect(html).toContain('data-chat="attach-remove" data-id="a1"')
    expect(html).toContain('data-chat="attach-remove" data-id="a2"')
    expect(html).toContain('aria-label="chat.attach.remove name=notes.txt"')
  })
  test('in the thread a chip shows the size the line gave, or that it is an image, and no remove', () => {
    const html = renderAttachments({ files: [{ name: 'a.pdf', size: '1.3 MB', path: '/x/1-a.pdf' }, { name: 's.png', image: true, path: '/x/2-s.png' }] }, {}, t)
    expect(html).toContain('<span class="p-pill-meta">1.3 MB</span>')
    expect(html).toContain('<span class="p-pill-meta">chat.attach.image</span>')
    expect(html).toContain('title="/x/1-a.pdf"')
    expect(html).not.toContain('attach-remove')
  })
  test('no files, nothing drawn', () => {
    expect(renderAttachments({ files: [] }, {}, t)).toBe('')
    expect(renderAttachments({}, {}, t)).toBe('')
  })
  test('escapes every value', () => {
    const html = renderAttachments({ files: [{ id: hostile, name: hostile, size: hostile, path: hostile }], removable: true }, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).not.toContain('"x"')
  })
})

describe('texts', () => {
  test('every key used exists in both languages with the same keys', () => {
    const source = readFileSync(new URL('./attachments.js', import.meta.url), 'utf8')
    const used = [...source.matchAll(/'(chat\.attach\.[a-zA-Z]+)'/g)].map((m) => m[1])
    expect(used.length).toBeGreaterThan(5)
    for (const lang of ['en', 'es']) for (const key of used) expect(TEXTS[lang][key], `${lang}: ${key}`).toBeTruthy()
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
  })
})
