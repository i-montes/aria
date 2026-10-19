// Run: bun test ui/chat/parts/files.test.js
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { TEXTS, filterFiles, renderFilePicker } from './files.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const hostile = '<script>alert("x")</script> & \'q\''
const files = [
  'src/queue/consumer.ts', 'src/queue/contract.ts', 'src/routes/send.ts', 'src/lib/push.ts', 'src/lib/env.ts', 'src/lib/env.test.ts',
  'docs/specs/push-queue.md', 'vite.config.ts', '.env.example', 'package.json', 'README.md', 'src/queue/', 'docs/',
]
const paths = (query, list = files) => filterFiles(list, query).map((x) => x.path)

describe('filterFiles', () => {
  test('no query keeps the order', () => {
    expect(paths('')).toEqual(files)
  })

  test('the file name beats the folder: a name prefix, then inside the name, then anywhere in the path', () => {
    const r = filterFiles(files, 'env')
    expect(r.map((x) => x.path)).toEqual(['src/lib/env.ts', 'src/lib/env.test.ts', '.env.example'])
    expect(r[0].score).toBe(0)
    expect(r[0].name).toBe('env.ts')
    expect(r[0].dir).toBe('src/lib/')
    expect(r[0].nameRanges).toEqual([[0, 3]])
    expect(r.find((x) => x.path === '.env.example').score).toBe(2) // ".env.example": "env" starts at 1
    const inPath = filterFiles(files, 'specs')
    expect(inPath.map((x) => x.path)).toEqual(['docs/specs/push-queue.md'])
    expect(inPath[0].score).toBe(10 + 'docs/specs/push-queue.md'.indexOf('specs'))
    expect(inPath[0].nameRanges).toEqual([])
    expect(inPath[0].dirRanges).toEqual([[5, 10]])
  })

  test('letters in order inside the path match when they stay close together', () => {
    expect(filterFiles(['src/queue/consumer.ts'], 'cmt')).toEqual([]) // too spread out (more than query length + 3)
    const near = filterFiles(['src/queue/consumer.ts'], 'cnsu')
    expect(near).toHaveLength(1)
    expect(near[0].score).toBeGreaterThanOrEqual(40)
    expect(near[0].nameRanges).toEqual([[0, 1], [2, 5]])
  })

  test('a folder ends with a slash, and ranks like a file by its last segment', () => {
    const r = filterFiles(files, 'queue')
    const folder = r.find((x) => x.path === 'src/queue/')
    expect(folder.isFolder).toBe(true)
    expect(folder.name).toBe('queue/')
    expect(folder.dir).toBe('src/')
    expect(folder.score).toBe(0)
    expect(r.find((x) => x.path === 'docs/specs/push-queue.md').score).toBe(1 + 'push-queue.md'.indexOf('queue'))
  })

  test('nothing matches gives an empty list', () => {
    expect(paths('zzz')).toEqual([])
  })
})

describe('renderFilePicker', () => {
  const picker = (extra = {}) => ({ items: files, query: '', active: 0, open: true, ...extra })

  test('one option per match, with a file or a folder icon, and the pick and hover emits', () => {
    const html = renderFilePicker(picker({ items: ['a/b.ts', 'docs/'] }), {}, t)
    expect(html.match(/data-chat="pick"/g)).toHaveLength(2)
    expect(html.match(/data-chat="picker-hover"/g)).toHaveLength(2)
    expect(html).toContain('p-files-icon is-file')
    expect(html).toContain('p-files-icon is-folder')
    expect(html).toContain('chat.files.count.other n=2')
    expect(html).toContain('chat.files.title')
  })

  test('the name is separated from its folder and the active row is marked', () => {
    const html = renderFilePicker(picker({ items: ['src/lib/env.ts', 'a.md'], active: 1 }), {}, t)
    expect(html).toContain('p-files-dir')
    expect(html.match(/p-files-dir/g)).toHaveLength(1) // a.md has no folder
    expect(html.match(/aria-selected="true"/g)).toHaveLength(1)
    expect(html).toMatch(/class="p-files-row on" data-chat="pick" data-index="1"/)
  })

  test('the matched letters are highlighted in the name and in the folder', () => {
    const html = renderFilePicker(picker({ items: ['docs/specs/a.md'], query: 'specs' }), {}, t)
    expect(html).toContain('<span class="p-files-dir-hl">specs</span>')
    const inName = renderFilePicker(picker({ items: ['docs/specs/a.md'], query: 'a.m' }), {}, t)
    expect(inName).toContain('<span class="p-files-hl">a.m</span>')
  })

  test('open and closed share one element', () => {
    expect(renderFilePicker(picker(), {}, t)).toContain('class="p-files is-open"')
    expect(renderFilePicker(picker({ open: false }), {}, t)).toContain('class="p-files"')
  })

  test('an empty result names the query, escaped', () => {
    const html = renderFilePicker(picker({ items: [], query: hostile }), {}, t)
    expect(html).not.toContain('<ul')
    expect(html).not.toContain('<script>')
    expect(html).toContain('chat.files.empty')
    expect(html).toContain('&lt;script&gt;')
  })

  test('paths are escaped', () => {
    const html = renderFilePicker(picker({ items: [`dir "${hostile}"/${hostile}.ts`] }), {}, t)
    expect(html).not.toContain('<script>')
    expect(html).toContain('&lt;script&gt;')
    expect(html).toContain('&quot;')
    expect(html).toContain('&amp;')
    const found = renderFilePicker(picker({ items: [`x/${hostile}.ts`], query: 'script' }), {}, t)
    expect(found).not.toContain('<script>')
  })
})

describe('texts', () => {
  test('every key used exists in both languages with the same keys', () => {
    const source = readFileSync(new URL('./files.js', import.meta.url), 'utf8')
    const used = [...source.matchAll(/\bt\('(chat\.[a-zA-Z.]+)'/g)].map((m) => m[1])
    expect(used.length).toBeGreaterThan(5)
    for (const lang of ['en', 'es']) for (const key of used) expect(TEXTS[lang][key], `${lang}: ${key}`).toBeTruthy()
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.files.')).toBe(true)
  })
})
