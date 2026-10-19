// Run: bun test ui/chat/parts/commands.test.js
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { TEXTS, clampActive, filterCommands, keepVisible, renderCommandPicker, segments, stepActive } from './commands.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const hostile = '<script>alert("x")</script> & \'q\''
const items = [
  { name: 'plan', description: 'Design and write a plan before building', source: 'extension' },
  { name: 'execute', description: 'Build the plan, task by task', source: 'extension' },
  { name: 'agents', description: 'Show the subagents that are running', source: 'extension' },
  { name: 'compact', description: 'Compact the context to save tokens', source: 'builtin' },
  { name: 'model', description: 'Choose which model Aria uses', source: 'builtin' },
  { name: 'memory', description: 'Search or save what Aria remembers', source: 'skill' },
]
const names = (query, list = items) => filterCommands(list, query).map((x) => x.item.name)

describe('filterCommands', () => {
  test('no query keeps every command in its own order', () => {
    expect(names('')).toEqual(items.map((i) => i.name))
    expect(names('   ')).toEqual(items.map((i) => i.name))
  })

  test('a prefix of the name ranks first, then anywhere in the name, then letters in order, then the description', () => {
    expect(names('m')).toEqual(['model', 'memory', 'compact'])
    const r = filterCommands(items, 'm')
    expect(r[0].item.name).toBe('model')
    expect(r[1].item.name).toBe('memory')
    expect(r[2].item.name).toBe('compact') // "m" inside the name
    expect(r[0].score).toBe(0)
    expect(r[2].score).toBe(1 + 'compact'.indexOf('m'))
    // the letters c-m-p in order inside "compact"
    const fuzzy = filterCommands(items, 'cmp')
    expect(fuzzy.map((x) => x.item.name)).toEqual(['compact'])
    expect(fuzzy[0].nameRanges).toEqual([[0, 1], [2, 4]])
    expect(fuzzy[0].score).toBe(20 + 3)
    // only the description matches
    const byDesc = filterCommands(items, 'subagents')
    expect(byDesc.map((x) => x.item.name)).toEqual(['agents'])
    expect(byDesc[0].score).toBe(60)
    expect(byDesc[0].descRanges).toEqual([[9, 18]])
  })

  test('matching ignores case and ties keep the original order', () => {
    expect(names('PLAN')).toEqual(['plan', 'execute']) // execute: "plan" is in its description
    const ties = [{ name: 'bb' }, { name: 'ba' }, { name: 'bc' }]
    expect(names('b', ties)).toEqual(['bb', 'ba', 'bc'])
  })

  test('nothing matches gives an empty list', () => {
    expect(names('zzz')).toEqual([])
  })

  test('a prefix highlights its range in the name', () => {
    const r = filterCommands(items, 'pl')
    expect(r[0].nameRanges).toEqual([[0, 2]])
  })
})

describe('active helpers', () => {
  test('segments split a text into plain and highlighted pieces', () => {
    expect(segments('compact', [[0, 1], [2, 4]])).toEqual([
      { text: 'c', hit: true }, { text: 'o', hit: false }, { text: 'mp', hit: true }, { text: 'act', hit: false },
    ])
    expect(segments('abc', [])).toEqual([{ text: 'abc', hit: false }])
    expect(segments('', [])).toEqual([{ text: '', hit: false }])
  })

  test('clampActive keeps the index inside the list', () => {
    expect(clampActive(5, 3)).toBe(2)
    expect(clampActive(-1, 3)).toBe(0)
    expect(clampActive(undefined, 3)).toBe(0)
    expect(clampActive(2, 0)).toBe(0)
  })

  test('stepActive wraps around both ways', () => {
    expect(stepActive(0, 3, 1)).toBe(1)
    expect(stepActive(2, 3, 1)).toBe(0)
    expect(stepActive(0, 3, -1)).toBe(2)
    expect(stepActive(0, 0, 1)).toBe(0)
  })

  test('keepVisible scrolls only as far as the active row needs', () => {
    const view = { scrollTop: 100, clientHeight: 200 }
    expect(keepVisible({ top: 150, height: 44 }, view)).toBe(100)
    expect(keepVisible({ top: 60, height: 44 }, view)).toBe(56)
    expect(keepVisible({ top: 400, height: 44 }, view)).toBe(248)
    expect(keepVisible({ top: 0, height: 44 }, { scrollTop: 100, clientHeight: 200 })).toBe(0)
  })
})

describe('renderCommandPicker', () => {
  const picker = (extra = {}) => ({ items, query: '', active: 0, open: true, ...extra })

  test('one option per match, with pick and hover emits', () => {
    const html = renderCommandPicker(picker(), {}, t)
    expect(html.match(/data-chat="pick"/g)).toHaveLength(6)
    expect(html.match(/data-chat="picker-hover"/g)).toHaveLength(6)
    expect(html).toContain('data-index="5"')
    expect(html).toContain('role="listbox"')
    expect(html).toContain('chat.commands.title')
    expect(html).toContain('chat.commands.count.other n=6')
  })

  test('the count is singular for one match', () => {
    expect(renderCommandPicker(picker({ query: 'plan', items: items.slice(0, 1) }), {}, t)).toContain('chat.commands.count.one n=1')
  })

  test('the active row is marked and the others are not; the index follows the filtered list', () => {
    const html = renderCommandPicker(picker({ query: 'm', active: 1 }), {}, t)
    expect(html.match(/aria-selected="true"/g)).toHaveLength(1)
    expect(html.match(/class="p-cmd-row on"/g)).toHaveLength(1)
    expect(html).toMatch(/class="p-cmd-row on" data-chat="pick" data-index="1"/)
    expect(renderCommandPicker(picker({ active: 99 }), {}, t)).toMatch(/class="p-cmd-row on" data-chat="pick" data-index="5"/)
  })

  test('the matched letters are highlighted', () => {
    const html = renderCommandPicker(picker({ query: 'pl' }), {}, t)
    expect(html).toContain('<span class="p-cmd-hl">pl</span>')
  })

  test('open and closed share one element so it can animate', () => {
    expect(renderCommandPicker(picker(), {}, t)).toContain('class="p-cmd is-open"')
    const closed = renderCommandPicker(picker({ open: false }), {}, t)
    expect(closed).toContain('class="p-cmd"')
    expect(closed).toContain('aria-hidden="true"')
  })

  test('an empty result explains itself and shows the query, escaped', () => {
    const html = renderCommandPicker(picker({ query: hostile }), {}, t)
    expect(html).not.toContain('<ul')
    expect(html).toContain('chat.commands.empty')
    expect(html).not.toContain('<script>')
    expect(html).toContain('&lt;script&gt;')
    expect(html).toContain('chat.commands.emptyHint')
  })

  test('a row shows only /name and its description: no tag, no shortcut', () => {
    const html = renderCommandPicker(picker({ items: [{ name: 'agents', description: 'Show the subagents', hint: 'alt+a', source: 'builtin' }, { name: 'memory', description: 'd', source: 'skill' }] }), {}, t)
    expect(html).not.toContain('>alt+a</kbd>')
    expect(html).not.toContain('>skill</kbd>')
    expect(html).not.toContain('>builtin</kbd>')
    expect(html).toContain('/</span>')
  })

  test('the description is one line in the row, and the whole of it is in the tooltip and shown on hover', () => {
    const long = 'Plan with the planning model, then write the spec and split it into phases before building anything'
    const html = renderCommandPicker(picker({ items: [{ name: 'plan', description: long, source: 'extension' }] }), {}, t)
    expect(html).toContain(`title="${long}"`)
    expect(html).toContain('p-cmd-desc')
    const css = readFileSync(new URL('./commands.css', import.meta.url), 'utf8')
    expect(css).toMatch(/\.p-cmd-desc\s*\{[^}]*text-overflow:\s*ellipsis/)
    expect(css).toMatch(/\.p-cmd-row:hover \.p-cmd-desc[^{]*\{[^}]*white-space:\s*normal/)
  })

  test('a description with markup in its tooltip is escaped', () => {
    const html = renderCommandPicker(picker({ items: [{ name: 'x', description: hostile }] }), {}, t)
    expect(html).not.toContain('title="<script>')
    expect(html).toContain('title="&lt;script&gt;')
  })

  test('names, descriptions and sources are escaped', () => {
    const bad = [{ name: hostile, description: hostile, source: hostile }]
    const html = renderCommandPicker(picker({ items: bad }), {}, t)
    expect(html).not.toContain('<script>')
    expect(html).not.toContain('alert("x")')
    expect(html).toContain('&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &#39;q&#39;')
    const highlighted = renderCommandPicker(picker({ items: bad, query: 'script' }), {}, t)
    expect(highlighted).not.toContain('<script>')
    expect(highlighted).toContain('<span class="p-cmd-hl">script</span>')
  })
})

describe('texts', () => {
  test('every key used exists in both languages with the same keys', () => {
    const source = readFileSync(new URL('./commands.js', import.meta.url), 'utf8')
    const used = [...source.matchAll(/\bt\('(chat\.[a-zA-Z.]+)'/g)].map((m) => m[1])
    expect(used.length).toBeGreaterThan(5)
    for (const lang of ['en', 'es']) for (const key of used) expect(TEXTS[lang][key], `${lang}: ${key}`).toBeTruthy()
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.commands.')).toBe(true)
  })
})
