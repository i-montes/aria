// Run: bun test ui/chat/parts/memory.test.js
import { describe, expect, test } from 'bun:test'
import {
  MEMORY_TYPES,
  TEXTS,
  agoLabel,
  highlight,
  initialMemoryUi,
  memoryExcerpt,
  nextSort,
  reduceMemoryUi,
  renderMemory,
  searchWords,
  selectMemories,
  stamp,
} from './memory.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const EVIL = `<script>alert("x")</script> & 'q'`
const DAY = 86400000
const NOW = Date.UTC(2026, 9, 18, 18)
const mem = (id, over = {}) => ({
  id, type: 'finding', title: `Title ${id}`, key: `k/${id}`, content: `Content of ${id}`, rev: 1, dup: 0,
  created: NOW - 5 * DAY, updated: NOW - 2 * DAY, seen: NOW - DAY, personal: false, pinned: false,
  session: 'ses_1', tool: 'mem_save', review: null, expires: null, ...over,
})
const memories = [
  mem(1, { title: 'Vite define needs JSON.stringify', content: 'Values passed to define must be stringified', updated: NOW - 12 * DAY }),
  mem(2, { type: 'decision', title: 'Use pnpm', key: null, content: 'pnpm everywhere', updated: NOW - DAY, rev: 3, pinned: true }),
  mem(3, { type: 'identity', title: 'Who you are', personal: true, updated: NOW }),
  mem(4, { title: 'Gone', forgotten: true }),
]
const props = { memories, now: NOW }

describe('pure helpers', () => {
  test('searchWords lowercases and splits', () => {
    expect(searchWords('  Vite  Define ')).toEqual(['vite', 'define'])
    expect(searchWords('')).toEqual([])
    expect(searchWords(undefined)).toEqual([])
  })
  test('highlight splits text into hit and plain segments, merging overlaps', () => {
    expect(highlight('Vite define', ['vite'])).toEqual([{ t: 'Vite', hit: true }, { t: ' define', hit: false }])
    expect(highlight('abcd', ['ab', 'bc'])).toEqual([{ t: 'abc', hit: true }, { t: 'd', hit: false }])
    expect(highlight('abc', [])).toEqual([{ t: 'abc', hit: false }])
    expect(highlight('', ['a'])).toEqual([{ t: '', hit: false }])
    expect(highlight(null, ['a'])).toEqual([{ t: '', hit: false }])
  })
  test('memoryExcerpt flattens whitespace and centers a late hit', () => {
    expect(memoryExcerpt('a\n\nb   c', [])).toBe('a b c')
    const long = `${'x '.repeat(80)}needle tail`
    expect(memoryExcerpt(long, ['needle']).startsWith('…')).toBe(true)
    expect(memoryExcerpt('needle early', ['needle'])).toBe('needle early')
  })
  test('agoLabel buckets days', () => {
    expect(agoLabel(NOW, NOW, t)).toBe('chat.memory.ago.today')
    expect(agoLabel(NOW - DAY, NOW, t)).toBe('chat.memory.ago.yesterday')
    expect(agoLabel(NOW - 5 * DAY, NOW, t)).toBe('chat.memory.ago.days n=5')
    expect(agoLabel(NOW - 35 * DAY, NOW, t)).toBe('chat.memory.ago.weeks n=5')
    expect(agoLabel(NOW + DAY, NOW, t)).toBe('chat.memory.ago.today')
    expect(agoLabel(undefined, NOW, t)).toBe('chat.memory.ago.today')
  })
  test('stamp is UTC minute precision', () => {
    expect(stamp(Date.UTC(2026, 9, 14, 8, 5))).toBe('2026-10-14 08:05')
    expect(stamp(undefined)).toBe('—')
  })
  test('nextSort cycles through the four orders', () => {
    expect(nextSort('updated')).toBe('created')
    expect(nextSort('created')).toBe('revisions')
    expect(nextSort('revisions')).toBe('title')
    expect(nextSort('title')).toBe('updated')
    expect(nextSort('???')).toBe('created')
  })
  test('the seven types of the design', () => {
    expect(MEMORY_TYPES.map((x) => x.id)).toEqual(['finding', 'decision', 'pattern', 'session_summary', 'manual', 'learning', 'identity'])
  })
})

describe('selecting', () => {
  test('forgotten memories are out; pinned come first; counts follow the facets', () => {
    const r = selectMemories(memories, initialMemoryUi())
    expect(r.active).toHaveLength(3)
    expect(r.shown.map((m) => m.id)).toEqual([2, 3, 1])
    expect(r.typeCounts).toMatchObject({ all: 3, finding: 1, decision: 1, identity: 1, pattern: 0 })
    expect(r.pinnedCount).toBe(1)
    expect(r.personalCount).toBe(1)
    expect(r.keyedCount).toBe(2)
  })
  test('text search needs every word and looks in title, content, key and type', () => {
    expect(selectMemories(memories, { ...initialMemoryUi(), query: 'vite stringified' }).shown.map((m) => m.id)).toEqual([1])
    expect(selectMemories(memories, { ...initialMemoryUi(), query: 'identity' }).shown.map((m) => m.id)).toEqual([3])
    expect(selectMemories(memories, { ...initialMemoryUi(), query: 'zzz' }).shown).toEqual([])
  })
  test('type, pinned and personal filters combine; type counts ignore the type filter', () => {
    const r = selectMemories(memories, { ...initialMemoryUi(), type: 'decision' })
    expect(r.shown.map((m) => m.id)).toEqual([2])
    expect(r.typeCounts.all).toBe(3)
    expect(selectMemories(memories, { ...initialMemoryUi(), pinnedOnly: true }).shown.map((m) => m.id)).toEqual([2])
    expect(selectMemories(memories, { ...initialMemoryUi(), personalOnly: true }).shown.map((m) => m.id)).toEqual([3])
  })
  test('sorts', () => {
    const base = { ...initialMemoryUi() }
    expect(selectMemories(memories, { ...base, sort: 'revisions' }).shown.map((m) => m.id)).toEqual([2, 3, 1])
    expect(selectMemories(memories, { ...base, sort: 'title' }).shown.map((m) => m.id)).toEqual([2, 1, 3])
    expect(selectMemories(memories, { ...base, sort: 'created' }).shown).toHaveLength(3)
    expect(selectMemories(memories, { ...base, sort: 'unknown' }).shown).toHaveLength(3)
  })
  test('junk input does not throw', () => {
    expect(selectMemories(undefined, undefined).shown).toEqual([])
    expect(selectMemories([null, {}, { id: 1 }], {}).shown.length).toBeGreaterThan(0)
  })
})

describe('ui transitions', () => {
  test('defaults', () => {
    expect(initialMemoryUi()).toEqual({ query: '', type: 'all', pinnedOnly: false, personalOnly: false, sort: 'updated', open: null })
  })
  test('search, clear, filter and the toggles close the open row', () => {
    const open = { ...initialMemoryUi(), open: '2' }
    expect(reduceMemoryUi(open, 'search', 'abc')).toMatchObject({ query: 'abc', open: null })
    expect(reduceMemoryUi({ ...open, query: 'a' }, 'clear')).toMatchObject({ query: '', open: null })
    expect(reduceMemoryUi(open, 'filter', 'decision')).toMatchObject({ type: 'decision', open: null })
    expect(reduceMemoryUi(open, 'pinned-only')).toMatchObject({ pinnedOnly: true, open: null })
    expect(reduceMemoryUi(open, 'personal-only')).toMatchObject({ personalOnly: true, open: null })
    expect(reduceMemoryUi(open, 'forget')).toMatchObject({ open: null })
  })
  test('sort keeps the open row; select toggles', () => {
    const open = { ...initialMemoryUi(), open: '2' }
    expect(reduceMemoryUi(open, 'sort', 'title')).toMatchObject({ sort: 'title', open: '2' })
    expect(reduceMemoryUi(open, 'sort')).toMatchObject({ sort: 'created' })
    expect(reduceMemoryUi(open, 'select', '2').open).toBe(null)
    expect(reduceMemoryUi(open, 'select', '3').open).toBe('3')
    expect(reduceMemoryUi(initialMemoryUi(), 'select', 7).open).toBe(7)
  })
  test('an unknown filter value falls back to all; unknown actions change nothing', () => {
    expect(reduceMemoryUi(initialMemoryUi(), 'filter', 'nonsense').type).toBe('all')
    expect(reduceMemoryUi(initialMemoryUi(), 'filter', 'finding').type).toBe('finding')
    const ui = initialMemoryUi()
    expect(reduceMemoryUi(ui, 'bogus')).toEqual(ui)
    expect(reduceMemoryUi(undefined, 'clear')).toEqual(initialMemoryUi())
  })
})

describe('the panel', () => {
  const html = renderMemory(props, initialMemoryUi(), t)
  test('header: back, title, project and active count', () => {
    expect(html).toContain('data-chat="memory-back"')
    expect(html).toContain('chat.memory.title')
    expect(html).toContain('engram · aria')
    expect(html).toContain('chat.memory.active n=3')
  })
  test('search field is an input with data-chat-field; clear shows only with a query', () => {
    expect(html).toContain('data-chat-field="memory-search"')
    expect(html).not.toContain('data-chat="memory-clear"')
    const withQuery = renderMemory(props, { ...initialMemoryUi(), query: 'vite' }, t)
    expect(withQuery).toContain('data-chat="memory-clear"')
    expect(withQuery).toContain('value="vite"')
    expect(html).toContain('chat.memory.textSearch')
  })
  test('type chips with counts, the pressed one marked', () => {
    expect(html.match(/data-chat="memory-filter"/g)).toHaveLength(8)
    expect(html).toContain('data-value="all"')
    expect(html).toContain('data-value="session_summary"')
    const pressed = renderMemory(props, { ...initialMemoryUi(), type: 'decision' }, t)
    expect(pressed).toContain('aria-pressed="true"')
    expect(pressed.match(/aria-pressed="true"/g)).toHaveLength(1)
  })
  test('pinned, personal and sort controls', () => {
    expect(html).toContain('data-chat="memory-pinned-only"')
    expect(html).toContain('data-chat="memory-personal-only"')
    expect(html).toContain('data-chat="memory-sort" data-value="created"')
    expect(html).toContain('chat.memory.sort.updated')
  })
  test('rows: one per shown memory, with select, pin icon, revisions and personal badge', () => {
    expect(html.match(/data-chat="memory-select"/g)).toHaveLength(3)
    expect(html).toContain('data-id="2"')
    expect(html).not.toContain('Gone')
    expect(html).toContain('chat.memory.rev n=3')
    expect(html).toContain('chat.memory.personal')
    expect(html).toContain('p-memory-pin-icon')
  })
  test('the detail is rendered only for the open row, with fields and actions', () => {
    expect(html).not.toContain('p-memory-detail')
    const open = renderMemory(props, { ...initialMemoryUi(), open: '2' }, t)
    expect(open.match(/class="p-memory-detail"/g)).toHaveLength(1)
    expect(open).toContain('aria-expanded="true"')
    expect(open).toContain('topic_key')
    expect(open).toContain('revision_count')
    expect(open).toContain('data-chat="memory-pin" data-id="2"')
    expect(open).toContain('data-chat="memory-forget" data-id="2"')
    expect(open).toContain('chat.memory.unpin')
    expect(open).toContain('chat.memory.embeddingNone')
    const other = renderMemory(props, { ...initialMemoryUi(), open: '1' }, t)
    expect(other).toContain('chat.memory.pin')
    expect(other).not.toContain('review_after')
    const dated = renderMemory({ ...props, memories: [mem(9, { review: NOW + DAY, expires: NOW + 2 * DAY })] }, { ...initialMemoryUi(), open: 9 }, t)
    expect(dated).toContain('review_after')
    expect(dated).toContain('expires_at')
  })
  test('search hits are highlighted in the title', () => {
    const hit = renderMemory(props, { ...initialMemoryUi(), query: 'vite' }, t)
    expect(hit).toContain('<span class="hit">Vite</span>')
  })
  test('empty results explain themselves', () => {
    const none = renderMemory(props, { ...initialMemoryUi(), query: 'zzz' }, t)
    expect(none).toContain('chat.memory.emptyQuery q=zzz')
    expect(none).toContain('chat.memory.emptyHint')
    expect(renderMemory({}, undefined, t)).toContain('chat.memory.empty')
    expect(renderMemory(undefined, undefined, t)).toContain('chat.memory.empty')
  })
  test('undo bar appears with the forgotten title', () => {
    expect(html).not.toContain('data-chat="memory-undo"')
    const undo = renderMemory({ ...props, undo: { id: 4, title: 'Gone' } }, initialMemoryUi(), t)
    expect(undo).toContain('data-chat="memory-undo"')
    expect(undo).toContain('chat.memory.forgot title=Gone')
  })
  test('footer: keyed, pinned and embeddings', () => {
    expect(html).toContain('chat.memory.keyed n=2')
    expect(html).toContain('chat.memory.pinnedN n=1')
    expect(html).toContain('chat.memory.embeddings n=0')
  })
  test('unknown types are shown neutrally, with underscores as spaces', () => {
    const odd = renderMemory({ memories: [mem(1, { type: 'brand_new' })], now: NOW }, initialMemoryUi(), t)
    expect(odd).toContain('brand new')
  })
  test('every text field is escaped, including ids, query and project', () => {
    const evil = mem(EVIL, { type: EVIL, title: EVIL, key: EVIL, content: EVIL, session: EVIL, tool: EVIL, rev: 2, pinned: true })
    const ui = { ...initialMemoryUi(), open: EVIL }
    const out = renderMemory({ memories: [evil], now: NOW, project: EVIL, undo: { id: 1, title: EVIL } }, ui, t)
    const query = renderMemory({ memories: [evil], now: NOW }, { ...initialMemoryUi(), query: EVIL }, t)
    for (const page of [out, query]) {
      expect(page).not.toContain('<script')
      expect(page).not.toContain('alert("x")')
      expect(page).toContain('&lt;script&gt;')
      expect(page).toContain('&amp;')
    }
    expect(out).toContain('&#39;q&#39;')
  })
})

test('texts: both languages have the same keys, prefixed', () => {
  expect(Object.keys(TEXTS.en).sort()).toEqual(Object.keys(TEXTS.es).sort())
  for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.memory.')).toBe(true)
})
