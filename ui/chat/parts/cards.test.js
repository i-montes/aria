// Run: bun test ui/chat/parts/cards.test.js
import { describe, expect, test } from 'bun:test'
import { parseCardTitle } from './cards.js'
import { createChat, reduce } from '../state.js'
import { INLINE_CARDS, renderThread } from '../thread.js'

const title = (object) => JSON.stringify({ aria: 1, ...object })

describe('parseCardTitle', () => {
  test('sandbox card', () => {
    const card = parseCardTitle(title({ kind: 'sandbox', summary: 'S', detail: 'pnpm i', jev: 'ok' }))
    expect(card).toEqual({ kind: 'sandbox', summary: 'S', detail: 'pnpm i', jev: 'ok', options: [] })
    expect('aria' in card).toBe(false)
  })

  test('sandbox jev may be an object; other types are dropped', () => {
    expect(parseCardTitle(title({ kind: 'sandbox', jev: { review: 'r' } })).jev).toEqual({ review: 'r' })
    expect(parseCardTitle(title({ kind: 'sandbox', jev: 5 })).jev).toBe('')
    expect(parseCardTitle(title({ kind: 'sandbox' }))).toEqual({ kind: 'sandbox', summary: '', detail: '', jev: '', options: [] })
  })

  test('quota card', () => {
    expect(parseCardTitle(title({ kind: 'quota', percent: 92, freeModel: 'big-pickle', countdown: 30 }))).toEqual({
      kind: 'quota', percent: 92, freeModel: 'big-pickle', countdown: 30, options: [],
    })
    const bad = parseCardTitle(title({ kind: 'quota', percent: 'x', countdown: -1 }))
    expect(bad.percent).toBe(0)
    expect(bad.countdown).toBe(null)
    expect(parseCardTitle(title({ kind: 'quota', percent: 400 })).percent).toBe(100)
  })

  test('question card keeps only object questions', () => {
    const card = parseCardTitle(title({ kind: 'question', questions: [{ question: 'Q', options: [] }, 'junk', null, 4] }))
    expect(card.kind).toBe('question')
    expect(card.questions).toEqual([{ question: 'Q', options: [] }])
    expect(parseCardTitle(title({ kind: 'question' })).questions).toEqual([])
    expect(parseCardTitle(title({ kind: 'question', questions: 'x' })).questions).toEqual([])
  })

  test('write card', () => {
    expect(parseCardTitle(title({ kind: 'write', path: 'a/b.rs', summary: 'Edit' }))).toEqual({ kind: 'write', path: 'a/b.rs', summary: 'Edit', detail: '', options: [] })
  })

  test('anything else is null and never throws', () => {
    const cases = [
      undefined, null, 5, {}, [], '', 'Allow?', '{', '{"aria":1', '[]', '"x"', 'null', '{}',
      '{"kind":"sandbox"}', '{"aria":2,"kind":"sandbox"}', '{"aria":"1","kind":"sandbox"}',
      '{"aria":1,"kind":"nope"}', '{"aria":1}', '{"aria":1,"kind":5}', '{"aria":1,"kind":"__proto__"}', '{"aria":1,"kind":"toString"}',
    ]
    for (const input of cases) expect(parseCardTitle(input)).toBe(null)
  })

  test('surrounding whitespace is fine', () => {
    expect(parseCardTitle(`  \n${title({ kind: 'write' })}  `)?.kind).toBe('write')
  })

  test('hostile payloads do not pollute or throw', () => {
    const card = parseCardTitle('{"aria":1,"kind":"write","__proto__":{"x":1},"path":"p"}')
    expect(card.path).toBe('p')
    expect({}.x).toBeUndefined()
  })
})

describe('options carried by a card', () => {
  test('sandbox, write and quota keep the options the extension accepts back, in order', () => {
    const options = [{ value: 'Yes, just this once', label: 'Yes, just this once', primary: true }, { value: 'No', label: 'No' }]
    for (const kind of ['sandbox', 'write', 'quota']) {
      expect(parseCardTitle(JSON.stringify({ aria: 1, kind, options })).options).toEqual(options)
    }
  })

  test('bad options are dropped, not trusted', () => {
    const card = parseCardTitle(JSON.stringify({ aria: 1, kind: 'sandbox', options: [{ value: 'ok', label: 'Ok' }, { value: 3 }, 'x', null, { label: 'no value' }] }))
    expect(card.options).toEqual([{ value: 'ok', label: 'Ok' }])
    expect(parseCardTitle(JSON.stringify({ aria: 1, kind: 'sandbox', options: 'nope' })).options).toEqual([])
  })

  test('a write card keeps its detail too', () => {
    expect(parseCardTitle(JSON.stringify({ aria: 1, kind: 'write', path: '~/a', detail: 'edit ~/a' })).detail).toBe('edit ~/a')
  })

  test('kind connector routes to the permission card', () => {
    const options = [{ value: 'Yes, just this once', label: 'Yes, just this once', primary: true }, { value: 'No', label: 'No' }]
    const raw = { aria: 1, kind: 'connector', summary: 'Notion wants to create pages', detail: 'title: Weekly notes', options }
    expect(parseCardTitle(JSON.stringify(raw))).toEqual({ kind: 'connector', summary: 'Notion wants to create pages', detail: 'title: Weekly notes', options })
    expect(parseCardTitle(JSON.stringify({ aria: 1, kind: 'connector' }))).toEqual({ kind: 'connector', summary: '', detail: '', options: [] })
    expect(INLINE_CARDS).toContain('connector')
    const state = reduce(createChat(), { type: 'chatDialog', id: 'c1', method: 'select', request: { title: JSON.stringify(raw), options: options.map((o) => o.value) }, hidden: false })
    const html = renderThread(state, {}, (key) => key)
    expect(html).toContain('class="p-perm"')
    expect(html).toContain('Notion wants to create pages')
    expect(html.match(/data-chat="permission"/g)).toHaveLength(2)
    expect(html).toContain('data-choice="Yes, just this once"')
  })
})
