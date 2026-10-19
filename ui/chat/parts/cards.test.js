// Run: bun test ui/chat/parts/cards.test.js
import { describe, expect, test } from 'bun:test'
import { parseCardTitle } from './cards.js'

const title = (object) => JSON.stringify({ aria: 1, ...object })

describe('parseCardTitle', () => {
  test('sandbox card', () => {
    const card = parseCardTitle(title({ kind: 'sandbox', summary: 'S', detail: 'pnpm i', jev: 'ok' }))
    expect(card).toEqual({ kind: 'sandbox', summary: 'S', detail: 'pnpm i', jev: 'ok' })
    expect('aria' in card).toBe(false)
  })

  test('sandbox jev may be an object; other types are dropped', () => {
    expect(parseCardTitle(title({ kind: 'sandbox', jev: { review: 'r' } })).jev).toEqual({ review: 'r' })
    expect(parseCardTitle(title({ kind: 'sandbox', jev: 5 })).jev).toBe('')
    expect(parseCardTitle(title({ kind: 'sandbox' }))).toEqual({ kind: 'sandbox', summary: '', detail: '', jev: '' })
  })

  test('quota card', () => {
    expect(parseCardTitle(title({ kind: 'quota', percent: 92, freeModel: 'big-pickle', countdown: 30 }))).toEqual({
      kind: 'quota', percent: 92, freeModel: 'big-pickle', countdown: 30,
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
    expect(parseCardTitle(title({ kind: 'write', path: 'a/b.rs', summary: 'Edit' }))).toEqual({ kind: 'write', path: 'a/b.rs', summary: 'Edit' })
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
