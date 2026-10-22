// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { activePath, rotationSeed } from './seed.ts'

let clock = 0
const at = () => new Date(Date.UTC(2026, 9, 21, 9, 0, clock++)).toISOString()
const user = (id: string, parentId: string | null, text: string) => ({
  type: 'message',
  id,
  parentId,
  timestamp: at(),
  message: { role: 'user', content: text, timestamp: 0 },
})
const assistant = (id: string, parentId: string | null, text: string) => ({
  type: 'message',
  id,
  parentId,
  timestamp: at(),
  message: { role: 'assistant', content: [{ type: 'text', text }], timestamp: 0 },
})
const compaction = (id: string, parentId: string, summary: string, firstKeptEntryId: string) => ({
  type: 'compaction',
  id,
  parentId,
  timestamp: at(),
  summary,
  firstKeptEntryId,
  tokensBefore: 1000,
})
const texts = (messages: any[]) =>
  messages.map((m) => (typeof m.content === 'string' ? m.content : m.content.map((c: any) => c.text).join('')))

describe('the rotation seed', () => {
  test('is null without a compaction', () => {
    const entries = [
      user('a', null, 'hello'),
      assistant('b', 'a', 'hi'),
      { type: 'custom_message', id: 'c', parentId: 'b', timestamp: at(), customType: 'clock', content: '[now]', display: false },
    ]
    expect(rotationSeed(entries, 'old-session')).toBeNull()
    expect(rotationSeed([], 'old-session')).toBeNull()
  })

  test('carries the latest compaction summary and the kept messages in order, on the active path', () => {
    const entries = [
      user('u1', null, 'one'),
      assistant('a1', 'u1', 'reply one'),
      user('u2', 'a1', 'two'),
      assistant('a2', 'u2', 'reply two'),
      compaction('c1', 'a2', 'first summary', 'u2'),
      user('u3', 'c1', 'three'),
      assistant('a3', 'u3', 'reply three'),
      // an abandoned branch from a3: never on the active path
      user('x1', 'a3', 'abandoned'),
      assistant('x2', 'x1', 'abandoned reply'),
      compaction('xc', 'x2', 'abandoned summary', 'x1'),
      // the active branch from a3
      user('u4', 'a3', 'four'),
      { type: 'model_change', id: 'm1', parentId: 'u4', timestamp: at(), provider: 'p', modelId: 'm' },
      assistant('a4', 'm1', 'reply four'),
      { type: 'custom_message', id: 'k1', parentId: 'a4', timestamp: at(), customType: 'clock', content: '[now]', display: false },
      compaction('c2', 'k1', 'second summary', 'u4'),
      user('u5', 'c2', 'five'),
      assistant('a5', 'u5', 'reply five'),
    ]
    const path = activePath(entries, 'a5')
    expect(path.map((e) => e.id)).toEqual(['u1', 'a1', 'u2', 'a2', 'c1', 'u3', 'a3', 'u4', 'm1', 'a4', 'k1', 'c2', 'u5', 'a5'])

    const seed = rotationSeed(path, 'old-session')
    expect(seed).not.toBeNull()
    expect(seed!.summary).toBe('second summary')
    expect(seed!.from).toBe('old-session')
    expect(texts(seed!.messages)).toEqual(['four', 'reply four', 'five', 'reply five'])
    expect(seed!.messages[0]).toBe((entries[10] as any).message)
  })

  test('without the kept entry on the path, carries what follows the compaction', () => {
    const entries = [
      user('u1', null, 'one'),
      compaction('c1', 'u1', 'summary', 'gone'),
      user('u2', 'c1', 'two'),
    ]
    const seed = rotationSeed(entries, 's')
    expect(seed!.summary).toBe('summary')
    expect(texts(seed!.messages)).toEqual(['two'])
  })

  test('an empty leaf gives an empty path', () => {
    expect(activePath([user('u1', null, 'one')], null)).toEqual([])
  })
})
