// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { rotationSeed } from './seed.ts'

type Entry = { type: string; id: string; parentId: string | null; [key: string]: any }

// The entries from the root to `leafId`, as Pi's `SessionManager.getBranch()` returns them.
function activePath(entries: Entry[], leafId: string | null): Entry[] {
  const byId = new Map(entries.map((e) => [e.id, e]))
  const path: Entry[] = []
  let current = leafId ? byId.get(leafId) : undefined
  while (current) {
    path.push(current)
    current = current.parentId ? byId.get(current.parentId) : undefined
  }
  return path.reverse()
}

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
const custom = (id: string, parentId: string, customType: string, content: string, display = true, details?: unknown) => ({
  type: 'custom_message',
  id,
  parentId,
  timestamp: at(),
  customType,
  content,
  display,
  details,
})
const text = (content: any) => (typeof content === 'string' ? content : content.map((c: any) => c.text).join(''))
// One line per carried item: `role: text` for a message, `customType: text` for a custom message.
const texts = (items: any[]) =>
  items.map((i) => (i.kind === 'message' ? `${i.message.role}: ${text(i.message.content)}` : `${i.customType}: ${text(i.content)}`))

describe('the rotation seed', () => {
  test('is null without a compaction', () => {
    const entries = [user('a', null, 'hello'), assistant('b', 'a', 'hi'), custom('c', 'b', 'clock', '[now]', false)]
    expect(rotationSeed(entries, 'old-session')).toBeNull()
    expect(rotationSeed([], 'old-session')).toBeNull()
  })

  test('carries the latest compaction summary and the kept context in order, on the active path', () => {
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
      custom('k1', 'a4', 'clock', '[now]', false),
      compaction('c2', 'k1', 'second summary', 'u4'),
      user('u5', 'c2', 'five'),
      custom('bg', 'u5', 'aria-background', 'job done', true, { job: 1 }),
      assistant('a5', 'bg', 'reply five'),
    ]
    const path = activePath(entries, 'a5')
    expect(path.map((e) => e.id)).toEqual(['u1', 'a1', 'u2', 'a2', 'c1', 'u3', 'a3', 'u4', 'm1', 'a4', 'k1', 'c2', 'u5', 'bg', 'a5'])

    const seed = rotationSeed(path, 'old-session')
    expect(seed).not.toBeNull()
    expect(seed!.summary).toBe('second summary')
    expect(seed!.from).toBe('old-session')
    expect(texts(seed!.items)).toEqual([
      'user: four',
      'assistant: reply four',
      'clock: [now]',
      'user: five',
      'aria-background: job done',
      'assistant: reply five',
    ])
    expect(seed!.items[0]).toEqual({ kind: 'message', message: (entries[10] as any).message })
    expect(seed!.items[2]).toEqual({ kind: 'custom', customType: 'clock', content: '[now]', display: false, details: undefined })
    expect(seed!.items[4]).toEqual({ kind: 'custom', customType: 'aria-background', content: 'job done', display: true, details: { job: 1 } })
  })

  test('skips a system message and an older aria-earlier seed among the kept entries, like Pi', () => {
    const entries = [
      custom('e0', null as any, 'aria-earlier', 'older summary', true, { from: 'older' }),
      { type: 'message', id: 's1', parentId: 'e0', timestamp: at(), message: { role: 'system', content: 'prompt', timestamp: 0 } },
      user('u1', 's1', 'one'),
      compaction('c1', 'u1', 'summary', 'e0'),
      user('u2', 'c1', 'two'),
    ]
    expect(texts(rotationSeed(entries, 's')!.items)).toEqual(['user: one', 'user: two'])
  })

  test('applies context edits the way Pi does: a replacement, or the entry dropped', () => {
    const entries = [
      user('u1', null, 'one'),
      assistant('a1', 'u1', 'long reply'),
      custom('k1', 'a1', 'aria-background', 'long output', true),
      user('u2', 'k1', 'two'),
      compaction('c1', 'u2', 'summary', 'a1'),
      { type: 'context_edit', id: 'e1', parentId: 'c1', timestamp: at(), targetId: 'a1', replacement: { content: 'short reply' } },
      { type: 'context_edit', id: 'e2', parentId: 'e1', timestamp: at(), targetId: 'k1', replacement: { content: 'short output' } },
      { type: 'context_edit', id: 'e3', parentId: 'e2', timestamp: at(), targetId: 'u2', replacement: null },
      user('u3', 'e3', 'three'),
    ]
    const items = rotationSeed(entries, 's')!.items as any[]
    expect(texts(items)).toEqual(['assistant: short reply', 'aria-background: short output', 'user: three'])
    expect(items[0].message.content).toEqual([{ type: 'text', text: 'short reply' }])
    expect(items[0].message.role).toBe('assistant')
    expect((entries[1] as any).message.content[0].text).toBe('long reply')
  })

  test('without the kept entry on the path, carries what follows the compaction', () => {
    const entries = [user('u1', null, 'one'), compaction('c1', 'u1', 'summary', 'gone'), user('u2', 'c1', 'two')]
    const seed = rotationSeed(entries, 's')
    expect(seed!.summary).toBe('summary')
    expect(texts(seed!.items)).toEqual(['user: two'])
  })
})
