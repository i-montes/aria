// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import rotate from './index.ts'
import { isRotating } from '../lib/rotation.ts'
import { COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX } from './seed.ts'

function command() {
  let registered: any
  rotate({ registerCommand: (name: string, options: any) => (registered = { name, ...options }) } as any)
  return registered
}

function stubCtx(path: any[]) {
  const calls: any[] = []
  const next = {
    appendCustomMessageEntry: (...args: any[]) => calls.push(['custom', ...args]),
    appendMessage: (message: any) => calls.push(['message', message]),
  }
  const ctx = {
    sessionManager: { getBranch: () => path, getSessionId: () => 'old-id', getSessionFile: () => '/sessions/old.jsonl' },
    newSession: async (options: any) => {
      calls.push(['newSession', options.parentSession])
      await options.setup(next)
      return { cancelled: false }
    },
  }
  return { ctx, calls }
}

const hello = { role: 'user', content: 'hello', timestamp: 0 }
const reply = { role: 'assistant', content: [{ type: 'text', text: 'hi' }], timestamp: 0 }

describe('/aria-rotate', () => {
  test('is registered as aria-rotate', () => {
    expect(command().name).toBe('aria-rotate')
  })

  test('does nothing without a compaction', async () => {
    const { ctx, calls } = stubCtx([{ type: 'message', id: 'u', parentId: null, timestamp: '', message: hello }])
    await command().handler('', ctx)
    expect(calls).toEqual([])
  })

  test('starts a new session from the old file: the framed summary first, then the kept context in order', async () => {
    const { ctx, calls } = stubCtx([
      { type: 'message', id: 'u', parentId: null, timestamp: '', message: hello },
      { type: 'compaction', id: 'c', parentId: 'u', timestamp: '', summary: 'S', firstKeptEntryId: 'u', tokensBefore: 1 },
      { type: 'custom_message', id: 'k', parentId: 'c', timestamp: '', customType: 'aria-background', content: 'done', display: true, details: { job: 2 } },
      { type: 'message', id: 'a', parentId: 'k', timestamp: '', message: reply },
    ])
    await command().handler('', ctx)
    expect(calls).toEqual([
      ['newSession', '/sessions/old.jsonl'],
      ['custom', 'aria-earlier', `${COMPACTION_SUMMARY_PREFIX}S${COMPACTION_SUMMARY_SUFFIX}`, true, { from: 'old-id', summary: 'S' }],
      ['message', hello],
      ['custom', 'aria-background', 'done', true, { job: 2 }],
      ['message', reply],
    ])
  })

  test('marks the switch while the new session starts, and clears the mark after, even when it fails', async () => {
    const path = [
      { type: 'message', id: 'u', parentId: null, timestamp: '', message: hello },
      { type: 'compaction', id: 'c', parentId: 'u', timestamp: '', summary: 'S', firstKeptEntryId: 'u', tokensBefore: 1 },
    ]
    const seen: boolean[] = []
    const { ctx } = stubCtx(path)
    ctx.newSession = async () => (seen.push(isRotating()), { cancelled: false })
    await command().handler('', ctx)
    expect(seen).toEqual([true])
    expect(isRotating()).toBe(false)
    ctx.newSession = async () => {
      throw new Error('no')
    }
    await expect(command().handler('', ctx)).rejects.toThrow('no')
    expect(isRotating()).toBe(false)
  })

  test("frames the summary exactly as Pi's compaction summary", () => {
    expect(COMPACTION_SUMMARY_PREFIX).toBe('The conversation history before this point was compacted into the following summary:\n\n<summary>\n')
    expect(COMPACTION_SUMMARY_SUFFIX).toBe('\n</summary>')
  })
})
