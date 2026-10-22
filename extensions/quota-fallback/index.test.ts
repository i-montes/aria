// Run: bun test extensions
// The quota fallback with a fake `pi` across Aria's session rotation: the switch to the free model is remembered.
import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { markRotating } from '../lib/rotation.ts'

// The usage cache and the switch decision live in the temp folder: a folder of the test's own, set before the
// extension is loaded, so a real Pi's cache is never read or written.
const scratch = mkdtempSync(join(tmpdir(), 'quota-fallback-'))
const previous = process.env.TMPDIR
let fallback: typeof import('./index.ts').default
let writeCachedUsage: typeof import('./fallback.ts').writeCachedUsage
let freeModel: typeof import('./fallback.ts').freeModel

beforeAll(async () => {
  process.env.TMPDIR = scratch
  fallback = (await import('./index.ts')).default
  ;({ writeCachedUsage, freeModel } = await import('./fallback.ts'))
})

afterAll(() => {
  if (previous === undefined) delete process.env.TMPDIR
  else process.env.TMPDIR = previous
  rmSync(scratch, { recursive: true, force: true })
})

const PAID = { provider: 'opencode-go', id: 'paid-model', cost: { input: 1, output: 1 } }

const usage = (percent: number) => {
  const window = { status: 'ok', percent, resetsAt: '2026-10-22T00:00:00Z' }
  return { rolling: window, weekly: { ...window, percent: 0 }, monthly: { ...window, percent: 0 } } as any
}

function fakePi() {
  const pi = {
    set: [] as any[],
    handlers: new Map<string, ((event: any, ctx: any) => any)[]>(),
    on(name: string, fn: any) {
      pi.handlers.set(name, [...(pi.handlers.get(name) ?? []), fn])
    },
    setModel: async (model: any) => {
      pi.set.push(model)
      return true
    },
    fire: async (name: string, event: any, ctx: any) => {
      for (const fn of pi.handlers.get(name) ?? []) await fn(event, ctx)
    },
  }
  return pi
}

const ctx = (model: any) => ({
  model,
  modelRegistry: { find: (provider: string, id: string) => ({ provider, id, cost: { input: 0, output: 0 } }) },
})

/** An instance that switched PAID for the free model at a spent quota. */
async function switched() {
  writeCachedUsage(usage(99))
  const pi = fakePi()
  fallback(pi as any)
  await pi.fire('session_start', { type: 'session_start', reason: 'startup' }, ctx(PAID))
  expect(pi.set.map((m) => m.id)).toEqual([freeModel().id])
  return pi
}

describe('a rotation', () => {
  test('keeps the switch, and the renewed quota brings the paid model back', async () => {
    const before = await switched()
    const free = before.set[0]
    markRotating(true)
    const after = fakePi()
    try {
      await before.fire('session_shutdown', { type: 'session_shutdown', reason: 'new' }, ctx(free))
      fallback(after as any)
      // Pi restarts the session on the default model; Aria puts the tab's model back, so this start decides nothing.
      await after.fire('session_start', { type: 'session_start', reason: 'new' }, ctx(PAID))
      expect(after.set).toEqual([])
    } finally {
      markRotating(false)
    }
    writeCachedUsage(usage(10))
    await after.fire('before_agent_start', { type: 'before_agent_start' }, ctx(free))
    expect(after.set.map((m) => m.id)).toEqual([PAID.id])
  })

  test('a new session that is not a rotation forgets the switch', async () => {
    const before = await switched()
    const free = before.set[0]
    await before.fire('session_shutdown', { type: 'session_shutdown', reason: 'new' }, ctx(free))
    const after = fakePi()
    fallback(after as any)
    writeCachedUsage(usage(10))
    await after.fire('session_start', { type: 'session_start', reason: 'new' }, ctx(free))
    await after.fire('before_agent_start', { type: 'before_agent_start' }, ctx(free))
    expect(after.set).toEqual([])
  })
})
