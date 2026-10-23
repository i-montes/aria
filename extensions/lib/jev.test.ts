import { describe, expect, test } from 'bun:test'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { chain, classifyWithJev, failureWait, FREE_JEV, offUntil, PAID_JEV, pauses } from './jev.ts'

const QUOTA = {
  stopReason: 'error',
  answers: {},
  errorMessage: 'Server requested 16342s retry delay (max: 60s). System One API returned 429',
}
const FUNDS = {
  stopReason: 'error',
  errorMessage: 'System One API error (402): {"error":{"message":"Upstream request failed: Insufficient account funds"}}',
}
const ANSWER = { stopReason: 'stop', answers: { route: 'quick' } }
const OTHER_FREE = { provider: 'openrouter', id: 'inception/mercury-decide:free' }
const TYPESAFE_JEV = { provider: 'typesafe', id: 'jev-latest' }
const JEV_ROUTER = { provider: 'openrouter', id: 'typesafe/jev-1.13' }

const key = (m: { provider: string; id: string }) => `${m.provider}/${m.id}`

// A model registry with `available` classifiers, each answering as `answers` says; it records which were asked.
function registry(available: { provider: string; id: string }[], answers: Record<string, any>) {
  const asked: string[] = []
  return {
    asked,
    getAvailableOfType: async () => available,
    classify: async (model: { provider: string; id: string }) => {
      asked.push(key(model))
      return answers[key(model)] ?? ANSWER
    },
  }
}

const file = () => join(mkdtempSync(join(tmpdir(), 'jev-')), 'jev-paused.json')

describe('Jev, model by model', () => {
  test('what a failure costs: a quota waits as long as it asks, no funds an hour, anything else nothing', () => {
    expect(failureWait(QUOTA)).toBe(16342)
    expect(failureWait({ stopReason: 'error', errorMessage: 'System One API returned 429' })).toBe(3600)
    expect(failureWait(FUNDS)).toBe(3600)
    expect(failureWait({ stopReason: 'error', errorMessage: 'network down' })).toBeNull()
    expect(failureWait(ANSWER)).toBeNull()
  })

  test('the order: free Jev, paid Jev, Jev elsewhere, other free classifiers, the rest', () => {
    const rest = { provider: 'openai', id: 'gpt-6-luna' }
    expect(chain([rest, OTHER_FREE, PAID_JEV, JEV_ROUTER, FREE_JEV]).map(key)).toEqual(
      [FREE_JEV, PAID_JEV, JEV_ROUTER, OTHER_FREE, rest].map(key),
    )
  })

  test("TypeSafe's own Jev answers when it is the only one reachable, and comes after OpenCode's when both are", async () => {
    const alone = registry([TYPESAFE_JEV], {})
    expect(await classifyWithJev(alone, {}, {}, { file: file(), now: () => 0 })).toEqual(ANSWER)
    expect(alone.asked).toEqual([key(TYPESAFE_JEV)])
    expect(chain([TYPESAFE_JEV, FREE_JEV]).map(key)).toEqual([FREE_JEV, TYPESAFE_JEV].map(key))
  })

  test('the free Jev answers while it has quota', async () => {
    const r = registry([FREE_JEV, PAID_JEV], {})
    expect(await classifyWithJev(r, {}, {}, { file: file(), now: () => 0 })).toEqual(ANSWER)
    expect(r.asked).toEqual([key(FREE_JEV)])
  })

  test('only the spent one steps aside: free out → paid; paid without funds → the next free one', async () => {
    const f = file()
    const notes: string[] = []
    const r = registry([FREE_JEV, PAID_JEV, OTHER_FREE], { [key(FREE_JEV)]: QUOTA, [key(PAID_JEV)]: FUNDS })
    expect(await classifyWithJev(r, {}, {}, { file: f, now: () => 0, onNote: (n) => notes.push(n) })).toEqual(ANSWER)
    expect(r.asked).toEqual([FREE_JEV, PAID_JEV, OTHER_FREE].map(key))
    expect(pauses(f)).toEqual({ [key(FREE_JEV)]: 16342 * 1000, [key(PAID_JEV)]: 3600 * 1000 })
    expect(notes.length).toBe(1)
    expect(notes[0]).toContain(OTHER_FREE.id)

    r.asked.length = 0
    await classifyWithJev(r, {}, {}, { file: f, now: () => 60_000, onNote: (n) => notes.push(n) })
    expect(r.asked).toEqual([key(OTHER_FREE)], 'the paused ones are not asked again')
    expect(notes.length).toBe(1, 'told once')
  })

  test('each comes back at its own hour', async () => {
    const f = file()
    const r = registry([FREE_JEV, PAID_JEV, OTHER_FREE], { [key(FREE_JEV)]: QUOTA, [key(PAID_JEV)]: FUNDS })
    await classifyWithJev(r, {}, {}, { file: f, now: () => 0 })
    r.asked.length = 0
    await classifyWithJev(registry([], {}), {}, {}, { file: f, now: () => 0 })
    const back = registry([FREE_JEV, PAID_JEV, OTHER_FREE], {})
    await classifyWithJev(back, {}, {}, { file: f, now: () => 3600 * 1000 })
    expect(back.asked).toEqual([key(PAID_JEV)], 'an hour later the paid one is tried again, the free one still waits')
    const later = registry([FREE_JEV, PAID_JEV, OTHER_FREE], {})
    await classifyWithJev(later, {}, {}, { file: f, now: () => 16342 * 1000 })
    expect(later.asked).toEqual([key(FREE_JEV)], 'the free one is back')
  })

  test('a failure that is not about quota or funds moves on for this question only', async () => {
    const f = file()
    const r = registry([FREE_JEV, PAID_JEV], { [key(FREE_JEV)]: { stopReason: 'error', errorMessage: 'timeout' } })
    expect(await classifyWithJev(r, {}, {}, { file: f, now: () => 0 })).toEqual(ANSWER)
    expect(pauses(f)).toEqual({})
  })

  test('with nothing left it says so once, with the earliest hour one comes back', async () => {
    const notes: string[] = []
    const r = registry([FREE_JEV, PAID_JEV], { [key(FREE_JEV)]: QUOTA, [key(PAID_JEV)]: FUNDS })
    const result = await classifyWithJev(r, {}, {}, { file: file(), now: () => 0, onNote: (n) => notes.push(n) })
    expect(result).toBeNull()
    expect(notes.length).toBe(1)
    expect(notes[0]).toContain('no classifier is left')
  })

  test('while nothing can answer, routing is off until the first one comes back', async () => {
    const f = file()
    await classifyWithJev(registry([FREE_JEV, PAID_JEV], { [key(FREE_JEV)]: QUOTA, [key(PAID_JEV)]: FUNDS }), {}, {}, { file: f, now: () => 0 })
    expect(offUntil(f, 0)).toBe(3600 * 1000)
    expect(offUntil(f, 3600 * 1000)).toBe(16342 * 1000, 'the paid one is back: only the free one still waits')
    expect(offUntil(f, 16342 * 1000)).toBeNull()
  })

  test('no pause file, or a broken one, is no pause', () => {
    expect(pauses(join(tmpdir(), 'no-such-dir', 'x.json'))).toEqual({})
  })
})
