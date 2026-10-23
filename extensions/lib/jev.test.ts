import { describe, expect, test } from 'bun:test'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { classifyWithJev, FREE_JEV, PAID_JEV, pausedUntil, pauseNote, quotaWait } from './jev.ts'

const QUOTA = {
  stopReason: 'error',
  answers: {},
  errorMessage: 'Server requested 16342s retry delay (max: 60s). System One API returned 429',
}
const ANSWER = { stopReason: 'stop', answers: { route: 'quick' } }

// A model registry that answers each model with the given result, and records which models were asked.
function registry(answers: Record<string, any>) {
  const asked: string[] = []
  return {
    asked,
    findOfType: (_type: string, provider: string, id: string) => ({ provider, id }),
    classify: async (model: { id: string }) => {
      asked.push(model.id)
      return answers[model.id]
    },
  }
}

const file = () => join(mkdtempSync(join(tmpdir(), 'jev-')), 'jev-free-paused.json')

describe('Jev, free and paid', () => {
  test('a quota answer says how long to wait; anything else does not', () => {
    expect(quotaWait(QUOTA)).toBe(16342)
    expect(quotaWait({ stopReason: 'error', errorMessage: 'System One API returned 429' })).toBe(3600)
    expect(quotaWait({ stopReason: 'error', errorMessage: 'network down' })).toBeNull()
    expect(quotaWait(ANSWER)).toBeNull()
    expect(quotaWait(null)).toBeNull()
  })

  test('the free Jev answers while it has quota', async () => {
    const r = registry({ [FREE_JEV.id]: ANSWER })
    expect(await classifyWithJev(r, {}, {}, { file: file(), now: () => 1000 })).toEqual(ANSWER)
    expect(r.asked).toEqual([FREE_JEV.id])
  })

  test('out of quota, the same question goes to the paid Jev, and it keeps answering until the free one is back', async () => {
    const f = file()
    const paused: number[] = []
    const r = registry({ [FREE_JEV.id]: QUOTA, [PAID_JEV.id]: ANSWER })
    const result = await classifyWithJev(r, {}, {}, { file: f, now: () => 1000, onPause: (until) => paused.push(until) })
    expect(result).toEqual(ANSWER)
    expect(r.asked).toEqual([FREE_JEV.id, PAID_JEV.id])
    expect(paused).toEqual([1000 + 16342 * 1000])
    expect(pausedUntil(f)).toBe(1000 + 16342 * 1000)

    r.asked.length = 0
    await classifyWithJev(r, {}, {}, { file: f, now: () => 1000 + 16341 * 1000 })
    expect(r.asked).toEqual([PAID_JEV.id], 'still paused: the free one is not asked')

    r.asked.length = 0
    await classifyWithJev(r, {}, {}, { file: f, now: () => 1000 + 16342 * 1000 })
    expect(r.asked[0]).toBe(FREE_JEV.id, 'the hour came: back to the free one')
  })

  test('a pause is shared through its file: another session goes straight to the paid Jev', async () => {
    const f = file()
    await classifyWithJev(registry({ [FREE_JEV.id]: QUOTA, [PAID_JEV.id]: ANSWER }), {}, {}, { file: f, now: () => 0 })
    const other = registry({ [FREE_JEV.id]: QUOTA, [PAID_JEV.id]: ANSWER })
    await classifyWithJev(other, {}, {}, { file: f, now: () => 5 })
    expect(other.asked).toEqual([PAID_JEV.id])
  })

  test('the pause says whether the paid Jev answered', async () => {
    const heard: [number, any][] = []
    const broke = { stopReason: 'error', errorMessage: 'System One API error (402): Insufficient account funds' }
    const r = registry({ [FREE_JEV.id]: QUOTA, [PAID_JEV.id]: broke })
    await classifyWithJev(r, {}, {}, { file: file(), now: () => 0, onPause: (until, paid) => heard.push([until, paid]) })
    expect(heard).toEqual([[16342 * 1000, broke]])
    expect(pauseNote(16342 * 1000, ANSWER)).toContain('the paid Jev answers until')
    expect(pauseNote(16342 * 1000, broke)).toContain('Insufficient account funds')
  })

  test('no pause file, or a broken one, is no pause', () => {
    expect(pausedUntil(join(tmpdir(), 'no-such-dir', 'x.json'))).toBe(0)
  })
})
