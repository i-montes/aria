import { describe, expect, test } from 'bun:test'
import { Verifier } from './flow.ts'

const make = (outcomes: { ok: boolean; output: string }[]) => {
  const sent: string[] = []
  const statuses: unknown[] = []
  let at = 0
  const v = new Verifier({
    checks: () => [{ name: 'test', command: 'make test' }],
    run: async () => ({ ...outcomes[Math.min(at++, outcomes.length - 1)], timedOut: false }),
    tell: (text) => sent.push(text),
    status: (value) => statuses.push(value),
  })
  return { v, sent, statuses }
}

describe('checks after editing', () => {
  test('a failure is fed back once per run', async () => {
    const { v, sent, statuses } = make([{ ok: false, output: 'line\n2 fail' }])
    v.edited()
    await v.runEnded()
    expect(sent).toHaveLength(1)
    expect(sent[0]).toContain('make test')
    expect(sent[0]).toContain('2 fail')
    expect(statuses.at(-1)).toEqual({ ok: false })
    // The model's fix is another run; failing again is said in the footer, not fed back again.
    v.edited()
    await v.runEnded()
    expect(sent).toHaveLength(1)
  })

  test('all passed says so and feeds nothing', async () => {
    const { v, sent, statuses } = make([{ ok: true, output: ' 43 pass\n 0 fail' }])
    v.edited()
    await v.runEnded()
    expect(sent).toEqual([])
    expect(statuses.at(-1)).toEqual({ ok: true, passed: 43 })
  })

  test('a run without edits checks nothing; off checks nothing', async () => {
    const { v, statuses } = make([{ ok: true, output: '' }])
    await v.runEnded()
    expect(statuses).toEqual([])
    v.setOn(false)
    v.edited()
    await v.runEnded()
    expect(statuses).toEqual([])
  })
})
