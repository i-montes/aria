// Run: bun test ~/.pi/agent/extensions/quota-fallback
import { describe, expect, test } from 'bun:test'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { choiceFrom, decide, decisionKey, FREE_MODEL, isFree, isPaidGo, readDecision, writeDecision, readCachedUsage, writeCachedUsage } from './fallback.ts'
import type { GoUsage } from '../jev-router/usage.ts'

const usage = (rolling: number, weekly: number, status = 'ok'): GoUsage => ({
  rolling: { status, percent: rolling },
  weekly: { status: 'ok', percent: weekly, resetsAt: '2026-10-19T00:00:00.000Z' },
  monthly: { status: 'ok', percent: 10 },
})
const deepseek = { provider: 'opencode-go', id: 'deepseek-v4.1-flash', cost: { input: 0.15, output: 0.6 } }
const free = { provider: 'opencode-go', id: FREE_MODEL.id, cost: { input: 0, output: 0 } }
const zen = { provider: 'opencode', id: 'claude-opus-5-5', cost: { input: 4, output: 20 } }

describe('isPaidGo', () => {
  test('a priced Go model', () => expect(isPaidGo(deepseek)).toBe(true))
  test('a free Go model', () => expect(isPaidGo(free)).toBe(false))
  test('a -free id without cost data', () => expect(isPaidGo({ provider: 'opencode-go', id: 'cat-jam-free' })).toBe(false))
  test('another provider is not ours to switch', () => expect(isPaidGo(zen)).toBe(false))
  test('no model', () => expect(isPaidGo(undefined)).toBe(false))
})

describe('isFree', () => {
  test('a -free id', () => expect(isFree({ provider: 'opencode-go', id: 'cat-jam-free' })).toBe(true))
  test('zero prices on any provider', () => expect(isFree({ provider: 'local', id: 'x', cost: { input: 0, output: 0 } })).toBe(true))
  test('a priced model', () => expect(isFree(deepseek)).toBe(false))
  test('no cost data is not free', () => expect(isFree({ provider: 'opencode-go', id: 'glm-5.3' })).toBe(false))
  test('no model', () => expect(isFree(null)).toBe(false))
})

describe('decide', () => {
  test('below the threshold nothing changes', () => {
    expect(decide({ usage: usage(10, 97), current: deepseek, switchedFrom: null, threshold: 98 })).toBeNull()
  })
  test('weekly at the threshold switches a paid Go model to the free one', () => {
    const d = decide({ usage: usage(10, 98), current: deepseek, switchedFrom: null, threshold: 98 })
    expect(d?.kind).toBe('switch')
    expect(d?.kind === 'switch' && d.window).toBe('weekly')
  })
  test('a 5-hour window marked exhausted switches even under the threshold', () => {
    expect(decide({ usage: usage(40, 50, 'exhausted'), current: deepseek, switchedFrom: null, threshold: 98 })?.kind).toBe('switch')
  })
  test('already on the free model: nothing to do', () => {
    expect(decide({ usage: usage(10, 99), current: free, switchedFrom: deepseek, threshold: 98 })).toBeNull()
  })
  test('a model of another provider is left alone', () => {
    expect(decide({ usage: usage(10, 99), current: zen, switchedFrom: null, threshold: 98 })).toBeNull()
  })
  test('after the reset it goes back to the model it replaced', () => {
    const d = decide({ usage: usage(1, 2), current: free, switchedFrom: deepseek, threshold: 98 })
    expect(d).toEqual({ kind: 'restore', to: deepseek })
  })
  test('no restore when someone moved off the free model meanwhile', () => {
    expect(decide({ usage: usage(1, 2), current: zen, switchedFrom: deepseek, threshold: 98 })).toBeNull()
  })
  test('unknown usage changes nothing', () => {
    expect(decide({ usage: null, current: deepseek, switchedFrom: null, threshold: 98 })).toBeNull()
    expect(decide({ usage: null, current: free, switchedFrom: deepseek, threshold: 98 })).toBeNull()
  })
})

describe('usage cache', () => {
  const dir = mkdtempSync('/tmp/quota-fallback-')
  const file = join(dir, 'usage.json')
  test('fresh entries are read back', () => {
    writeCachedUsage(usage(1, 99), file, 1000)
    expect(readCachedUsage(file, 1000 + 59_000, 60_000)?.weekly.percent).toBe(99)
  })
  test('stale entries are ignored', () => {
    writeCachedUsage(usage(1, 99), file, 1000)
    expect(readCachedUsage(file, 1000 + 61_000, 60_000)).toBeNull()
  })
  test('a missing or broken file is ignored', () => {
    expect(readCachedUsage(join(dir, 'nope.json'), 0, 60_000)).toBeNull()
    writeFileSync(file, '{broken')
    expect(readCachedUsage(file, 0, 60_000)).toBeNull()
  })
})

describe('switch decision', () => {
  const near = { window: 'weekly', percent: 98, resetsAt: '2026-10-19T00:00:00.000Z' }
  test('a pick wins', () => {
    expect(choiceFrom('S', 'S', 'N', 1000)).toBe('switch')
    expect(choiceFrom('N', 'S', 'N', 1000)).toBe('stay')
  })
  test('no answer at the timeout switches', () => expect(choiceFrom(undefined, 'S', 'N', 300_000)).toBe('switch'))
  test('Esc before the timeout stays', () => expect(choiceFrom(undefined, 'S', 'N', 5_000)).toBe('stay'))
  test('a new window or a full quota asks again', () => {
    expect(decisionKey(near)).not.toBe(decisionKey({ ...near, resetsAt: '2026-10-26T00:00:00.000Z' }))
    expect(decisionKey(near)).not.toBe(decisionKey({ ...near, percent: 100 }))
  })
  test('the decision is shared through a file, for its window only', () => {
    const file = join(mkdtempSync('/tmp/qf-decision-'), 'd.json')
    expect(readDecision(decisionKey(near), file)).toBeNull()
    writeDecision(decisionKey(near), 'stay', file)
    expect(readDecision(decisionKey(near), file)).toBe('stay')
    expect(readDecision(decisionKey({ ...near, percent: 100 }), file)).toBeNull()
  })
})
