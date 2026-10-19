// Run: bun test ~/.pi/agent/extensions/autocompact
import { describe, expect, test } from 'bun:test'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { applyPercent, DEFAULT_PERCENT, MIN_RESERVE, parsePercent, readPercent, reserveFor, syncSettings, writePercent } from './autocompact.ts'

const big = { provider: 'opencode-go', id: 'big', contextWindow: 1_000_000 }
const small = { provider: 'opencode-go', id: 'small', contextWindow: 128_000 }

describe('parsePercent', () => {
  test('integers 0-100', () => expect([parsePercent('0'), parsePercent(' 40 '), parsePercent('100'), parsePercent('35%')]).toEqual([0, 40, 100, 35]))
  test('anything else', () => expect([parsePercent('101'), parsePercent('-1'), parsePercent('4.5'), parsePercent('abc'), parsePercent('')]).toEqual([null, null, null, null, null]))
})

describe('reserveFor', () => {
  test('40% of 1M compacts at 400k', () => expect(1_000_000 - reserveFor(1_000_000, 40)).toBe(400_000))
  test('40% of 128k compacts at 51.2k', () => expect(128_000 - reserveFor(128_000, 40)).toBe(51_200))
  test('never below the summary floor', () => expect(reserveFor(128_000, 100)).toBe(MIN_RESERVE))
})

describe('applyPercent', () => {
  const settings = { defaultModel: 'x', compaction: { keepRecentTokens: 30000, modelOverrides: { 'opencode-go/big': { keepRecentTokens: 5000, reserveTokens: 1 }, 'other/kept': { reserveTokens: 7 } } } }
  test('sets every model and keeps the rest', () => {
    const next = applyPercent(settings, [big, small], 40)
    expect(next.defaultModel).toBe('x')
    expect(next.compaction.keepRecentTokens).toBe(30000)
    expect(next.compaction.modelOverrides['opencode-go/big']).toEqual({ keepRecentTokens: 5000, reserveTokens: 600_000 })
    expect(next.compaction.modelOverrides['opencode-go/small']).toEqual({ reserveTokens: 76_800 })
    expect(next.compaction.modelOverrides['other/kept']).toEqual({ reserveTokens: 7 })
  })
  test('0 turns it off without touching the reserves', () => {
    const next = applyPercent(settings, [big], 0)
    expect(next.compaction.enabled).toBe(false)
    expect(next.compaction.modelOverrides['opencode-go/big'].reserveTokens).toBe(1)
  })
  test('a percent after 0 turns it back on', () => expect(applyPercent({ compaction: { enabled: false } }, [big], 40).compaction.enabled).toBeUndefined())
  test('models without a window are skipped', () => expect(applyPercent({}, [{ provider: 'p', id: 'm' }], 40).compaction.modelOverrides).toEqual({}))
})

describe('files', () => {
  const dir = mkdtempSync('/tmp/autocompact-')
  test('percent defaults to 40 and round-trips', () => {
    const f = join(dir, 'p.json')
    expect(readPercent(f)).toBe(DEFAULT_PERCENT)
    writePercent(25, f)
    expect(readPercent(f)).toBe(25)
    writePercent(DEFAULT_PERCENT, f)
    expect(readPercent(f)).toBe(DEFAULT_PERCENT)
  })
  test('syncSettings writes only on change', () => {
    const f = join(dir, 's.json')
    writeFileSync(f, JSON.stringify({ theme: 'dark' }))
    expect(syncSettings([big], 40, f)).toBe(true)
    expect(JSON.parse(readFileSync(f, 'utf8')).compaction.modelOverrides['opencode-go/big'].reserveTokens).toBe(600_000)
    expect(syncSettings([big], 40, f)).toBe(false)
  })
})
