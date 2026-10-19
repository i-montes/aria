// Run: bun test extensions
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ariaConfig, writeAriaSetting } from './config.ts'

const TODAY = {
  version: 1,
  models: { plan: { provider: 'opencode-go', id: 'glm-5.3' }, planThinking: 'high', allowPayPerUse: false },
  quota: { switchAt: 98, planSwitchAt: 90, freeModel: { provider: 'opencode-go', id: 'cat-jam-free' }, askFirst: true, countdownMs: 300000, onNoAnswer: 'switch' },
  router: { enabled: true },
  effort: { mode: 'auto', fixedLevel: null, maxOnFree: true },
  compaction: { percent: 40 },
  style: 'very-short',
  sandbox: { extraWritable: [], askToLeave: true, jevReview: true },
  subagents: { saveSessions: false, turnLimit: null, planExecute: true },
}

let dir = ''
const saved = process.env.PI_CODING_AGENT_DIR
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'aria-config-'))
  process.env.PI_CODING_AGENT_DIR = dir
})
afterEach(() => {
  if (saved === undefined) delete process.env.PI_CODING_AGENT_DIR
  else process.env.PI_CODING_AGENT_DIR = saved
})
const write = (value: unknown) => writeFileSync(join(dir, 'aria.json'), typeof value === 'string' ? value : JSON.stringify(value))

describe("the extensions' settings (aria.json)", () => {
  test("without aria.json every value is today's default", () => {
    expect(ariaConfig()).toEqual(TODAY)
  })

  test('a missing or wrong-typed field falls back field by field', () => {
    write({ quota: { switchAt: 'lots' }, effort: { maxOnFree: false, fixedLevel: 'high' }, subagents: { turnLimit: 12 }, sandbox: { extraWritable: ['/srv', 3] } })
    const config = ariaConfig()
    expect(config.quota.switchAt).toBe(98)
    expect(config.effort.maxOnFree).toBe(false)
    expect(config.effort.fixedLevel).toBe('high')
    expect(config.subagents.turnLimit).toBe(12)
    expect(config.sandbox.extraWritable).toEqual(['/srv'])
    expect(config.compaction.percent).toBe(40)
  })

  test('a file that does not parse is ignored', () => {
    write('{ nope')
    expect(ariaConfig()).toEqual(TODAY)
  })

  test('a provider without a free tier (no quota section) keeps the default fallback values', () => {
    write({ router: { enabled: false } })
    expect(ariaConfig().router.enabled).toBe(false)
    expect(ariaConfig().quota.switchAt).toBe(98)
  })

  test('is read again on every call', () => {
    write({ compaction: { percent: 70 } })
    expect(ariaConfig().compaction.percent).toBe(70)
    write({ compaction: { percent: 55 } })
    expect(ariaConfig().compaction.percent).toBe(55)
  })
})

describe("writing Aria's settings (ARIA_SETTINGS)", () => {
  const savedSettings = process.env.ARIA_SETTINGS
  afterEach(() => {
    if (savedSettings === undefined) delete process.env.ARIA_SETTINGS
    else process.env.ARIA_SETTINGS = savedSettings
  })

  test('keeps every other key and writes atomically', () => {
    const file = join(dir, 'settings.json')
    writeFileSync(file, JSON.stringify({ version: 1, effort: { mode: 'auto', fixedLevel: null, maxOnFree: true }, extra: { kept: true } }))
    process.env.ARIA_SETTINGS = file
    expect(writeAriaSetting(['effort', 'fixedLevel'], 'high')).toBe(true)
    const after = JSON.parse(require('node:fs').readFileSync(file, 'utf8'))
    expect(after).toEqual({ version: 1, effort: { mode: 'auto', fixedLevel: 'high', maxOnFree: true }, extra: { kept: true } })
    expect(require('node:fs').statSync(file).mode & 0o777).toBe(0o600)
    expect(require('node:fs').readdirSync(dir).filter((f: string) => f.includes('.tmp'))).toEqual([])
  })

  test('outside Aria it writes nothing', () => {
    delete process.env.ARIA_SETTINGS
    expect(writeAriaSetting(['compaction', 'percent'], 70)).toBe(false)
  })

  test('a settings file that does not parse is left alone', () => {
    const file = join(dir, 'settings.json')
    writeFileSync(file, '{ nope')
    process.env.ARIA_SETTINGS = file
    expect(writeAriaSetting(['compaction', 'percent'], 70)).toBe(false)
    expect(require('node:fs').readFileSync(file, 'utf8')).toBe('{ nope')
  })
})
