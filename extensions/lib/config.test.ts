// Run: bun test extensions
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ariaConfig, writeAriaSetting } from './config.ts'

const TODAY = {
  version: 1,
  models: { plan: { provider: 'opencode-go', id: 'mimo-v2.6-pro' }, planThinking: 'high', allowPayPerUse: false },
  quota: { switchAt: 98, planSwitchAt: 90, freeModel: { provider: 'opencode-go', id: 'cat-jam-free' }, askFirst: true, countdownMs: 300000, onNoAnswer: 'switch' },
  router: { enabled: true, backend: 'opencode' },
  effort: { mode: 'auto', fixedLevel: null, maxOnFree: true },
  compaction: { percent: 40 },
  style: 'very-short',
  assistant: { language: '' },
  sandbox: { extraWritable: [], askToLeave: true, jevReview: true },
  subagents: { saveSessions: false, turnLimit: null, planExecute: true },
  connectors: {},
  noAsk: false,
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

  test('the conversation language is read by name, and empty when absent or not a string', () => {
    expect(ariaConfig().assistant.language).toBe('')
    write({ assistant: { language: 'Spanish' } })
    expect(ariaConfig().assistant.language).toBe('Spanish')
    write({ assistant: { language: 3 } })
    expect(ariaConfig().assistant.language).toBe('')
  })

  test('connectors default to empty', () => {
    expect(ariaConfig().connectors).toEqual({})
    write({ connectors: 'all' })
    expect(ariaConfig().connectors).toEqual({})
  })

  test('an unknown level reads as ask', () => {
    write({ connectors: { notion: 'read', tavily: 'allow', rea: 'ask', figma: 'sometimes', other: 3 } })
    expect(ariaConfig().connectors).toEqual({ notion: 'read', tavily: 'allow', rea: 'ask', figma: 'ask', other: 'ask' })
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

  test("the router's backend is the Jev's provider: TypeSafe, else OpenCode", () => {
    write({ router: { enabled: true, backend: 'typesafe' } })
    expect(ariaConfig().router).toEqual({ enabled: true, backend: 'typesafe' })
    write({ router: { enabled: true, backend: 'opencode' } })
    expect(ariaConfig().router.backend).toBe('opencode')
    write({ router: { enabled: true } })
    expect(ariaConfig().router.backend).toBe('opencode')
    write({ router: { enabled: true, backend: 'somewhere' } })
    expect(ariaConfig().router.backend).toBe('opencode')
  })

  test('is read again on every call', () => {
    write({ compaction: { percent: 70 } })
    expect(ariaConfig().compaction.percent).toBe(70)
    write({ compaction: { percent: 55 } })
    expect(ariaConfig().compaction.percent).toBe(55)
  })
})

describe("a scheduled tab's fixed permissions (ARIA_GRANTS)", () => {
  const savedGrants = process.env.ARIA_GRANTS
  afterEach(() => {
    if (savedGrants === undefined) delete process.env.ARIA_GRANTS
    else process.env.ARIA_GRANTS = savedGrants
  })

  test('grants set the connector levels and turn asking off', () => {
    write({ connectors: { notion: 'ask', tavily: 'allow' }, sandbox: { extraWritable: ['/a'], askToLeave: true, jevReview: false } })
    process.env.ARIA_GRANTS = JSON.stringify({ connectors: { notion: 'read', tavily: 'read' }, sandbox: { askToLeave: false, extraWritable: ['/out'] } })
    const config = ariaConfig()
    expect(config.connectors).toEqual({ notion: 'read', tavily: 'read' })
    expect(config.sandbox).toEqual({ extraWritable: ['/out'], askToLeave: false, jevReview: false })
  })

  test('invalid grants change nothing', () => {
    write({ connectors: { notion: 'allow' } })
    for (const bad of ['{ nope', '[]', '"x"', JSON.stringify({ connectors: 'all', sandbox: { askToLeave: 'no', extraWritable: [1] } })]) {
      process.env.ARIA_GRANTS = bad
      const config = ariaConfig()
      expect(config.connectors).toEqual({ notion: 'allow' })
      expect(config.sandbox.askToLeave).toBe(true)
      expect(config.sandbox.extraWritable).toEqual([])
    }
  })

  test('a scheduled tab never asks: noAsk comes only from the grants', () => {
    write({ noAsk: true })
    expect(ariaConfig().noAsk).toBe(false)
    process.env.ARIA_GRANTS = JSON.stringify({ noAsk: true })
    expect(ariaConfig().noAsk).toBe(true)
  })

  test('a granted level outside read, ask and allow reads as ask', () => {
    process.env.ARIA_GRANTS = JSON.stringify({ connectors: { notion: 'everything' } })
    expect(ariaConfig().connectors).toEqual({ notion: 'ask' })
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
