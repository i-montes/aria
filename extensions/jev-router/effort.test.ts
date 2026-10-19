// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { DEFAULTS } from '../lib/config.ts'
import { effortFor, saveEffort } from './effort.ts'

const config = (effort: Partial<typeof DEFAULTS.effort>) => ({ ...DEFAULTS, effort: { ...DEFAULTS.effort, ...effort } })

describe('the reasoning effort for a message', () => {
  test('auto: a quick route drops to low, a think route goes at least to high', () => {
    expect(effortFor({ route: 'quick', free: false, base: 'medium', fixed: null, config: DEFAULTS })).toBe('low')
    expect(effortFor({ route: 'think', free: false, base: 'medium', fixed: null, config: DEFAULTS })).toBe('high')
  })
  test('a free model runs at its top level', () => expect(effortFor({ route: 'quick', free: true, base: 'medium', fixed: null, config: DEFAULTS })).toBe('max'))
  test('maxOnFree off lets a free model follow the route', () =>
    expect(effortFor({ route: 'quick', free: true, base: 'medium', fixed: null, config: config({ maxOnFree: false }) })).toBe('low'))
  test('fixed mode ignores the route', () =>
    expect(effortFor({ route: 'quick', free: true, base: 'medium', fixed: null, config: config({ mode: 'fixed', fixedLevel: 'high' }) })).toBe('high'))
  test('a level picked by hand wins (the owner’s effort file, outside Aria)', () =>
    expect(effortFor({ route: 'think', free: false, base: 'medium', fixed: 'minimal', config: DEFAULTS })).toBe('minimal'))
  test('without a route (router off) the session level stays', () =>
    expect(effortFor({ route: undefined, free: false, base: 'medium', fixed: null, config: DEFAULTS })).toBe('medium'))
})

describe('/effort under Aria', () => {
  const { mkdtempSync, readFileSync, writeFileSync, existsSync } = require('node:fs')
  const { tmpdir } = require('node:os')
  const { join } = require('node:path')
  const run = (aria: boolean, body: (dir: string, settings: string) => void) => {
    const dir = mkdtempSync(join(tmpdir(), 'aria-effort-'))
    const settings = join(dir, 'settings.json')
    writeFileSync(settings, JSON.stringify({ effort: { mode: 'auto', fixedLevel: null, maxOnFree: true } }))
    const saved = process.env.ARIA_SETTINGS
    if (aria) process.env.ARIA_SETTINGS = settings
    else delete process.env.ARIA_SETTINGS
    try { body(dir, settings) } finally { if (saved === undefined) delete process.env.ARIA_SETTINGS; else process.env.ARIA_SETTINGS = saved }
  }
  test('a level fixes it in the settings; auto goes back', () => run(true, (dir, settings) => {
    saveEffort('high', join(dir, 'effort.json'))
    expect(JSON.parse(readFileSync(settings, 'utf8')).effort).toEqual({ mode: 'fixed', fixedLevel: 'high', maxOnFree: true })
    saveEffort(null, join(dir, 'effort.json'))
    expect(JSON.parse(readFileSync(settings, 'utf8')).effort).toEqual({ mode: 'auto', fixedLevel: null, maxOnFree: true })
    expect(existsSync(join(dir, 'effort.json'))).toBe(false)
  }))
  test('outside Aria the effort file keeps it, as before', () => run(false, (dir) => {
    saveEffort('high', join(dir, 'effort.json'))
    expect(JSON.parse(readFileSync(join(dir, 'effort.json'), 'utf8')).level).toBe('high')
  }))
})
