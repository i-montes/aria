// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { DEFAULTS } from '../lib/config.ts'
import { effortFor } from './effort.ts'

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
