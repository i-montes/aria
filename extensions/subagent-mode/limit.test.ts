// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { DEFAULTS } from '../lib/config.ts'
import { turnLimitReached } from './limit.ts'

describe('the subagents turn limit (settings)', () => {
  test('no limit by default', () => expect(turnLimitReached(500, DEFAULTS)).toBe(false))
  test('stops at the limit', () => {
    const config = { ...DEFAULTS, subagents: { ...DEFAULTS.subagents, turnLimit: 3 } }
    expect(turnLimitReached(2, config)).toBe(false)
    expect(turnLimitReached(3, config)).toBe(true)
  })
})
