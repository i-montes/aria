// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { bashRoute } from './route.ts'

const base = { covered: false, outside: false, hasBwrap: true, canAsk: true, platform: 'linux' }

describe('where a bash command runs', () => {
  test('inside bwrap by default', () => expect(bashRoute(base)).toBe('boxed'))
  test('straight outside when the session already allowed it', () => expect(bashRoute({ ...base, covered: true })).toBe('plain'))
  test('outside after asking, when the model asks for it', () => expect(bashRoute({ ...base, outside: true })).toBe('ask'))
  test('refused when it must ask and nobody can answer', () => expect(bashRoute({ ...base, outside: true, canAsk: false })).toBe('refuse-nobody'))

  describe('without bwrap (macOS): nothing runs unconfined without the user', () => {
    const mac = { ...base, hasBwrap: false, platform: 'darwin' }
    test('every command asks first', () => expect(bashRoute(mac)).toBe('ask'))
    test('a command the session allowed still runs', () => expect(bashRoute({ ...mac, covered: true })).toBe('plain'))
    test('without a UI it is refused', () => expect(bashRoute({ ...mac, canAsk: false })).toBe('refuse-nobody'))
  })
  test('on Linux a missing bwrap is a broken install: still refused, as before', () => {
    expect(bashRoute({ ...base, hasBwrap: false })).toBe('refuse-no-bwrap')
    expect(bashRoute({ ...base, hasBwrap: false, covered: true })).toBe('plain')
    expect(bashRoute({ ...base, hasBwrap: false, outside: true })).toBe('ask')
  })
})

describe('leaving the sandbox switched off in the settings', () => {
  const base = { covered: false, outside: false, hasBwrap: true, canAsk: true, platform: 'linux', mayLeave: false }
  test('asking to leave is refused without asking, as not allowed', () => expect(bashRoute({ ...base, outside: true })).toBe('refuse-not-allowed'))
  test('a normal command still runs in the box', () => expect(bashRoute(base)).toBe('boxed'))
  test('a command the session already allowed still runs', () => expect(bashRoute({ ...base, outside: true, covered: true })).toBe('plain'))
})
