// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { bashRoute } from './route.ts'

const base = { covered: false, outside: false, hasBwrap: true, canAsk: true }

describe('where a bash command runs', () => {
  test('inside bwrap by default', () => expect(bashRoute(base)).toBe('boxed'))
  test('straight outside when the session already allowed it', () => expect(bashRoute({ ...base, covered: true })).toBe('plain'))
  test('outside after asking, when the model asks for it', () => expect(bashRoute({ ...base, outside: true })).toBe('ask'))
  test('refused when it must ask and nobody can answer', () => expect(bashRoute({ ...base, outside: true, canAsk: false })).toBe('refuse-nobody'))

  describe('without a working box (macOS, or Linux with no bwrap or one the system blocks): nothing runs unconfined without the user', () => {
    for (const platform of ['darwin', 'linux']) {
      const none = { ...base, hasBwrap: false, platform }
      test(`${platform}: every command asks first`, () => expect(bashRoute(none)).toBe('ask'))
      test(`${platform}: a command the session allowed still runs`, () => expect(bashRoute({ ...none, covered: true })).toBe('plain'))
      test(`${platform}: without a UI it is refused`, () => expect(bashRoute({ ...none, canAsk: false })).toBe('refuse-nobody'))
      test(`${platform}: leaving the sandbox off in the settings refuses`, () => expect(bashRoute({ ...none, mayLeave: false })).toBe('refuse-not-allowed'))
    }
  })
})

describe('leaving the sandbox switched off in the settings', () => {
  const base = { covered: false, outside: false, hasBwrap: true, canAsk: true, mayLeave: false }
  test('asking to leave is refused without asking, as not allowed', () => expect(bashRoute({ ...base, outside: true })).toBe('refuse-not-allowed'))
  test('a normal command still runs in the box', () => expect(bashRoute(base)).toBe('boxed'))
  test('a command the session already allowed still runs', () => expect(bashRoute({ ...base, outside: true, covered: true })).toBe('plain'))
})
