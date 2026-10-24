// Run: bun test ui/keys.test.js
import { describe, expect, test } from 'bun:test'
import { isAgentsKey, isDictateKey } from './keys.js'

describe('the chat shortcuts are read by the key place, not by the character it types', () => {
  test('Alt+A on Linux', () => {
    expect(isAgentsKey({ key: 'a', code: 'KeyA', altKey: true })).toBe(true)
  })

  test('Option+A on a Mac types another character; the place is what counts', () => {
    expect(isAgentsKey({ key: 'å', code: 'KeyA', altKey: true })).toBe(true)
  })

  test('Alt+A needs Alt and the A place', () => {
    expect(isAgentsKey({ key: 'a', code: 'KeyA', altKey: false })).toBe(false)
    expect(isAgentsKey({ key: 'd', code: 'KeyD', altKey: true })).toBe(false)
    // another layout whose A key is elsewhere: the place wins, as for Alt+D
    expect(isAgentsKey({ key: 'a', code: 'KeyQ', altKey: true })).toBe(false)
  })

  test('Alt+D keeps working the same way', () => {
    expect(isDictateKey({ key: '∂', code: 'KeyD', altKey: true })).toBe(true)
    expect(isDictateKey({ key: 'd', code: 'KeyD', altKey: false })).toBe(false)
    expect(isDictateKey({ key: 'a', code: 'KeyA', altKey: true })).toBe(false)
  })
})
