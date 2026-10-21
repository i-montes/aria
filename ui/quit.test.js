// Run: bun test ui/quit.test.js
import { describe, expect, test } from 'bun:test'
import { QUIT_WINDOW_MS, isQuitKey, quitPress } from './quit.js'

describe('quitting with Cmd+Q', () => {
  test('the first press only asks; a second one inside the window quits', () => {
    expect(quitPress(null, 1000)).toEqual({ quit: false, armedAt: 1000 })
    expect(quitPress(1000, 1000 + QUIT_WINDOW_MS)).toEqual({ quit: true, armedAt: null })
  })

  test('a second press after the window asks again', () => {
    expect(quitPress(1000, 1001 + QUIT_WINDOW_MS)).toEqual({ quit: false, armedAt: 1001 + QUIT_WINDOW_MS })
  })

  test('only Cmd+Q on a Mac is the key, with no other modifier', () => {
    const key = (extra) => ({ key: 'q', metaKey: true, altKey: false, ctrlKey: false, shiftKey: false, ...extra })
    expect(isQuitKey(key(), true)).toBe(true)
    expect(isQuitKey(key({ key: 'Q' }), true)).toBe(true)
    expect(isQuitKey(key(), false)).toBe(false) // Super+Q on Omarchy belongs to the compositor
    expect(isQuitKey(key({ altKey: true }), true)).toBe(false)
    expect(isQuitKey(key({ metaKey: false }), true)).toBe(false)
    expect(isQuitKey(key({ key: 'w' }), true)).toBe(false)
  })
})
