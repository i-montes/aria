// Run: bun test ui/keys.test.js
import { describe, expect, test } from 'bun:test'
import { isAgentsKey, isDictateKey } from './keys.js'

const alt = (key, code, extra = {}) => ({ key, code, altKey: true, ctrlKey: false, metaKey: false, ...extra })

describe('the chat shortcuts follow the key labelled with the letter, on any layout', () => {
  test('Alt+A on a QWERTY keyboard', () => {
    expect(isAgentsKey(alt('a', 'KeyA'))).toBe(true)
  })

  test('Option+A on a Mac types another character; the key place still fires it', () => {
    expect(isAgentsKey(alt('å', 'KeyA'))).toBe(true)
  })

  test('Alt+A on AZERTY: the key labelled A sends the code KeyQ', () => {
    expect(isAgentsKey(alt('a', 'KeyQ'))).toBe(true)
    expect(isAgentsKey(alt('A', 'KeyQ', { shiftKey: true }))).toBe(true)
  })

  test('Alt+A needs Alt, and no Ctrl or Cmd with it', () => {
    expect(isAgentsKey(alt('a', 'KeyA', { altKey: false }))).toBe(false)
    expect(isAgentsKey(alt('å', 'KeyA', { metaKey: true }))).toBe(false) // Cmd+Option+A
    expect(isAgentsKey(alt('a', 'KeyA', { ctrlKey: true }))).toBe(false) // Ctrl+Alt+A
    expect(isAgentsKey(alt('a', 'KeyQ', { metaKey: true }))).toBe(false)
  })

  test('another letter, or a named key, is not Alt+A', () => {
    expect(isAgentsKey(alt('d', 'KeyD'))).toBe(false)
    expect(isAgentsKey(alt('Dead', 'KeyE'))).toBe(false)
    expect(isAgentsKey(alt('Alt', 'AltLeft'))).toBe(false)
  })

  test('Alt+D works the same way', () => {
    expect(isDictateKey(alt('d', 'KeyD'))).toBe(true)
    expect(isDictateKey(alt('∂', 'KeyD'))).toBe(true) // Option+D on a Mac
    expect(isDictateKey(alt('d', 'KeyE'))).toBe(true) // a layout with D elsewhere (Dvorak)
    expect(isDictateKey(alt('d', 'KeyD', { altKey: false }))).toBe(false)
    expect(isDictateKey(alt('∂', 'KeyD', { metaKey: true }))).toBe(false)
    expect(isDictateKey(alt('d', 'KeyD', { ctrlKey: true }))).toBe(false)
    expect(isDictateKey(alt('a', 'KeyA'))).toBe(false)
  })
})
