// Run: bun test ui
import { describe, expect, test } from 'bun:test'
import { defaultShortcut, display, fromKeyEvent, label, parseShortcut } from './shortcut.js'

describe('the shortcut, as the page reads and records it', () => {
  test('parsing mirrors the dock: forgiving, canonical, per platform', () => {
    expect(label(parseShortcut('super+shift+a', 'linux'))).toBe('SUPER + SHIFT + A')
    expect(label(parseShortcut(' Shift + SUPER+ a ', 'linux'))).toBe('SUPER + SHIFT + A')
    expect(label(parseShortcut('ctrl + alt + space', 'linux'))).toBe('CTRL + ALT + Space')
    expect(label(parseShortcut('shift + cmd + opt + k', 'mac'))).toBe('Command + Option + Shift + K')
    expect(display(parseShortcut('ctrl+alt+f5', 'linux'))).toBe('Ctrl + Alt + F5')
    for (const text of ['', 'A', 'SUPER', 'SUPER + AB', 'SUPER + F13', 'SUPER + F05', 'SUPER + Return', 'HYPER + A', 'SUPER + A + B', 'Command + A']) {
      expect(parseShortcut(text, 'linux')).toBeNull()
    }
    expect(parseShortcut('SUPER + A', 'mac')).toBeNull()
    expect(display(defaultShortcut('linux'))).toBe('Super + A')
    expect(display(defaultShortcut('mac'))).toBe('Option + Space')
  })

  test('a key press becomes the canonical shortcut', () => {
    expect(fromKeyEvent({ key: 'a', code: 'KeyA', metaKey: true }, 'linux')).toEqual({ label: 'SUPER + A' })
    expect(fromKeyEvent({ key: '!', code: 'Digit1', shiftKey: true, ctrlKey: true }, 'linux')).toEqual({ label: 'CTRL + SHIFT + 1' })
    expect(fromKeyEvent({ key: ' ', code: 'Space', altKey: true }, 'mac')).toEqual({ label: 'Option + Space' })
    expect(fromKeyEvent({ key: 'F12', code: 'F12', ctrlKey: true, metaKey: true }, 'mac')).toEqual({ label: 'Command + Control + F12' })
    expect(fromKeyEvent({ key: 'x', ctrlKey: true }, 'linux')).toEqual({ label: 'CTRL + X' }) // no code: the key itself
    expect(fromKeyEvent({ key: 'Control', code: 'ControlLeft', ctrlKey: true }, 'linux')).toEqual({ pending: true })
    expect(fromKeyEvent({ key: 'k', code: 'KeyK' }, 'linux')).toEqual({ error: 'needsModifier' })
    expect(fromKeyEvent({ key: 'Tab', code: 'Tab', altKey: true }, 'linux')).toEqual({ error: 'unsupported' })
    expect(fromKeyEvent({ key: 'F13', code: 'F13', altKey: true }, 'linux')).toEqual({ error: 'unsupported' })
  })
})
