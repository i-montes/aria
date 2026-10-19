// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { STYLE_NOTE, styleNote } from './style.ts'

describe('the answer-style reminder after each message', () => {
  test("very-short (and no setting) is the owner's own note", () => {
    expect(styleNote('very-short')).toBe(STYLE_NOTE)
    expect(styleNote(undefined)).toBe(STYLE_NOTE)
    expect(STYLE_NOTE).toContain('at most ~80 words')
  })
  test('the style follows the setting', () => {
    const notes = ['brief', 'detailed', 'technical', 'teaching'].map(styleNote)
    expect(new Set([...notes, STYLE_NOTE]).size).toBe(5)
    expect(styleNote('detailed')).not.toContain('80 words')
  })
})
