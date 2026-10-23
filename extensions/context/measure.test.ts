import { describe, expect, test } from 'bun:test'
import { measure, tokens } from './measure.ts'

describe('what fills the context', () => {
  test('measures each part', () => {
    expect(tokens('abcd'.repeat(10))).toBe(10)
    expect(tokens('abc')).toBe(1)
    const parts = measure({
      systemPrompt: 'x'.repeat(4000),
      skills: [{ name: 'tdd', description: 'y'.repeat(400) }],
      tools: [{ name: 'read', description: 'z'.repeat(396), parameters: {} }, { name: 'mcp_notion', description: 'w'.repeat(796), parameters: {} }],
      memory: 'm'.repeat(80),
      usedTokens: 5_000,
    })
    expect(parts.system).toBe(1000 - 100)
    expect(parts.skills).toBe(100)
    expect(parts.tools).toBe(101 + 203)
    expect(parts.memory).toBe(20)
    expect(parts.messages).toBe(5_000 - 900 - 100 - 304 - 20)
  })

  test('without a known total, messages are what is left at least zero', () => {
    const parts = measure({ systemPrompt: 'x'.repeat(40), skills: [], tools: [], memory: '', usedTokens: null })
    expect(parts.messages).toBe(0)
  })
})
