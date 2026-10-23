// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import secrets, { savedName, secretCard } from './index.ts'

describe('the secret card', () => {
  test('names the secret and its variable, never a value', () => {
    expect(JSON.parse(secretCard('github-token', 'the CI'))).toEqual({
      aria: 1,
      kind: 'secret',
      name: 'github-token',
      purpose: 'the CI',
      env: 'ARIA_SECRET_GITHUB_TOKEN',
    })
  })

  test("only the page's saved answer counts", () => {
    expect(savedName('saved:github-token')).toBe('github-token')
    expect(savedName('ghp_0123456789abcdefghij')).toBeNull()
    expect(savedName(undefined)).toBeNull()
  })
})

describe('every tool output is masked', () => {
  const handlers: Record<string, Function> = {}
  secrets({ registerTool() {}, on: (name: string, fn: Function) => (handlers[name] = fn) } as any)

  test('a key in a file read reaches the model masked, with the note', () => {
    const out = handlers.tool_result({ input: { path: '.env' }, content: [{ type: 'text', text: 'OPENAI_API_KEY=sk-proj-0123456789abcdefXYZ' }] })
    expect(out.content[0].text).not.toContain('sk-proj')
    expect(out.content[0].text).toContain('secret_import')
    expect(out.content[0].text).toContain('`.env`')
  })

  test('an output with nothing to mask is left alone', () => {
    expect(handlers.tool_result({ content: [{ type: 'text', text: 'PORT=3000' }] })).toBeUndefined()
  })
})
