// Run: bun test extensions
// The language the tool tells the model to write in: the conversation's, named when Aria's config carries it.
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { inLanguage, toolTexts } from './texts.ts'

let dir = ''
const saved = process.env.PI_CODING_AGENT_DIR
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'aria-ask-'))
  process.env.PI_CODING_AGENT_DIR = dir
})
afterEach(() => {
  if (saved === undefined) delete process.env.PI_CODING_AGENT_DIR
  else process.env.PI_CODING_AGENT_DIR = saved
})

// What the tool shows the model: its description and the two fields that carry the language.
const registered = () => toolTexts()
const texts = (tool: any) => JSON.stringify(tool)

describe('ask_user language', () => {
  test('inLanguage names the language when it is known', () => {
    expect(inLanguage('')).toBe("the conversation's language")
    expect(inLanguage(' ')).toBe("the conversation's language")
    expect(inLanguage('French')).toBe("French, the conversation's language")
  })

  test('without aria.json the tool says the conversation\'s language, never Spanish', () => {
    const tool = registered()
    expect(tool.description).toContain("Write questions and options in the conversation's language.")
    expect(texts(tool)).not.toContain('Spanish')
    expect(texts(tool)).not.toContain('Alcance')
  })

  test('with assistant.language in aria.json the tool names it, in the description and in the question field', () => {
    writeFileSync(join(dir, 'aria.json'), JSON.stringify({ assistant: { language: 'Portuguese' } }))
    const tool = registered()
    expect(tool.description).toContain("Write questions and options in Portuguese, the conversation's language.")
    expect(tool.question).toBe("The full question, in Portuguese, the conversation's language, ending with a question mark.")
    expect(texts(tool)).not.toContain('Spanish')
  })
})
