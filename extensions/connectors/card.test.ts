// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { connectorCard, preview } from './card.ts'

describe("a connector's card", () => {
  test('the_preview_is_short_and_readable', () => {
    const input: Record<string, unknown> = { title: 'Weekly notes', body: 'x'.repeat(1000), parent: { page_id: 'abc', nested: { deep: [1, 2] } } }
    for (let i = 0; i < 20; i++) input[`extra${i}`] = i
    const out = preview(input)
    const lines = out.split('\n')
    expect(lines.length).toBeLessThanOrEqual(12)
    expect(lines[0]).toBe('title: Weekly notes')
    expect(lines[1].startsWith('body: xxx')).toBe(true)
    expect(lines[1].length).toBeLessThanOrEqual('body: '.length + 300)
    expect(lines[2]).toBe('parent: {"page_id":"abc","nested":{"deep":[1,2]}}')
    expect(lines[11]).toContain('more')
    for (const line of lines) expect(line).not.toContain('\n')
    // A multi-line string stays on its line.
    expect(preview({ text: 'one\ntwo' })).toBe('text: one two')
    expect(preview({})).toBe('(no arguments)')
    expect(preview(undefined)).toBe('(no arguments)')
    expect(preview('just text')).toBe('just text')
  })

  test('the_card_is_a_connector_card', () => {
    const options = ['Yes, just this once', "Yes, and don't ask again for Notion create pages this session", 'No']
    const json = connectorCard('Notion', 'create pages', 'title: Weekly notes', options)
    expect(JSON.parse(json)).toEqual({
      aria: 1,
      kind: 'connector',
      summary: 'Notion wants to create pages',
      detail: 'title: Weekly notes',
      options: [
        { value: options[0], label: options[0], primary: true },
        { value: options[1], label: options[1] },
        { value: 'No', label: 'No' },
      ],
    })
  })
})
