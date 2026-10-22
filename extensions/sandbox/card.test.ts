// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { parseCardTitle } from '../../ui/chat/parts/cards.js'
import { canAsk, sandboxCard } from './card.ts'

describe("the chat's permission card", () => {
  test('the dock reads what the sandbox sends: kind, fields and its four options, the first primary', () => {
    const title = sandboxCard('sandbox', { summary: 'Restarts the database', detail: 'docker restart pg', jev: '✓ matches the description' }, ['Once', 'For docker', 'All', 'No'])
    const card = parseCardTitle(title)
    expect(card.kind).toBe('sandbox')
    expect(card.summary).toBe('Restarts the database')
    expect(card.detail).toBe('docker restart pg')
    expect(card.jev).toBe('✓ matches the description')
    expect(card.options).toEqual([
      { value: 'Once', label: 'Once', primary: true },
      { value: 'For docker', label: 'For docker' },
      { value: 'All', label: 'All' },
      { value: 'No', label: 'No' },
    ])
  })

  test('a write card carries the path', () => {
    const card = parseCardTitle(sandboxCard('write', { path: '~/notes.md', detail: 'write ~/notes.md' }, ['Once', 'No']))
    expect(card.kind).toBe('write')
    expect(card.path).toBe('~/notes.md')
  })

  test('only the terminal and Aria have someone to ask', () => {
    expect(canAsk({ mode: 'tui', hasUI: true })).toBe(true)
    expect(canAsk({ mode: 'rpc', hasUI: true })).toBe(true)
    expect(canAsk({ mode: 'json', hasUI: false })).toBe(false)
    expect(canAsk({ mode: 'print', hasUI: false })).toBe(false)
    expect(canAsk({ mode: 'rpc', hasUI: false })).toBe(false)
    expect(canAsk(undefined)).toBe(false)
  })
})

describe('a scheduled tab (ARIA_GRANTS noAsk)', () => {
  test('nobody can be asked there', () => {
    const saved = process.env.ARIA_GRANTS
    const rpc = { hasUI: true, mode: 'rpc' }
    expect(canAsk(rpc)).toBe(true)
    process.env.ARIA_GRANTS = JSON.stringify({ noAsk: true })
    expect(canAsk(rpc)).toBe(false)
    if (saved === undefined) delete process.env.ARIA_GRANTS
    else process.env.ARIA_GRANTS = saved
  })
})
