// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { parseCardTitle } from '../../ui/chat/parts/cards.js'
import { quotaQuestion } from './ask.ts'
import { ASK_TIMEOUT_MS, FREE_MODEL } from './fallback.ts'

const spent = { window: '5-hour', percent: 98 }

describe('the quota question', () => {
  test("in Aria's dock it is the chat's quota card, with the two choices and the countdown", () => {
    const q = quotaQuestion({ mode: 'rpc', hasUI: true }, spent, 'glm-5.3')
    expect(q).not.toBeNull()
    const card = parseCardTitle(q!.title)
    expect(card.kind).toBe('quota')
    expect(card.percent).toBe(98)
    expect(card.freeModel).toBe(`${FREE_MODEL.provider}/${FREE_MODEL.id}`)
    expect(card.countdown).toBe(ASK_TIMEOUT_MS / 1000)
    expect(card.options.map((o: any) => o.value)).toEqual(q!.options)
    expect(card.options[0].role).toBe('free')
    expect(card.options[1].role).toBe('keep')
  })

  test('in the terminal it is the same text as always', () => {
    const q = quotaQuestion({ mode: 'tui', hasUI: true }, spent, 'glm-5.3')
    expect(q!.title.startsWith('OpenCode Go 5-hour quota at 98%.')).toBe(true)
    expect(q!.options).toEqual([`Switch to ${FREE_MODEL.id} now`, 'Stay on glm-5.3'])
  })

  test('with nobody to ask there is no question (it switches, as before)', () => {
    expect(quotaQuestion({ mode: 'json', hasUI: false }, spent, 'x')).toBeNull()
    expect(quotaQuestion(undefined, spent, 'x')).toBeNull()
  })
})
