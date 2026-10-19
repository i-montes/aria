// Run: bun test ~/.pi/agent/extensions/sandbox
import { describe, expect, test } from 'bun:test'
import { parseReview, reviewLine, reviewRequest } from './review.ts'

const answer = (p: Record<string, number>) => ({
  stopReason: 'stop',
  answers: Object.fromEntries(Object.entries(p).map(([k, v]) => [k, { type: 'bool', probability: v }])),
})

describe('review', () => {
  test('asks about the match and every flag', () =>
    expect(Object.keys(reviewRequest('rm -rf x', 'Cleans up').questions)).toEqual(['matches', 'deletes', 'sends', 'installs', 'system']))
  test('a matching, harmless command', () => expect(reviewLine(parseReview(answer({ matches: 0.9, deletes: 0.1, sends: 0, installs: 0, system: 0 })))).toBe('Jev: ✓ matches the description'))
  test('a mismatch with flags', () =>
    expect(reviewLine(parseReview(answer({ matches: 0.2, deletes: 0.8, sends: 0.7, installs: 0.1, system: 0 })))).toBe(
      'Jev: ⚠ does NOT match the description · deletes or overwrites data · sends data out',
    ))
  test('a failed call', () => expect(reviewLine(parseReview({ stopReason: 'error' }))).toBe('Jev: no review (unavailable)'))
  test('a missing answer', () => expect(parseReview({ stopReason: 'stop', answers: {} })).toBeNull())
})
