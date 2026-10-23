// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { CLASSIFIER, hintLine, jevStatus, methodNote, parseHints } from './route.ts'

const hints = (over: Record<string, unknown> = {}) => ({ route: 'focused', routeP: 0.9, memory: false, web: false, notion: false, readOnly: false, ...over }) as any

describe("the route line the dock shows", () => {
  test('each route has the words the chat knows', () => {
    expect(jevStatus(hints({ route: 'quick' })).mode).toBe('quick')
    expect(jevStatus(hints({ route: 'focused' })).mode).toBe('focused task')
    expect(jevStatus(hints({ route: 'parallel' })).mode).toBe('work to split')
    expect(jevStatus(hints({ route: 'think' })).mode).toBe('needs thinking')
  })

  test('the flags become what Aria does, in order; none gives no action', () => {
    expect(jevStatus(hints({ memory: true, web: true, notion: true, readOnly: true })).action).toBe('searches memory · searches the web · checks Notion · read-only')
    expect(jevStatus(hints()).action).toBe('')
  })
})

test('the classifier is the free Jev, by its exact id (jev-1.13 is paid)', () => {
  expect(CLASSIFIER).toEqual({ provider: 'opencode', id: 'jev-1.13-free' })
})

describe('offering a plan for sizable work', () => {
  const result = (route: string, sizable: number, changes = 0.9) => ({
    stopReason: 'stop',
    answers: {
      route: { type: 'choice', choice: route, probabilities: { [route]: 0.9 } },
      sizable: { type: 'bool', probability: sizable },
      changes: { type: 'bool', probability: changes },
    },
  })

  test('sizable work that changes things gets plan=offer', () => {
    const h = parseHints(result('focused', 0.85))!
    expect(h.planWorth).toBe(true)
    expect(hintLine(h)).toEndWith('plan=offer')
  })

  test('small work, a doubt, a question or a read-only analysis does not', () => {
    expect(parseHints(result('focused', 0.2))!.planWorth).toBe(false)
    expect(parseHints(result('focused', 0.55))!.planWorth).toBe(false)
    expect(parseHints(result('think', 0.9))!.planWorth).toBe(false)
    expect(parseHints(result('quick', 0.9))!.planWorth).toBe(false)
    expect(parseHints(result('parallel', 0.9, 0.1))!.planWorth).toBe(false)
    expect(hintLine(parseHints(result('focused', 0.2))!)).not.toContain('plan=')
  })

  test('a focused task is asked for its outcome first', () => {
    const note = methodNote(parseHints(result('focused', 0.2))!)
    expect(note).toContain('code in a project with tests')
    expect(note).toContain('see it fail')
    expect(note).toContain('Otherwise just do it and check the result once', 'a file, a setting or a command needs no failing check first')
    expect(methodNote(parseHints(result('focused', 0.2, 0.1))!)).toBe('', 'read-only: nothing to prove')
    expect(methodNote(parseHints(result('focused', 0.9))!)).toBe('', 'sizable work is offered a plan instead')
    expect(methodNote(parseHints(result('quick', 0.1))!)).toBe('')
  })

  test('no method names in hints', () => {
    for (const route of ['quick', 'focused', 'parallel', 'think']) {
      for (const sizable of [0.1, 0.9]) {
        const h = parseHints(result(route, sizable))
        if (!h) continue
        const said = hintLine(h) + methodNote(h)
        expect(said).not.toMatch(/\b(ODD|SDD|TDD)\b/)
      }
    }
  })
})
