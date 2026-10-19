// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { CLASSIFIER, jevStatus } from './route.ts'

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
