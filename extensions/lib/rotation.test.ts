// Run: bun test extensions
import { afterEach, describe, expect, test } from 'bun:test'
import { handOver, isRotating, markRotating, takeOver } from './rotation.ts'

afterEach(() => markRotating(false))

describe('the rotation handover', () => {
  test('hands state to the next instance only during a rotation, at its session_start', () => {
    markRotating(true)
    expect(isRotating()).toBe(true)
    handOver('k', { a: 1 })
    expect(takeOver('k', { type: 'session_start', reason: 'startup' })).toBeUndefined()
    expect(takeOver('k', { type: 'session_start', reason: 'new' })).toEqual({ a: 1 })
    expect(takeOver('k', { type: 'session_start', reason: 'new' })).toBeUndefined()
  })

  test('outside a rotation nothing is kept', () => {
    handOver('k', 1)
    expect(takeOver('k', { type: 'session_start', reason: 'new' })).toBeUndefined()
  })

  test('clearing the mark drops what was not taken back', () => {
    markRotating(true)
    handOver('k', 1)
    markRotating(false)
    expect(isRotating()).toBe(false)
    markRotating(true)
    expect(takeOver('k', { type: 'session_start', reason: 'new' })).toBeUndefined()
  })
})
