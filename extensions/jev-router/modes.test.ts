// Run: bun test ~/.pi/agent/extensions/jev-router/modes.test.ts
import { describe, expect, test } from 'bun:test'
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { isPlanPath, notAPlan, planMayWrite } from './modes.ts'
import { exhausted, parseGoUsage } from './usage.ts'

const widget = join(homedir(), 'Projects', 'acme', 'checkout-widget')
const gateway = join(homedir(), 'Projects', 'acme', 'checkout-api')

describe('planMayWrite', () => {
  test('spec in the repo Pi runs in', () => expect(planMayWrite('docs/superpowers/specs/x.md', widget)).toBe(true))
  test('plan in another repo under ~/Projects', () => expect(planMayWrite(`${gateway}/docs/superpowers/plans/x.md`, widget)).toBe(true))
  test('.git of the repo (commit)', () => expect(planMayWrite(`${widget}/.git/index`, widget)).toBe(true))
  test('/tmp', () => expect(planMayWrite('/tmp/notes.md', widget)).toBe(true))
  test('source code is refused', () => expect(planMayWrite('src/main.ts', widget)).toBe(false))
  test('outside ~/Projects is refused', () => expect(planMayWrite(join(homedir(), '.bashrc'), widget)).toBe(false))
})

test('isPlanPath', () => {
  expect(isPlanPath(`${widget}/docs/superpowers/plans/2026-10-14-x.md`)).toBe(true)
  expect(isPlanPath(`${widget}/docs/superpowers/specs/2026-10-14-x.md`)).toBe(false)
})

describe('notAPlan', () => {
  const dir = mkdtempSync('/tmp/notaplan-')
  mkdirSync(join(dir, 'specs'))
  mkdirSync(join(dir, 'plans'))
  writeFileSync(join(dir, 'specs', 'a.md'), '# Design\n### Task 1: x\n')
  writeFileSync(join(dir, 'plans', 'a.md'), '# X Implementation Plan\n\n### Task 1: x\n')
  writeFileSync(join(dir, 'plans', 'b.md'), '# Notes\nsome text\n')
  test('a spec is refused', () => expect(notAPlan(join(dir, 'specs', 'a.md'), dir)).toContain('spec'))
  test('a plan passes', () => expect(notAPlan('plans/a.md', dir)).toBeNull())
  test('a file without tasks is refused', () => expect(notAPlan('plans/b.md', dir)).toContain('Task 1'))
  test('a missing file is refused', () => expect(notAPlan('plans/zzz.md', dir)).toContain("Can't find"))
})

describe('Go usage', () => {
  const body = (r: number, w: number) => ({
    usage: { rolling: { status: 'ok', percent: r }, weekly: { status: 'ok', percent: w }, monthly: { status: 'ok', percent: 46 } },
  })
  test('weekly at 93% is exhausted', () => expect(exhausted(parseGoUsage(body(0, 93))!, 90)?.name).toBe('weekly'))
  test('5 h first', () => expect(exhausted(parseGoUsage(body(95, 93))!, 90)?.name).toBe('5-hour'))
  test('below the threshold', () => expect(exhausted(parseGoUsage(body(10, 50))!, 90)).toBeNull())
  test('garbage parses to null', () => expect(parseGoUsage({ nope: 1 })).toBeNull())
})
