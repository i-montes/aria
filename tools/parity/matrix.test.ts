import { describe, expect, test } from 'bun:test'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { parseMatrix, problems, type Row, setupItems } from './matrix.ts'

const row = (over: Partial<Row>): Row => ({ item: 1, what: 'x', check: 'test', where: 'a.rs::t', result: '—', ...over })
const files: Record<string, string> = { 'a.rs': 'fn t() {}', 'e.ts': '' }
const exists = (p: string) => p in files
const read = (p: string) => files[p] ?? ''

describe('the parity matrix', () => {
  test('parses the table rows whose first cell is a number', () => {
    const md = '| Item | What | Check | Where | Result |\n|---|---|---|---|---|\n| 7 | Daily model | test | `a.rs::t` | pass |\n| x | not a row |\n'
    expect(parseMatrix(md)).toEqual([{ item: 7, what: 'Daily model', check: 'test', where: 'a.rs::t', result: 'pass' }])
  })

  test("reads the setup's numbered items", () => {
    expect(setupItems('## A\n\n1. **Pi**\n2. **Packages:**\n   - sub\n10. **Free**\n')).toEqual([1, 2, 10])
  })

  test('every setup item needs a row', () => {
    expect(problems([row({})], [1, 2], exists, read)).toEqual(['item 2 has no row'])
  })

  test('a test row names a file and a test that exist', () => {
    expect(problems([row({ where: 'b.rs::t' })], [1], exists, read)).toEqual(['item 1: b.rs does not exist'])
    expect(problems([row({ where: 'a.rs::gone' })], [1], exists, read)).toEqual(['item 1: a.rs has no test gone'])
    expect(problems([row({ where: 'a.rs' })], [1], exists, read)).toEqual([])
  })

  test('eval and script rows name a file that exists', () => {
    expect(problems([row({ check: 'eval', where: 'e.ts' })], [1], exists, read)).toEqual([])
    expect(problems([row({ check: 'script', where: 'nope.sh' })], [1], exists, read)).toEqual(['item 1: nope.sh does not exist'])
  })

  test('manual, difference, out-of-scope and n/a rows need a where text', () => {
    for (const check of ['manual', 'difference', 'n/a'] as const) {
      expect(problems([row({ check, where: '' })], [1], exists, read)).toEqual(['item 1: no reason or steps'])
    }
  })

  test("out-of-scope is only for the owner's exclusions", () => {
    expect(problems([row({ item: 6, check: 'out-of-scope', where: 'OpenPencil' })], [6], exists, read)).toEqual([])
    expect(problems([row({ item: 7, check: 'out-of-scope', where: 'no' })], [7], exists, read)).toEqual(['item 7 cannot be out of scope'])
  })

  test('an unknown check kind is a problem', () => {
    expect(problems([row({ check: 'maybe' as Row['check'] })], [1], exists, read)).toEqual(['item 1: unknown check maybe'])
  })

  test('the real matrix is sound', () => {
    const root = join(import.meta.dir, '..', '..')
    const at = (p: string) => join(root, p)
    const rows = parseMatrix(readFileSync(at('docs/parity.md'), 'utf8'))
    const items = setupItems(readFileSync(at('ARIA-SETUP.md'), 'utf8'))
    expect(items.length).toBe(37)
    expect(problems(rows, items, (p) => existsSync(at(p)), (p) => readFileSync(at(p), 'utf8'))).toEqual([])
  })
})
