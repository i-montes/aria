import { describe, expect, test } from 'bun:test'
import { checksFor, runCheck } from './checks.ts'

describe('the checks of a project', () => {
  test('checks are found per project', () => {
    const node = { packageJson: JSON.stringify({ scripts: { lint: 'eslint .', typecheck: 'tsc', test: 'vitest', build: 'x' } }), bunLock: true }
    expect(checksFor('linux', node).map((c) => c.command)).toEqual(['bun run lint', 'bun run typecheck', 'bun run test'])
    expect(checksFor('linux', { ...node, bunLock: false }).map((c) => c.command)).toEqual(['npm run lint', 'npm run typecheck', 'npm run test'])
    expect(checksFor('linux', { cargoToml: true }).map((c) => c.command)).toEqual(['cargo clippy -q', 'cargo test -q'])
    expect(checksFor('linux', { pyproject: true }).map((c) => c.command)).toEqual(['ruff check', 'pytest -q'])
    expect(checksFor('linux', {})).toEqual([])
    const own = { verifyJson: JSON.stringify({ lint: 'make lint', test: 'make test' }), cargoToml: true }
    expect(checksFor('linux', own).map((c) => c.command)).toEqual(['make lint', 'make test'])
  })

  test('mac runs no checks: there is no box there, and nobody would be asked', () => {
    expect(checksFor('darwin', { cargoToml: true })).toEqual([])
    expect(checksFor('darwin', { verifyJson: JSON.stringify({ test: 'make test' }) })).toEqual([])
  })

  test('a check that runs too long is stopped', async () => {
    const started = Date.now()
    const got = await runCheck('sleep 5', '/tmp', 200)
    expect(got.ok).toBe(false)
    expect(got.timedOut).toBe(true)
    expect(Date.now() - started).toBeLessThan(2_000)
  })

  test('a check reports its output and outcome', async () => {
    const fine = await runCheck('echo 43 pass', '/tmp', 5_000)
    expect(fine).toMatchObject({ ok: true, timedOut: false })
    expect(fine.output).toContain('43 pass')
    const bad = await runCheck('echo broken; exit 1', '/tmp', 5_000)
    expect(bad.ok).toBe(false)
    expect(bad.output).toContain('broken')
  })
})

describe('checks that leave children behind', () => {
  test('a check whose child holds its output still ends at the limit', async () => {
    const started = Date.now()
    const got = await runCheck('setsid sleep 30 & sleep 60', '/tmp', 500)
    expect(got.timedOut).toBe(true)
    expect(Date.now() - started).toBeLessThan(3_000)
  })

  test("npm's placeholder test is not a check", () => {
    const pkg = { packageJson: JSON.stringify({ scripts: { test: 'echo "Error: no test specified" && exit 1', lint: 'eslint .' } }) }
    expect(checksFor('linux', pkg).map((c) => c.command)).toEqual(['npm run lint'])
  })
})
