// Run: bun test extensions/sandbox/browser.test.ts
// The verify-in-browser skill's shape works inside the sandbox: one boxed bash call opens a page, reads it and closes
// the browser, leaving no process behind.
import { describe, expect, test } from 'bun:test'
import { execFileSync, spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { shellQuote, wrapCommand } from './policy.ts'

const ready = existsSync('/usr/bin/bwrap') && !!Bun.which('agent-browser')

const leftovers = (pattern: string) => {
  try {
    return execFileSync('pgrep', ['-f', pattern], { encoding: 'utf8' }).trim()
  } catch {
    return '' // pgrep exits 1 when nothing matches
  }
}

describe.skipIf(!ready)('checking a page in the browser from the sandbox', () => {
  test('one boxed call opens the page, reads it and closes the browser', () => {
    const project = mkdtempSync('/tmp/sbx-browser-')
    mkdirSync(join(project, '.git'))
    const page = join(project, 'index.html')
    writeFileSync(page, '<!doctype html><title>Check</title><h1>Hello</h1><button>Copy</button>')
    const session = `check-test-${process.pid}-${Date.now()}`
    const s = shellQuote(session)
    const command = [
      `agent-browser --session ${s} batch --bail`,
      shellQuote(`open file://${page}`),
      shellQuote('set viewport 390 844'),
      shellQuote('snapshot -i'),
      shellQuote('eval document.documentElement.scrollWidth'),
      `; st=$?; agent-browser --session ${s} close >/dev/null; exit $st`,
    ].join(' ')
    try {
      const run = spawnSync('bash', ['-c', wrapCommand(command, project)], { encoding: 'utf8', timeout: 90_000 })
      const output = `${run.stdout}${run.stderr}`
      expect(run.status, output).toBe(0)
      expect(output).toContain('button "Copy"')
      expect(output.split('\n').map((line) => line.trim())).toContain('390')
      expect(leftovers(session)).toBe('')
    } finally {
      spawnSync('agent-browser', ['--session', session, 'close'], { stdio: 'ignore', timeout: 30_000 })
      rmSync(project, { recursive: true, force: true })
    }
  }, 120_000)
})
