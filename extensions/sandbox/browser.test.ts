// Run: bun test extensions/sandbox/browser.test.ts
// The verify-in-browser skill's shape works inside the sandbox: one boxed bash call opens a page, reads it and closes
// the browser, leaving no process behind.
import { describe, expect, test } from 'bun:test'
import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { shellQuote, wrapCommand } from './policy.ts'

const ready = existsSync('/usr/bin/bwrap') && !!Bun.which('agent-browser')

// agent-browser's daemon learns its session from the environment (its command line is only the binary), and Chrome
// inherits it, so a leftover is a process whose environment names the session.
const leftovers = (session: string) => {
  const marker = `AGENT_BROWSER_SESSION=${session}`
  return readdirSync('/proc').filter((pid) => {
    if (!/^\d+$/.test(pid)) return false
    try {
      return readFileSync(`/proc/${pid}/environ`, 'utf8').split('\0').includes(marker)
    } catch {
      return false // not ours to read, or already gone
    }
  })
}

// The daemon and Chrome take a moment to exit after `close`.
const settle = (session: string, ms = 3000) => {
  const until = Date.now() + ms
  let left = leftovers(session)
  while (left.length && Date.now() < until) {
    Bun.sleepSync(100)
    left = leftovers(session)
  }
  return left
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
    let closed = false
    try {
      const run = spawnSync('bash', ['-c', wrapCommand(command, project)], { encoding: 'utf8', timeout: 90_000 })
      const output = `${run.stdout}${run.stderr}`
      closed = run.status === 0
      expect(run.status, output).toBe(0)
      expect(output).toContain('button "Copy"')
      expect(output.split('\n').map((line) => line.trim())).toContain('390')
      expect(settle(session)).toEqual([])
    } finally {
      // Only after a failed run: a host-side close of a closed session starts a daemon of its own.
      if (!closed) spawnSync('agent-browser', ['--session', session, 'close'], { stdio: 'ignore', timeout: 30_000 })
      rmSync(project, { recursive: true, force: true })
    }
  }, 120_000)
})
