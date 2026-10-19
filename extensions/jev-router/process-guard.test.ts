// Run: bun test ~/.pi/agent/extensions/jev-router/process-guard.test.ts
import { describe, expect, test } from 'bun:test'
import { checkKill, killTargets, type ProcessInfo } from './process-guard.ts'

// A fake process table: pid → [parent, cwd].
const table: Record<number, [number, string]> = {
  100: [1, '/home/u'], // the Pi process (root)
  200: [100, '/home/u/Projects/widget'], // started by the session
  201: [200, '/tmp'], // grandchild of the session, outside the project
  300: [1, '/home/u/Projects/widget'], // detached dev server of this project
  400: [1, '/home/u/Projects/other-app'], // another project's dev server
}
const info: ProcessInfo = {
  ppid: (pid) => table[pid]?.[0] ?? null,
  cwd: (pid) => table[pid]?.[1] ?? null,
}
const check = (cmd: string) => checkKill(cmd, 100, '/home/u/Projects/widget', info)

describe('commands that pick victims by pattern are refused', () => {
  for (const cmd of [
    'kill $(pgrep -f "vite --port 5173" | head -5)',
    'pkill -f vite',
    'killall node',
    'fuser -k -n tcp 5174',
    'fuser -n tcp -k 5174',
    'pgrep -f vite | xargs kill',
    'kill `pidof node`',
    'cd x && pkill vite; echo ok',
  ]) {
    test(cmd, () => expect(check(cmd).block).toBe(true))
  }
})

describe('kill with literal PIDs', () => {
  test("the session's own child", () => expect(check('kill 200').block).toBe(false))
  test("the session's grandchild, even outside the project", () => expect(check('kill -9 201').block).toBe(false))
  test('a detached process running in the project', () => expect(check('kill 300').block).toBe(false))
  test("another project's process is refused and named", () => {
    const v = check('kill 400')
    expect(v.block).toBe(true)
    expect(v.reason).toContain('other-app')
  })
  test('one foreign PID among several is enough to refuse', () => expect(check('kill 200 400').block).toBe(true))
  test('a PID that no longer exists is harmless', () => expect(check('kill 999').block).toBe(false))
  test('kill of a variable is refused', () => expect(check('kill $PID').block).toBe(true))
  test('a shell job is ours', () => expect(check('kill %1').block).toBe(false))
})

describe('commands without kill pass', () => {
  for (const cmd of ['ss -ltnp "sport = :5173"', 'pgrep -af vite', 'bun run dev --port 5173', 'echo "no pkill here"']) {
    test(cmd, () => expect(check(cmd).block).toBe(cmd.includes('pkill')))
  }
})

test('killTargets reads each segment', () => {
  expect(killTargets('kill 1 2; ls && kill -TERM 3')).toEqual([
    { pids: [1, 2], nonLiteral: false },
    { pids: [3], nonLiteral: false },
  ])
})
