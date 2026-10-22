// Run: bun test extensions
// A subagent's side (child.ts) of the watch: the lines it streams to the parent's socket carry what the dock's trace needs
// (times, the tool call's id, the model and its usage).
import { afterEach, expect, test } from 'bun:test'
import { mkdtempSync } from 'node:fs'
import { createServer, type Server } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { child } from './child.ts'

let server: Server | null = null
afterEach(() => {
  server?.close()
  server = null
})

function fakePi() {
  const handlers = new Map<string, ((event: any, ctx: any) => any)[]>()
  return {
    handlers,
    on(name: string, fn: any) {
      handlers.set(name, [...(handlers.get(name) ?? []), fn])
    },
    fire(name: string, event: any) {
      for (const fn of handlers.get(name) ?? []) fn(event, { mode: 'json' })
    },
  }
}

test('subagent lines carry ids, times and usage', async () => {
  const path = join(mkdtempSync(join(tmpdir(), 'aria-watch-')), 's.sock')
  const lines: any[] = []
  const connected = new Promise<void>((resolve) => {
    server = createServer((socket) => {
      resolve()
      let buf = ''
      socket.on('data', (d) => {
        buf += d.toString()
        let i: number
        while ((i = buf.indexOf('\n')) >= 0) {
          lines.push(JSON.parse(buf.slice(0, i)))
          buf = buf.slice(i + 1)
        }
      })
    }).listen(path)
  })
  const pi = fakePi()
  child(pi as any, path)
  pi.fire('session_start', {})
  await connected
  pi.fire('before_agent_start', { messageType: 'user', prompt: 'find X' })
  pi.fire('message_start', { message: { role: 'assistant' } })
  const usage = { input: 10, output: 4, cacheRead: 0, cacheWrite: 0, totalTokens: 14, cost: { total: 0.02 } }
  pi.fire('message_end', { message: { role: 'assistant', model: 'm', usage, stopReason: 'stop' } })
  pi.fire('tool_execution_start', { toolCallId: 'c1', toolName: 'bash', args: { command: 'ls' } })
  pi.fire('tool_execution_end', { toolCallId: 'c1', toolName: 'bash', isError: false, result: { content: [{ type: 'text', text: 'out' }] } })
  pi.fire('agent_end', {})
  for (let i = 0; i < 50 && lines.length < 6; i++) await Bun.sleep(10)

  const of = (t: string) => lines.find((l) => l.t === t)
  for (const l of lines) expect(typeof l.at).toBe('number')
  expect(of('prompt').text).toBe('find X')
  expect(of('llm_start')).toBeTruthy()
  expect(of('turn').model).toBe('m')
  expect(of('turn').usage.cost.total).toBe(0.02)
  expect(of('turn').stop).toBe('stop')
  expect(of('tool').id).toBe('c1')
  expect(of('tool').args).toBe('ls')
  expect(of('result').id).toBe('c1')
  expect(of('result').text).toBe('out')
  expect(of('end')).toBeTruthy()
})
