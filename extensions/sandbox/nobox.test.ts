// Run: bun test extensions
// A host where the box does not work (no bwrap, or one the system blocks) behaves as a Mac does: every command asks first,
// and the card says plainly that there is no sandbox. Where the box works nothing changes. Drives the extension's bash tool
// with a fake Pi.
import { describe, expect, mock, test } from 'bun:test'

// Pi's packages are not installed in the repo: the parts index.ts uses at load, faked (as own.test.ts does).
mock.module('@earendil-works/pi-coding-agent', () => ({ createBashToolDefinition: () => ({ name: 'bash', execute: async () => ({ content: [{ type: 'text', text: 'ran' }] }) }) }))
mock.module('typebox', () => ({ Type: { Object: (x: any) => x, String: (x: any) => x, Number: (x: any) => x, Boolean: (x: any) => x, Optional: (x: any) => x } }))
const { default: sandbox } = await import('./index.ts')

function bashTool(boxWorks: boolean) {
  const tools: Record<string, any> = {}
  const pi: any = { registerTool: (t: any) => (tools[t.name] = t), registerCommand() {}, on() {} }
  sandbox(pi, boxWorks)
  return tools.bash
}

function session(answer: string) {
  const asked: string[] = []
  const ctx = {
    mode: 'rpc',
    hasUI: true,
    cwd: process.cwd(),
    ui: { select: async (title: string, options: string[]) => (asked.push(title), options.find((o) => o.startsWith(answer))), setStatus() {}, notify() {} },
  }
  return { asked, ctx }
}

describe('no box on this machine', () => {
  test('a command asks first, the card says there is no sandbox, and a no stops it', async () => {
    const { asked, ctx } = session('No')
    const result = await bashTool(false).execute('1', { command: 'ls', reason: 'List files' }, undefined, undefined, ctx)
    expect(asked.length).toBe(1)
    expect(JSON.parse(asked[0]).detail).toMatch(/No sandbox is available on this machine/)
    expect(result.isError).toBe(true)
    expect(result.content[0].text).toBe('The user said no: the command did not run.')
  })
  test('a yes runs it, outside any box', async () => {
    const { asked, ctx } = session('Yes, just this once')
    const result = await bashTool(false).execute('1', { command: 'ls', reason: 'List files' }, undefined, undefined, ctx)
    expect(asked.length).toBe(1)
    expect(result.content[0].text).toBe('ran')
  })
  test('with nobody to ask it is refused, not run', async () => {
    const result = await bashTool(false).execute('1', { command: 'ls' }, undefined, undefined, { mode: 'print', hasUI: false, cwd: process.cwd() })
    expect(result.isError).toBe(true)
    expect(result.content[0].text).toMatch(/no sandbox on this system/)
  })
})

describe('a box that works', () => {
  test('a command runs in it without asking, and the card of a leave has no such note', async () => {
    const { asked, ctx } = session('No')
    const result = await bashTool(true).execute('1', { command: 'ls' }, undefined, undefined, ctx)
    expect(asked.length).toBe(0)
    expect(result.content[0].text).toBe('ran')
    await bashTool(true).execute('2', { command: 'ls', outsideSandbox: true, reason: 'x' }, undefined, undefined, ctx)
    expect(asked.length).toBe(1)
    expect(JSON.parse(asked[0]).detail).not.toMatch(/No sandbox is available/)
  })
})
