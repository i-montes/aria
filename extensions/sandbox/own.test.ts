// Run: bun test extensions
// Pi's write and edit never reach Aria's own files, whatever the session granted: the sandbox switched off for the
// session included. Drives the extension's tool_call handler with a fake Pi.
import { afterAll, beforeAll, describe, expect, mock, test } from 'bun:test'

// Pi's packages are not installed in the repo: the parts index.ts uses at load, faked.
mock.module('@earendil-works/pi-coding-agent', () => ({ createBashToolDefinition: () => ({ name: 'bash', execute: async () => ({ content: [] }) }) }))
mock.module('typebox', () => ({ Type: { Object: (x: any) => x, String: (x: any) => x, Number: (x: any) => x, Boolean: (x: any) => x, Optional: (x: any) => x } }))
const { default: sandbox } = await import('./index.ts')

const ENV = { PI_CODING_AGENT_DIR: '/srv/aria-home/agent', ARIA_PROJECTS_DIR: '/srv/work', ARIA_SETTINGS: '/srv/aria-home/settings.json' }

describe("Aria's own files under a session without sandbox", () => {
  const saved: Record<string, string | undefined> = {}
  beforeAll(() => {
    for (const [k, v] of Object.entries(ENV)) {
      saved[k] = process.env[k]
      process.env[k] = v
    }
  })
  afterAll(() => {
    for (const [k, v] of Object.entries(saved)) {
      if (v === undefined) delete process.env[k]
      else process.env[k] = v
    }
  })

  test('aria.json, settings.json and the attachments are refused, never asked', async () => {
    const handlers: Record<string, (event: any, ctx: any) => any> = {}
    const pi: any = { registerTool() {}, registerCommand() {}, on: (name: string, fn: any) => (handlers[name] = fn) }
    sandbox(pi)
    const asked: string[] = []
    const ctx = {
      mode: 'rpc',
      hasUI: true,
      cwd: '/srv/work',
      ui: { select: async (title: string, options: string[]) => (asked.push(title), options.find((o) => o.startsWith('Yes, no sandbox'))), setStatus() {} },
    }
    // The person turns the sandbox off for the session.
    expect(await handlers.tool_call({ toolName: 'write', input: { path: '/srv/elsewhere/x.txt' } }, ctx)).toBeUndefined()
    expect(asked.length).toBe(1)
    for (const path of ['/srv/aria-home/agent/aria.json', '/srv/aria-home/settings.json', '/srv/aria-home/attachments/s1/2026-10-21/1-a.png']) {
      for (const toolName of ['write', 'edit']) {
        const got = await handlers.tool_call({ toolName, input: { path } }, ctx)
        expect(got?.block).toBe(true)
        expect(got.reason).toContain('part of Aria itself')
      }
    }
    expect(asked.length).toBe(1)
    // Reading them stays possible (the model reads its attachments).
    expect(await handlers.tool_call({ toolName: 'read', input: { path: '/srv/aria-home/attachments/s1/2026-10-21/1-a.png' } }, ctx)).toBeUndefined()
  })
})
