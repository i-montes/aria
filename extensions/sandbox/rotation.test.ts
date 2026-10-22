// Run: bun test extensions
// Aria's rotation reloads the extensions with a new session: the sandbox's session grants are kept across it, and
// only across it. Drives the extension's handlers with a fake Pi.
import { afterAll, beforeAll, describe, expect, mock, test } from 'bun:test'
import { markRotating } from '../lib/rotation.ts'

// Pi's packages are not installed in the repo: the parts index.ts uses at load, faked.
mock.module('@earendil-works/pi-coding-agent', () => ({ createBashToolDefinition: () => ({ name: 'bash', execute: async () => ({ content: [] }) }) }))
mock.module('typebox', () => ({ Type: { Object: (x: any) => x, String: (x: any) => x, Number: (x: any) => x, Boolean: (x: any) => x, Optional: (x: any) => x } }))
const { default: sandbox } = await import('./index.ts')

const ENV = { PI_CODING_AGENT_DIR: '/srv/aria-home/agent', ARIA_PROJECTS_DIR: '/srv/work', ARIA_SETTINGS: '/srv/aria-home/settings.json' }
const OUTSIDE = { toolName: 'write', input: { path: '/srv/elsewhere/notes.txt' } }

/** A fresh instance of the extension, as Pi loads one for each session. */
function instance() {
  const handlers: Record<string, (event: any, ctx: any) => any> = {}
  sandbox({ registerTool() {}, registerCommand() {}, on: (name: string, fn: any) => (handlers[name] = fn) } as any)
  return handlers
}

/** A dock that allows writing in the folder for the session, and counts its questions. */
function dock() {
  const asked: string[] = []
  const ctx = {
    mode: 'rpc',
    hasUI: true,
    cwd: '/srv/work',
    ui: { select: async (title: string, options: string[]) => (asked.push(title), options.find((o) => o.includes('this session'))), setStatus() {} },
  }
  return { asked, ctx }
}

describe('the session grants and a rotation', () => {
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

  async function grantedThenNewSession(rotating: boolean) {
    const before = instance()
    const first = dock()
    await before.session_start({ type: 'session_start', reason: 'startup' }, first.ctx)
    expect(await before.tool_call(OUTSIDE, first.ctx)).toBeUndefined()
    expect(first.asked).toHaveLength(1)
    markRotating(rotating)
    try {
      await before.session_shutdown({ type: 'session_shutdown', reason: 'new' }, first.ctx)
      const after = instance()
      const next = dock()
      await after.session_start({ type: 'session_start', reason: 'new' }, next.ctx)
      expect(await after.tool_call(OUTSIDE, next.ctx)).toBeUndefined()
      return next.asked.length
    } finally {
      markRotating(false)
    }
  }

  test('are kept by the reloaded extension after a rotation', async () => {
    expect(await grantedThenNewSession(true)).toBe(0)
  })

  test('start over at any other new session', async () => {
    expect(await grantedThenNewSession(false)).toBe(1)
  })
})
