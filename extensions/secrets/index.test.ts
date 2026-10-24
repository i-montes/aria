// Run: bun test extensions
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { jobsClient } from '../background/client.ts'
import { fakeDock, socketPath, type FakeDock } from '../background/fake-dock.test.ts'
import secrets, { savedName, secretCard } from './index.ts'

describe('the secret card', () => {
  test('names the secret and its variable, never a value', () => {
    expect(JSON.parse(secretCard('github-token', 'the CI'))).toEqual({
      aria: 1,
      kind: 'secret',
      name: 'github-token',
      purpose: 'the CI',
      env: 'ARIA_SECRET_GITHUB_TOKEN',
    })
  })

  test("only the page's saved answer counts", () => {
    expect(savedName('saved:github-token')).toBe('github-token')
    expect(savedName('ghp_0123456789abcdefghij')).toBeNull()
    expect(savedName(undefined)).toBeNull()
  })
})

describe('every tool output is masked', () => {
  const handlers: Record<string, Function> = {}
  secrets({ registerTool() {}, on: (name: string, fn: Function) => (handlers[name] = fn) } as any)

  test('a key in a file read reaches the model masked, with the note', () => {
    const out = handlers.tool_result({ input: { path: '.env' }, content: [{ type: 'text', text: 'OPENAI_API_KEY=sk-proj-0123456789abcdefXYZ' }] })
    expect(out.content[0].text).not.toContain('sk-proj')
    expect(out.content[0].text).toContain('secret_import')
    expect(out.content[0].text).toContain('`.env`')
  })

  test('an output with nothing to mask is left alone', () => {
    expect(handlers.tool_result({ content: [{ type: 'text', text: 'PORT=3000' }] })).toBeUndefined()
  })
})

describe('secret_import', () => {
  let dock: FakeDock | null = null
  const saved = process.env.ARIA_JOBS_SOCK
  beforeEach(() => {
    delete process.env.ARIA_JOBS_SOCK
  })
  afterEach(async () => {
    jobsClient()?.close()
    await dock?.stop()
    dock = null
    if (saved === undefined) delete process.env.ARIA_JOBS_SOCK
    else process.env.ARIA_JOBS_SOCK = saved
  })

  const REFUSED = "Aria keeps a secret only from a file inside this session's project or the projects folder, and never from Aria's own folder or a hidden folder of your home folder."

  test("says which files work, sends the working folder, and hands the dock's one-sentence refusal to the model", async () => {
    dock = await fakeDock(socketPath(), (req) => (req.op === 'secret_import' ? { ok: false, error: REFUSED } : undefined))
    process.env.ARIA_JOBS_SOCK = dock.path
    const tools = new Map<string, any>()
    secrets({ registerTool: (tool: any) => tools.set(tool.name, tool), on() {} } as any)
    const tool = tools.get('secret_import')
    expect(tool.description).toContain('inside this project or the projects folder')
    const out = await tool.execute('call-1', { path: '/home/ana/.aws/credentials', name: 'aws' }, undefined, undefined, { mode: 'rpc', cwd: '/home/ana/Projects/acme/checkout-widget' })
    expect(out.isError).toBe(true)
    expect(out.content[0].text).toBe(REFUSED)
    expect(await dock.waitFor((l) => l.op === 'secret_import')).toMatchObject({ path: '/home/ana/.aws/credentials', cwd: '/home/ana/Projects/acme/checkout-widget' })
  })
})
