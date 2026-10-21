// Run: bun test extensions
// The background extension with a fake `pi` and a fake dock: the tools, the hidden line, and what Aria is told.
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { jobsClient } from './client.ts'
import { fakeDock, job, socketPath, type FakeDock } from './fake-dock.test.ts'
import background from './index.ts'

const DOCK_REFUSAL = "Background work needs Aria's dock."
const SUBAGENT_REFUSAL = 'Background work is not available inside a subagent.'

function fakePi() {
  const pi = {
    tools: new Map<string, any>(),
    handlers: new Map<string, ((event: any, ctx: any) => any)[]>(),
    messages: [] as { message: any; options: any; acksBefore: number }[],
    dock: null as FakeDock | null,
    registerTool(tool: any) {
      pi.tools.set(tool.name, tool)
    },
    on(name: string, fn: any) {
      pi.handlers.set(name, [...(pi.handlers.get(name) ?? []), fn])
    },
    sendMessage(message: any, options: any) {
      pi.messages.push({ message, options, acksBefore: pi.dock?.lines.filter((l) => l.op === 'ack').length ?? 0 })
    },
    fire: async (name: string, event: any, ctx: any) => {
      let out: any
      for (const fn of pi.handlers.get(name) ?? []) out = (await fn(event, ctx)) ?? out
      return out
    },
    run: (name: string, params: any, ctx: any) => pi.tools.get(name).execute('call-1', params, undefined, undefined, ctx),
  }
  return pi
}

const cwd = mkdtempSync(join(tmpdir(), 'aria-projects-'))
const rpc = { mode: 'rpc', cwd, hasUI: true }
const child = { mode: 'json', cwd, hasUI: false }
const text = (out: any) => out.content.map((c: any) => c.text).join('\n')

let dock: FakeDock | null = null
const saved = { sock: process.env.ARIA_JOBS_SOCK, sandbox: process.env.PI_SANDBOX }
beforeEach(() => {
  delete process.env.ARIA_JOBS_SOCK
  delete process.env.PI_SANDBOX
})
afterEach(async () => {
  jobsClient()?.close()
  await dock?.stop()
  dock = null
  if (saved.sock === undefined) delete process.env.ARIA_JOBS_SOCK
  else process.env.ARIA_JOBS_SOCK = saved.sock
  if (saved.sandbox === undefined) delete process.env.PI_SANDBOX
  else process.env.PI_SANDBOX = saved.sandbox
})

async function withDock(handle?: (req: any, dock: FakeDock) => any) {
  dock = await fakeDock(socketPath(), handle)
  process.env.ARIA_JOBS_SOCK = dock.path
  const pi = fakePi()
  pi.dock = dock
  background(pi as any)
  await pi.fire('session_start', { type: 'session_start' }, rpc)
  return pi
}

describe('the background extension', () => {
  test('background_agent sends the resolved folder and refuses one outside the projects folder', async () => {
    const pi = await withDock((req) => (req.op === 'start' ? { ok: true, job: job({ kind: 'agent', title: req.title, state: 'queued', number: 4, id: 'j4' }) } : undefined))
    const out = await pi.run('background_agent', { task: 'Fix the failing tests in shop.', title: 'Fix the tests', folder: 'shop', writes: true }, rpc)
    expect(out.isError).toBeFalsy()
    const start = await dock!.waitFor((l) => l.op === 'start')
    expect(start).toMatchObject({ kind: 'agent', task: 'Fix the failing tests in shop.', title: 'Fix the tests', folder: join(cwd, 'shop'), writes: true })
    expect(text(out)).toContain('#4')
    expect(text(out)).toContain('queued')
    expect(text(out)).toContain('told')

    const starts = dock!.lines.filter((l) => l.op === 'start').length
    const outside = await pi.run('background_agent', { task: 'x', title: 'x', folder: '/usr/share' }, rpc)
    expect(outside.isError).toBe(true)
    expect(text(outside)).toContain('outside')
    await Bun.sleep(30)
    expect(dock!.lines.filter((l) => l.op === 'start').length).toBe(starts)
  })

  test('without ARIA_JOBS_SOCK every tool refuses with "Background work needs Aria\'s dock."', async () => {
    process.env.PI_SANDBOX = '0'
    const pi = fakePi()
    background(pi as any)
    await pi.fire('session_start', { type: 'session_start' }, rpc)
    const params: Record<string, any> = {
      background_agent: { task: 't', title: 't' },
      background_list: {},
      background_output: { job: 'j1' },
      background_stop: { job: 'j1' },
      background_run: { command: 'bun dev', title: 'Dev' },
    }
    expect([...pi.tools.keys()].sort()).toEqual(Object.keys(params).sort())
    for (const [name, p] of Object.entries(params)) {
      const out = await pi.run(name, p, rpc)
      expect(out.isError).toBe(true)
      expect(text(out)).toBe(DOCK_REFUSAL)
    }
    expect(await pi.fire('before_agent_start', { type: 'before_agent_start', messageType: 'user', prompt: 'hi' }, rpc)).toBeUndefined()
  })

  test('in a subagent (mode json) the tools refuse and no context line is added', async () => {
    dock = await fakeDock()
    dock.jobs = [job()]
    process.env.ARIA_JOBS_SOCK = dock.path
    const pi = fakePi()
    background(pi as any)
    await pi.fire('session_start', { type: 'session_start' }, child)
    for (const name of pi.tools.keys()) {
      const out = await pi.run(name, { task: 't', title: 't', job: 'j1' }, child)
      expect(out.isError).toBe(true)
      expect(text(out)).toBe(SUBAGENT_REFUSAL)
    }
    expect(await pi.fire('before_agent_start', { type: 'before_agent_start', messageType: 'user', prompt: 'hi' }, child)).toBeUndefined()
    await Bun.sleep(50)
    expect(dock.lines).toEqual([])
  })

  test('a user message gets the hidden [background] line when something runs', async () => {
    const pi = await withDock()
    dock!.jobs = [job()]
    const out = await pi.fire('before_agent_start', { type: 'before_agent_start', messageType: 'user', prompt: 'hi' }, rpc)
    expect(out.message).toMatchObject({ customType: 'aria-background-now', display: false })
    expect(out.message.content).toStartWith('[background] running: #3 process "Dev server"')
    dock!.jobs = []
    expect(await pi.fire('before_agent_start', { type: 'before_agent_start', messageType: 'user', prompt: 'hi' }, rpc)).toBeUndefined()
  })

  test('a dock that does not answer in time adds no line', async () => {
    const pi = await withDock((req) => (req.op === 'list' ? null : undefined))
    const started = Date.now()
    expect(await pi.fire('before_agent_start', { type: 'before_agent_start', messageType: 'user', prompt: 'hi' }, rpc)).toBeUndefined()
    expect(Date.now() - started).toBeLessThan(600)
  })

  test('acks_only_after_send_message', async () => {
    const pi = await withDock()
    await dock!.waitFor((l) => l.op === 'hello')
    await Bun.sleep(20)
    dock!.push({ event: 'ended', seq: 7, by: 'self', job: job({ state: 'exited', exit: 1 }), lastLines: ['boom'] })
    await dock!.waitFor((l) => l.op === 'ack' && l.event === 7)
    expect(pi.messages.length).toBe(1)
    expect(pi.messages[0].acksBefore).toBe(0)
    expect(pi.messages[0].message).toMatchObject({ customType: 'aria-background', display: true, details: { by: 'self', event: 'ended' } })
    expect(pi.messages[0].message.content).toContain('boom')
    expect(pi.messages[0].options).toEqual({ triggerTurn: true, deliverAs: 'followUp' })

    dock!.push({ event: 'ended', seq: 8, by: 'stop', job: job({ state: 'stopped' }), lastLines: [] })
    await dock!.waitFor((l) => l.op === 'ack' && l.event === 8)
    expect(pi.messages[1].options).toBeUndefined()
  })

  test('an event given again (the ack was lost) is acked but not told twice', async () => {
    const pi = await withDock()
    await dock!.waitFor((l) => l.op === 'hello')
    await Bun.sleep(20)
    const event = { event: 'ended', seq: 7, by: 'self', job: job({ state: 'exited', exit: 1, endedAt: 5 }), lastLines: [] }
    dock!.push(event)
    await dock!.waitFor((l) => l.op === 'ack' && l.event === 7)
    dock!.lines.length = 0
    dock!.push(event)
    await dock!.waitFor((l) => l.op === 'ack' && l.event === 7)
    expect(pi.messages.length).toBe(1)
  })

  test('a sendMessage that throws is not acked and does not break the extension', async () => {
    const pi = await withDock()
    pi.sendMessage = () => {
      throw new Error('no session')
    }
    await dock!.waitFor((l) => l.op === 'hello')
    await Bun.sleep(20)
    dock!.push({ event: 'ended', seq: 3, by: 'self', job: job({ state: 'exited' }), lastLines: [] })
    await Bun.sleep(100)
    expect(dock!.lines.some((l) => l.op === 'ack')).toBe(false)
  })

  test('background_run exists only with PI_SANDBOX=0', async () => {
    const boxed = fakePi()
    background(boxed as any)
    expect(boxed.tools.has('background_run')).toBe(false)
    expect([...boxed.tools.keys()].sort()).toEqual(['background_agent', 'background_list', 'background_output', 'background_stop'])

    process.env.PI_SANDBOX = '0'
    const pi = await withDock((req) => (req.op === 'start' ? { ok: true, job: job({ title: req.title }) } : req.op === 'output' ? { ok: true, lines: ['ready on http://localhost:3000'] } : undefined))
    const out = await pi.run('background_run', { command: 'bun dev', title: 'Dev server' }, rpc)
    const start = await dock!.waitFor((l) => l.op === 'start')
    expect(start).toMatchObject({ kind: 'process', command: 'bun dev', line: 'bun dev', title: 'Dev server', folder: cwd })
    expect(text(out)).toContain('Started in the background as #3 (Dev server)')
    expect(text(out)).toContain('URL: http://localhost:3000')
  })

  test('the descriptions carry the rules for deciding', () => {
    process.env.PI_SANDBOX = '0'
    const pi = fakePi()
    background(pi as any)
    const agent = pi.tools.get('background_agent').description
    expect(agent).toContain('does not see this conversation')
    expect(agent).toContain('writes: false')
    expect(agent).toContain('worktree')
    expect(agent).toContain('heavy')
    expect(agent).toContain('6 words')
    expect(pi.tools.get('background_run').description).toContain('does not end by itself')
  })

  test('background_stop and background_output take #n and say when a job had already ended', async () => {
    const pi = await withDock((req) => {
      if (req.op === 'list') return { ok: true, jobs: [job()] }
      if (req.op === 'stop') return { ok: true, already: true }
      if (req.op === 'output') return { ok: true, lines: [`lines ${req.lines}`] }
    })
    expect(text(await pi.run('background_stop', { job: '#3' }, rpc))).toContain('already ended')
    expect(await dock!.waitFor((l) => l.op === 'stop')).toMatchObject({ job: 'j3' })
    expect(text(await pi.run('background_output', { job: 'j3', lines: 900 }, rpc))).toContain('lines 400')
  })
  test('background_run refuses a folder outside the projects folder', async () => {
    process.env.PI_SANDBOX = '0'
    const pi = await withDock()
    const out = await pi.run('background_run', { command: 'bun dev', title: 'Dev', folder: '/usr/share' }, rpc)
    expect(out.isError).toBe(true)
    expect(text(out)).toContain('outside')
    await Bun.sleep(30)
    expect(dock!.lines.some((l) => l.op === 'start')).toBe(false)
  })

  test('background_run waits at most about 3 s even when the dock stops answering', async () => {
    process.env.PI_SANDBOX = '0'
    const pi = await withDock((req) => (req.op === 'start' ? { ok: true, job: job() } : req.op === 'hello' ? undefined : null))
    const started = Date.now()
    const out = await pi.run('background_run', { command: 'bun dev', title: 'Dev server' }, rpc)
    expect(Date.now() - started).toBeLessThan(3300)
    expect(text(out)).toContain('Started in the background as #3')
  }, 8000)

  test('a second instance without a shutdown (a reload) takes over the events: an end is told once', async () => {
    const first = await withDock()
    const second = fakePi()
    second.dock = dock
    background(second as any)
    await second.fire('session_start', { type: 'session_start' }, rpc)
    await dock!.waitFor((l) => l.op === 'hello')
    await Bun.sleep(20)
    dock!.push({ event: 'ended', seq: 11, by: 'self', job: job({ state: 'exited', endedAt: 11 }), lastLines: [] })
    await dock!.waitFor((l) => l.op === 'ack' && l.event === 11)
    await Bun.sleep(30)
    expect(first.messages.length + second.messages.length).toBe(1)
    expect(second.messages.length).toBe(1)
    expect(dock!.lines.filter((l) => l.op === 'ack' && l.event === 11).length).toBe(1)
  })

  test('when the dock socket changes, the next message listens on the new client', async () => {
    const pi = await withDock()
    const other = await fakeDock(socketPath())
    try {
      process.env.ARIA_JOBS_SOCK = other.path
      pi.dock = other
      await pi.fire('before_agent_start', { type: 'before_agent_start', messageType: 'user', prompt: 'hi' }, rpc)
      await other.waitFor((l) => l.op === 'hello')
      await Bun.sleep(20)
      other.push({ event: 'ended', seq: 1, by: 'stop', job: job({ id: 'j8', number: 8, state: 'stopped', endedAt: 1 }), lastLines: [] })
      await other.waitFor((l) => l.op === 'ack' && l.event === 1)
      expect(pi.messages.map((m) => m.message.details.job.id)).toEqual(['j8'])
    } finally {
      jobsClient()?.close()
      await other.stop()
    }
  })
})
