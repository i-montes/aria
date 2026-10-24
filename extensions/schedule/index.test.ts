// Run: bun test extensions
// The schedule extension with a fake `pi` and a fake dock: its four tools and their refusals.
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { jobsClient } from '../background/client.ts'
import { fakeDock, socketPath, type FakeDock } from '../background/fake-dock.test.ts'
import schedule from './index.ts'

function fakePi() {
  const pi = {
    tools: new Map<string, any>(),
    registerTool(tool: any) {
      pi.tools.set(tool.name, tool)
    },
    on() {},
    run: (name: string, params: any, ctx: any) => pi.tools.get(name).execute('call-1', params, undefined, undefined, ctx),
  }
  return pi
}

const rpc = { mode: 'rpc', cwd: '/tmp', hasUI: true }
const child = { mode: 'json', cwd: '/tmp', hasUI: false }
const text = (out: any) => out.content.map((c: any) => c.text).join('\n')
const TASK = {
  id: 's-abcdefghijkl',
  title: 'Notion tasks',
  prompt: 'My tasks',
  when: { at: '07:30', days: ['mon', 'tue', 'wed', 'thu', 'fri'], date: null },
  grants: { connectors: { notion: 'read' }, writable: [] },
  enabled: true,
  next: Date.UTC(2026, 9, 22, 12, 30),
  lastResult: { at: 1, ok: true, summary: '3 tasks for today' },
}

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

async function withDock(handle?: (req: any) => any) {
  dock = await fakeDock(socketPath(), handle)
  process.env.ARIA_JOBS_SOCK = dock.path
  const pi = fakePi()
  schedule(pi as any)
  return pi
}

describe('the schedule extension', () => {
  test('create sends the fields and says when it runs', async () => {
    const pi = await withDock((req) => (req.op === 'schedule_create' ? { ok: true, task: TASK } : undefined))
    const out = await pi.run('schedule_create', { title: 'Notion tasks', prompt: 'My tasks', at: '07:30', days: ['mon', 'tue', 'wed', 'thu', 'fri'], connectors: { notion: 'read' } }, rpc)
    expect(out.isError).toBeFalsy()
    const sent = await dock!.waitFor((l) => l.op === 'schedule_create')
    expect(sent).toMatchObject({ title: 'Notion tasks', prompt: 'My tasks', at: '07:30', days: ['mon', 'tue', 'wed', 'thu', 'fri'], connectors: { notion: 'read' } })
    expect(text(out)).toContain('s-abcdefghijkl')
    expect(text(out)).toContain('07:30')
  })

  test('a task that writes waits for the person, and the model is told to say so', async () => {
    const pi = await withDock((req) => (req.op === 'schedule_create' ? { ok: true, task: { ...TASK, enabled: false, awaiting: true, grants: { connectors: { notion: 'allow' }, writable: [] } } } : undefined))
    const out = text(await pi.run('schedule_create', { title: 'Notion tasks', prompt: 'p', at: '07:30', connectors: { notion: 'allow' } }, rpc))
    expect(out).toContain('approve')
    expect(out).toContain('/schedules')
  })

  test('list shows every task with its next run and last result', async () => {
    const pi = await withDock((req) => (req.op === 'schedule_list' ? { ok: true, tasks: [TASK] } : undefined))
    const out = text(await pi.run('schedule_list', {}, rpc))
    expect(out).toContain('Notion tasks')
    expect(out).toContain('mon, tue, wed, thu, fri at 07:30')
    expect(out).toContain('notion: read')
    expect(out).toContain('3 tasks for today')
  })

  test('update and delete name the task', async () => {
    const pi = await withDock((req) => (req.op === 'schedule_update' ? { ok: true, task: { ...TASK, enabled: false } } : req.op === 'schedule_delete' ? { ok: true } : undefined))
    await pi.run('schedule_update', { task: 's-abcdefghijkl', enabled: false }, rpc)
    expect(await dock!.waitFor((l) => l.op === 'schedule_update')).toMatchObject({ task: 's-abcdefghijkl', enabled: false })
    await pi.run('schedule_delete', { task: 's-abcdefghijkl' }, rpc)
    expect(await dock!.waitFor((l) => l.op === 'schedule_delete')).toMatchObject({ task: 's-abcdefghijkl' })
  })

  test("the dock's refusal reaches the model", async () => {
    const pi = await withDock((req) => (req.op === 'schedule_create' ? { ok: false, error: 'there is no connector "gmail"' } : undefined))
    const out = await pi.run('schedule_create', { title: 't', prompt: 'p', at: '07:30', connectors: { gmail: 'read' } }, rpc)
    expect(out.isError).toBe(true)
    expect(text(out)).toContain('gmail')
  })

  test('the writable folders are said to be bounded, and the dock\'s refusal of one reaches the model', async () => {
    const rule = '/home/ana/.config: a scheduled task may write only inside the projects folder (/home/ana/Projects) or /tmp'
    const pi = await withDock((req) => (req.op === 'schedule_create' ? { ok: false, error: rule } : undefined))
    const tool = pi.tools.get('schedule_create')
    expect(tool.parameters.properties.writable.description).toContain('projects folder or /tmp')
    const out = await pi.run('schedule_create', { title: 't', prompt: 'p', at: '07:30', writable: ['/home/ana/.config'] }, rpc)
    expect(out.isError).toBe(true)
    expect(text(out)).toContain('projects folder')
    expect(text(out)).toContain('/tmp')
  })

  test('refuses in a subagent', async () => {
    const pi = await withDock()
    const out = await pi.run('schedule_list', {}, child)
    expect(out.isError).toBe(true)
    expect(text(out)).toContain('subagent')
  })

  test('refuses without a dock', async () => {
    const pi = fakePi()
    schedule(pi as any)
    const out = await pi.run('schedule_list', {}, rpc)
    expect(out.isError).toBe(true)
    expect(text(out)).toContain("Aria's dock")
  })

  test('the create description asks for a yes first', () => {
    const pi = fakePi()
    schedule(pi as any)
    expect(pi.tools.get('schedule_create').description).toContain('yes')
  })
})
