// Run: bun test extensions
// The jobs socket client against a fake dock: hello, answers by id, pushed events, reconnecting, and no dock.
import { afterEach, describe, expect, test } from 'bun:test'
import { JobsClient } from './client.ts'
import { fakeDock, socketPath, type FakeDock } from './fake-dock.ts'

let client: JobsClient | null = null
let dock: FakeDock | null = null
afterEach(async () => {
  client?.close()
  client = null
  await dock?.stop()
  dock = null
})

describe('the jobs client', () => {
  test('hello on connect and answers matched by id', async () => {
    dock = await fakeDock(undefined, (req) => (req.op === 'output' ? { ok: true, lines: [`out of ${req.job}`] } : undefined))
    client = new JobsClient(dock.path)
    await dock.waitFor((l) => l.op === 'hello')
    const [a, b] = await Promise.all([client.request('output', { job: 'j1' }), client.request('output', { job: 'j2' })])
    expect(a.lines).toEqual(['out of j1'])
    expect(b.lines).toEqual(['out of j2'])
    expect(dock.lines[0].op).toBe('hello')
  })

  test('pushed events reach onEvent', async () => {
    dock = await fakeDock()
    client = new JobsClient(dock.path)
    const seen: any[] = []
    client.onEvent((e) => seen.push(e))
    await dock.waitFor((l) => l.op === 'hello')
    await Bun.sleep(20)
    dock.push({ event: 'ended', seq: 4, by: 'self', job: { id: 'j1' }, lastLines: [] })
    for (let i = 0; i < 100 && !seen.length; i++) await Bun.sleep(10)
    expect(seen).toEqual([{ event: 'ended', seq: 4, by: 'self', job: { id: 'j1' }, lastLines: [] }])
  })

  test('an event that came before anyone listened is kept for the first listener', async () => {
    dock = await fakeDock()
    client = new JobsClient(dock.path)
    await dock.waitFor((l) => l.op === 'hello')
    await Bun.sleep(20)
    dock.push({ event: 'cut', seq: 2, jobs: [] })
    await Bun.sleep(50)
    const seen: any[] = []
    client.onEvent((e) => seen.push(e))
    expect(seen).toEqual([{ event: 'cut', seq: 2, jobs: [] }])
  })

  test('reconnects and says hello again after the dock comes back', async () => {
    dock = await fakeDock()
    client = new JobsClient(dock.path)
    await dock.waitFor((l) => l.op === 'hello')
    await dock.stop()
    dock.lines.length = 0
    await dock.start()
    await dock.waitFor((l) => l.op === 'hello', 3000)
    const answer = await client.request('list')
    expect(answer.ok).toBe(true)
  })

  test('a request without a dock rejects within its timeout', async () => {
    client = new JobsClient(socketPath())
    const started = Date.now()
    await expect(client.request('list', {}, 150)).rejects.toThrow()
    expect(Date.now() - started).toBeLessThan(1000)
  })

  test('acks go out as a line of their own', async () => {
    dock = await fakeDock()
    client = new JobsClient(dock.path)
    await dock.waitFor((l) => l.op === 'hello')
    client.ack(7)
    expect(await dock.waitFor((l) => l.op === 'ack')).toEqual({ op: 'ack', event: 7 })
  })
})
