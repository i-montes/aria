// A fake of Aria's dock for the background extension's tests: a Unix socket server that speaks the jobs protocol
// (one JSON object per line), records every line it gets, answers with `handle` (or a plain `ok`), and can push events.
import { mkdtempSync } from 'node:fs'
import { createServer, type Server, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

export type FakeDock = {
  path: string
  lines: any[]
  jobs: any[]
  push: (event: any) => void
  stop: () => Promise<void>
  start: () => Promise<void>
  waitFor: (pred: (line: any) => boolean, ms?: number) => Promise<any>
}

export const socketPath = () => join(mkdtempSync(join(tmpdir(), 'aria-bg-')), 'jobs.sock')

export async function fakeDock(path = socketPath(), handle?: (req: any, dock: FakeDock) => any): Promise<FakeDock> {
  let server: Server | null = null
  const sockets = new Set<Socket>()
  const dock: FakeDock = {
    path,
    lines: [],
    jobs: [],
    push: (event) => {
      for (const s of sockets) s.write(`${JSON.stringify(event)}\n`)
    },
    stop: async () => {
      for (const s of sockets) s.destroy()
      sockets.clear()
      await new Promise<void>((resolve) => (server ? server.close(() => resolve()) : resolve()))
      server = null
    },
    start: async () => {
      server = createServer((sock) => {
        sockets.add(sock)
        let buffer = ''
        sock.on('data', (chunk) => {
          buffer += chunk.toString()
          let at: number
          while ((at = buffer.indexOf('\n')) >= 0) {
            const line = buffer.slice(0, at)
            buffer = buffer.slice(at + 1)
            const req = JSON.parse(line)
            dock.lines.push(req)
            if (req.op === 'ack') continue
            const answer = handle?.(req, dock) ?? (req.op === 'list' ? { ok: true, jobs: dock.jobs } : { ok: true })
            if (answer === null) continue
            sock.write(`${JSON.stringify({ re: req.id, ...answer })}\n`)
          }
        })
        sock.on('close', () => sockets.delete(sock))
        sock.on('error', () => {})
      })
      await new Promise<void>((resolve) => server!.listen(path, () => resolve()))
    },
    waitFor: async (pred, ms = 3000) => {
      const end = Date.now() + ms
      while (Date.now() < end) {
        const found = dock.lines.find(pred)
        if (found) return found
        await Bun.sleep(10)
      }
      throw new Error('the fake dock never got the expected line')
    },
  }
  await dock.start()
  return dock
}

/** A job as the dock writes it, with the given fields changed. */
export const job = (fields: Record<string, unknown> = {}) => ({
  id: 'j3',
  number: 3,
  session: 's1',
  kind: 'process',
  title: 'Dev server',
  command: 'bun dev',
  task: null,
  folder: '~/Projects/shop',
  state: 'running',
  startedAt: Date.now(),
  endedAt: null,
  url: null,
  exit: null,
  isolation: null,
  worktree: null,
  writes: true,
  result: null,
  conflict: false,
  dismissed: false,
  ...fields,
})
