// The client of the dock's jobs socket (`ARIA_JOBS_SOCK`, one per session): one JSON object per line both ways.
// It says `hello` on every connect (the dock then gives the events not acked yet), matches answers to requests by
// `id`, hands pushed events to its listeners, and reconnects every second while the dock is away. The socket and its
// timers are unref'd, so the client never keeps Pi's process alive. `startProcess` is the shared start of a
// background process (`background_run`, here with `PI_SANDBOX=0` and in the sandbox extension otherwise).
import { createConnection, type Socket } from 'node:net'

export const DOCK_REFUSAL = "Background work needs Aria's dock."
const RECONNECT_MS = 1000
const DEFAULT_TIMEOUT_MS = 5000
const MAX_LINE = 4 * 1024 * 1024
const MAX_KEPT_EVENTS = 200

type Pending = { line: string; resolve: (v: any) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout>; sent: boolean }

export class JobsClient {
  readonly path: string
  private socket: Socket | null = null
  private connected = false
  private closed = false
  private buffer = ''
  private nextId = 1
  private pending = new Map<number, Pending>()
  private listeners = new Set<(e: any) => void>()
  private kept: any[] = []
  private retry: ReturnType<typeof setTimeout> | null = null

  constructor(path: string) {
    this.path = path
    this.connect()
  }

  private connect() {
    if (this.closed) return
    let socket: Socket
    try {
      socket = createConnection({ path: this.path })
    } catch {
      this.scheduleReconnect()
      return
    }
    this.socket = socket
    socket.unref()
    socket.setEncoding('utf8')
    socket.on('connect', () => {
      if (this.socket !== socket) return
      this.connected = true
      this.buffer = ''
      socket.write(`${JSON.stringify({ id: this.nextId++, op: 'hello' })}\n`)
      for (const p of this.pending.values()) {
        if (!p.sent) {
          p.sent = true
          socket.write(p.line)
        }
      }
    })
    socket.on('data', (chunk: string) => this.read(chunk))
    socket.on('error', () => {})
    socket.on('close', () => {
      if (this.socket !== socket) return
      this.socket = null
      this.connected = false
      // An answer to a line already written will not come on a new connection.
      for (const [id, p] of this.pending) {
        if (p.sent) {
          clearTimeout(p.timer)
          this.pending.delete(id)
          p.reject(new Error("Aria's dock closed the connection"))
        }
      }
      this.scheduleReconnect()
    })
  }

  private scheduleReconnect() {
    if (this.closed || this.retry) return
    this.retry = setTimeout(() => {
      this.retry = null
      this.connect()
    }, RECONNECT_MS)
    this.retry.unref?.()
  }

  private read(chunk: string) {
    this.buffer += chunk
    if (this.buffer.length > MAX_LINE && !this.buffer.includes('\n')) {
      this.buffer = ''
      return
    }
    let at: number
    while ((at = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, at)
      this.buffer = this.buffer.slice(at + 1)
      let msg: any
      try {
        msg = JSON.parse(line)
      } catch {
        continue
      }
      if (!msg || typeof msg !== 'object') continue
      if ('re' in msg) {
        const p = this.pending.get(msg.re)
        if (!p) continue
        clearTimeout(p.timer)
        this.pending.delete(msg.re)
        p.resolve(msg)
      } else if ('event' in msg) {
        this.emit(msg)
      }
    }
  }

  private emit(event: any) {
    if (!this.listeners.size) {
      this.kept.push(event)
      if (this.kept.length > MAX_KEPT_EVENTS) this.kept.shift()
      return
    }
    for (const fn of this.listeners) {
      try {
        fn(event)
      } catch {
        // a listener's failure never reaches the socket
      }
    }
  }

  /** Sends `op` with `fields` and resolves with the dock's answer (`ok` true or false); rejects on timeout. */
  request(op: string, fields: object = {}, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<any> {
    return new Promise((resolve, reject) => {
      if (this.closed) return reject(new Error('the jobs client is closed'))
      const id = this.nextId++
      const line = `${JSON.stringify({ ...fields, id, op })}\n`
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`Aria's dock did not answer ${op} in time`))
      }, timeoutMs)
      timer.unref?.()
      const p: Pending = { line, resolve, reject, timer, sent: false }
      this.pending.set(id, p)
      if (this.connected && this.socket) {
        p.sent = true
        this.socket.write(line)
      }
    })
  }

  /** Tells the dock an event was handled, so it is not given again. */
  ack(seq: number) {
    if (this.connected && this.socket) this.socket.write(`${JSON.stringify({ op: 'ack', event: seq })}\n`)
  }

  /** Listens to pushed events; events that came while nobody listened go to the first listener. Returns the unsubscribe. */
  onEvent(fn: (e: any) => void): () => void {
    this.listeners.add(fn)
    const kept = this.kept
    this.kept = []
    for (const e of kept) this.emit(e)
    return () => {
      this.listeners.delete(fn)
    }
  }

  close() {
    this.closed = true
    if (this.retry) clearTimeout(this.retry)
    this.retry = null
    for (const p of this.pending.values()) {
      clearTimeout(p.timer)
      p.reject(new Error('the jobs client is closed'))
    }
    this.pending.clear()
    this.listeners.clear()
    this.socket?.destroy()
    this.socket = null
    this.connected = false
  }

  get isClosed() {
    return this.closed
  }
}

// One client per process (kept on globalThis, so a reloaded extension reuses it); a new path replaces it.
const KEY = Symbol.for('aria.background.jobsClient')
const store = globalThis as unknown as Record<symbol, JobsClient | undefined>

/** The session's client, or null without `ARIA_JOBS_SOCK` (not under Aria's dock). */
export const jobsClient = (): JobsClient | null => {
  const path = process.env.ARIA_JOBS_SOCK
  const current = store[KEY]
  if (!path) return null
  if (current && !current.isClosed && current.path === path) return current
  current?.close()
  const client = new JobsClient(path)
  store[KEY] = client
  return client
}

const textResult = (text: string, isError = false, details: any = undefined) => ({
  content: [{ type: 'text' as const, text }],
  ...(isError ? { isError: true } : {}),
  details,
})

export const refusal = (text: string) => textResult(text, true)

const URL_RE = /\bhttps?:\/\/[^\s'"<>()\]]+/

const POLL_MS = 300
const WAIT_MS = 3000
const SHOWN_LINES = 30

/**
 * Starts a background process: `command` is what the person sees, `line` what runs (the sandbox's line, or the
 * command itself without the sandbox). Waits up to 3 s for its first output (less once a URL shows) and says what
 * started.
 */
export async function startProcess(o: { command: string; line: string; title: string; folder: string }): Promise<{ content: any[]; isError?: boolean; details?: any }> {
  const client = process.env.ARIA_JOBS_SOCK ? jobsClient() : null
  if (!client) return refusal(DOCK_REFUSAL)
  const env: Record<string, string> = {}
  for (const [k, v] of Object.entries(process.env)) if (typeof v === 'string') env[k] = v
  let answer: any
  try {
    answer = await client.request('start', { kind: 'process', command: o.command, line: o.line, title: o.title, folder: o.folder, env })
  } catch (e: any) {
    return refusal(`Could not start it: ${e?.message ?? e}`)
  }
  if (!answer?.ok || !answer.job) return refusal(`Could not start it: ${answer?.error ?? 'no answer from the dock'}`)
  const job = answer.job
  let lines: string[] = []
  let url: string | null = job.url ?? null
  let ended: any = null
  const end = Date.now() + WAIT_MS
  while (Date.now() < end && !url && !ended) {
    await new Promise((r) => setTimeout(r, Math.min(POLL_MS, Math.max(0, end - Date.now()))))
    try {
      const out = await client.request('output', { job: job.id, lines: SHOWN_LINES }, 1000)
      if (out?.ok && Array.isArray(out.lines)) lines = out.lines
      const found = lines.map((l) => l.match(URL_RE)?.[0]).find(Boolean)
      if (found) url = found
      const list = await client.request('list', {}, 1000)
      const now = Array.isArray(list?.jobs) ? list.jobs.find((j: any) => j.id === job.id) : null
      if (now?.url) url = now.url
      if (now && now.state !== 'running' && now.state !== 'queued') ended = now
    } catch {
      // keep waiting until the 3 s are over
    }
  }
  const parts = [`Started in the background as #${job.number} (${o.title}). Output so far:`, lines.length ? lines.join('\n') : '(nothing yet)']
  if (url) parts.push(`URL: ${url}`)
  if (ended) parts.push(`It has already ended (${ended.state}${ended.exit !== null && ended.exit !== undefined ? `, exit ${ended.exit}` : ''}).`)
  parts.push('Use background_output, background_stop.')
  return textResult(parts.join('\n'), false, { job: ended ?? job, url })
}
