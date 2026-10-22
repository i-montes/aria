// Subagent watch: lets the user observe, live, what subagents (child `pi` processes such as the
// ones `parallel` starts) are doing, without saving their sessions.
//
// - Parent (interactive session): opens a local Unix socket on session start and exports its path
//   in PI_WATCH_SOCK, which every child process inherits. `alt+a` or `/agents` opens an overlay
//   with each subagent's prompt, streamed text and tool calls; Esc returns to the session.
// - Child (`pi --mode json`, with PI_WATCH_SOCK set): streams its own events to that socket.
//
// Nothing is written to disk; the socket file lives in the temp dir and is removed on shutdown.
// Everything is best effort: if the socket is missing or breaks, subagents run as before.
import { existsSync, unlinkSync } from 'node:fs'
import { createServer, type Server, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { matchesKey, truncateToWidth, visibleWidth, wrapTextWithAnsi } from '@earendil-works/pi-tui'
import { child } from './child.ts'

const ENV = 'PI_WATCH_SOCK'
const MAX_ENTRIES = 2000
const MAX_AGENTS = 20

type Kind = 'prompt' | 'text' | 'thinking' | 'tool' | 'result' | 'error' | 'status'
interface Entry {
  kind: Kind
  text: string
}
interface Agent {
  id: number
  name: string
  status: 'running' | 'done' | 'failed'
  entries: Entry[]
  startedAt: number
  tools: number
}


// --- Parent side --------------------------------------------------------------------------------

function parent(pi: ExtensionAPI) {
  const agents: Agent[] = []
  let nextId = 1
  let server: Server | null = null
  let sockPath = ''
  let ui: any = null
  let redraw: (() => void) | null = null
  // Names and tasks of the latest `parallel` call, to name the subagents that connect.
  let pending: { name: string; task: string }[] = []

  const updateStatus = () => {
    const running = agents.filter((a) => a.status === 'running').length
    try {
      ui?.setStatus?.('subagent-watch', running ? `▶ ${running} subagent${running > 1 ? 's' : ''} · alt+a to watch` : undefined)
    } catch {
      // no UI
    }
  }
  const changed = () => {
    updateStatus()
    redraw?.()
  }

  const push = (agent: Agent, entry: Entry) => {
    agent.entries.push(entry)
    if (agent.entries.length > MAX_ENTRIES) agent.entries.splice(0, agent.entries.length - MAX_ENTRIES)
  }
  const appendStream = (agent: Agent, kind: 'text' | 'thinking', delta: string) => {
    const last = agent.entries.at(-1)
    if (last?.kind === kind) last.text += delta
    else push(agent, { kind, text: delta })
  }

  const nameFor = (prompt: string): string => {
    const i = pending.findIndex((p) => p.task && prompt.includes(p.task.slice(0, 160)))
    if (i >= 0) return pending.splice(i, 1)[0].name
    const line = prompt.replace(/^[\s\S]*?Task:\s*/, '').split('\n').find((l) => l.trim()) ?? 'subagent'
    return line.trim().slice(0, 30)
  }

  const onConnection = (socket: Socket) => {
    const agent: Agent = { id: nextId++, name: `#${nextId - 1}`, status: 'running', entries: [], startedAt: Date.now(), tools: 0 }
    agents.push(agent)
    if (agents.length > MAX_AGENTS) agents.splice(0, agents.length - MAX_AGENTS)
    changed()
    let buffer = ''
    socket.setEncoding('utf8')
    socket.on('data', (chunk: string) => {
      buffer += chunk
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        let m: any
        try {
          m = JSON.parse(line)
        } catch {
          continue
        }
        if (m.t === 'prompt') {
          agent.name = nameFor(m.text)
          push(agent, { kind: 'prompt', text: m.text })
        } else if (m.t === 'text' || m.t === 'thinking') appendStream(agent, m.t, String(m.delta ?? ''))
        else if (m.t === 'turn') push(agent, { kind: 'status', text: '' })
        else if (m.t === 'tool') {
          agent.tools++
          push(agent, { kind: 'tool', text: `${m.name}  ${m.args ?? ''}` })
        } else if (m.t === 'result') push(agent, { kind: m.error ? 'error' : 'result', text: m.text })
        else if (m.t === 'end') agent.status = 'done'
      }
      changed()
    })
    const close = () => {
      if (agent.status === 'running') agent.status = agent.entries.length ? 'done' : 'failed'
      changed()
    }
    socket.on('close', close)
    socket.on('error', close)
  }

  pi.on('session_start', (_event: any, ctx: any) => {
    if (ctx?.mode !== 'tui') return
    ui = ctx.ui
    if (server) return
    sockPath = join(tmpdir(), `pi-watch-${process.pid}.sock`)
    try {
      if (existsSync(sockPath)) unlinkSync(sockPath)
      server = createServer(onConnection)
      server.on('error', () => {
        server = null
      })
      server.listen(sockPath)
      server.unref()
      process.env[ENV] = sockPath
    } catch {
      server = null
    }
  })

  pi.on('session_shutdown', () => {
    server?.close()
    server = null
    try {
      if (sockPath && existsSync(sockPath)) unlinkSync(sockPath)
    } catch {
      // already gone
    }
    delete process.env[ENV]
  })

  pi.on('tool_call', (event: any) => {
    if (event.toolName !== 'parallel') return
    const input = event.input ?? {}
    const list: any[] = [
      ...(Array.isArray(input.tasks) ? input.tasks : []),
      ...(Array.isArray(input.chain) ? input.chain : []),
      ...(Array.isArray(input.team?.tasks) ? input.team.tasks.map((t: any) => ({ ...t, name: t.name ?? t.id ?? t.assignee })) : []),
    ]
    if (typeof input.task === 'string') list.push({ name: 'subagent', task: input.task })
    if (input.race?.task) for (const m of input.race.models ?? []) list.push({ name: `carrera ${m}`, task: input.race.task })
    pending = list.map((t, i) => {
      const name = String(t.name ?? t.agent ?? `tarea ${i + 1}`)
      return { name: t.agent && t.agent !== name ? `${name} (${t.agent})` : name, task: String(t.task ?? '') }
    })
  })

  const open = async (ctx: any) => {
    if (ctx?.mode !== 'tui') return
    if (!agents.length) {
      ctx.ui.notify('No subagents in this session yet.', 'info')
      return
    }
    await ctx.ui.custom(
      (tui: any, theme: any, _kb: any, done: (r: null) => void) => {
        const view = new WatchView(agents, theme, () => tui.terminal.rows, done)
        redraw = () => {
          view.invalidate()
          tui.requestRender()
        }
        return view
      },
      { overlay: true, overlayOptions: { width: '92%', maxHeight: '90%', anchor: 'center' } },
    )
    redraw = null
  }

  pi.registerShortcut('alt+a', { description: 'Watch live what the subagents are doing', handler: open })
  pi.registerCommand('agents', { description: 'Watch live what the subagents are doing', handler: (_args: string, ctx: any) => open(ctx) })
}

class WatchView {
  private selected: number
  private scroll = 0 // lines from the bottom; 0 = follow the end
  private cache: { width: number; lines: string[] } | null = null

  constructor(
    private agents: Agent[],
    private theme: any,
    private rows: () => number,
    private done: (r: null) => void,
  ) {
    const running = agents.findIndex((a) => a.status === 'running')
    this.selected = running >= 0 ? running : agents.length - 1
  }

  invalidate() {
    this.cache = null
  }

  handleInput(data: string) {
    const page = Math.max(1, this.height() - 6)
    if (matchesKey(data, 'escape') || matchesKey(data, 'q')) return this.done(null)
    if (matchesKey(data, 'right') || matchesKey(data, 'tab')) this.select(this.selected + 1)
    else if (matchesKey(data, 'left') || matchesKey(data, 'shift+tab')) this.select(this.selected - 1)
    else if (matchesKey(data, 'up')) this.scroll += 1
    else if (matchesKey(data, 'down')) this.scroll = Math.max(0, this.scroll - 1)
    else if (matchesKey(data, 'pageUp')) this.scroll += page
    else if (matchesKey(data, 'pageDown')) this.scroll = Math.max(0, this.scroll - page)
    else if (matchesKey(data, 'end')) this.scroll = 0
    else if (matchesKey(data, 'home')) this.scroll = Number.MAX_SAFE_INTEGER
    else return
    this.invalidate()
  }

  private select(i: number) {
    const n = this.agents.length
    this.selected = ((i % n) + n) % n
    this.scroll = 0
    this.invalidate()
  }

  private height() {
    return Math.max(8, Math.floor(this.rows() * 0.9))
  }

  private body(agent: Agent, width: number): string[] {
    const t = this.theme
    const out: string[] = []
    // `prefix` marks the first line; `rest` indents the following ones.
    const add = (text: string, style: (s: string) => string, prefix = '', rest = prefix) => {
      text.split('\n').forEach((line, i) => {
        for (const w of wrapTextWithAnsi((i ? rest : prefix) + line, width)) out.push(style(w))
      })
    }
    for (const e of agent.entries) {
      if (e.kind === 'prompt') {
        add('Tarea', (s) => t.fg('accent', t.bold ? t.bold(s) : s))
        add(e.text.trim(), (s) => t.fg('muted', s), '  ')
        out.push('')
      } else if (e.kind === 'text') {
        add(e.text.trim(), (s) => s)
      } else if (e.kind === 'thinking') {
        add(e.text.trim(), (s) => t.fg('dim', s), '  ∴ ')
      } else if (e.kind === 'tool') {
        out.push('')
        add(e.text, (s) => t.fg('toolTitle', s), '▸ ', '  ')
      } else if (e.kind === 'result' || e.kind === 'error') {
        const lines = e.text.trimEnd().split('\n')
        const shown = lines.slice(0, 12).join('\n')
        add(shown, (s) => t.fg(e.kind === 'error' ? 'error' : 'dim', s), '  │ ')
        if (lines.length > 12) out.push(t.fg('muted', `  │ … ${lines.length - 12} more lines`))
      }
    }
    if (agent.status === 'running') out.push('', t.fg('accent', '● trabajando…'))
    return out
  }

  render(width: number): string[] {
    if (this.cache?.width === width) return this.cache.lines
    const t = this.theme
    const inner = Math.max(20, width - 4)
    const agent = this.agents[this.selected]
    const icon = (a: Agent) => (a.status === 'running' ? t.fg('accent', '●') : a.status === 'done' ? t.fg('success', '✓') : t.fg('error', '✗'))
    const tabs = this.agents
      .map((a, i) => {
        const label = `${icon(a)} ${a.name}`
        return i === this.selected ? t.fg('accent', `[${label}${t.fg('accent', ']')}`) : ` ${label} `
      })
      .join(t.fg('muted', '│'))
    const elapsed = Math.round((Date.now() - agent.startedAt) / 1000)
    const meta = t.fg('muted', `${agent.tools} herramientas · ${elapsed}s`)
    const body = this.body(agent, inner)
    const viewport = this.height() - 5
    const maxScroll = Math.max(0, body.length - viewport)
    this.scroll = Math.min(this.scroll, maxScroll)
    const end = body.length - this.scroll
    const visible = body.slice(Math.max(0, end - viewport), end)
    while (visible.length < viewport) visible.push('')
    const border = (s: string) => t.fg('border', s)
    const pad = (s: string) => {
      const line = truncateToWidth(s, inner)
      return `${border('│')} ${line}${' '.repeat(Math.max(0, inner - visibleWidth(line)))} ${border('│')}`
    }
    const footer = this.scroll ? `↑ ${this.scroll} lines up · End to follow` : 'following live'
    const lines = [
      border(`╭${'─'.repeat(inner + 2)}╮`),
      pad(`${truncateToWidth(tabs, inner - visibleWidth(meta) - 2)}  ${meta}`),
      border(`├${'─'.repeat(inner + 2)}┤`),
      ...visible.map(pad),
      pad(t.fg('muted', `←/→ agente · ↑/↓ PgUp/PgDn desplazar · ${footer} · Esc volver`)),
      border(`╰${'─'.repeat(inner + 2)}╯`),
    ]
    this.cache = { width, lines }
    return lines
  }
}

export default function (pi: ExtensionAPI) {
  const path = process.env[ENV]
  // A child started by a subagent tool inherits the parent's socket path and runs in JSON mode.
  // The mode is only known per event, so both sides register and check it there.
  if (path) {
    let isChild: boolean | null = null
    const childPi = new Proxy(pi, {
      get(target, prop) {
        if (prop !== 'on') return (target as any)[prop]
        return (event: string, handler: any) =>
          target.on(event as any, (e: any, ctx: any) => {
            if (isChild === null) isChild = ctx?.mode === 'json' || ctx?.mode === 'print'
            return isChild ? handler(e, ctx) : undefined
          })
      },
    })
    child(childPi as ExtensionAPI, path)
  }
  parent(pi)
}
