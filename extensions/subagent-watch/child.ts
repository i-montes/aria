// Subagent watch, the child's side: a subagent (`pi --mode json` with PI_WATCH_SOCK set) streams its own events to the
// parent's socket, one JSON line each. The dock (crates/aria/src/watch.rs) reads them for the subagents view and for
// the `/debug` trace: every line carries `at` (the child's clock), tool lines the call's id, and a finished assistant
// message its model and usage. Kept apart from index.ts so it loads without Pi's TUI package (and can be tested).
import { createConnection, type Socket } from 'node:net'
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'

export const MAX_RESULT_CHARS = 4000

export const textOf = (content: unknown): string => {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content
    .map((b: any) => (b?.type === 'text' ? b.text : b?.type === 'image' ? '[imagen]' : ''))
    .filter(Boolean)
    .join('\n')
}

export const argsPreview = (name: string, args: any): string => {
  if (!args || typeof args !== 'object') return ''
  if (name === 'bash') return String(args.command ?? '')
  if (name === 'read' || name === 'write' || name === 'edit') return String(args.path ?? '')
  if (name === 'codemode') return String(args.code ?? '')
  const s = JSON.stringify(args)
  return s.length > 400 ? `${s.slice(0, 400)}…` : s
}

// --- Child side ---------------------------------------------------------------------------------

export function child(pi: ExtensionAPI, path: string) {
  let sock: Socket | null = null
  const send = (msg: Record<string, unknown>) => {
    if (!sock || sock.destroyed) return
    try {
      sock.write(`${JSON.stringify({ ...msg, at: Date.now() })}\n`)
    } catch {
      // the parent went away; keep working
    }
  }

  pi.on('session_start', () => {
    try {
      sock = createConnection(path)
      sock.on('error', () => {
        sock = null
      })
      sock.unref()
    } catch {
      sock = null
    }
  })
  pi.on('session_shutdown', () => sock?.end())

  pi.on('before_agent_start', (event: any) => {
    if (event.messageType === 'user') send({ t: 'prompt', text: String(event.prompt ?? '') })
  })
  pi.on('message_update', (event: any) => {
    const e = event.assistantMessageEvent
    if (e?.type === 'text_delta') send({ t: 'text', delta: e.delta })
    else if (e?.type === 'thinking_delta') send({ t: 'thinking', delta: e.delta })
  })
  pi.on('message_start', (event: any) => {
    if (event.message?.role === 'assistant') send({ t: 'llm_start' })
  })
  pi.on('message_end', (event: any) => {
    const m = event.message
    if (m?.role === 'assistant') send({ t: 'turn', model: m.model, usage: m.usage, stop: m.stopReason, error: m.errorMessage })
  })
  pi.on('tool_execution_start', (event: any) => {
    send({ t: 'tool', id: event.toolCallId, name: event.toolName, args: argsPreview(event.toolName, event.args) })
  })
  pi.on('tool_execution_end', (event: any) => {
    send({ t: 'result', id: event.toolCallId, text: textOf(event.result?.content).slice(0, MAX_RESULT_CHARS), error: !!event.isError })
  })
  pi.on('agent_end', () => send({ t: 'end' }))
}
