// The chat as plain data and a pure reducer: dock messages (and the Pi session events inside
// `chatEvent`) go in, a conversation to draw comes out. Nothing here touches the DOM.
//
// Entries of `messages`:
//   { kind: 'user', text }
//   { kind: 'assistant', text, thinking, done, texts, thoughts }   (texts/thoughts: streamed blocks by index)
//   { kind: 'tools', calls: [{ id, name, args, status: 'running'|'done'|'error', summary, result }] }
//   { kind: 'notice', tone: 'info'|'warn'|'error', key?, vars?, text? }

export function createChat() {
  return { workdir: '', running: false, engine: 'idle', model: null, thinking: null, problem: null, messages: [], dialogs: [] }
}

const RESULT_LIMIT = 4000

/** A short label for a tool call: the command, the path, or the tool's name. */
export function toolSummary(name, args = {}) {
  if (name === 'bash' && typeof args.command === 'string' && args.command) return args.command.split('\n')[0]
  if (['read', 'write', 'edit'].includes(name)) {
    const path = args.path ?? args.file_path
    if (typeof path === 'string' && path) return path
  }
  return name
}

const textOf = (content) =>
  typeof content === 'string' ? content : (content ?? []).filter((block) => block.type === 'text').map((block) => block.text).join('')

const joined = (blocks) =>
  Object.keys(blocks)
    .map(Number)
    .sort((a, b) => a - b)
    .map((index) => blocks[index])
    .join('')

const notice = (tone, extra) => ({ kind: 'notice', tone, ...extra })

const withMessages = (state, messages) => ({ ...state, messages })
const push = (state, entry) => withMessages(state, [...state.messages, entry])
const replaceLast = (state, entry) => withMessages(state, [...state.messages.slice(0, -1), entry])

function lastStreamingAssistant(state) {
  const last = state.messages.at(-1)
  return last?.kind === 'assistant' && !last.done ? last : null
}

function streamUpdate(state, update) {
  const current = lastStreamingAssistant(state)
  if (!current) return state
  const field = update.type.startsWith('thinking') ? 'thoughts' : 'texts'
  const kind = update.type.split('_')[1] // start | delta | end
  const blocks = { ...current[field] }
  if (kind === 'delta') blocks[update.contentIndex] = (blocks[update.contentIndex] ?? '') + update.delta
  else if (kind === 'start') blocks[update.contentIndex] = blocks[update.contentIndex] ?? ''
  else if (kind === 'end' && typeof update.content === 'string') blocks[update.contentIndex] = update.content
  else return state
  const next = { ...current, [field]: blocks }
  next.text = joined(next.texts)
  next.thinking = joined(next.thoughts)
  return replaceLast(state, next)
}

function endAssistant(state, message) {
  const text = textOf(message.content)
  const thinking = (message.content ?? []).filter((block) => block.type === 'thinking').map((block) => block.thinking).join('')
  const current = lastStreamingAssistant(state)
  const base = current ?? { kind: 'assistant', texts: {}, thoughts: {} }
  const done = { ...base, text, thinking: thinking || base.thinking || '', done: true }
  const rest = current ? state.messages.slice(0, -1) : state.messages
  if (!done.text && !done.thinking) return withMessages(state, rest) // only tool calls: the tools show up as chips
  return withMessages(state, [...rest, done])
}

function startTool(state, event) {
  const call = {
    id: event.toolCallId,
    name: event.toolName,
    args: event.args ?? {},
    status: 'running',
    summary: toolSummary(event.toolName, event.args),
    result: '',
  }
  const last = state.messages.at(-1)
  if (last?.kind === 'tools') return replaceLast(state, { ...last, calls: [...last.calls, call] })
  return push(state, { kind: 'tools', calls: [call] })
}

function finishTool(state, event) {
  const resultText = textOf(event.result?.content).slice(0, RESULT_LIMIT)
  const messages = state.messages.map((entry) =>
    entry.kind !== 'tools' || !entry.calls.some((call) => call.id === event.toolCallId)
      ? entry
      : {
          ...entry,
          calls: entry.calls.map((call) =>
            call.id === event.toolCallId ? { ...call, status: event.isError ? 'error' : 'done', result: resultText } : call,
          ),
        },
  )
  return withMessages(state, messages)
}

const TONES = { warning: 'warn', error: 'error' }

function fromHistory(messages) {
  const entries = []
  const calls = new Map()
  for (const message of messages ?? []) {
    if (message.role === 'user') {
      entries.push({ kind: 'user', text: textOf(message.content) })
    } else if (message.role === 'assistant') {
      const text = textOf(message.content)
      const thinking = (message.content ?? []).filter((block) => block.type === 'thinking').map((block) => block.thinking).join('')
      if (text || thinking) entries.push({ kind: 'assistant', text, thinking, done: true, texts: { 0: text }, thoughts: { 0: thinking } })
      const toolCalls = (message.content ?? []).filter((block) => block.type === 'toolCall')
      if (toolCalls.length) {
        const group = {
          kind: 'tools',
          calls: toolCalls.map((block) => {
            const call = { id: block.id, name: block.name, args: block.arguments ?? {}, status: 'done', summary: toolSummary(block.name, block.arguments), result: '' }
            calls.set(block.id, call)
            return call
          }),
        }
        entries.push(group)
      }
    } else if (message.role === 'toolResult') {
      const call = calls.get(message.toolCallId)
      if (call) {
        call.status = message.isError ? 'error' : 'done'
        call.result = textOf(message.content).slice(0, RESULT_LIMIT)
      }
    }
  }
  return entries
}

function reduceEvent(state, event) {
  switch (event.type) {
    case 'agent_start':
      return { ...state, running: true }
    case 'agent_settled':
      return { ...state, running: false }
    case 'message_start':
      return event.message?.role === 'assistant'
        ? push(state, { kind: 'assistant', text: '', thinking: '', done: false, texts: {}, thoughts: {} })
        : state
    case 'message_update':
      return streamUpdate(state, event.assistantMessageEvent ?? {})
    case 'message_end':
      return event.message?.role === 'assistant' ? endAssistant(state, event.message) : state
    case 'tool_execution_start':
      return startTool(state, event)
    case 'tool_execution_end':
      return finishTool(state, event)
    case 'auto_retry_start':
      return push(state, notice('warn', { key: 'chat.retry', vars: { attempt: event.attempt, max: event.maxAttempts } }))
    case 'compaction_start':
      return push(state, notice('info', { key: 'chat.compacting' }))
    case 'engine_exited':
      return push({ ...state, running: false }, notice('error', { key: 'chat.engine.exited' }))
    case 'command_failed':
      return push(state, notice('error', { text: event.error ?? event.command }))
    case 'ui_notice':
      return event.request?.method === 'notify'
        ? push(state, notice(TONES[event.request.notifyType] ?? 'info', { text: event.request.message }))
        : state
    case 'history':
      // Only into an empty page: it can arrive after the first message or while a reply streams.
      return state.messages.length ? state : withMessages(state, fromHistory(event.messages))
    case 'session_reset':
      return withMessages(state, [])
    default:
      return state
  }
}

/** `message` is a dock message (see crates/aria-core/src/bridge.rs). */
export function reduce(state, message) {
  switch (message.type) {
    case 'chatEvent':
      return reduceEvent(state, message.event ?? {})
    case 'chatState':
      return {
        ...state,
        workdir: message.workdir,
        running: message.running,
        model: message.model ?? null,
        thinking: message.thinking ?? null,
        problem: message.problem ?? null,
      }
    case 'chatEngine':
      return { ...state, engine: message.state }
    case 'chatDialog': {
      const dialog = { id: message.id, method: message.method, request: message.request }
      return { ...state, dialogs: [...state.dialogs.filter((d) => d.id !== message.id), dialog] }
    }
    case 'chatDialogClosed':
      return { ...state, dialogs: state.dialogs.filter((d) => d.id !== message.id) }
    default:
      return state
  }
}

export const addUser = (state, text) => push(state, { kind: 'user', text })
export const answerDialog = (state, id) => ({ ...state, dialogs: state.dialogs.filter((d) => d.id !== id) })
