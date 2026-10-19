// The chat as plain data and a pure reducer: dock messages (and the Pi session events inside
// `chatEvent`) go in, a conversation to draw comes out. Nothing here touches the DOM.
//
import { parseCardTitle } from './parts/cards.js'

// Entries of `messages`:
//   { kind: 'user', text }
//   { kind: 'assistant', text, thinking, done, texts, thoughts }   (texts/thoughts: streamed blocks by index)
//   { kind: 'tools', calls: [{ id, name, args, status: 'running'|'done'|'error', summary, result }] }
//   { kind: 'notice', tone: 'info'|'warn'|'error', key?, vars?, text? }
//   { kind: 'note', label, text?, key? }          a one-line fact (model changed, stopped, sandbox answer…)
//   { kind: 'divider', key, vars? }                 a separator (the context was compacted)
//   { kind: 'jev', mode, action }                   the router's route line before an answer
//   { kind: 'answers', answers: [{ header, selected, talk }] }   what you answered to a question card
// `queue` holds messages waiting behind the run; `phase` names what the run is doing now.

export function createChat() {
  return {
    workdir: '', running: false, engine: 'idle', model: null, thinking: null, context: null, problem: null,
    messages: [], dialogs: [], queue: [], delivered: [], phase: 'working', startedAt: null,
  }
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

/** What the run is doing, from the tool that just started (a code the working indicator has a text for). */
export function phaseFor(name = '') {
  if (/engram|mem_/.test(name)) return 'searching memory'
  if (/tavily/.test(name)) return 'searching the web'
  if (/notion/.test(name)) return 'checking Notion'
  if (/^(parallel|subagent)/.test(name)) return 'delegating'
  if (name === 'bash') return 'running command'
  if (name === 'read') return 'reading files'
  if (name === 'write' || name === 'edit') return 'writing files'
  if (['grep', 'find', 'ls'].includes(name)) return 'reading files'
  return 'working'
}

const modelShort = (model) => String(model ?? '').split('/').pop()

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
      return { ...state, running: true, startedAt: Date.now(), phase: 'working' }
    case 'agent_settled': {
      const settled = { ...state, running: false, startedAt: null, phase: 'working', delivered: [] }
      return event.aborted ? push(settled, { kind: 'note', label: 'aria', key: 'chat.stoppedByYou' }) : settled
    }
    case 'message_start':
      if (event.message?.role === 'assistant') {
        return push(state, { kind: 'assistant', text: '', thinking: '', done: false, texts: {}, thoughts: {} })
      }
      if (event.message?.role === 'user') {
        // A message that waited in the queue is now being answered: it moves into the thread. Pi takes it out of
        // the queue (a `queue_update`) just BEFORE announcing it, so `delivered` remembers what left the queue.
        const text = textOf(event.message.content)
        const waiting = state.queue.findIndex((q) => q.text === text)
        if (waiting >= 0) return push({ ...state, queue: state.queue.filter((_, i) => i !== waiting) }, { kind: 'user', text })
        const left = state.delivered.indexOf(text)
        if (left >= 0) return push({ ...state, delivered: state.delivered.filter((_, i) => i !== left) }, { kind: 'user', text })
        return state // the echo of a message the page already showed
      }
      return state
    case 'queue_update': {
      const queue = [...(event.steering ?? []), ...(event.followUp ?? [])].map((text) => ({ text }))
      // What was queued and no longer is has been taken by the run (or cancelled by us, which `clearQueue` handles first).
      const remaining = [...queue]
      const gone = []
      for (const old of state.queue) {
        const at = remaining.findIndex((q) => q.text === old.text)
        if (at >= 0) remaining.splice(at, 1)
        else gone.push(old.text)
      }
      return { ...state, queue, delivered: [...state.delivered, ...gone] }
    }
    case 'aria_jev':
      return push(state, { kind: 'jev', mode: event.mode, action: event.action })
    case 'message_update': {
      const update = event.assistantMessageEvent ?? {}
      const next = streamUpdate(state, update)
      if (update.type?.startsWith('thinking')) return { ...next, phase: 'thinking' }
      return update.type === 'text_delta' && next.phase === 'thinking' ? { ...next, phase: 'working' } : next
    }
    case 'message_end':
      return event.message?.role === 'assistant' ? endAssistant(state, event.message) : state
    case 'tool_execution_start':
      return { ...startTool(state, event), phase: phaseFor(event.toolName) }
    case 'tool_execution_end':
      return finishTool(state, event)
    case 'auto_retry_start':
      return push(state, notice('warn', { key: 'chat.retry', vars: { attempt: event.attempt, max: event.maxAttempts } }))
    case 'compaction_start':
      return { ...state, phase: 'compacting context' }
    case 'compaction_end':
      return event.aborted || !event.result
        ? { ...state, phase: 'working' }
        : push({ ...state, phase: 'working' }, { kind: 'divider', key: 'chat.compacted' })
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
      return { ...withMessages(state, []), queue: [], delivered: [], dialogs: [] }
    default:
      return state
  }
}

/** `message` is a dock message (see crates/aria-core/src/bridge.rs). */
export function reduce(state, message) {
  switch (message.type) {
    case 'chatEvent':
      return reduceEvent(state, message.event ?? {})
    case 'chatState': {
      const next = {
        ...state,
        workdir: message.workdir,
        running: message.running,
        model: message.model ?? null,
        thinking: message.thinking ?? null,
        context: message.context ?? null,
        problem: message.problem ?? null,
      }
      // The model changing under a conversation leaves a note (the first report is not a change).
      return state.model && next.model && state.model !== next.model
        ? push(next, { kind: 'note', label: 'model', text: modelShort(next.model) })
        : next
    }
    case 'chatEngine':
      return { ...state, engine: message.state }
    case 'chatDialog': {
      const dialog = { id: message.id, method: message.method, request: message.request, card: parseCardTitle(message.request?.title) }
      return { ...state, dialogs: [...state.dialogs.filter((d) => d.id !== message.id), dialog] }
    }
    case 'chatDialogClosed':
      return { ...state, dialogs: state.dialogs.filter((d) => d.id !== message.id) }
    default:
      return state
  }
}

export const addUser = (state, text) => push(state, { kind: 'user', text })
export const addQueued = (state, text) => ({ ...state, queue: [...state.queue, { text }] })
export const clearQueue = (state, at) => ({ ...state, queue: at === undefined ? [] : state.queue.filter((_, i) => i !== at) })

/** The question card's answers as the answers card draws them: its header, what was picked, whether it was only talked about. */
export function answersForCard(questions, answers) {
  return (answers ?? []).map((a) => {
    const asked = (questions ?? []).find((q) => q.question === a.question)
    return { header: asked?.header || a.question, selected: a.answers ?? [], talk: a.talk === true }
  })
}

const CARD_LABEL = { sandbox: 'sandbox', write: 'sandbox', quota: 'model', question: 'ask' }

/** The dialog is answered: it goes away, and a card leaves what was chosen in the thread. `choice` is the
 * option's text (or the answers array of a question card; nothing for a skipped one). */
export function answerDialog(state, id, choice) {
  const dialog = state.dialogs.find((d) => d.id === id)
  const next = { ...state, dialogs: state.dialogs.filter((d) => d.id !== id) }
  const kind = dialog?.card?.kind
  if (!kind) return next
  if (kind === 'question') {
    return Array.isArray(choice) ? push(next, { kind: 'answers', answers: choice }) : push(next, { kind: 'note', label: 'ask', key: 'chat.askSkipped' })
  }
  return choice ? push(next, { kind: 'note', label: CARD_LABEL[kind] ?? kind, key: `chat.result.${choice}`, text: String(choice) }) : next
}
