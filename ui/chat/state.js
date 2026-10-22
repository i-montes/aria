// The chat as plain data and a pure reducer: dock messages (and the Pi session events inside
// `chatEvent`) go in, a conversation to draw comes out. Nothing here touches the DOM.
//
import { splitAttached } from './parts/attachments.js'
import { parseCardTitle } from './parts/cards.js'

// Entries of `messages`:
//   { kind: 'user', text, files? }   files: what it carries (attachments.js); `text` never holds the dock's lines
//   { kind: 'assistant', text, thinking, done, texts, thoughts }   (texts/thoughts: streamed blocks by index)
//   { kind: 'tools', calls: [{ id, name, args, status: 'running'|'done'|'error', summary, result }] }
//   { kind: 'notice', tone: 'info'|'warn'|'error', key?, vars?, text? }
//   { kind: 'note', label, text?, key?, vars? }   a one-line fact (model changed, stopped, sandbox answer, a background job ended…)
//   { kind: 'divider', key, vars? }                 a separator (the context was compacted)
//   { kind: 'earlier', id, summary }                what came before this session's history, as its summary (a
//                                                   compaction's, or the `aria-earlier` message a rotation starts with)
//   { kind: 'jev', mode, action }                   the router's route line before an answer
//   { kind: 'answers', answers: [{ header, selected, talk }] }   what you answered to a question card
// `earlier` pages back through older history: { more, cursor, loading, count, stale } (count: the entries prepended so
// far; stale: a history came while a page was on its way, so that page belongs to the paging before it and is dropped).
// `queue` holds messages waiting behind the run ({ text, steer?, files?, raw? }); `phase` names what the run is doing now.
//
// Attachments: the dock adds a line per file to the message it gives Pi (`[attached: …]`), so Pi's echo, its queue and
// the history hold those lines while the page's own entry (drawn before the dock answers) does not. Every entry keeps
// the text without the lines and the files apart, and messages are matched by that text. A queue entry from Pi also
// keeps `raw`, the text with the lines, which is what goes back to Pi when the queue is rebuilt (cancelling one).

export function createChat() {
  return {
    workdir: '', running: false, engine: 'idle', model: null, thinking: null, context: null, problem: null,
    messages: [], dialogs: [], queue: [], delivered: [], phase: 'working', startedAt: null, quota: [], giveBack: null, agents: [], turnFrom: 0,
    // The dock's turn watchdog stopped the current run (its notice says so; the settle then is not "stopped by you").
    stoppedByWatchdog: false,
    // Until Pi sends the session's history (empty for a new one) the thread shows that it is loading, timed from here.
    loaded: false, loadingSince: Date.now(),
    earlier: { ...NO_EARLIER },
  }
}

const NO_EARLIER = { more: false, cursor: null, loading: false, count: 0, stale: false }

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

/** A user entry from a text that may end with the dock's attachment lines. */
function userEntry(raw) {
  const { text, files } = splitAttached(raw)
  return files.length ? { kind: 'user', text, files } : { kind: 'user', text }
}

/** A queue entry from Pi's text. */
function queueEntry(raw, steer) {
  const { text, files } = splitAttached(raw)
  return { text, ...(steer ? { steer } : {}), ...(files.length ? { raw, files } : {}) }
}

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
  // Only the answer and the reasoning stream into the thread. A tool call's arguments (`toolcall_*`, JSON) are not
  // text: the call shows up as a chip when it runs.
  const field = update.type.startsWith('thinking') ? 'thoughts' : update.type.startsWith('text') ? 'texts' : null
  if (!field) return state
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

// ---- the turn's group (design: tool-chips): one accordion per answer ---------------------------
//
// What Aria does on the way (its reasoning, every tool call) goes into a `tools` entry, in order; a reasoning step is a
// call with `think: true` and its `text`. What it writes is said to the reader as a message, also between tools: the
// tools after it start a new group below it, so a question asked on the way is not folded away.

let stepSeq = 0

/** The group steps go into: the last `tools` entry after the last user message and the last thing Aria said (and after
 * `from`, where the run started), or -1. */
function turnGroup(messages, from = 0) {
  for (let i = messages.length - 1; i >= from; i--) {
    if (messages[i].kind === 'user' || (messages[i].kind === 'assistant' && messages[i].text)) return -1
    if (messages[i].kind === 'tools') return i
  }
  return -1
}

/** Adds steps to this turn's group, creating it (at the end) when the turn has none yet. */
function addSteps(messages, steps, from = 0) {
  if (!steps.length) return messages
  const at = turnGroup(messages, from)
  if (at < 0) return [...messages, { kind: 'tools', calls: steps }]
  const next = messages.slice()
  next[at] = { ...next[at], calls: [...next[at].calls, ...steps] }
  return next
}

/** A message's reasoning and text as steps of the group (empty ones are left out). */
const thinkSteps = (...texts) =>
  texts.filter((text) => text && text.trim()).map((text) => ({ id: `think-${++stepSeq}`, name: 'thinking', think: true, text: text.trim(), status: 'done' }))

const thinkingOf = (content) => (Array.isArray(content) ? content : []).filter((block) => block.type === 'thinking').map((block) => block.thinking).join('')
const hasToolCalls = (content) => Array.isArray(content) && content.some((block) => block.type === 'toolCall')

function endAssistant(state, message) {
  const text = textOf(message.content)
  const thinking = thinkingOf(message.content)
  const current = lastStreamingAssistant(state)
  const rest = current ? state.messages.slice(0, -1) : state.messages
  // Its reasoning is a step of the group; its text is a message, whether it answers or comes between tools.
  const grouped = addSteps(rest, thinkSteps(thinking), state.turnFrom)
  if (!text) return withMessages(state, grouped)
  const base = current ?? { kind: 'assistant', texts: {}, thoughts: {} }
  return withMessages(state, [...grouped, { ...base, text, thinking: '', done: true }])
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
  return withMessages(state, addSteps(state.messages, [call], state.turnFrom))
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

// ---- background work: the background extension's `aria-background` custom message, a note in the thread ---------

/** How a job ended, from the message's `details` ({ job, jobs, by, event }, extensions/background/index.ts). */
function backgroundOutcome(details) {
  if (details?.event === 'cut') return 'cut'
  if (details?.by === 'stop') return 'stopped'
  const job = details?.job ?? {}
  if (job.state === 'done') return job.conflict ? 'conflict' : 'done'
  if (job.state === 'exited') return 'exited'
  if (job.state === 'stopped' || job.state === 'cut') return job.state
  return 'failed'
}

/** The note for a shown `aria-background` custom message, or null for any other message (or a hidden one). */
function backgroundNote(message) {
  if (message?.role !== 'custom' || message.customType !== 'aria-background' || !message.display) return null
  const details = message.details ?? {}
  const outcome = backgroundOutcome(details)
  const title =
    outcome === 'cut' && Array.isArray(details.jobs) && details.jobs.length
      ? details.jobs.map((j) => j?.title ?? '').filter(Boolean).join(', ')
      : String(details.job?.title ?? '')
  const vars = outcome === 'exited' ? { title, exit: details.job?.exit ?? '?' } : { title }
  return { kind: 'note', label: 'background', key: `chat.bg.note.${outcome}`, vars }
}

const TONES = { warning: 'warn', error: 'error' }

/** The summary of what came before, or null: a compaction's (not rotated yet) or a rotation's seed message, whose
 * `details.summary` is the raw summary (its `content` is Pi's framing of it, which the page never shows). */
function earlierSummary(message) {
  if (message?.role === 'compactionSummary') return String(message.summary ?? '')
  if (message?.role === 'custom' && message.customType === 'aria-earlier') return String(message.details?.summary ?? '')
  return null
}

function fromHistory(messages) {
  let entries = []
  const calls = new Map()
  for (const [at, message] of (messages ?? []).entries()) {
    if (message.role === 'user') {
      entries.push(userEntry(textOf(message.content)))
    } else if (message.role === 'assistant') {
      const text = textOf(message.content)
      const thinking = thinkingOf(message.content)
      const toolCalls = (message.content ?? []).filter((block) => block.type === 'toolCall')
      if (toolCalls.length) {
        const made = toolCalls.map((block) => {
          const call = { id: block.id, name: block.name, args: block.arguments ?? {}, status: 'done', summary: toolSummary(block.name, block.arguments), result: '' }
          calls.set(block.id, call)
          return call
        })
        entries = addSteps(entries, thinkSteps(thinking))
        if (text) entries.push({ kind: 'assistant', text, thinking: '', done: true, texts: { 0: text }, thoughts: {} })
        entries = addSteps(entries, made)
      } else {
        entries = addSteps(entries, thinkSteps(thinking))
        if (text) entries.push({ kind: 'assistant', text, thinking: '', done: true, texts: { 0: text }, thoughts: {} })
      }
    } else if (earlierSummary(message) !== null) {
      // A stable id (the old session it summarises, or its place), so a history sent again keeps it open.
      entries.push({ kind: 'earlier', id: `earlier-${message.details?.from ?? at}`, summary: earlierSummary(message) })
    } else if (message.role === 'custom') {
      const note = backgroundNote(message)
      if (note) entries.push(note)
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
      // A new run starts a new group, even when its question's echo never became an entry.
      return { ...state, running: true, startedAt: Date.now(), phase: 'working', turnFrom: state.messages.length, stoppedByWatchdog: false }
    case 'agent_settled': {
      const settled = { ...state, running: false, startedAt: null, phase: 'working', delivered: [], stoppedByWatchdog: false }
      // The dock's watchdog stopped it: its notice already says why, and it was not the person.
      if (!event.aborted || state.stoppedByWatchdog) return settled
      return push(settled, { kind: 'note', label: 'aria', key: 'chat.stoppedByYou' })
    }
    case 'message_start':
      if (event.message?.role === 'assistant') {
        return push(state, { kind: 'assistant', text: '', thinking: '', done: false, texts: {}, thoughts: {} })
      }
      if (event.message?.role === 'user') {
        // A message that waited in the queue is now being answered: it moves into the thread. Pi takes it out of
        // the queue (a `queue_update`) just BEFORE announcing it, so `delivered` remembers what left the queue.
        const entry = userEntry(textOf(event.message.content))
        const text = entry.text
        const waiting = state.queue.findIndex((q) => q.text === text)
        if (waiting >= 0) return push({ ...state, queue: state.queue.filter((_, i) => i !== waiting) }, entry)
        const left = state.delivered.indexOf(text)
        if (left >= 0) return push({ ...state, delivered: state.delivered.filter((_, i) => i !== left) }, entry)
        return state // the echo of a message the page already showed
      }
      if (event.message?.role === 'custom') {
        const note = backgroundNote(event.message)
        return note ? push(state, note) : state
      }
      return state
    case 'queue_update': {
      const queue = [...(event.steering ?? []).map((text) => queueEntry(text, true)), ...(event.followUp ?? []).map((text) => queueEntry(text))]
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
      // One asked with /compact while nothing runs shows the working line too, until it ends.
      return state.running
        ? { ...state, phase: 'compacting context' }
        : { ...state, phase: 'compacting context', running: true, compacting: true, startedAt: Date.now() }
    case 'compaction_end': {
      const ended = state.compacting ? { ...state, running: false, compacting: false, startedAt: null } : state
      return event.aborted || !event.result
        ? { ...ended, phase: 'working' }
        : push({ ...ended, phase: 'working' }, { kind: 'divider', key: 'chat.compacted' })
    }
    case 'engine_exited':
      return push({ ...state, running: false }, notice('error', { key: 'chat.engine.exited' }))
    case 'command_failed': {
      // A refused prompt (Pi was compacting, a command that needed a queue) leaves the thread or the queue, and its
      // text goes back to the composer (`giveBack`), so nothing looks sent that was not.
      let next = state
      if (typeof event.text === 'string') {
        const queued = state.queue.findIndex((q) => q.text === event.text)
        const shown = state.messages.findLastIndex((m) => m.kind === 'user' && m.text === event.text)
        if (queued >= 0) next = { ...state, queue: state.queue.filter((_, i) => i !== queued) }
        else if (shown >= 0) next = withMessages(state, state.messages.filter((_, i) => i !== shown))
        next = { ...next, giveBack: event.text }
      }
      return push(next, notice('error', { text: event.error ?? event.command }))
    }
    case 'ui_notice': {
      const request = event.request ?? {}
      if (request.method === 'notify') {
        const tone = TONES[request.notifyType] ?? 'info'
        // The dock's own notices (the turn watchdog) name a key, so the page says them in its language.
        if (typeof request.key === 'string') {
          const next = request.key === 'chat.watchdog.stopped' ? { ...state, stoppedByWatchdog: true } : state
          return push(next, notice(tone, request.vars ? { key: request.key, vars: request.vars } : { key: request.key }))
        }
        return push(state, notice(tone, { text: request.message }))
      }
      // The router's route line (jev-router sends it as the status `aria-jev`).
      if (request.method === 'setStatus' && request.statusKey === 'aria-jev') {
        let route = null
        try {
          route = JSON.parse(request.statusText)
        } catch {
          return state
        }
        return typeof route?.mode === 'string' && route.mode ? push(state, { kind: 'jev', mode: route.mode, action: typeof route.action === 'string' ? route.action : '' }) : state
      }
      return state
    }
    case 'history': {
      // The conversation so far. It can arrive after the owner already typed (a cold start): what Pi had before that
      // goes in front, and so do notices that came first. While only a reply streams (no message of the owner on the
      // page), it is left alone.
      const past = fromHistory(event.messages)
      // Pages shown before it are dropped: the paging starts again from this history, whose summary replaces any shown.
      const shown = state.messages.slice(state.earlier.count).filter((m) => m.kind !== 'earlier' || !past.some((p) => p.kind === 'earlier'))
      state = { ...withMessages(state, shown), loaded: true, turnFrom: Math.max(0, (state.turnFrom ?? 0) - state.earlier.count), earlier: { ...NO_EARLIER, more: past[0]?.kind === 'earlier', loading: state.earlier.loading, stale: state.earlier.loading } }
      if (!state.messages.length) return withMessages(state, past)
      const first = state.messages.find((m) => m.kind === 'user')
      if (!first) return state.running || state.messages.some((m) => m.kind === 'assistant' && !m.done) ? state : withMessages(state, [...past, ...state.messages])
      const at = past.findLastIndex((m) => m.kind === 'user' && m.text === first.text)
      return withMessages(state, [...(at >= 0 ? past.slice(0, at) : past), ...state.messages])
    }
    case 'earlier': {
      // A page of older messages goes above what is shown; indexes into the thread move with it.
      // A page asked before a history came again belongs to the paging before it: dropped (ask again).
      if (state.earlier.stale) return { ...state, earlier: { ...state.earlier, loading: false, stale: false } }
      const page = fromHistory(event.messages)
      return {
        ...withMessages(state, [...page, ...state.messages]),
        turnFrom: (state.turnFrom ?? 0) + page.length,
        earlier: { more: event.more === true, cursor: event.cursor ?? null, loading: false, count: state.earlier.count + page.length, stale: false },
      }
    }
    case 'session_reset':
      return { ...withMessages(state, []), queue: [], delivered: [], dialogs: [], earlier: { ...NO_EARLIER } }
    default:
      return state
  }
}

/** The dialog's card, if its title is one; a button may only send an option the dialog itself accepts (a title cannot
 * show "No" and answer "allow everything"). A card whose options were all foreign falls back to the plain dialog. */
function checkedCard(request) {
  const card = parseCardTitle(request?.title)
  if (!card || !Array.isArray(card.options) || !card.options.length) return card
  const accepted = Array.isArray(request.options) ? request.options : []
  const options = card.options.filter((o) => accepted.includes(o.value))
  return options.length ? { ...card, options } : null
}

/** `message` is a dock message (see crates/aria-core/src/bridge.rs). */
export function reduce(state, message) {
  switch (message.type) {
    case 'chatEvent':
      return reduceEvent(state, message.event ?? {})
    case 'chatState': {
      // A compaction Pi is doing with no run of its own (one Aria asked for) keeps the working line: a later state, or
      // the one a reloaded page starts from, must not hide it.
      const compacting = Boolean(message.compacting)
      const next = {
        ...state,
        workdir: message.workdir,
        running: message.running || compacting,
        compacting,
        phase: compacting ? 'compacting context' : state.phase,
        startedAt: compacting && !state.compacting ? (state.startedAt ?? Date.now()) : state.startedAt,
        model: message.model ?? null,
        thinking: message.thinking ?? null,
        context: message.context ?? null,
        problem: message.problem ?? null,
      }
      // The model changing under a conversation leaves a note (the first report is not a change, and neither is Pi
      // resuming a session on another model than the settings' daily one, before anything is on the page).
      return state.messages.length && state.model && next.model && state.model !== next.model
        ? push(next, { kind: 'note', label: 'model', text: modelShort(next.model) })
        : next
    }
    case 'chatAgents':
      return { ...state, agents: Array.isArray(message.agents) ? message.agents : state.agents }
    case 'chatQuota':
      return { ...state, quota: Array.isArray(message.windows) ? message.windows : state.quota }
    case 'chatEngine':
      return message.state === 'stopped' || message.state === 'exited'
        ? { ...state, engine: message.state, running: false, startedAt: null, loaded: true }
        : { ...state, engine: message.state }
    case 'chatDialog': {
      const dialog = { id: message.id, method: message.method, request: message.request, card: checkedCard(message.request) }
      return { ...state, dialogs: [...state.dialogs.filter((d) => d.id !== message.id), dialog] }
    }
    case 'chatDialogClosed':
      return { ...state, dialogs: state.dialogs.filter((d) => d.id !== message.id) }
    default:
      return state
  }
}

// A file the page attached is an image by its type, as the dock's lines say for history.
const marked = (file) => ('image' in file ? file : { ...file, image: String(file.mime ?? '').startsWith('image/') })
const withFiles = (entry, files) => (files?.length ? { ...entry, files: files.map(marked) } : entry)
/** `files`: what the message carries ([{ name, mime, size }]), drawn as chips before the dock has answered. */
export const addUser = (state, text, files) => push(state, withFiles({ kind: 'user', text }, files))
/** "Show earlier": `[state, message]`, the message to post (`chatEarlier`, with the last cursor once there is one), or
 * null while a page is on its way or when there is nothing older. */
export function askEarlier(state) {
  const { more, loading, cursor } = state.earlier
  if (!more || loading) return [state, null]
  return [{ ...state, earlier: { ...state.earlier, loading: true } }, cursor ? { type: 'chatEarlier', cursor } : { type: 'chatEarlier' }]
}
/** A notice from the page itself (an attachment refused). */
export const addNotice = (state, tone, key, vars) => push(state, notice(tone, vars ? { key, vars } : { key }))
/** A one-line fact in the thread that did not come from Pi (a provider was connected). */
export const addNote = (state, label, text) => push(state, { kind: 'note', label, text })
/** `steer`: the message goes in at the next gap between the run's steps (Pi's steering), not when the run ends. */
export const addQueued = (state, text, steer = false, files) => ({ ...state, queue: [...state.queue, withFiles(steer ? { text, steer } : { text }, files)] })
/** The composer took back a refused message's text. */
export const tookBack = (state) => ({ ...state, giveBack: null })
export const clearQueue = (state, at) => ({ ...state, queue: at === undefined ? [] : state.queue.filter((_, i) => i !== at) })

/** The question card's answers as the answers card draws them: its header, what was picked, whether it was only talked about. */
export function answersForCard(questions, answers) {
  return (answers ?? []).map((a) => {
    const asked = (questions ?? []).find((q) => q.question === a.question)
    return { header: asked?.header || a.question, selected: a.answers ?? [], talk: a.talk === true }
  })
}

const CARD_LABEL = { sandbox: 'sandbox', write: 'sandbox', connector: 'connector', quota: 'model', question: 'ask' }

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
  if (!choice) return next
  // The known codes have a sentence ("allowed once"); an extension's own option is shown as it was written.
  const known = ['allow', 'always', 'deny', 'free', 'keep'].includes(choice)
  return push(next, known ? { kind: 'note', label: CARD_LABEL[kind] ?? kind, key: `chat.result.${choice}`, text: String(choice) } : { kind: 'note', label: CARD_LABEL[kind] ?? kind, text: String(choice) })
}
