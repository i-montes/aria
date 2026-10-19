// Run: bun test ui
import { describe, expect, test } from 'bun:test'
import { addQueued, addUser, answerDialog, answersForCard, clearQueue, createChat, reduce, toolSummary } from './state.js'

const ev = (event) => ({ type: 'chatEvent', event })
const run = (state, ...messages) => messages.reduce(reduce, state)
const delta = (index, text) => ev({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', contentIndex: index, delta: text } })
const thinkingDelta = (index, text) => ev({ type: 'message_update', assistantMessageEvent: { type: 'thinking_delta', contentIndex: index, delta: text } })
const assistantStart = ev({ type: 'message_start', message: { role: 'assistant', content: [] } })
const assistantEnd = (content) => ev({ type: 'message_end', message: { role: 'assistant', content } })
const toolStart = (id, name, args = {}) => ev({ type: 'tool_execution_start', toolCallId: id, toolName: name, args })
const toolEnd = (id, name, isError = false, text = 'ok') =>
  ev({ type: 'tool_execution_end', toolCallId: id, toolName: name, isError, result: { content: [{ type: 'text', text }] } })

describe('a run', () => {
  test('agent_start sets running and agent_settled clears it', () => {
    let state = run(createChat(), ev({ type: 'agent_start' }))
    expect(state.running).toBe(true)
    state = reduce(state, ev({ type: 'agent_settled', aborted: false }))
    expect(state.running).toBe(false)
  })

  test('text deltas build one assistant message and message_end replaces it with the final text', () => {
    let state = run(createChat(), assistantStart, delta(0, 'Hel'), delta(0, 'lo'))
    expect(state.messages).toEqual([{ kind: 'assistant', text: 'Hello', thinking: '', done: false, texts: { 0: 'Hello' }, thoughts: {} }])
    state = reduce(state, assistantEnd([{ type: 'text', text: 'Hello there' }]))
    expect(state.messages).toHaveLength(1)
    expect(state.messages[0]).toMatchObject({ kind: 'assistant', text: 'Hello there', done: true })
  })

  test('two text blocks join in index order', () => {
    const state = run(createChat(), assistantStart, delta(1, 'B'), delta(0, 'A'))
    expect(state.messages[0].text).toBe('AB')
  })

  test('thinking deltas go to thinking, not to the text', () => {
    const state = run(createChat(), assistantStart, thinkingDelta(0, 'hmm'), delta(1, 'answer'))
    expect(state.messages[0]).toMatchObject({ thinking: 'hmm', text: 'answer' })
  })

  test('an assistant message with nothing to show is dropped when it ends', () => {
    const state = run(createChat(), assistantStart, assistantEnd([{ type: 'toolCall', id: 'c1', name: 'bash', arguments: {} }]))
    expect(state.messages).toEqual([])
  })

  test('the echo of the user message is not added twice', () => {
    let state = addUser(createChat(), 'hi')
    state = reduce(state, ev({ type: 'message_start', message: { role: 'user', content: 'hi' } }))
    state = reduce(state, ev({ type: 'message_end', message: { role: 'user', content: 'hi' } }))
    expect(state.messages).toEqual([{ kind: 'user', text: 'hi' }])
  })
})

describe('tool calls', () => {
  test('consecutive calls group in order and take their results', () => {
    const state = run(
      createChat(),
      toolStart('a', 'bash', { command: 'ls -la\nsecond' }),
      toolStart('b', 'read', { path: 'src/main.rs' }),
      toolEnd('a', 'bash'),
      toolEnd('b', 'read', true, 'boom'),
    )
    expect(state.messages).toHaveLength(1)
    const [group] = state.messages
    expect(group.kind).toBe('tools')
    expect(group.calls.map((c) => [c.id, c.status, c.summary])).toEqual([
      ['a', 'done', 'ls -la'],
      ['b', 'error', 'src/main.rs'],
    ])
    expect(group.calls[1].result).toBe('boom')
  })

  test('an assistant message after tools starts a new entry', () => {
    const state = run(createChat(), toolStart('a', 'bash'), toolEnd('a', 'bash'), assistantStart, delta(0, 'done'))
    expect(state.messages.map((m) => m.kind)).toEqual(['tools', 'assistant'])
  })

  test('tools after an assistant message start a new group', () => {
    const state = run(createChat(), assistantStart, delta(0, 'looking'), assistantEnd([{ type: 'text', text: 'looking' }]), toolStart('a', 'ls'))
    expect(state.messages.map((m) => m.kind)).toEqual(['assistant', 'tools'])
  })

  test('toolSummary cases', () => {
    expect(toolSummary('bash', { command: 'git status\nmore' })).toBe('git status')
    expect(toolSummary('read', { path: 'a/b.rs' })).toBe('a/b.rs')
    expect(toolSummary('write', { file_path: 'c.rs' })).toBe('c.rs')
    expect(toolSummary('edit', { path: 'd.rs' })).toBe('d.rs')
    expect(toolSummary('grep', { pattern: 'x' })).toBe('grep')
    expect(toolSummary('bash', {})).toBe('bash')
  })
})

describe('dialogs', () => {
  test('a dialog is queued and closed by chatDialogClosed', () => {
    let state = reduce(createChat(), { type: 'chatDialog', id: 'u1', method: 'select', request: { title: 'Allow?', options: ['Allow'] }, hidden: false })
    expect(state.dialogs).toEqual([{ id: 'u1', method: 'select', request: { title: 'Allow?', options: ['Allow'] }, card: null }])
    state = reduce(state, { type: 'chatDialogClosed', id: 'u1' })
    expect(state.dialogs).toEqual([])
  })

  test('the same dialog arriving twice (a replay) is kept once', () => {
    const open = { type: 'chatDialog', id: 'u1', method: 'confirm', request: { title: 'x' }, hidden: false }
    expect(run(createChat(), open, open).dialogs).toHaveLength(1)
  })

  test('answerDialog removes it at once', () => {
    const state = run(createChat(), { type: 'chatDialog', id: 'u1', method: 'confirm', request: {}, hidden: false })
    expect(answerDialog(state, 'u1').dialogs).toEqual([])
  })
})

describe('the engine and notices', () => {
  test('engine_exited adds a notice and stops running', () => {
    const state = run(createChat(), ev({ type: 'agent_start' }), ev({ type: 'engine_exited', code: 3 }))
    expect(state.running).toBe(false)
    expect(state.messages.at(-1)).toMatchObject({ kind: 'notice', tone: 'error', key: 'chat.engine.exited' })
  })

  test('auto_retry_start adds a notice with the attempt', () => {
    const state = run(createChat(), ev({ type: 'auto_retry_start', attempt: 1, maxAttempts: 3, delayMs: 2000, errorMessage: '529' }))
    expect(state.messages.at(-1)).toMatchObject({ kind: 'notice', tone: 'warn', key: 'chat.retry', vars: { attempt: 1, max: 3 } })
  })

  test('a failed command shows its error', () => {
    const state = run(createChat(), ev({ type: 'command_failed', command: 'prompt', error: 'Agent is already streaming' }))
    expect(state.messages.at(-1)).toMatchObject({ kind: 'notice', tone: 'error', text: 'Agent is already streaming' })
  })

  test('extension notices become notices; other ui requests are ignored', () => {
    let state = run(createChat(), ev({ type: 'ui_notice', request: { method: 'notify', message: 'careful', notifyType: 'warning' } }))
    expect(state.messages.at(-1)).toMatchObject({ kind: 'notice', tone: 'warn', text: 'careful' })
    state = reduce(state, ev({ type: 'ui_notice', request: { method: 'setStatus', statusKey: 'x', statusText: 'y' } }))
    expect(state.messages).toHaveLength(1)
  })

  test('chatState sets the folder, model, level, problem and running', () => {
    const state = reduce(createChat(), { type: 'chatState', workdir: '/home/u/Projects', running: true, model: 'opencode-go/glm-5.3', thinking: 'high', problem: 'pi-missing' })
    expect(state).toMatchObject({ workdir: '/home/u/Projects', running: true, model: 'opencode-go/glm-5.3', thinking: 'high', problem: 'pi-missing' })
    expect(reduce(state, { type: 'chatState', workdir: '/home/u/Projects', running: false, model: null, thinking: null, problem: null }).problem).toBeNull()
  })

  test('chatEngine records the engine state', () => {
    expect(reduce(createChat(), { type: 'chatEngine', state: 'ready' }).engine).toBe('ready')
  })

  test('session_reset clears the conversation', () => {
    const state = run(addUser(createChat(), 'hi'), ev({ type: 'session_reset' }))
    expect(state.messages).toEqual([])
  })

  test('unknown events and messages are ignored', () => {
    const before = createChat()
    expect(reduce(before, ev({ type: 'something_new' }))).toEqual(before)
    expect(reduce(before, { type: 'saved', id: 1 })).toEqual(before)
  })
})

describe('history', () => {
  const history = [
    { role: 'user', content: 'list files' },
    { role: 'assistant', content: [{ type: 'thinking', thinking: 'ok' }, { type: 'text', text: 'Sure.' }, { type: 'toolCall', id: 't1', name: 'bash', arguments: { command: 'ls' } }] },
    { role: 'toolResult', toolCallId: 't1', toolName: 'bash', isError: false, content: [{ type: 'text', text: 'a b' }] },
    { role: 'assistant', content: [{ type: 'text', text: 'There they are.' }] },
    { role: 'user', content: [{ type: 'text', text: 'thanks' }] },
  ]

  test('past messages become entries in order', () => {
    const state = reduce(createChat(), ev({ type: 'history', messages: history }))
    expect(state.messages.map((m) => m.kind)).toEqual(['user', 'assistant', 'tools', 'assistant', 'user'])
    expect(state.messages[0].text).toBe('list files')
    expect(state.messages[1]).toMatchObject({ text: 'Sure.', thinking: 'ok', done: true })
    expect(state.messages[2].calls[0]).toMatchObject({ id: 't1', status: 'done', summary: 'ls', result: 'a b' })
    expect(state.messages[4].text).toBe('thanks')
  })

  test('history never replaces a conversation already on the page (it can arrive after the first message)', () => {
    const state = run(addUser(createChat(), 'just typed'), ev({ type: 'history', messages: history }))
    expect(state.messages).toEqual([{ kind: 'user', text: 'just typed' }])
  })

  test('history does not cut a reply that is already streaming', () => {
    const state = run(createChat(), ev({ type: 'message_start', message: { role: 'assistant' } }), ev({ type: 'history', messages: history }), ev({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', contentIndex: 0, delta: 'hi' } }))
    expect(state.messages.at(-1)).toMatchObject({ kind: 'assistant', text: 'hi' })
    expect(state.messages).toHaveLength(1)
  })

  test('a failed tool result marks the call as an error', () => {
    const failed = [history[1], { ...history[2], isError: true }]
    expect(reduce(createChat(), ev({ type: 'history', messages: failed })).messages[1].calls[0].status).toBe('error')
  })
})

describe('queued messages', () => {
  test('a message queued behind a run waits in the queue and not in the thread', () => {
    const state = addQueued(run(createChat(), ev({ type: 'agent_start' })), 'then run the tests')
    expect(state.queue).toEqual([{ text: 'then run the tests' }])
    expect(state.messages).toEqual([])
  })

  test('queue_update replaces the queue with what Pi reports', () => {
    const state = run(addQueued(createChat(), 'a'), ev({ type: 'queue_update', steering: [], followUp: ['a', 'b'] }))
    expect(state.queue).toEqual([{ text: 'a' }, { text: 'b' }])
    expect(reduce(state, ev({ type: 'queue_update', steering: [], followUp: [] })).queue).toEqual([])
  })

  test('when Pi delivers a queued message it moves into the thread', () => {
    const state = run(
      createChat(),
      ev({ type: 'queue_update', steering: [], followUp: ['next one'] }),
      ev({ type: 'message_start', message: { role: 'user', content: 'next one' } }),
    )
    expect(state.queue).toEqual([])
    expect(state.messages).toEqual([{ kind: 'user', text: 'next one' }])
  })

  test('Pi removes a message from the queue BEFORE it announces it: it still reaches the thread', () => {
    // agent-session.js emits queue_update (without the message) and then message_start for it
    const state = run(
      addQueued(createChat(), 'then this'),
      ev({ type: 'queue_update', steering: [], followUp: [] }),
      ev({ type: 'message_start', message: { role: 'user', content: 'then this' } }),
      ev({ type: 'message_end', message: { role: 'user', content: 'then this' } }),
    )
    expect(state.queue).toEqual([])
    expect(state.messages).toEqual([{ kind: 'user', text: 'then this' }])
  })

  test('the same text queued twice is delivered twice, and a direct send is still not doubled', () => {
    let state = run(createChat(), ev({ type: 'queue_update', steering: [], followUp: ['again', 'again'] }))
    state = run(state, ev({ type: 'queue_update', steering: [], followUp: ['again'] }), ev({ type: 'message_start', message: { role: 'user', content: 'again' } }))
    state = run(state, ev({ type: 'queue_update', steering: [], followUp: [] }), ev({ type: 'message_start', message: { role: 'user', content: 'again' } }))
    expect(state.messages.map((m) => m.text)).toEqual(['again', 'again'])
    const direct = run(addUser(createChat(), 'hello'), ev({ type: 'message_start', message: { role: 'user', content: 'hello' } }))
    expect(direct.messages).toHaveLength(1)
  })

  test('what left the queue but was never announced does not linger past the end of the run', () => {
    let state = run(addQueued(createChat(), 'cancelled'), ev({ type: 'queue_update', steering: [], followUp: [] }))
    expect(state.delivered).toEqual(['cancelled'])
    state = run(state, ev({ type: 'agent_settled', aborted: false }))
    expect(state.delivered).toEqual([])
  })

  test('cancelling one queued message keeps the others, in order', () => {
    const state = clearQueue(run(createChat(), ev({ type: 'queue_update', steering: [], followUp: ['a', 'b', 'c'] })), 1)
    expect(state.queue.map((q) => q.text)).toEqual(['a', 'c'])
  })

  test('cancelling the queue empties it', () => {
    expect(clearQueue(addQueued(createChat(), 'a')).queue).toEqual([])
  })
})

describe('phase, elapsed time and the notes around a run', () => {
  test('the working phase follows the tool that runs', () => {
    const phase = (name, args = {}) => run(createChat(), ev({ type: 'agent_start' }), toolStart('a', name, args)).phase
    expect(phase('bash')).toBe('running command')
    expect(phase('read')).toBe('reading files')
    expect(phase('write')).toBe('writing files')
    expect(phase('edit')).toBe('writing files')
    expect(phase('grep')).toBe('reading files')
    expect(phase('mcp__engram__mem_search')).toBe('searching memory')
    expect(phase('mcp__tavily__tavily_search')).toBe('searching the web')
    expect(phase('mcp__notion__notion-search')).toBe('checking Notion')
    expect(phase('parallel')).toBe('delegating')
    expect(phase('something-new')).toBe('working')
  })

  test('thinking deltas set the phase to thinking, and it resets when the run settles', () => {
    let state = run(createChat(), ev({ type: 'agent_start' }), assistantStart, thinkingDelta(0, 'hmm'))
    expect(state.phase).toBe('thinking')
    state = reduce(state, ev({ type: 'agent_settled', aborted: false }))
    expect(state.phase).toBe('working')
    expect(state.startedAt).toBeNull()
  })

  test('the run remembers when it started', () => {
    const before = Date.now()
    const state = run(createChat(), ev({ type: 'agent_start' }))
    expect(state.startedAt).toBeGreaterThanOrEqual(before)
  })

  test('compaction shows its phase and leaves a divider when it ends', () => {
    let state = run(createChat(), ev({ type: 'compaction_start', reason: 'threshold' }))
    expect(state.phase).toBe('compacting context')
    state = reduce(state, ev({ type: 'compaction_end', reason: 'threshold', result: { tokensBefore: 150000 }, aborted: false }))
    expect(state.messages.at(-1)).toMatchObject({ kind: 'divider', key: 'chat.compacted' })
  })

  test('a run you stopped says so', () => {
    const state = run(createChat(), ev({ type: 'agent_start' }), ev({ type: 'agent_settled', aborted: true }))
    expect(state.messages.at(-1)).toMatchObject({ kind: 'note', label: 'aria', key: 'chat.stoppedByYou' })
  })

  test('a model change leaves a note', () => {
    const state = run(createChat(), { type: 'chatState', workdir: '/w', running: false, model: 'a/one', thinking: null, context: null, problem: null }, { type: 'chatState', workdir: '/w', running: false, model: 'a/two', thinking: null, context: null, problem: null })
    expect(state.messages.at(-1)).toMatchObject({ kind: 'note', label: 'model', text: 'two' })
  })

  test('the first model report is not a change', () => {
    const state = run(createChat(), { type: 'chatState', workdir: '/w', running: false, model: 'a/one', thinking: null, context: null, problem: null })
    expect(state.messages).toEqual([])
  })

  test('the context percent is kept from the state', () => {
    expect(reduce(createChat(), { type: 'chatState', workdir: '/w', running: false, model: null, thinking: null, context: 42, problem: null }).context).toBe(42)
  })
})

describe('the route line from the router', () => {
  test('a jev event becomes a route line before the next answer', () => {
    const state = run(createChat(), ev({ type: 'aria_jev', mode: 'focused task', action: 'searches memory' }))
    expect(state.messages.at(-1)).toEqual({ kind: 'jev', mode: 'focused task', action: 'searches memory' })
  })
})

describe('a card sent by an extension (a dialog whose title is JSON)', () => {
  test('the dialog keeps its parsed card so the view can choose the renderer', () => {
    const title = JSON.stringify({ aria: 1, kind: 'quota', percent: 98, freeModel: 'opencode-go/cat-jam-free', countdown: 300 })
    const state = reduce(createChat(), { type: 'chatDialog', id: 'u1', method: 'select', request: { title, options: ['Switch', 'Keep'] }, hidden: false })
    expect(state.dialogs[0].card).toMatchObject({ kind: 'quota', percent: 98 })
  })

  test('an ordinary dialog has no card', () => {
    const state = reduce(createChat(), { type: 'chatDialog', id: 'u1', method: 'select', request: { title: 'Allow?', options: ['Yes'] }, hidden: false })
    expect(state.dialogs[0].card).toBeNull()
  })

  test('answering a card leaves a resolved note with the chosen option', () => {
    const title = JSON.stringify({ aria: 1, kind: 'sandbox', summary: 'rm -rf build' })
    let state = reduce(createChat(), { type: 'chatDialog', id: 'u1', method: 'select', request: { title, options: ['Allow once', 'Block'] }, hidden: false })
    state = answerDialog(state, 'u1', 'deny')
    expect(state.messages.at(-1)).toMatchObject({ kind: 'note', label: 'sandbox', key: 'chat.result.deny' })
  })
})

describe('what the question card leaves in the thread', () => {
  const questions = [
    { question: 'Which fallback?', header: 'Fallback', options: [{ label: 'Git hash' }, { label: 'Version' }] },
    { question: 'Which checks?', header: 'Checks', multi: true, options: [{ label: 'Unit' }, { label: 'Lint' }] },
  ]

  test('answers carry the card\'s header and the picked labels', () => {
    const rows = answersForCard(questions, [{ question: 'Which fallback?', answers: ['Git hash'] }, { question: 'Which checks?', answers: ['Unit', 'Lint'] }])
    expect(rows).toEqual([
      { header: 'Fallback', selected: ['Git hash'], talk: false },
      { header: 'Checks', selected: ['Unit', 'Lint'], talk: false },
    ])
  })

  test('a question marked "talk about this" says so, and an unknown question keeps its own text as the header', () => {
    const rows = answersForCard(questions, [{ question: 'Which fallback?', answers: [], talk: true }, { question: 'Something else?', answers: ['x'] }])
    expect(rows[0]).toMatchObject({ header: 'Fallback', selected: [], talk: true })
    expect(rows[1]).toMatchObject({ header: 'Something else?', selected: ['x'] })
  })

  test('answering a question card stores those rows', () => {
    const title = JSON.stringify({ aria: 1, kind: 'question', questions })
    let state = reduce(createChat(), { type: 'chatDialog', id: 'u1', method: 'input', request: { title }, hidden: false })
    state = answerDialog(state, 'u1', [{ header: 'Fallback', selected: ['Git hash'], talk: false }])
    expect(state.messages.at(-1)).toEqual({ kind: 'answers', answers: [{ header: 'Fallback', selected: ['Git hash'], talk: false }] })
  })
})
