// Run: bun test ui
import { describe, expect, test } from 'bun:test'
import { addNotice, addQueued, addUser, answerDialog, answersForCard, askEarlier, clearQueue, createChat, reduce, toolSummary } from './state.js'

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

  test('a notice of the dock carries a translation key and its values, not English text', () => {
    const request = { method: 'notify', key: 'chat.watchdog.stopped', vars: { minutes: 10 }, notifyType: 'warning' }
    const state = run(createChat(), ev({ type: 'ui_notice', request }))
    expect(state.messages.at(-1)).toEqual({ kind: 'notice', tone: 'warn', key: 'chat.watchdog.stopped', vars: { minutes: 10 } })
  })

  test('chatState sets the folder, model, level, problem and running', () => {
    const state = reduce(createChat(), { type: 'chatState', workdir: '/home/u/Projects', running: true, model: 'opencode-go/glm-5.3', thinking: 'high', problem: 'pi-missing' })
    expect(state).toMatchObject({ workdir: '/home/u/Projects', running: true, model: 'opencode-go/glm-5.3', thinking: 'high', problem: 'pi-missing' })
    expect(reduce(state, { type: 'chatState', workdir: '/home/u/Projects', running: false, model: null, thinking: null, problem: null }).problem).toBeNull()
  })

  test('a compaction the state reports keeps the working line, even after a reload', () => {
    // Aria compacts a large session on its own, with no run: the state says so, and a later state does not hide it.
    const fresh = reduce(createChat(), { type: 'chatState', workdir: '/w', running: false, compacting: true, model: null, thinking: null, problem: null })
    expect(fresh).toMatchObject({ running: true, compacting: true, phase: 'compacting context' })
    const started = reduce(createChat(), { type: 'chatEvent', event: { type: 'compaction_start' } })
    const again = reduce(started, { type: 'chatState', workdir: '/w', running: false, compacting: true, model: null, thinking: null, problem: null })
    expect(again).toMatchObject({ running: true, compacting: true, phase: 'compacting context' })
    expect(again.startedAt).toBe(started.startedAt)
    const ended = reduce(reduce(again, { type: 'chatEvent', event: { type: 'compaction_end', result: { summary: 's' } } }), { type: 'chatState', workdir: '/w', running: false, compacting: false, model: null, thinking: null, problem: null })
    expect(ended).toMatchObject({ running: false, compacting: false })
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
    expect(state.messages.map((m) => m.kind)).toEqual(['user', 'tools', 'assistant', 'tools', 'assistant', 'user'])
    expect(state.messages[0].text).toBe('list files')
    // the reasoning is a step of the group; the text written before the tool is said to the reader, and splits the group
    expect(state.messages[1].calls.map((c) => (c.think ? c.text : c.id))).toEqual(['ok'])
    expect(state.messages[2]).toMatchObject({ kind: 'assistant', text: 'Sure.', done: true })
    expect(state.messages[3].calls[0]).toMatchObject({ id: 't1', status: 'done', summary: 'ls', result: 'a b' })
    expect(state.messages[4]).toMatchObject({ kind: 'assistant', text: 'There they are.', done: true })
    expect(state.messages[5].text).toBe('thanks')
  })

  test('history never replaces a conversation already on the page: it goes in front of it', () => {
    const state = run(addUser(createChat(), 'just typed'), ev({ type: 'history', messages: history }))
    expect(state.messages.at(-1)).toEqual({ kind: 'user', text: 'just typed' })
    expect(state.messages[0]).toEqual({ kind: 'user', text: 'list files' })
  })

  test('history does not cut a reply that is already streaming', () => {
    const state = run(createChat(), ev({ type: 'message_start', message: { role: 'assistant' } }), ev({ type: 'history', messages: history }), ev({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', contentIndex: 0, delta: 'hi' } }))
    expect(state.messages.at(-1)).toMatchObject({ kind: 'assistant', text: 'hi' })
    expect(state.messages).toHaveLength(1)
  })

  test('a resumed session on another model than the daily one keeps its history (no note before it)', () => {
    const guess = { type: 'chatState', workdir: '/w', running: false, model: 'opencode-go/deepseek-v4.1-flash' }
    const real = { ...guess, model: 'opencode-go/mimo-v2.6-pro' }
    const state = run(createChat(), guess, real, ev({ type: 'history', messages: history }))
    expect(state.messages.map((m) => m.kind)).toEqual(['user', 'tools', 'assistant', 'tools', 'assistant', 'user'])
  })

  test('a notice that arrives before the history stays after it', () => {
    const notice = ev({ type: 'ui_notice', request: { method: 'notify', message: 'careful', notifyType: 'warning' } })
    const state = run(createChat(), notice, ev({ type: 'history', messages: history }))
    expect(state.messages.map((m) => m.kind)).toEqual(['user', 'tools', 'assistant', 'tools', 'assistant', 'user', 'notice'])
  })

  test('a failed tool result marks the call as an error', () => {
    const failed = [history[1], { ...history[2], isError: true }]
    const calls = reduce(createChat(), ev({ type: 'history', messages: failed })).messages.flatMap((m) => m.calls ?? [])
    expect(calls.find((c) => !c.think).status).toBe('error')
  })
})

describe('queued messages', () => {
  test('a message queued behind a run waits in the queue and not in the thread', () => {
    const state = addQueued(run(createChat(), ev({ type: 'agent_start' })), 'then run the tests')
    expect(state.queue).toEqual([{ text: 'then run the tests' }])
    expect(state.messages).toEqual([])
  })

  test('a message for the next gap is marked so, and Pi\'s steering list keeps the mark', () => {
    let state = addQueued(run(createChat(), ev({ type: 'agent_start' })), 'stop and look at the logs', true)
    expect(state.queue).toEqual([{ text: 'stop and look at the logs', steer: true }])
    state = reduce(state, ev({ type: 'queue_update', steering: ['stop and look at the logs'], followUp: ['later'] }))
    expect(state.queue).toEqual([{ text: 'stop and look at the logs', steer: true }, { text: 'later' }])
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

  test('a compaction asked while idle shows the working line until it ends', () => {
    let state = run(createChat(), ev({ type: 'compaction_start', reason: 'manual' }))
    expect(state.running).toBe(true)
    expect(state.phase).toBe('compacting context')
    expect(Number.isFinite(state.startedAt)).toBe(true)
    state = reduce(state, ev({ type: 'compaction_end', reason: 'manual', result: { tokensBefore: 150000 }, aborted: false }))
    expect(state.running).toBe(false)
    expect(state.messages.at(-1)).toMatchObject({ kind: 'divider', key: 'chat.compacted' })
  })

  test('a compaction inside a run leaves the run going', () => {
    let state = run(createChat(), ev({ type: 'agent_start' }), ev({ type: 'compaction_start', reason: 'threshold' }))
    state = reduce(state, ev({ type: 'compaction_end', reason: 'threshold', aborted: true }))
    expect(state.running).toBe(true)
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

  test('a run the watchdog stopped says so once, not also "stopped by you"; the next stop is yours again', () => {
    const watchdog = ev({ type: 'ui_notice', request: { method: 'notify', key: 'chat.watchdog.stopped', vars: { minutes: 10 }, notifyType: 'warning' } })
    let state = run(createChat(), ev({ type: 'agent_start' }), watchdog, ev({ type: 'agent_settled', aborted: true }))
    expect(state.messages.at(-1)).toMatchObject({ kind: 'notice', key: 'chat.watchdog.stopped' })
    expect(state.messages.some((m) => m.key === 'chat.stoppedByYou')).toBe(false)
    state = run(state, ev({ type: 'agent_start' }), ev({ type: 'agent_settled', aborted: true }))
    expect(state.messages.at(-1)).toMatchObject({ kind: 'note', key: 'chat.stoppedByYou' })
  })

  test('a model change under a conversation leaves a note', () => {
    const one = run(createChat(), { type: 'chatState', workdir: '/w', running: false, model: 'a/one', thinking: null, context: null, problem: null })
    const state = run(addUser(one, 'hi'), { type: 'chatState', workdir: '/w', running: false, model: 'a/two', thinking: null, context: null, problem: null })
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

describe('the provider quota', () => {
  const windows = [{ id: '5h', label: '5h', percent: 42, resetsAt: null }, { id: 'week', label: 'Week', percent: 17, resetsAt: null }]

  test('chatQuota stores the windows for the session', () => {
    expect(reduce(createChat(), { type: 'chatQuota', windows }).quota).toEqual(windows)
  })

  test('an empty answer clears them (a model whose provider has nothing to read)', () => {
    const state = run(createChat(), { type: 'chatQuota', windows }, { type: 'chatQuota', windows: [] })
    expect(state.quota).toEqual([])
  })

  test('a new conversation keeps the quota (it is the account\'s, not the conversation\'s)', () => {
    const state = run(createChat(), { type: 'chatQuota', windows }, ev({ type: 'session_reset' }))
    expect(state.quota).toEqual(windows)
  })

  test('a malformed message is ignored', () => {
    expect(reduce(createChat(), { type: 'chatQuota', windows: 'nope' }).quota).toEqual([])
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
    state = reduce(state, { type: 'chatDialog', id: 'u2', method: 'select', request: { title, options: ['Allow git for this session'] }, hidden: false })
    state = answerDialog(state, 'u2', 'Allow git for this session')
    expect(state.messages.at(-1)).toEqual({ kind: 'note', label: 'sandbox', text: 'Allow git for this session' })
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

describe('review fixes before 2b', () => {
  test('a refused prompt leaves the thread, and the page learns the text to give back', () => {
    let state = addUser(createChat(), 'hello')
    state = reduce(state, ev({ type: 'command_failed', command: 'prompt', error: 'Cannot submit a prompt while compaction is in progress', text: 'hello' }))
    expect(state.messages.map((m) => m.kind)).toEqual(['notice'])
    expect(state.giveBack).toBe('hello')
  })

  test('a refused queued prompt leaves the queue', () => {
    let state = addQueued(run(createChat(), ev({ type: 'agent_start' })), 'later')
    state = reduce(state, ev({ type: 'command_failed', command: 'prompt', error: 'no', text: 'later' }))
    expect(state.queue).toEqual([])
    expect(state.giveBack).toBe('later')
  })

  test('a failure without a text changes nothing but the notice', () => {
    const state = reduce(addUser(createChat(), 'x'), ev({ type: 'command_failed', command: 'set_model', error: 'no' }))
    expect(state.messages.map((m) => m.kind)).toEqual(['user', 'notice'])
    expect(state.giveBack).toBeNull()
  })

  test('history that arrives after the first message goes before it', () => {
    const history = [
      { role: 'user', content: 'earlier question' },
      { role: 'assistant', content: [{ type: 'text', text: 'earlier answer' }] },
      { role: 'user', content: 'just typed' },
    ]
    const state = reduce(addUser(createChat(), 'just typed'), ev({ type: 'history', messages: history }))
    expect(state.messages.map((m) => m.text)).toEqual(['earlier question', 'earlier answer', 'just typed'])
  })

  test('history without the typed message still goes before it', () => {
    const state = reduce(addUser(createChat(), 'new'), ev({ type: 'history', messages: [{ role: 'user', content: 'old' }] }))
    expect(state.messages.map((m) => m.text)).toEqual(['old', 'new'])
  })

  test('a stopped engine is not running', () => {
    const state = run(createChat(), ev({ type: 'agent_start' }), { type: 'chatEngine', state: 'stopped' })
    expect(state.running).toBe(false)
    expect(state.engine).toBe('stopped')
  })
})

describe('the route line from the router (a status)', () => {
  const status = (statusText) => ev({ type: 'ui_notice', request: { method: 'setStatus', statusKey: 'aria-jev', statusText } })

  test('a jev status becomes a route line', () => {
    const state = run(addUser(createChat(), 'hi'), status(JSON.stringify({ mode: 'focused task', action: 'searches memory' })))
    expect(state.messages.at(-1)).toEqual({ kind: 'jev', mode: 'focused task', action: 'searches memory' })
  })

  test('a broken or empty jev status, or another status, is ignored', () => {
    const before = addUser(createChat(), 'hi')
    expect(run(before, status('not json')).messages).toEqual(before.messages)
    expect(run(before, status(undefined)).messages).toEqual(before.messages)
    expect(run(before, status(JSON.stringify({ action: 'x' }))).messages).toEqual(before.messages)
    expect(run(before, ev({ type: 'ui_notice', request: { method: 'setStatus', statusKey: 'sandbox', statusText: '🔒 sandbox' } })).messages).toEqual(before.messages)
  })
})

describe('the subagents of a session', () => {
  const agents = [{ id: 'a1', role: 'security', task: 'Review', status: 'working · 1 tool', kind: 'active', steps: [] }]

  test('chatAgents stores them for the session', () => {
    expect(reduce(createChat(), { type: 'chatAgents', agents }).agents).toEqual(agents)
  })

  test('a malformed list is ignored, and a new chat starts with none', () => {
    expect(createChat().agents).toEqual([])
    expect(reduce(createChat(), { type: 'chatAgents', agents: 'x' }).agents).toEqual([])
  })
})

describe('a card can only answer with the options the dialog really has', () => {
  test('options in the title that the dialog does not accept are dropped', () => {
    const title = JSON.stringify({ aria: 1, kind: 'sandbox', summary: 'x', options: [{ value: 'Yes, no sandbox for the rest of the session', label: 'No' }, { value: 'No', label: 'No' }] })
    const state = reduce(createChat(), { type: 'chatDialog', id: 'u1', method: 'select', request: { title, options: ['No'] }, hidden: false })
    expect(state.dialogs[0].card.options).toEqual([{ value: 'No', label: 'No' }])
  })

  test('when none of them is accepted the card falls back to the plain dialog', () => {
    const title = JSON.stringify({ aria: 1, kind: 'sandbox', options: [{ value: 'evil', label: 'Ok' }] })
    const state = reduce(createChat(), { type: 'chatDialog', id: 'u1', method: 'select', request: { title, options: ['No'] }, hidden: false })
    expect(state.dialogs[0].card).toBeNull()
  })

  test('a card with no options of its own keeps working (the defaults are answered by the dialog)', () => {
    const title = JSON.stringify({ aria: 1, kind: 'question', questions: [] })
    expect(reduce(createChat(), { type: 'chatDialog', id: 'u1', method: 'input', request: { title }, hidden: false }).dialogs[0].card.kind).toBe('question')
  })
})

describe('a tool call being written', () => {
  const update = (e) => ev({ type: 'message_update', assistantMessageEvent: e })
  test('its arguments (JSON) never show as the answer while they stream', () => {
    const state = run(
      createChat(),
      ev({ type: 'message_start', message: { role: 'assistant' } }),
      delta(0, 'Let me look.'),
      update({ type: 'toolcall_start', contentIndex: 1 }),
      update({ type: 'toolcall_delta', contentIndex: 1, delta: '{"command":"ls' }),
      update({ type: 'toolcall_delta', contentIndex: 1, delta: ' src"}' }),
      update({ type: 'toolcall_end', contentIndex: 1, toolCall: { id: 't1', name: 'bash', arguments: { command: 'ls src' } } }),
    )
    const last = state.messages.at(-1)
    expect(last.text).toBe('Let me look.')
    expect(last.thinking ?? '').toBe('')
  })
})

describe('one accordion per answer (design: tool-chips)', () => {
  const update = (e) => ev({ type: 'message_update', assistantMessageEvent: e })
  const toolStart = (id, name, args = {}) => ev({ type: 'tool_execution_start', toolCallId: id, toolName: name, args })
  const toolEnd = (id) => ev({ type: 'tool_execution_end', toolCallId: id, result: { content: [{ type: 'text', text: 'ok' }] }, isError: false })
  const step = (thinking, calls) => [
    ev({ type: 'message_start', message: { role: 'assistant' } }),
    update({ type: 'thinking_delta', contentIndex: 0, delta: thinking }),
    ev({ type: 'message_end', message: { role: 'assistant', content: [{ type: 'thinking', thinking }, ...calls.map(([id, name]) => ({ type: 'toolCall', id, name, arguments: {} }))] } }),
    ...calls.flatMap(([id, name]) => [toolStart(id, name), toolEnd(id)]),
  ]
  const turn = () =>
    run(
      createChat(),
      ev({ type: 'message_start', message: { role: 'user', content: 'fix it' } }),
      ev({ type: 'agent_start' }),
      ...step('Look at the file first.', [['t1', 'read']]),
      ...step('Now change it.', [['t2', 'edit']]),
      ...step('Check the build.', [['t3', 'bash']]),
      ev({ type: 'message_start', message: { role: 'assistant' } }),
      update({ type: 'thinking_delta', contentIndex: 0, delta: 'All good.' }),
      update({ type: 'text_delta', contentIndex: 1, delta: 'Fixed.' }),
      ev({ type: 'message_end', message: { role: 'assistant', content: [{ type: 'thinking', thinking: 'All good.' }, { type: 'text', text: 'Fixed.' }] } }),
    )

  test('the thinking and every tool of a turn go into a single group before the answer', () => {
    const kinds = turn().messages.map((m) => m.kind)
    expect(kinds.filter((k) => k === 'tools')).toHaveLength(1)
    expect(kinds.slice(-2)).toEqual(['tools', 'assistant'])
  })

  test('the group lists thinking rows and calls in the order they happened', () => {
    const group = turn().messages.find((m) => m.kind === 'tools')
    expect(group.calls.map((c) => (c.think ? `think:${c.text}` : c.name))).toEqual([
      'think:Look at the file first.', 'read', 'think:Now change it.', 'edit', 'think:Check the build.', 'bash', 'think:All good.',
    ])
  })

  test('the answer keeps only its text', () => {
    expect(turn().messages.at(-1)).toMatchObject({ kind: 'assistant', text: 'Fixed.' })
  })

  test('text written between tools is a message, and the tools after it start a new group', () => {
    const state = run(
      createChat(),
      ev({ type: 'agent_start' }),
      ev({ type: 'message_start', message: { role: 'assistant' } }),
      update({ type: 'text_delta', contentIndex: 0, delta: 'Let me look.' }),
      ev({ type: 'message_end', message: { role: 'assistant', content: [{ type: 'text', text: 'Let me look.' }, { type: 'toolCall', id: 't1', name: 'read', arguments: {} }] } }),
      toolStart('t1', 'read'),
    )
    expect(state.messages.map((m) => m.kind)).toEqual(['assistant', 'tools'])
    expect(state.messages[0]).toMatchObject({ kind: 'assistant', text: 'Let me look.', done: true })
    expect(state.messages[1].calls[0]).toMatchObject({ id: 't1', name: 'read' })
  })

  test('the next question starts a new group', () => {
    const state = run(turn(), ev({ type: 'agent_settled', aborted: false }), ev({ type: 'message_start', message: { role: 'user', content: 'again' } }), ev({ type: 'agent_start' }), ...step('Again.', [['t9', 'read']]))
    expect(state.messages.filter((m) => m.kind === 'tools')).toHaveLength(2)
  })

  test('history is grouped the same way', () => {
    const history = [
      { role: 'user', content: 'fix it' },
      { role: 'assistant', content: [{ type: 'thinking', thinking: 'Look.' }, { type: 'toolCall', id: 'h1', name: 'read', arguments: {} }] },
      { role: 'toolResult', toolCallId: 'h1', content: [{ type: 'text', text: 'ok' }] },
      { role: 'assistant', content: [{ type: 'thinking', thinking: 'Edit.' }, { type: 'toolCall', id: 'h2', name: 'edit', arguments: {} }] },
      { role: 'toolResult', toolCallId: 'h2', content: [{ type: 'text', text: 'ok' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'Fixed.' }] },
    ]
    const state = reduce(createChat(), ev({ type: 'history', messages: history }))
    expect(state.messages.map((m) => m.kind)).toEqual(['user', 'tools', 'assistant'])
    expect(state.messages[1].calls.map((c) => c.think ? 'think' : c.name)).toEqual(['think', 'read', 'think', 'edit'])
  })
})

describe('background work in the thread', () => {
  const job = (over = {}) => ({ id: 'j1', number: 1, session: 's1', kind: 'process', title: 'bun dev', folder: '~/p', state: 'exited', exit: 1, conflict: false, ...over })
  const custom = (details, over = {}) => ({ role: 'custom', customType: 'aria-background', content: 'Background #1 …', display: true, details, ...over })
  const live = (message) => ev({ type: 'message_start', message })
  const noteOf = (message) => run(createChat(), live(message)).messages

  test('an aria-background custom message becomes a background note, live and from history', () => {
    const exited = custom({ job: job(), jobs: undefined, by: 'self', event: 'ended' })
    const want = { kind: 'note', label: 'background', key: 'chat.bg.note.exited', vars: { title: 'bun dev', exit: 1 } }
    expect(noteOf(exited)).toEqual([want])
    const past = reduce(createChat(), ev({ type: 'history', messages: [{ role: 'user', content: 'run it' }, exited] })).messages
    expect(past).toEqual([{ kind: 'user', text: 'run it' }, want])

    const outcome = (details) => noteOf(custom(details))[0]?.key
    expect(outcome({ job: job({ state: 'failed', exit: null }), by: 'self', event: 'ended' })).toBe('chat.bg.note.failed')
    expect(outcome({ job: job({ state: 'stopped' }), by: 'stop', event: 'ended' })).toBe('chat.bg.note.stopped')
    expect(outcome({ job: job({ kind: 'agent', state: 'done', title: 'fix the tests' }), by: 'self', event: 'ended' })).toBe('chat.bg.note.done')
    expect(outcome({ job: job({ kind: 'agent', state: 'done', conflict: true }), by: 'self', event: 'ended' })).toBe('chat.bg.note.conflict')
    expect(outcome({ job: job({ kind: 'agent', state: 'failed' }), by: 'self', event: 'ended' })).toBe('chat.bg.note.failed')
    const cut = noteOf(custom({ job: null, jobs: [job({ title: 'bun dev', state: 'cut' }), job({ id: 'j2', title: 'fix it', state: 'cut' })], by: null, event: 'cut' }))[0]
    expect(cut).toEqual({ kind: 'note', label: 'background', key: 'chat.bg.note.cut', vars: { title: 'bun dev, fix it' } })
  })

  test('a hidden custom message (display false) adds nothing', () => {
    expect(noteOf(custom({ job: job(), by: 'self', event: 'ended' }, { display: false }))).toEqual([])
    expect(noteOf({ role: 'custom', customType: 'aria-background-now', content: '[background] …', display: false })).toEqual([])
    expect(noteOf({ role: 'custom', customType: 'other', content: 'x', display: true, details: { job: job() } })).toEqual([])
    const past = reduce(createChat(), ev({ type: 'history', messages: [custom({ job: job() }, { display: false }), { role: 'custom', customType: 'other', display: true }] })).messages
    expect(past).toEqual([])
  })
})

describe('attachments', () => {
  const files = [{ name: 'notes.txt', mime: 'text/plain', size: 2048 }]
  const lines = '\n\n[attached: notes.txt (text/plain, 2 KB) at /x/1-notes.txt]'
  test('a sent message shows its files at once, and Pi\'s echo with the lines is not a second bubble', () => {
    let state = addUser(createChat(), 'look', files)
    expect(state.messages).toEqual([{ kind: 'user', text: 'look', files: [{ ...files[0], image: false }] }])
    state = run(state, ev({ type: 'message_start', message: { role: 'user', content: [{ type: 'text', text: 'look' + lines }, { type: 'image', data: 'x' }] } }))
    expect(state.messages.length).toBe(1)
  })
  test('a_queued_message_keeps_its_attachments', () => {
    let state = addQueued(run(createChat(), ev({ type: 'agent_start' })), 'look', false, files)
    expect(state.queue).toEqual([{ text: 'look', files: [{ ...files[0], image: false }] }])
    // Pi's queue holds the text the dock built, lines and all: the entry keeps them to give back on a re-queue
    state = run(state, ev({ type: 'queue_update', steering: [], followUp: ['look' + lines] }))
    expect(state.queue).toEqual([{ text: 'look', raw: 'look' + lines, files: [{ name: 'notes.txt', mime: 'text/plain', size: '2 KB', path: '/x/1-notes.txt', image: false }] }])
    state = run(state, ev({ type: 'queue_update', steering: [], followUp: [] }), ev({ type: 'message_start', message: { role: 'user', content: 'look' + lines } }))
    expect(state.queue).toEqual([])
    expect(state.messages).toEqual([{ kind: 'user', text: 'look', files: [{ name: 'notes.txt', mime: 'text/plain', size: '2 KB', path: '/x/1-notes.txt', image: false }] }])
  })
  test('the page\'s own bubble marks an image as history does', () => {
    const state = addUser(createChat(), 'see', [{ name: 's.png', mime: 'image/png', size: 7 }, { name: 'a.txt', mime: 'text/plain', size: 7 }])
    expect(state.messages[0].files.map((f) => f.image)).toEqual([true, false])
  })
  test('a steering message keeps them too', () => {
    const state = run(addQueued(createChat(), 'now', true, files), ev({ type: 'queue_update', steering: ['now' + lines], followUp: [] }))
    expect(state.queue[0].steer).toBe(true)
    expect(state.queue[0].files[0].name).toBe('notes.txt')
  })
  test('history replays the lines as files', () => {
    const state = reduce(createChat(), ev({ type: 'history', messages: [{ role: 'user', content: [{ type: 'text', text: 'see' + lines }] }] }))
    expect(state.messages).toEqual([{ kind: 'user', text: 'see', files: [{ name: 'notes.txt', mime: 'text/plain', size: '2 KB', path: '/x/1-notes.txt', image: false }] }])
  })
  test('a refused message with files leaves the thread and its text goes back', () => {
    const state = run(addUser(createChat(), 'look', files), ev({ type: 'command_failed', command: 'prompt', error: 'disk full', text: 'look' }))
    expect(state.messages.filter((m) => m.kind === 'user')).toEqual([])
    expect(state.giveBack).toBe('look')
  })
  test('a note from the page goes into the thread', () => {
    const state = addNotice(createChat(), 'warn', 'chat.attach.tooMany', { max: 10 })
    expect(state.messages).toEqual([{ kind: 'notice', tone: 'warn', key: 'chat.attach.tooMany', vars: { max: 10 } }])
  })
})

describe('earlier in this conversation (paged history)', () => {
  const framed = 'The conversation history before this point was compacted into the following summary:\n\n<summary>\nWe fixed the build.\n</summary>'
  const seed = { role: 'custom', customType: 'aria-earlier', content: framed, display: true, details: { from: 'old-id', summary: 'We fixed the build.' } }
  const compaction = { role: 'compactionSummary', summary: 'We talked about tests.', tokensBefore: 120000 }
  const kept = [{ role: 'user', content: 'and now?' }, { role: 'assistant', content: [{ type: 'text', text: 'Now the docs.' }] }]
  const page = (texts, cursor, more) => ev({ type: 'earlier', messages: texts.flatMap((text) => [{ role: 'user', content: text }, { role: 'assistant', content: [{ type: 'text', text: `re: ${text}` }] }]), cursor, more })

  test('a rotated segment\'s seed message is one earlier entry with the raw summary, never the framed text', () => {
    const state = reduce(createChat(), ev({ type: 'history', messages: [seed, ...kept] }))
    expect(state.messages[0]).toMatchObject({ kind: 'earlier', summary: 'We fixed the build.' })
    expect(state.messages.map((m) => m.kind)).toEqual(['earlier', 'user', 'assistant'])
    expect(JSON.stringify(state.messages)).not.toContain('<summary>')
  })

  test('a compaction summary not rotated yet is the same entry', () => {
    const state = reduce(createChat(), ev({ type: 'history', messages: [compaction, ...kept] }))
    expect(state.messages[0]).toMatchObject({ kind: 'earlier', summary: 'We talked about tests.' })
  })

  test('other custom messages are not earlier entries', () => {
    const other = { ...seed, customType: 'aria-other' }
    expect(reduce(createChat(), ev({ type: 'history', messages: [other, ...kept] })).messages.map((m) => m.kind)).toEqual(['user', 'assistant'])
  })

  test('there is more to show when the history starts with an earlier entry, and not otherwise', () => {
    expect(reduce(createChat(), ev({ type: 'history', messages: [seed, ...kept] })).earlier).toMatchObject({ more: true, cursor: null, loading: false })
    expect(reduce(createChat(), ev({ type: 'history', messages: kept })).earlier.more).toBe(false)
  })

  test('asking: the first time without a cursor, then with the last one; it is loading until the reply', () => {
    let state = reduce(createChat(), ev({ type: 'history', messages: [seed, ...kept] }))
    let message
    ;[state, message] = askEarlier(state)
    expect(message).toEqual({ type: 'chatEarlier' })
    expect(state.earlier.loading).toBe(true)
    expect(askEarlier(state)[1]).toBeNull() // one request at a time
    state = reduce(state, page(['one'], { segment: 0, index: 40 }, true))
    expect(state.earlier.loading).toBe(false)
    ;[state, message] = askEarlier(state)
    expect(message).toEqual({ type: 'chatEarlier', cursor: { segment: 0, index: 40 } })
  })

  test('nothing is asked when there is no more', () => {
    const state = reduce(createChat(), ev({ type: 'history', messages: kept }))
    expect(askEarlier(state)).toEqual([state, null])
  })

  test('a reply goes above what is shown, in order, and keeps the rest; a second one goes above it', () => {
    let state = reduce(createChat(), ev({ type: 'history', messages: [seed, ...kept] }))
    state = reduce(askEarlier(state)[0], page(['b1', 'b2'], { segment: 0, index: 4 }, true))
    expect(state.messages.map((m) => m.text ?? m.kind)).toEqual(['b1', 're: b1', 'b2', 're: b2', 'earlier', 'and now?', 'Now the docs.'])
    expect(state.earlier).toMatchObject({ more: true, cursor: { segment: 0, index: 4 } })
    state = reduce(askEarlier(state)[0], page(['a1'], { segment: 0, index: 0 }, false))
    expect(state.messages.map((m) => m.text ?? m.kind)).toEqual(['a1', 're: a1', 'b1', 're: b1', 'b2', 're: b2', 'earlier', 'and now?', 'Now the docs.'])
    expect(state.earlier).toMatchObject({ more: false, cursor: { segment: 0, index: 0 }, loading: false })
  })

  test('the running turn\'s start moves with what was prepended', () => {
    let state = run(createChat(), ev({ type: 'history', messages: [seed, ...kept] }), ev({ type: 'agent_start' }))
    const from = state.turnFrom
    state = reduce(state, page(['x'], { segment: 0, index: 0 }, false))
    expect(state.turnFrom).toBe(from + 2)
  })

  test('a history that comes again replaces the pages shown before it instead of repeating them', () => {
    let state = reduce(createChat(), ev({ type: 'history', messages: [seed, ...kept] }))
    state = reduce(state, page(['b1'], { segment: 0, index: 2 }, false))
    state = reduce(state, ev({ type: 'history', messages: [seed, ...kept] }))
    expect(state.messages.map((m) => m.text ?? m.kind)).toEqual(['earlier', 'and now?', 'Now the docs.'])
    expect(state.earlier).toMatchObject({ more: true, cursor: null, count: 0 })
  })

  test('a reset forgets the paging', () => {
    let state = reduce(createChat(), ev({ type: 'history', messages: [seed, ...kept] }))
    state = reduce(state, ev({ type: 'session_reset' }))
    expect(state.earlier).toEqual(createChat().earlier)
  })
})

describe('earlier: review fixes', () => {
  const seed = { role: 'custom', customType: 'aria-earlier', content: 'framed', display: true, details: { from: 'old-id', summary: 'S' } }
  const kept = [{ role: 'user', content: 'and now?' }]
  const history = ev({ type: 'history', messages: [seed, ...kept] })
  const page = ev({ type: 'earlier', messages: [{ role: 'user', content: 'older' }], cursor: { segment: 0, index: 0 }, more: false })

  test('the summary keeps its id when the history comes again (an open summary stays open)', () => {
    const a = reduce(createChat(), history).messages[0].id
    const b = run(createChat(), history, history).messages[0].id
    expect(a).toBe('earlier-old-id')
    expect(b).toBe(a)
    expect(reduce(createChat(), ev({ type: 'history', messages: [{ role: 'compactionSummary', summary: 'S' }] })).messages[0].id).toBe('earlier-0')
  })

  test('a history arriving while a page is on its way: still loading, that page is dropped, then asking works again', () => {
    let state = reduce(createChat(), history)
    state = askEarlier(state)[0]
    state = reduce(state, history)
    expect(state.earlier).toMatchObject({ loading: true, stale: true })
    expect(askEarlier(state)[1]).toBeNull() // page 1 is not asked twice
    state = reduce(state, page)
    expect(state.messages.map((m) => m.text ?? m.kind)).toEqual(['earlier', 'and now?'])
    expect(state.earlier).toMatchObject({ loading: false, stale: false, more: true, count: 0 })
    expect(askEarlier(state)[1]).toEqual({ type: 'chatEarlier' })
  })
})
