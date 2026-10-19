// Run: bun test ui
import { describe, expect, test } from 'bun:test'
import { addUser, answerDialog, createChat, reduce, toolSummary } from './state.js'

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
    expect(state.dialogs).toEqual([{ id: 'u1', method: 'select', request: { title: 'Allow?', options: ['Allow'] } }])
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
