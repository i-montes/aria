// Run: bun test ui
import { describe, expect, test } from 'bun:test'
import { chatTexts } from './texts.js'
import { addQueued, addUser, answerDialog, createChat, reduce } from './state.js'
import { renderThread, threadItems } from './thread.js'

const texts = chatTexts().en
const t = (key, vars) => (texts[key] ?? key).replace(/\{(\w+)\}/g, (_, name) => vars?.[name] ?? '')
const ev = (event) => ({ type: 'chatEvent', event })
const run = (state, ...messages) => messages.reduce(reduce, state)
const dialog = (id, card, extra = {}) => ({ type: 'chatDialog', id, method: 'select', request: { title: JSON.stringify({ aria: 1, ...card }), options: ['x'], ...extra }, hidden: false })

describe('the thread', () => {
  test('an empty session shows the design\'s start hint once its history came back empty', () => {
    const html = renderThread(run(createChat(), ev({ type: 'history', messages: [] })), {}, t)
    expect(html).toContain('NEW SESSION')
    expect(html).toContain('type /')
  })

  test('until its history arrives a session shows the working line, loading the conversation', () => {
    const html = renderThread(createChat(), {}, t)
    expect(html).toContain('Loading the conversation')
    expect(html).toContain('thread-working')
    expect(html).not.toContain('NEW SESSION')
  })

  test('a session whose Pi cannot start shows no loader', () => {
    const state = run(createChat(), { type: 'chatState', workdir: '/w', running: false, model: null, problem: 'node-missing' })
    expect(renderThread(state, {}, t)).not.toContain('Loading the conversation')
  })

  test('user and assistant text is escaped and the user message gets a DOM id for the index', () => {
    let state = addUser(createChat(), '<script>alert(1)</script>')
    state = run(state, ev({ type: 'message_start', message: { role: 'assistant' } }), ev({ type: 'message_end', message: { role: 'assistant', content: [{ type: 'text', text: 'a < b & "c"' }] } }))
    const html = renderThread(state, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).toContain('&lt;script&gt;alert(1)&lt;/script&gt;')
    expect(html).toContain('id="msg-0"')
    expect(html).toContain('a &lt; b &amp; ')
  })

  test('tool calls render as a group that knows its position', () => {
    const state = run(
      createChat(),
      ev({ type: 'tool_execution_start', toolCallId: 'a', toolName: 'bash', args: { command: 'ls' } }),
      ev({ type: 'tool_execution_end', toolCallId: 'a', toolName: 'bash', isError: false, result: { content: [{ type: 'text', text: 'ok' }] } }),
    )
    const html = renderThread(state, {}, t)
    expect(html).toContain('data-chat="toggle-tools" data-index="0"')
  })

  test('the working line shows while running, with the phase of the running tool', () => {
    const state = run(createChat(), ev({ type: 'agent_start' }), ev({ type: 'tool_execution_start', toolCallId: 'a', toolName: 'read', args: { path: 'x' } }))
    const html = renderThread(state, { now: Date.now() }, t)
    expect(html).toContain('Reading files')
    expect(html).toContain('data-since=')
    expect(renderThread(run(createChat(), ev({ type: 'history', messages: [] })), {}, t)).not.toContain('p-working')
  })

  test('queued messages show after the thread with a cancel', () => {
    const html = renderThread(addQueued(addUser(createChat(), 'first'), 'second'), {}, t)
    expect(html.indexOf('first')).toBeLessThan(html.indexOf('second'))
    expect(html).toContain('data-chat="cancel-queued"')
  })

  test('notes, dividers, route lines and answers each go through their part', () => {
    let state = run(createChat(), ev({ type: 'compaction_end', result: { tokensBefore: 1 }, aborted: false }))
    state = { ...state, messages: [...state.messages, { kind: 'jev', mode: 'focused task', action: 'searches memory' }] }
    state = { ...state, messages: [...state.messages, { kind: 'note', label: 'engram', text: 'saved <it>' }, { kind: 'answers', answers: [{ header: 'Scope', selected: ['Small'] }] }] }
    const html = renderThread(state, {}, t)
    expect(html).toContain('p-jev')
    expect(html).toContain('context compacted')
    expect(html).toContain('saved &lt;it&gt;')
    expect(html).toContain('Scope')
  })

  test('a background note fills its text with the job: "background · bun dev ended (exit 1)"', () => {
    const state = { ...createChat(), messages: [{ kind: 'note', label: 'background', key: 'chat.bg.note.exited', vars: { title: 'bun <dev>', exit: 1 } }] }
    const html = renderThread(state, {}, t)
    expect(html).toContain('<span class="p-note-label">background</span>')
    expect(html).toContain('bun &lt;dev&gt; ended (exit 1)')
  })

  test('a sandbox card waits in the thread with its three choices; a quota card with its two', () => {
    let state = run(createChat(), dialog('u1', { kind: 'sandbox', summary: 'rm -rf <build>', detail: 'rm -rf build' }), dialog('u2', { kind: 'quota', percent: 98, freeModel: 'opencode-go/cat-jam-free', countdown: 300 }))
    const html = renderThread(state, {}, t)
    expect(html).toContain('&lt;build&gt;')
    expect(html.match(/data-chat="permission"/g)).toHaveLength(3)
    expect(html.match(/data-chat="quota"/g)).toHaveLength(2)
    expect(html).toContain('data-choice="always"')
  })

  test('an answered card leaves a note and no card', () => {
    let state = run(createChat(), dialog('u1', { kind: 'sandbox', summary: 'rm -rf build' }))
    state = answerDialog(state, 'u1', 'allow')
    const html = renderThread(state, {}, t)
    expect(html).not.toContain('data-chat="permission"')
    expect(html).toContain('allowed once')
  })

  test('a write card without a summary is headed by its path', () => {
    const html = renderThread(run(createChat(), dialog('w1', { kind: 'write', path: '~/notes.md' })), {}, t)
    expect(html).toContain('~/notes.md')
  })

  test('the question card is not in the thread (it is an overlay) and plain dialogs are not either', () => {
    const state = run(createChat(), dialog('u1', { kind: 'question', questions: [{ question: 'Which?', header: 'Pick', options: [{ label: 'A' }, { label: 'B' }] }] }), { type: 'chatDialog', id: 'u2', method: 'confirm', request: { title: 'Plain?' }, hidden: false })
    const html = renderThread(state, {}, t)
    expect(html).not.toContain('data-chat="question"')
    expect(html).not.toContain('Plain?')
  })

  test('a notice shows its text escaped', () => {
    const html = renderThread(run(createChat(), ev({ type: 'command_failed', command: 'prompt', error: 'nope <b>' })), {}, t)
    expect(html).toContain('nope &lt;b&gt;')
    expect(html).toContain('tone-error')
  })
})

describe("the turn's group opens while Aria works and folds when it answers", () => {
  const group = { kind: 'tools', calls: [{ id: 'c1', name: 'read', args: {}, status: 'done', summary: 'a.ts', result: '' }] }
  const working = { ...createChat(), running: true, turnFrom: 0, messages: [group] }
  const answered = { ...working, running: false, messages: [group, { kind: 'assistant', text: 'Done.', done: true }] }
  const isOpen = (html) => /class="p-tools[^"]*is-open/.test(html)

  test('open while the run goes', () => expect(isOpen(renderThread(working, {}, t))).toBe(true))
  test('folded once the answer is there', () => expect(isOpen(renderThread(answered, {}, t))).toBe(false))
  test('closed by hand while working, it stays closed', () => expect(isOpen(renderThread(working, { collapsed: new Set([0]) }, t))).toBe(false))
  test('opened by hand afterwards, it stays open', () => expect(isOpen(renderThread(answered, { expanded: new Set([0]) }, t))).toBe(true))
  test('an earlier turn is not live', () => {
    const state = { ...working, turnFrom: 1, messages: [group, { kind: 'user', text: 'again' }] }
    expect(isOpen(renderThread(state, {}, t))).toBe(false)
  })
})

describe('the thread as items', () => {
  const streaming = () =>
    run(createChat(), ev({ type: 'history', messages: [] }), { type: 'chatState', workdir: '/w', running: true }, ev({ type: 'message_start', message: { role: 'assistant', content: [] } }), ev({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', contentIndex: 0, delta: 'Hola' } }))

  test('one string per element, and joined they are the thread', () => {
    const state = run(addUser(createChat(), 'hi'), ev({ type: 'history', messages: [] }))
    const items = threadItems(state, {}, t)
    expect(items.every((item) => item.startsWith('<') && item.length > 0)).toBe(true)
    expect(items.join('')).toBe(renderThread(state, {}, t))
  })

  test('an entry that did not change is not rendered again (entries are never changed in place)', () => {
    const state = run(addUser(createChat(), 'first question'), ev({ type: 'history', messages: [] }))
    threadItems(state, {}, t)
    state.messages[0].text = 'changed in place'
    expect(threadItems(state, {}, t).join('')).toContain('first question')
  })

  test('a streamed word renders only the entry it changed', () => {
    const before = streaming()
    const after = reduce(before, ev({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', contentIndex: 0, delta: ' mundo' } }))
    expect(after.messages.at(-1)).not.toBe(before.messages.at(-1))
    expect(threadItems(after, {}, t).join('')).toContain('Hola mundo')
  })

  test('a tool group renders again when it opens or closes', () => {
    const state = run(createChat(), ev({ type: 'history', messages: [] }), ev({ type: 'tool_execution_start', toolCallId: 'a', toolName: 'bash', args: { command: 'ls' } }))
    const closed = threadItems(state, {}, t).join('')
    const open = threadItems(state, { expanded: new Set([0]) }, t).join('')
    expect(open).not.toBe(closed)
  })
})
