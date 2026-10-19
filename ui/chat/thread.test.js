// Run: bun test ui
import { describe, expect, test } from 'bun:test'
import { chatTexts } from './texts.js'
import { addQueued, addUser, answerDialog, createChat, reduce } from './state.js'
import { renderThread } from './thread.js'

const texts = chatTexts().en
const t = (key, vars) => (texts[key] ?? key).replace(/\{(\w+)\}/g, (_, name) => vars?.[name] ?? '')
const ev = (event) => ({ type: 'chatEvent', event })
const run = (state, ...messages) => messages.reduce(reduce, state)
const dialog = (id, card, extra = {}) => ({ type: 'chatDialog', id, method: 'select', request: { title: JSON.stringify({ aria: 1, ...card }), options: ['x'], ...extra }, hidden: false })

describe('the thread', () => {
  test('an empty session shows the design\'s start hint', () => {
    const html = renderThread(createChat(), {}, t)
    expect(html).toContain('NEW SESSION')
    expect(html).toContain('type /')
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
    expect(renderThread(createChat(), {}, t)).not.toContain('p-working')
  })

  test('queued messages show after the thread with a cancel', () => {
    const html = renderThread(addQueued(addUser(createChat(), 'first'), 'second'), {}, t)
    expect(html.indexOf('first')).toBeLessThan(html.indexOf('second'))
    expect(html).toContain('data-chat="cancel-queued"')
  })

  test('notes, dividers, route lines and answers each go through their part', () => {
    let state = run(createChat(), ev({ type: 'aria_jev', mode: 'focused task', action: 'searches memory' }), ev({ type: 'compaction_end', result: { tokensBefore: 1 }, aborted: false }))
    state = { ...state, messages: [...state.messages, { kind: 'note', label: 'engram', text: 'saved <it>' }, { kind: 'answers', answers: [{ header: 'Scope', selected: ['Small'] }] }] }
    const html = renderThread(state, {}, t)
    expect(html).toContain('p-jev')
    expect(html).toContain('context compacted')
    expect(html).toContain('saved &lt;it&gt;')
    expect(html).toContain('Scope')
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
