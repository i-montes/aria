// Run: bun test ui
import { describe, expect, test } from 'bun:test'
import { chatTexts } from './texts.js'
import { createI18n } from '../i18n.js'
import { readFileSync } from 'node:fs'
import { addQueued, addUser, answerDialog, createChat, reduce } from './state.js'
import { renderThread, threadItems } from './thread.js'

const texts = chatTexts().en
const t = (key, vars) => (texts[key] ?? key).replace(/\{(\w+)\}/g, (_, name) => vars?.[name] ?? '')
const ev = (event) => ({ type: 'chatEvent', event })
const run = (state, ...messages) => messages.reduce(reduce, state)
const dialog = (id, card, extra = {}) => ({ type: 'chatDialog', id, method: 'select', request: { title: JSON.stringify({ aria: 1, ...card }), options: ['x'], ...extra }, hidden: false })

describe('the watchdog notice is said in the page\'s language, with the catalogs the page boots with', () => {
  const catalog = (code) => JSON.parse(readFileSync(new URL(`../i18n/${code}.json`, import.meta.url), 'utf8'))
  const parts = chatTexts()
  const i18n = createI18n({ en: { ...catalog('en'), ...parts.en }, es: { ...catalog('es'), ...parts.es } })
  const stopped = (minutes) => ev({ type: 'ui_notice', request: { method: 'notify', key: 'chat.watchdog.stopped', vars: { minutes }, notifyType: 'warning' } })
  for (const [language, minutes, text] of [
    ['en', 10, 'Aria stopped this turn: nothing came back for 10 minutes.'],
    ['es', 10, 'Aria detuvo este turno: no llegó nada en 10 minutos.'],
    ['en', 1, 'Aria stopped this turn: nothing came back for 1 minute.'],
    ['es', 1, 'Aria detuvo este turno: no llegó nada en 1 minuto.'],
  ]) {
    test(`${language}, ${minutes} min`, () => {
      i18n.setLanguage(language)
      const html = renderThread(run(createChat(), ev({ type: 'history', messages: [] }), stopped(minutes)), {}, i18n.t)
      expect(html).toContain(text)
      expect(html).not.toContain('chat.watchdog.stopped')
    })
  }
})

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

  test('a plan card waits in the thread as the document and the phases, with one button per option', () => {
    const options = [{ value: 'approve', label: 'Approve and build', primary: true }, { value: 'change', label: 'Change something' }]
    const card = { kind: 'plan', title: 'Sign-up <flow>', path: 'docs/plans/x.md', sections: [{ heading: 'Goal', text: 'Let people sign up.' }], phases: [{ name: 'Sign up', tasks: ['Form', 'Mail'] }], options }
    const state = run(createChat(), dialog('p1', card, { options: ['approve', 'change'] }))
    const html = renderThread(state, {}, t)
    expect(html).toContain('p-doc-card')
    expect(html).toContain('Sign-up &lt;flow&gt;')
    expect(html).toContain('docs/plans/x.md')
    expect(html).toContain('Draft')
    expect(html).toContain('Let people sign up.')
    expect(html).toContain('Sign up')
    expect(html).toContain('Form')
    expect(html).toContain('class="p-phases')
    expect(html.match(/<button/g)).toHaveLength(2)
    expect(html).toContain('data-chat="permission" data-id="p1" data-choice="approve"')
    expect(html).toContain('data-chat="permission" data-id="p1" data-choice="change"')
    expect(html).toContain('Approve and build')
    expect(html.indexOf('p-doc-card')).toBeLessThan(html.indexOf('class="p-phases'))
    expect(html.indexOf('class="p-phases')).toBeLessThan(html.indexOf('data-choice="approve"'))
  })

  test('an approved plan stays in the thread as a record: Approved, no buttons', () => {
    const plan = { kind: 'plan', status: 'approved', title: 'Sign-up', path: 'docs/plans/x.md', sections: [], phases: [{ name: 'Sign up', tasks: ['a'] }] }
    const html = renderThread({ ...createChat(), messages: [plan] }, {}, t)
    expect(html).toContain('Approved')
    expect(html).toContain('is-static')
    expect(html).toContain('class="p-phases')
    expect(html).not.toContain('<button')
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

  const pulses = (html) => (html.match(/p-tools-pulse/g) ?? []).length
  const second = { kind: 'tools', calls: [{ id: 'c2', name: 'bash', args: {}, status: 'done', summary: 'ls', result: '' }] }
  const message = { kind: 'assistant', text: 'Found it, now checking the tests.', done: true }

  test('a message to the owner closes the chain before it: only the group after it is live', () => {
    const html = renderThread({ ...working, messages: [group, message, second] }, {}, t)
    expect(pulses(html)).toBe(1)
    const groups = html.match(/class="p-tools(?: [^"]*)?"/g)
    expect(groups).toEqual(['class="p-tools"', 'class="p-tools is-open is-live"'])
  })

  test('a message after the last group closes it too, and nothing pulses', () => {
    const html = renderThread({ ...working, messages: [group, message] }, {}, t)
    expect(pulses(html)).toBe(0)
    expect(isOpen(html)).toBe(false)
  })

  test('a group closed by a message can still be opened by hand', () => {
    expect(isOpen(renderThread({ ...working, messages: [group, message] }, { expanded: new Set([0]) }, t))).toBe(true)
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

describe('attachments in the thread', () => {
  test('a sent and a queued message show their files as chips', () => {
    const files = [{ name: 'notes.txt', mime: 'text/plain', size: 2048 }]
    const state = addQueued(addUser(run(createChat(), ev({ type: 'history', messages: [] })), 'look', files), 'then this', false, [{ name: 'b.pdf', size: 10 }])
    const html = renderThread(state, {}, t)
    expect(html).toContain('notes.txt')
    expect(html).toContain('2 KB')
    expect(html).toContain('b.pdf')
    expect(html).not.toContain('attach-remove')
  })
})

describe('earlier in this conversation', () => {
  // A summary as Pi writes it (pi-coding-agent's compaction prompt: ## Goal, ## Progress, ### Done, ## Next Steps…).
  const piSummary = '## Goal\nMake RELEASE optional.\n\n## Progress\n### Done\n- [x] **env.ts** reads it lazily\n\n## Next Steps\n1. Update the docs'
  const seed = { role: 'custom', customType: 'aria-earlier', content: 'framed <summary>x</summary>', display: true, details: { from: 'a', summary: piSummary } }
  const kept = [{ role: 'user', content: 'and now?' }]
  const withSeed = () => reduce(createChat(), ev({ type: 'history', messages: [seed, ...kept] }))

  test('the earlier entry is the chat\'s divider with the summary folded under it as a tool row', () => {
    const state = withSeed()
    const html = renderThread(state, {}, t)
    expect(html).toContain('class="p-divider"')
    expect(html).toContain('Earlier in this conversation')
    const id = state.messages[0].id
    expect(html).toContain(`data-chat="toggle-tool-row" data-id="${id}" aria-expanded="false"`)
    expect(html).toContain('<div class="p-tools-detail">') // folded: the detail is there, closed
    expect(html).not.toContain('p-tools-cmd') // no chip: it would show "## Goal"
    expect(html).not.toContain('##')
    expect(html).not.toContain('framed')
  })

  test('opened, the row shows the whole summary', () => {
    const state = withSeed()
    const html = renderThread(state, { openRows: new Set([state.messages[0].id]) }, t)
    expect(html).toContain('aria-expanded="true"')
    expect(html).toContain('<div class="p-tools-detail is-open">')
    // the chat's markdown, not the raw text: headings, a list, bold; no markers left
    expect(html).toContain('p-tools-md p-md')
    expect(html).toContain('<p class="p-md-h p-md-h2">Goal</p>')
    expect(html).toContain('<strong>env.ts</strong>')
    expect(html).toContain('Update the docs')
    expect(html).not.toContain('##')
    expect(html).not.toContain('**')
  })

  test('"Show earlier" sits above the first message while there is more, and says it is loading once asked', () => {
    const state = withSeed()
    const items = threadItems(state, {}, t)
    expect(items[0]).toContain('data-chat="chat-earlier"')
    expect(items[0]).toContain('Show earlier')
    const loading = threadItems({ ...state, earlier: { ...state.earlier, loading: true } }, {}, t)[0]
    expect(loading).toContain('Loading earlier messages')
    expect(loading).not.toContain('data-chat="chat-earlier"')
  })

  test('no button without more to show', () => {
    const state = reduce(withSeed(), ev({ type: 'earlier', messages: [{ role: 'user', content: 'first' }], cursor: { segment: 0, index: 0 }, more: false }))
    expect(renderThread(state, {}, t)).not.toContain('chat-earlier')
    expect(renderThread(reduce(createChat(), ev({ type: 'history', messages: kept })), {}, t)).not.toContain('chat-earlier')
  })

  test('both languages have the texts', () => {
    const { en, es } = chatTexts()
    expect(es['chat.earlier.divider']).toBe('Antes en esta conversación')
    expect(es['chat.earlier.show']).toBe('Ver lo anterior')
    for (const key of ['chat.earlier.divider', 'chat.earlier.show', 'chat.earlier.loading', 'chat.earlier.summary']) {
      expect(en[key]).toBeTruthy()
      expect(es[key]).toBeTruthy()
    }
  })
})
