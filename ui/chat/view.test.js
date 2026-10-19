// Run: bun test ui
// The chat's HTML is built by pure functions, so it is tested without a browser.
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { addUser, createChat, reduce } from './state.js'
import { insertFile, pickerQuery, renderDialog, renderEmpty, renderMessages, renderPath, renderProblem, keepRenameField, renderTabs, renderUndo, shortPath } from './view.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const ev = (event) => ({ type: 'chatEvent', event })
const run = (state, ...messages) => messages.reduce(reduce, state)

describe('messages', () => {
  test('user and assistant text is escaped', () => {
    let state = addUser(createChat(), '<script>alert(1)</script>')
    state = run(
      state,
      ev({ type: 'message_start', message: { role: 'assistant' } }),
      ev({ type: 'message_end', message: { role: 'assistant', content: [{ type: 'text', text: 'a < b & "c"' }] } }),
    )
    const html = renderMessages(state, t)
    expect(html).not.toContain('<script>')
    expect(html).toContain('&lt;script&gt;alert(1)&lt;/script&gt;')
    expect(html).toContain('a &lt; b &amp; &quot;c&quot;')
    expect(html).toContain('class="msg user"')
    expect(html).toContain('class="msg assistant"')
  })

  test('a tool group shows a count and keeps its rows hidden until expanded', () => {
    const state = run(
      createChat(),
      ev({ type: 'tool_execution_start', toolCallId: 'a', toolName: 'bash', args: { command: 'ls' } }),
      ev({ type: 'tool_execution_start', toolCallId: 'b', toolName: 'read', args: { path: 'x<y.rs' } }),
      ev({ type: 'tool_execution_end', toolCallId: 'a', toolName: 'bash', isError: false, result: { content: [{ type: 'text', text: 'out' }] } }),
    )
    const closed = renderMessages(state, t)
    expect(closed).toContain('chat.tools n=2')
    expect(closed).not.toContain('class="tool-row ')
    const open = renderMessages(state, t, new Set([0]))
    expect(open.match(/class="tool-row /g)).toHaveLength(2)
    expect(open).toContain('x&lt;y.rs')
    expect(open).toContain('status-done')
    expect(open).toContain('status-running')
  })

  test('the working indicator shows only while running', () => {
    const idle = renderMessages(createChat(), t)
    expect(idle).not.toContain('class="working"')
    const busy = renderMessages(reduce(createChat(), ev({ type: 'agent_start' })), t)
    expect(busy).toContain('class="working"')
    expect(busy).toContain('chat.working')
  })

  test('notices use their key and variables, or their text', () => {
    const state = run(
      createChat(),
      ev({ type: 'auto_retry_start', attempt: 2, maxAttempts: 3 }),
      ev({ type: 'command_failed', command: 'prompt', error: 'nope <b>' }),
    )
    const html = renderMessages(state, t)
    expect(html).toContain('chat.retry attempt=2,max=3')
    expect(html).toContain('nope &lt;b&gt;')
    expect(html).toContain('notice tone-warn')
    expect(html).toContain('notice tone-error')
  })

  test('an empty chat shows a start hint', () => {
    expect(renderEmpty(createChat(), t)).toContain('chat.hint.empty')
    expect(renderEmpty(addUser(createChat(), 'hi'), t)).toBe('')
  })
})

describe('the working folder', () => {
  test('the path is always shown, shortened under the home folder, with the full path as its title', () => {
    const html = renderPath({ ...createChat(), workdir: '/home/ana/Projects' })
    expect(html).toContain('~/Projects')
    expect(html).toContain('title="/home/ana/Projects"')
  })

  test('shortPath handles Linux and Mac homes and leaves other paths alone', () => {
    expect(shortPath('/home/ana/Projects/x')).toBe('~/Projects/x')
    expect(shortPath('/Users/ana/Code')).toBe('~/Code')
    expect(shortPath('/srv/code')).toBe('/srv/code')
    expect(shortPath('/home/ana')).toBe('~')
    expect(shortPath('')).toBe('')
  })

  test('an unknown folder still shows something', () => {
    expect(renderPath(createChat())).toContain('chat.path.unknown')
  })
})

describe('dialogs', () => {
  const dialog = (method, request) => ({ id: 'u<1', method, request })

  test('select renders one button per option plus cancel, escaped', () => {
    const html = renderDialog(dialog('select', { title: 'Allow <rm>?', options: ['Allow', 'Block & log'] }), t)
    expect(html).toContain('Allow &lt;rm&gt;?')
    expect(html.match(/data-chat="answer"/g)).toHaveLength(2)
    expect(html).toContain('data-value="Block &amp; log"')
    expect(html).toContain('data-chat="cancel"')
    expect(html).toContain('data-id="u&lt;1"')
  })

  test('confirm renders yes and no', () => {
    const html = renderDialog(dialog('confirm', { title: 'Switch model?', message: 'Free model' }), t)
    expect(html).toContain('Free model')
    expect(html).toContain('data-chat="confirm" data-id="u&lt;1" data-value="true"')
    expect(html).toContain('data-value="false"')
    expect(html).toContain('chat.yes')
    expect(html).toContain('chat.no')
  })

  test('input and editor render a field and OK, and keep the prefill', () => {
    const input = renderDialog(dialog('input', { title: 'Name?', placeholder: 'type…' }), t)
    expect(input).toContain('<input')
    expect(input).toContain('data-chat="submit"')
    const editor = renderDialog(dialog('editor', { title: 'Edit', prefill: 'line <1>' }), t)
    expect(editor).toContain('<textarea')
    expect(editor).toContain('line &lt;1&gt;')
  })
})

describe('problems', () => {
  test('a problem names what to do', () => {
    expect(renderProblem({ problem: 'pi-missing' }, t)).toContain('chat.problem.pi-missing')
    expect(renderProblem({ problem: null }, t)).toBe('')
  })

  test('every problem the chat can report has a string in both languages', () => {
    const rust = readFileSync(new URL('../../crates/aria/src/chat.rs', import.meta.url), 'utf8').split('#[cfg(test)]')[0]
    const core = readFileSync(new URL('../../crates/aria-core/src/pi.rs', import.meta.url), 'utf8').split('#[cfg(test)]')[0]
    const codes = new Set([
      ...[...rust.matchAll(/(?:Err\(|map_err\(\|_\| |=> )"([a-z]+-[a-z-]+)"/g)].map((m) => m[1]),
      ...[...core.matchAll(/=> "([a-z]+-[a-z]+)"/g)].map((m) => m[1]),
    ])
    expect(codes.has('pi-missing')).toBe(true)
    expect(codes.has('node-missing')).toBe(true)
    for (const lang of ['en', 'es']) {
      const catalog = JSON.parse(readFileSync(new URL(`../i18n/${lang}.json`, import.meta.url), 'utf8'))
      for (const code of codes) expect(catalog[`chat.problem.${code}`], `${lang}: chat.problem.${code}`).toBeDefined()
    }
  })
})

describe('pickers', () => {
  test('pickerQuery finds a slash command at the start and an @ file at the caret', () => {
    expect(pickerQuery('/ef', 3)).toEqual({ kind: 'command', query: 'ef', start: 0 })
    expect(pickerQuery('/effort high', 12)).toBeNull() // after a space it is an argument
    expect(pickerQuery('look at @src/ma', 15)).toEqual({ kind: 'file', query: 'src/ma', start: 8 })
    expect(pickerQuery('mail me@x.com', 13)).toBeNull() // an @ inside a word is not a file
    expect(pickerQuery('hello', 5)).toBeNull()
  })

  test('insertFile replaces the @ token with the path and one space', () => {
    expect(insertFile('look at @src/ma now', 15, 8, 'src/main.rs')).toEqual({ text: 'look at @src/main.rs now', caret: 21 })
    expect(insertFile('see @sr', 7, 4, 'src/a.rs')).toEqual({ text: 'see @src/a.rs ', caret: 14 })
  })
})

describe('the tab bar', () => {
  const tab = (id, extra = {}) => ({ id, title: id, status: 'idle', unread: false, ...extra })
  const bar = (sessions, active = sessions[0].id, closed = null) => ({ sessions, active, closed })

  test('titles are escaped and the active tab is marked', () => {
    const html = renderTabs(bar([tab('a', { title: '<img onerror=x>' }), tab('b')], 'b'), t)
    expect(html).not.toContain('<img')
    expect(html).toContain('&lt;img onerror=x&gt;')
    expect(html).toMatch(/aria-selected="true"[^>]*data-session="b"|data-session="b"[^>]*aria-selected="true"/)
    expect(html.match(/aria-selected="true"/g)).toHaveLength(1)
  })

  test('an untitled session shows the placeholder title', () => {
    expect(renderTabs(bar([tab('a', { title: '' })]), t)).toContain('chat.untitled')
  })

  test('each tab carries its status, and unread shows', () => {
    const html = renderTabs(bar([tab('a', { status: 'working' }), tab('b', { status: 'needs' }), tab('c', { status: 'done', unread: true })]), t)
    expect(html).toContain('status-working')
    expect(html).toContain('status-needs')
    expect(html).toContain('status-done')
    expect(html).toContain('unread')
  })

  test('close buttons exist only when there is more than one session', () => {
    expect(renderTabs(bar([tab('a')]), t)).not.toContain('data-chat="close-tab"')
    expect(renderTabs(bar([tab('a'), tab('b')]), t).match(/data-chat="close-tab"/g)).toHaveLength(2)
  })

  test('the plus button opens a new session', () => {
    expect(renderTabs(bar([tab('a')]), t)).toContain('data-chat="add"')
  })

  test('more than four sessions show an overflow count and, open, a menu with the rest', () => {
    const sessions = Array.from({ length: 6 }, (_, i) => tab(`s${i}`))
    const closed = renderTabs(bar(sessions), t)
    expect(closed).toContain('chat.tab.more n=2')
    expect(closed).not.toContain('role="menu"')
    const open = renderTabs(bar(sessions), t, { menuOpen: true })
    expect(open).toContain('role="menu"')
    expect(open.match(/data-chat="select"[^>]*data-session="s[45]"/g)).toHaveLength(2)
  })

  test('a hidden session that needs you shows on the overflow button', () => {
    const sessions = Array.from({ length: 6 }, (_, i) => tab(`s${i}`, i === 5 ? { status: 'needs' } : {}))
    const html = renderTabs(bar(sessions), t)
    expect(html).toMatch(/class="tab-more needs"/)
    expect(html).toContain('chat.tab.moreNeeds')
    const calm = renderTabs(bar(sessions.map((s) => ({ ...s, status: 'idle' }))), t)
    expect(calm).not.toMatch(/tab-more needs/)
  })

  test('a hidden session that finished unread also shows on the overflow button', () => {
    const sessions = Array.from({ length: 6 }, (_, i) => tab(`s${i}`, i === 4 ? { status: 'done', unread: true } : {}))
    expect(renderTabs(bar(sessions), t)).toMatch(/class="tab-more unread"/)
  })

  test('the plus is disabled at eight sessions', () => {
    const eight = Array.from({ length: 8 }, (_, i) => tab(`s${i}`))
    expect(renderTabs(bar(eight), t)).toMatch(/data-chat="add"[^>]*disabled|disabled[^>]*data-chat="add"/)
    expect(renderTabs(bar(eight.slice(0, 3)), t)).not.toMatch(/data-chat="add"[^>]*disabled/)
  })

  test('the tab bar is left alone while a rename field is open', () => {
    expect(keepRenameField({ renaming: 'a' }, true)).toBe(true)
    expect(keepRenameField({ renaming: 'a' }, false)).toBe(false) // the field has to be drawn first
    expect(keepRenameField({ renaming: null }, true)).toBe(false)
  })

  test('a tab being renamed shows a field with its title', () => {
    const html = renderTabs(bar([tab('a', { title: 'My <plan>' })]), t, { renaming: 'a' })
    expect(html).toContain('data-chat-rename')
    expect(html).toContain('value="My &lt;plan&gt;"')
  })

  test('the undo bar names the closed session', () => {
    expect(renderUndo(bar([tab('a')]), t)).toBe('')
    const html = renderUndo(bar([tab('a')], 'a', { entry: tab('x', { title: 'Old <one>' }), index: 1 }), t)
    expect(html).toContain('Old &lt;one&gt;')
    expect(html).toContain('data-chat="undo"')
  })
})

describe('texts', () => {
  test('every text key the chat view and state use exists in both languages', () => {
    const read = (file) => readFileSync(new URL(file, import.meta.url), 'utf8')
    const source = read('./view.js') + read('./state.js')
    const keys = new Set([...source.matchAll(/\bt\('(chat\.[a-zA-Z.-]+)'/g)].map((m) => m[1]))
    for (const status of ['working', 'needs', 'done', 'idle']) keys.add(`chat.status.${status}`)
    for (const key of [...source.matchAll(/key: '(chat\.[a-zA-Z.-]+)'/g)].map((m) => m[1])) keys.add(key)
    expect(keys.size).toBeGreaterThan(20)
    for (const lang of ['en', 'es']) {
      const catalog = JSON.parse(read(`../i18n/${lang}.json`))
      for (const key of keys) expect(catalog[key], `${lang}: ${key}`).toBeDefined()
    }
  })
})
