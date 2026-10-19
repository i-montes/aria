// Run: bun test ui/chat/parts/tabs.test.js
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { TEXTS, renderSessionTabs } from './tabs.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const hostile = '<script>alert("x")</script> & \'q\''
const tab = (id, extra = {}) => ({ id, title: id, status: 'idle', unread: false, ...extra })
const bar = (sessions, active = sessions[0].id, closed = null) => ({ sessions, active, closed })
const render = (tabs, ui = {}) => renderSessionTabs(tabs, ui, t)

// The same assertions as the old `renderTabs` tests in view.test.js, against the new renderer.
describe('the tab bar (what the old renderTabs guaranteed)', () => {
  test('titles are escaped and the active tab is marked', () => {
    const html = render(bar([tab('a', { title: '<img onerror=x>' }), tab('b')], 'b'))
    expect(html).not.toContain('<img')
    expect(html).toContain('&lt;img onerror=x&gt;')
    expect(html).toMatch(/aria-selected="true"[^>]*data-session="b"|data-session="b"[^>]*aria-selected="true"/)
    expect(html.match(/aria-selected="true"/g)).toHaveLength(1)
  })

  test('an untitled session shows the placeholder title', () => {
    expect(render(bar([tab('a', { title: '' })]))).toContain('chat.untitled')
  })

  test('each tab carries its status, and unread shows', () => {
    const html = render(bar([tab('a', { status: 'working' }), tab('b', { status: 'needs' }), tab('c', { status: 'done', unread: true })]))
    expect(html).toContain('status-working')
    expect(html).toContain('status-needs')
    expect(html).toContain('status-done')
    expect(html).toContain('unread')
  })

  test('close buttons exist only when there is more than one session', () => {
    expect(render(bar([tab('a')]))).not.toContain('data-chat="close-tab"')
    expect(render(bar([tab('a'), tab('b')])).match(/data-chat="close-tab"/g)).toHaveLength(2)
  })

  test('the plus button opens a new session, unless it is hidden', () => {
    expect(render(bar([tab('a')]))).toContain('data-chat="add"')
    expect(render(bar([tab('a')]), { hideNew: true })).not.toContain('data-chat="add"')
  })

  test('more than four sessions show an overflow count and, open, a menu with the rest', () => {
    const sessions = Array.from({ length: 6 }, (_, i) => tab(`s${i}`))
    const closed = render(bar(sessions))
    expect(closed).toContain('chat.tab.more n=2')
    expect(closed).not.toContain('role="menu"')
    const open = render(bar(sessions), { menuOpen: true })
    expect(open).toContain('role="menu"')
    expect(open.match(/role="menuitem" data-chat="select" data-session="s[45]"/g)).toHaveLength(2)
  })

  test('a hidden session that needs you shows on the overflow button', () => {
    const sessions = Array.from({ length: 6 }, (_, i) => tab(`s${i}`, i === 5 ? { status: 'needs' } : {}))
    const html = render(bar(sessions))
    expect(html).toMatch(/class="p-tabs-more needs"/)
    expect(html).toContain('chat.tab.moreNeeds')
    const calm = render(bar(sessions.map((s) => ({ ...s, status: 'idle' }))))
    expect(calm).not.toMatch(/p-tabs-more needs/)
    expect(calm).not.toContain('chat.tab.moreNeeds')
  })

  test('a hidden session that finished unread also shows on the overflow button', () => {
    const sessions = Array.from({ length: 6 }, (_, i) => tab(`s${i}`, i === 4 ? { status: 'done', unread: true } : {}))
    expect(render(bar(sessions))).toMatch(/class="p-tabs-more unread"/)
  })

  test('the plus is disabled at eight sessions', () => {
    const eight = Array.from({ length: 8 }, (_, i) => tab(`s${i}`))
    expect(render(bar(eight))).toMatch(/data-chat="add"[^>]*disabled|disabled[^>]*data-chat="add"/)
    expect(render(bar(eight.slice(0, 3)))).not.toMatch(/data-chat="add"[^>]*disabled/)
  })

  test('a tab being renamed shows a field with its title', () => {
    const html = render(bar([tab('a', { title: 'My <plan>' })]), { renaming: 'a' })
    expect(html).toContain('data-chat-rename')
    expect(html).toContain('data-session="a"')
    expect(html).toContain('value="My &lt;plan&gt;"')
    expect(html).not.toContain('role="tab"') // the field replaces the button
  })

  test('the undo bar names the closed session', () => {
    expect(render(bar([tab('a')]))).not.toContain('data-chat="undo"')
    const html = render(bar([tab('a')], 'a', { entry: tab('x', { title: 'Old <one>' }), index: 1 }))
    expect(html).toContain('Old &lt;one&gt;')
    expect(html).toContain('data-chat="undo"')
    expect(render(bar([tab('a')], 'a', { entry: tab('x', { title: 'Old' }), index: 1 }), { hideUndo: true })).not.toContain('data-chat="undo"')
    expect(render(bar([tab('a')], 'a', { entry: tab('x', { title: '' }), index: 1 }))).toContain('chat.undo.closed title=chat.untitled')
  })
})

describe('the design behaviours', () => {
  test('maxVisible sets how many tabs show; the active one always shows', () => {
    const sessions = Array.from({ length: 5 }, (_, i) => tab(`s${i}`))
    const two = render(bar(sessions, 's4'), { maxVisible: 2 })
    expect(two.match(/role="tab"/g)).toHaveLength(2)
    expect(two).toMatch(/data-session="s4"[^>]*>|aria-selected="true"/)
    expect(two).toContain('chat.tab.more n=3')
    const six = render(bar(sessions), { maxVisible: 6 })
    expect(six.match(/role="tab"/g)).toHaveLength(5)
    expect(six).not.toContain('p-tabs-more')
    expect(render(bar(sessions), {}).match(/role="tab"/g)).toHaveLength(4) // default 4
    expect(render(bar(sessions), { maxVisible: 1 }).match(/role="tab"/g)).toHaveLength(2) // at least 2
  })

  test('the overflow button announces itself, the menu closes when nothing is hidden', () => {
    const sessions = Array.from({ length: 6 }, (_, i) => tab(`s${i}`))
    const html = render(bar(sessions), { menuOpen: true })
    expect(html).toContain('aria-expanded="true"')
    expect(html).toContain('data-chat="menu"')
    expect(html).toContain('chat.tabs.moreLabel n=2')
    expect(render(bar(sessions.slice(0, 3)), { menuOpen: true })).not.toContain('role="menu"')
  })

  test('menu rows show each hidden session with its state', () => {
    const sessions = Array.from({ length: 6 }, (_, i) => tab(`s${i}`, i === 5 ? { status: 'needs' } : i === 4 ? { status: 'working' } : {}))
    const html = render(bar(sessions), { menuOpen: true })
    expect(html).toContain('chat.status.needs')
    expect(html).toContain('chat.status.working')
  })

  const WINDOWS = [{ id: '5h', label: '5h', percent: 42, resetsAt: null }, { id: 'week', label: 'Week', percent: 17, resetsAt: null }]

  test('hovering a tab shows the same quota card as the header chip, and only with more than one session', () => {
    expect(render(bar([tab('a')]))).not.toContain('role="tooltip"')
    const sessions = [tab('a', { quotaWindows: WINDOWS, provider: 'OpenCode Go', model: 'GLM 5', context: 42.4 }), tab('b', { quotaWindows: WINDOWS }), tab('c')]
    const html = render(bar(sessions))
    expect(html.match(/role="tooltip"/g)).toHaveLength(3)
    expect(html.match(/class="p-qb-card"/g)).toHaveLength(3)
    expect(html).toContain('OpenCode Go')
    expect(html).toContain('p-qb-hero')
    expect(html).toContain('width: 42%')
    expect(html).toContain('chat.quotabars.window.5h')
    expect(html).toContain('GLM 5')
    expect(html).toContain('>42%<')
  })

  test('the card of a tab starts with the tab\'s title and state', () => {
    const html = render(bar([tab('a', { title: 'Fix the build', status: 'working', quotaWindows: WINDOWS }), tab('b')]))
    const first = html.slice(html.indexOf('class="p-qb-card"'))
    expect(first.indexOf('Fix the build')).toBeLessThan(first.indexOf('p-qb-hero'))
    expect(first).toContain('chat.status.working')
  })

  test('a session whose provider has no quota to read says so, and a free model says that', () => {
    const none = render(bar([tab('a', { quotaWindows: [] }), tab('b')]))
    expect(none).toContain('chat.quotabars.none')
    const free = render(bar([tab('a', { free: true }), tab('b')]))
    expect(free).toContain('chat.quotabars.free')
  })

  test('a tab has no native tooltip of its own (its card is the hover), only an accessible name', () => {
    const html = render(bar([tab('a', { title: '' }), tab('b', { title: 'Plan' })]))
    const mains = html.match(/<button[^>]*class="p-tabs-main"[^>]*>/g)
    expect(mains).toHaveLength(2)
    for (const main of mains) {
      expect(main).not.toMatch(/\stitle=/)
      expect(main).toContain('aria-label=')
    }
  })

  test('tooltips of the right half of the bar open to the left', () => {
    const html = render(bar([tab('a'), tab('b'), tab('c'), tab('d')]))
    const tips = html.match(/class="p-tabs-tip[^"]*"/g)
    expect(tips).toEqual(['class="p-tabs-tip"', 'class="p-tabs-tip"', 'class="p-tabs-tip is-right"', 'class="p-tabs-tip is-right"'])
  })

  test('the tooltip state follows the tab: idle, working, needs, done', () => {
    const html = render(bar([tab('a'), tab('b', { status: 'working' }), tab('c', { status: 'needs' }), tab('d', { status: 'idle', unread: true })]))
    expect(html).toContain('chat.status.working')
    expect(html).toContain('chat.status.needs')
    expect(html).toContain('chat.status.done') // unread counts as done
    expect(html).toContain('chat.status.idle')
  })

  test('selecting, closing and the plus carry the session id', () => {
    const html = render(bar([tab('a'), tab('b')]))
    expect(html).toContain('data-chat="select" data-session="a"')
    expect(html).toContain('data-chat="close-tab" data-session="b"')
    expect(html).toContain('p-tabs-x')
  })

  test('the plus label explains the limit', () => {
    const eight = Array.from({ length: 8 }, (_, i) => tab(`s${i}`))
    expect(render(bar(eight))).toContain('chat.tabs.limit')
    expect(render(bar(eight.slice(0, 2)))).toContain('aria-label="chat.new"')
  })

  test('a tab names its state to screen readers', () => {
    expect(render(bar([tab('a', { status: 'needs' })]))).toContain('aria-label="a, chat.status.needs"')
    expect(render(bar([tab('a')]))).toContain('aria-label="a"')
  })
})

describe('escaping', () => {
  test('titles, ids, models and the closed title are escaped everywhere', () => {
    const sessions = Array.from({ length: 6 }, (_, i) => tab(`id${i}"<`, { title: hostile, model: hostile, status: i === 5 ? 'needs' : 'idle' }))
    const html = render(bar(sessions, sessions[0].id, { entry: tab('z', { title: hostile }), index: 0 }), { menuOpen: true, renaming: sessions[1].id, quota: 10, hasQuota: true })
    expect(html).not.toContain('<script>')
    expect(html).not.toContain('alert("x")')
    expect(html).toContain('&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &#39;q&#39;')
    expect(html).toContain('data-session="id0&quot;&lt;"')
    expect(html).not.toContain('id0"<')
  })

  test('numbers coming from the data are clamped, not injected', () => {
    const html = render(bar([tab('a', { quotaWindows: [{ id: '5h', label: '5h', percent: '1"><script>' }], context: '9999' }), tab('b')]))
    expect(html).not.toContain('<script>')
    expect(html).not.toContain('<b>')
    expect(html).toContain('>100%<') // context clamps to 100
  })
})

describe('texts', () => {
  test('every key used exists in both languages with the same keys', () => {
    const source = readFileSync(new URL('./tabs.js', import.meta.url), 'utf8')
    const used = [...source.matchAll(/\bt\('(chat\.[a-zA-Z.]+)'/g)].map((m) => m[1])
    for (const status of ['working', 'needs', 'done', 'idle']) used.push(`chat.status.${status}`)
    expect(used.length).toBeGreaterThan(10)
    for (const lang of ['en', 'es']) for (const key of used) expect(TEXTS[lang][key], `${lang}: ${key}`).toBeTruthy()
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
  })

  test('the reused catalog keys keep the wording the app already has', () => {
    const en = JSON.parse(readFileSync(new URL('../../i18n/en.json', import.meta.url), 'utf8'))
    const es = JSON.parse(readFileSync(new URL('../../i18n/es.json', import.meta.url), 'utf8'))
    for (const key of Object.keys(TEXTS.en)) {
      if (en[key] !== undefined) expect(TEXTS.en[key], key).toBe(en[key])
      if (es[key] !== undefined) expect(TEXTS.es[key], key).toBe(es[key])
    }
  })
})
