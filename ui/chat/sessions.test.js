// Run: bun test ui
import { describe, expect, test } from 'bun:test'
import { createChat } from './state.js'
import {
  addSession, applyStatus, closeSession, createTabs, MAX_SESSIONS, renameSession, selectSession, setTitleIfEmpty, titleFromMessage,
  toSaved, undoClose, visibleTabs,
  addScheduled,
  hideScheduled,
  regularCount,
  removeSession,
  shownSessions,
  addFromDock,
} from './sessions.js'

let counter = 0
const makeId = () => `id-${++counter}`
const tab = (id, extra = {}) => ({ id, title: id, status: 'idle', unread: false, ...extra })
const tabs = (...ids) => ({ sessions: ids.map((id) => tab(id)), active: ids[0], closed: null })
const ids = (t) => t.sessions.map((s) => s.id)

describe('starting', () => {
  test('nothing saved gives one fresh active session', () => {
    counter = 0
    const t = createTabs(null, makeId)
    expect(t.sessions).toEqual([{ id: 'id-1', title: '', status: 'idle', unread: false }])
    expect(t.active).toBe('id-1')
  })

  test('saved sessions come back idle, with the saved one active', () => {
    const t = createTabs({ sessions: [{ id: 'a', title: 'Fix' }, { id: 'b', title: 'Plan' }], active: 'b' }, makeId)
    expect(ids(t)).toEqual(['a', 'b'])
    expect(t.active).toBe('b')
    expect(t.sessions[0]).toMatchObject({ title: 'Fix', status: 'idle', unread: false })
  })

  test('an active id that is not in the list falls back to the first', () => {
    expect(createTabs({ sessions: [{ id: 'a', title: '' }], active: 'zzz' }, makeId).active).toBe('a')
  })

  test('toSaved keeps ids, titles and the active one only', () => {
    const t = { sessions: [tab('a', { status: 'working', unread: true })], active: 'a', closed: null }
    expect(toSaved(t)).toEqual({ sessions: [{ id: 'a', title: 'a' }], active: 'a' })
  })
})

describe('adding, selecting, closing', () => {
  test('a new session is appended, empty-titled and active', () => {
    const t = addSession(tabs('a', 'b'), 'c')
    expect(ids(t)).toEqual(['a', 'b', 'c'])
    expect(t.active).toBe('c')
    expect(t.sessions[2].title).toBe('')
  })

  test('there are never more than eight', () => {
    let t = tabs('s0')
    for (let i = 1; i < 12; i++) t = addSession(t, `s${i}`)
    expect(t.sessions).toHaveLength(MAX_SESSIONS)
    expect(t.active).toBe('s7')
  })

  test('selecting a session makes it active and marks it read', () => {
    const t = selectSession({ ...tabs('a', 'b'), sessions: [tab('a'), tab('b', { status: 'done', unread: true })] }, 'b')
    expect(t.active).toBe('b')
    expect(t.sessions[1].unread).toBe(false)
  })

  test('selecting an unknown id changes nothing', () => {
    const t = tabs('a', 'b')
    expect(selectSession(t, 'zzz')).toEqual(t)
  })

  test('closing the active tab activates its neighbour and remembers it for undo', () => {
    const t = closeSession({ ...tabs('a', 'b', 'c'), active: 'b' }, 'b')
    expect(ids(t)).toEqual(['a', 'c'])
    expect(t.active).toBe('c')
    expect(t.closed).toMatchObject({ entry: { id: 'b' }, index: 1 })
  })

  test('closing the last tab of the list picks the one before it', () => {
    expect(closeSession({ ...tabs('a', 'b'), active: 'b' }, 'b').active).toBe('a')
  })

  test('closing a background tab keeps the active one', () => {
    expect(closeSession(tabs('a', 'b', 'c'), 'c').active).toBe('a')
  })

  test('the last session is never closed', () => {
    const t = tabs('a')
    expect(closeSession(t, 'a')).toEqual(t)
  })

  test('undo puts the tab back where it was, inactive', () => {
    const closed = closeSession(tabs('a', 'b', 'c'), 'b')
    const t = undoClose(closed)
    expect(ids(t)).toEqual(['a', 'b', 'c'])
    expect(t.active).toBe('a')
    expect(t.closed).toBeNull()
  })

  test('undo with nothing closed, or with a full list, changes nothing', () => {
    const t = tabs('a', 'b')
    expect(undoClose(t)).toEqual(t)
    let full = tabs('s0')
    for (let i = 1; i < 8; i++) full = addSession(full, `s${i}`)
    const closed = closeSession(full, 's3')
    const refilled = addSession(closed, 'extra')
    expect(undoClose(refilled).sessions).toHaveLength(8)
    expect(ids(undoClose(refilled))).not.toContain('s3')
  })
})

describe('titles', () => {
  test('rename trims, and ignores an empty title', () => {
    expect(renameSession(tabs('a'), 'a', '  New plan  ').sessions[0].title).toBe('New plan')
    expect(renameSession(tabs('a'), 'a', '   ').sessions[0].title).toBe('a')
  })

  test('the first message becomes the title: first line, short, with an ellipsis', () => {
    expect(titleFromMessage('Fix the failing build\nplease')).toBe('Fix the failing build')
    expect(titleFromMessage('  lots   of    spaces ')).toBe('lots of spaces')
    const long = titleFromMessage('x'.repeat(60))
    expect(long).toHaveLength(28)
    expect(long.endsWith('…')).toBe(true)
    expect(titleFromMessage('\n\n   ')).toBe('')
  })

  test('only an untitled session takes its title from the first message', () => {
    const t = { sessions: [tab('a', { title: '' }), tab('b', { title: 'Mine' })], active: 'a', closed: null }
    expect(setTitleIfEmpty(t, 'a', 'Hello there').sessions[0].title).toBe('Hello there')
    expect(setTitleIfEmpty(t, 'b', 'Hello there').sessions[1].title).toBe('Mine')
  })
})

describe('status', () => {
  const chat = (extra) => ({ ...createChat(), ...extra })
  const one = (extra = {}) => ({ sessions: [tab('a', extra)], active: 'x', closed: null })

  test('a running session is working', () => {
    expect(applyStatus(one(), 'a', chat({ running: true }), false).sessions[0].status).toBe('working')
  })

  test('an open dialog means it needs you, even while running', () => {
    const dialogs = [{ id: 'u1', method: 'select', request: {} }]
    expect(applyStatus(one(), 'a', chat({ running: true, dialogs }), false).sessions[0].status).toBe('needs')
  })

  test('a background run that settles is done and unread', () => {
    const t = applyStatus(one({ status: 'working' }), 'a', chat({ running: false }), false)
    expect(t.sessions[0]).toMatchObject({ status: 'done', unread: true })
  })

  test('an unread done stays done until the tab is opened', () => {
    const t = applyStatus(one({ status: 'done', unread: true }), 'a', chat({}), false)
    expect(t.sessions[0]).toMatchObject({ status: 'done', unread: true })
  })

  test('the tab you are looking at is never unread', () => {
    const t = applyStatus(one({ status: 'working' }), 'a', chat({ running: false }), true)
    expect(t.sessions[0]).toMatchObject({ status: 'idle', unread: false })
  })

  test('an unknown id changes nothing', () => {
    const t = one()
    expect(applyStatus(t, 'zzz', chat({ running: true }), false)).toEqual(t)
  })
})

describe('what fits in the bar', () => {
  const many = (n, active) => ({ sessions: Array.from({ length: n }, (_, i) => tab(`s${i}`)), active, closed: null })

  test('up to four show, the rest go to the overflow', () => {
    const { visible, overflow } = visibleTabs(many(6, 's0'))
    expect(visible.map((s) => s.id)).toEqual(['s0', 's1', 's2', 's3'])
    expect(overflow.map((s) => s.id)).toEqual(['s4', 's5'])
  })

  test('the active tab is always visible, taking the last slot', () => {
    const { visible, overflow } = visibleTabs(many(6, 's5'))
    expect(visible.map((s) => s.id)).toEqual(['s0', 's1', 's2', 's5'])
    expect(overflow.map((s) => s.id)).toEqual(['s3', 's4'])
  })

  test('few tabs have no overflow', () => {
    expect(visibleTabs(many(2, 's0')).overflow).toEqual([])
  })
})

describe('scheduled tabs', () => {
  const base = () => createTabs({ sessions: [{ id: 'a', title: 'A' }], active: 'a' }, () => 'new')

  test('they do not count against the eight, and come in without taking the focus', () => {
    let tabs = base()
    for (let n = 0; n < 7; n++) tabs = addSession(tabs, `r${n}`)
    tabs = addScheduled(tabs, 's-1', 'Notion tasks', 's-1')
    expect(tabs.sessions).toHaveLength(9)
    expect(regularCount(tabs)).toBe(8)
    expect(tabs.active).toBe('r6')
    expect(addSession(tabs, 'r9')).toBe(tabs)
    expect(addScheduled(tabs, 's-1', 'Notion tasks', 's-1')).toBe(tabs)
  })

  test('a scheduled tab shows only while it has news, works or is active', () => {
    let tabs = addScheduled(base(), 's-1', 'Notion tasks', 's-1')
    expect(shownSessions(tabs).map((s) => s.id)).toEqual(['a'])
    tabs = { ...tabs, sessions: tabs.sessions.map((s) => (s.id === 's-1' ? { ...s, unread: true, status: 'done' } : s)) }
    expect(shownSessions(tabs).map((s) => s.id)).toEqual(['a', 's-1'])
    tabs = hideScheduled(selectSession(tabs, 's-1'), 's-1')
    expect(tabs.active).toBe('a')
    expect(shownSessions(tabs).map((s) => s.id)).toEqual(['a'])
  })

  test('they are saved with their task and come back', () => {
    const tabs = addScheduled(base(), 's-1', 'Notion tasks', 's-1')
    const saved = toSaved(tabs)
    expect(saved.sessions[1]).toEqual({ id: 's-1', title: 'Notion tasks', schedule: 's-1' })
    expect(saved.sessions[0]).toEqual({ id: 'a', title: 'A' })
    expect(createTabs(saved, () => 'x').sessions[1].schedule).toBe('s-1')
  })

  test('a list of only scheduled tabs gets a regular one, and it is the active one', () => {
    const tabs = createTabs({ sessions: [{ id: 's-1', title: 'Notion tasks', schedule: 's-1' }], active: 's-1' }, () => 'fresh')
    expect(tabs.sessions.map((s) => s.id)).toEqual(['s-1', 'fresh'])
    expect(tabs.active).toBe('fresh')
  })

  test('a deleted task removes its tab', () => {
    let tabs = selectSession(addScheduled(base(), 's-1', 'Notion tasks', 's-1'), 's-1')
    tabs = removeSession(tabs, 's-1')
    expect(tabs.sessions.map((s) => s.id)).toEqual(['a'])
    expect(tabs.active).toBe('a')
  })
})

describe('tabs the dock opens (a terminal)', () => {
  const base = () => createTabs({ sessions: [{ id: 'a', title: 'A' }], active: 'a' }, () => 'new')

  test('come in at the end without taking the focus, once', () => {
    const tabs = addFromDock(base(), 'b', '')
    expect(tabs.sessions.map((s) => s.id)).toEqual(['a', 'b'])
    expect(tabs.active).toBe('a')
    expect(addFromDock(tabs, 'b', '')).toBe(tabs)
  })

  test('the one a terminal closed goes away', () => {
    const tabs = removeSession(addFromDock(base(), 'b', ''), 'b')
    expect(tabs.sessions.map((s) => s.id)).toEqual(['a'])
  })
})
