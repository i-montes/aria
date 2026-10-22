// The dock's tabs as plain data (the design's session-tabs): which sessions exist, which is active,
// what each one is doing, and the tab that was just closed (for undo). Pure functions; the chat view
// keeps one conversation per session and Rust keeps one Pi per session.

export const MAX_SESSIONS = 8
const TITLE_LIMIT = 28

// A scheduled task's tab carries `schedule` (its task id): the dock made it, it does not count against the eight, and
// the bar shows it only while it has something to show (`shownSessions`).
const entry = (id, title = '', schedule = undefined) => ({ id, title, status: 'idle', unread: false, ...(schedule ? { schedule } : {}) })
const withSessions = (tabs, sessions) => ({ ...tabs, sessions })
const find = (tabs, id) => tabs.sessions.find((s) => s.id === id)

/** `saved` is `{ sessions: [{ id, title, schedule? }], active }` from `~/.aria/sessions.json`, or nothing. There is
 * always a regular tab, and a regular tab is active when the saved one is not there. */
export function createTabs(saved, makeId) {
  const sessions = (saved?.sessions ?? []).map((s) => entry(s.id, s.title ?? '', s.schedule))
  if (!sessions.some((s) => !s.schedule)) sessions.push(entry(makeId()))
  const active = sessions.some((s) => s.id === saved?.active && !s.schedule) ? saved.active : sessions.find((s) => !s.schedule).id
  return { sessions, active, closed: null }
}

export const toSaved = (tabs) => ({
  sessions: tabs.sessions.map(({ id, title, schedule }) => (schedule ? { id, title, schedule } : { id, title })),
  active: tabs.active,
})

/** The tabs the person opened (the scheduled ones aside). */
export const regularCount = (tabs) => tabs.sessions.filter((s) => !s.schedule).length

export function addSession(tabs, id) {
  if (regularCount(tabs) >= MAX_SESSIONS) return tabs
  return { ...withSessions(tabs, [...tabs.sessions, entry(id)]), active: id, closed: null }
}

export function selectSession(tabs, id) {
  if (!find(tabs, id)) return tabs
  return { ...withSessions(tabs, tabs.sessions.map((s) => (s.id === id ? { ...s, unread: false } : s))), active: id }
}

export function closeSession(tabs, id) {
  const index = tabs.sessions.findIndex((s) => s.id === id)
  if (index < 0 || tabs.sessions.length <= 1) return tabs
  const rest = tabs.sessions.filter((s) => s.id !== id)
  const active = tabs.active === id ? rest[Math.min(index, rest.length - 1)].id : tabs.active
  return { sessions: rest, active, closed: { entry: tabs.sessions[index], index } }
}

export function undoClose(tabs) {
  if (!tabs.closed || tabs.sessions.length >= MAX_SESSIONS) return { ...tabs, closed: null }
  const sessions = [...tabs.sessions]
  sessions.splice(Math.min(tabs.closed.index, sessions.length), 0, { ...tabs.closed.entry, status: 'idle', unread: false })
  return { ...withSessions(tabs, sessions), closed: null }
}

export function renameSession(tabs, id, title) {
  const clean = String(title ?? '').trim()
  if (!clean || !find(tabs, id)) return tabs
  return withSessions(tabs, tabs.sessions.map((s) => (s.id === id ? { ...s, title: clean } : s)))
}

/** The first line of the message, whitespace collapsed, cut to 28 characters with an ellipsis. */
export function titleFromMessage(text) {
  const line = String(text ?? '').split('\n').map((l) => l.trim()).find(Boolean) ?? ''
  const clean = line.replace(/\s+/g, ' ')
  return clean.length > TITLE_LIMIT ? `${clean.slice(0, TITLE_LIMIT - 1)}…` : clean
}

export function setTitleIfEmpty(tabs, id, text) {
  const current = find(tabs, id)
  if (!current || current.title) return tabs
  return renameSession(tabs, id, titleFromMessage(text))
}

/** What the tab should show, from the session's conversation: needs you > working > done (unread) > idle. */
export function applyStatus(tabs, id, chat, isActive) {
  const current = find(tabs, id)
  if (!current) return tabs
  let { status, unread } = current
  if (chat.dialogs.length) status = 'needs'
  else if (chat.running) status = 'working'
  else if (current.status === 'working' && !isActive) [status, unread] = ['done', true]
  else if (current.status === 'done' && current.unread && !isActive) status = 'done'
  else [status, unread] = ['idle', isActive ? false : unread]
  if (isActive && !chat.dialogs.length && !chat.running) unread = false
  if (status === current.status && unread === current.unread) return tabs
  return withSessions(tabs, tabs.sessions.map((s) => (s.id === id ? { ...s, status, unread } : s)))
}

/** A scheduled task's tab the dock just made: added at the end, never taking the focus. */
export function addScheduled(tabs, id, title, schedule) {
  if (find(tabs, id)) return tabs
  return withSessions(tabs, [...tabs.sessions, entry(id, title, schedule)])
}

/** The tabs the bar shows: every regular one, and a scheduled one while it is active, unread, working or waiting. */
export const shownSessions = (tabs) =>
  tabs.sessions.filter((s) => !s.schedule || s.id === tabs.active || s.unread || s.status === 'working' || s.status === 'needs')

/** Closing a scheduled tab only hides it (its task stays): it is read, and the focus goes back to a regular tab. */
export function hideScheduled(tabs, id) {
  const sessions = tabs.sessions.map((s) => (s.id === id ? { ...s, unread: false, status: 'idle' } : s))
  const active = tabs.active === id ? (sessions.find((s) => !s.schedule)?.id ?? tabs.active) : tabs.active
  return { ...withSessions(tabs, sessions), active }
}

/** A deleted task's tab goes for good. */
export function removeSession(tabs, id) {
  if (!find(tabs, id)) return tabs
  const rest = tabs.sessions.filter((s) => s.id !== id)
  const active = tabs.active === id ? (rest.find((s) => !s.schedule)?.id ?? rest[0]?.id) : tabs.active
  return { ...withSessions(tabs, rest), active }
}

/** The bar shows `max` tabs; the active one always does (it takes the last slot); the rest overflow. */
export function visibleTabs(tabs, max = 4) {
  const activeAt = tabs.sessions.findIndex((s) => s.id === tabs.active)
  let visible = tabs.sessions.slice(0, max)
  if (activeAt >= max) visible = [...visible.slice(0, max - 1), tabs.sessions[activeAt]]
  const shown = new Set(visible.map((s) => s.id))
  return { visible, overflow: tabs.sessions.filter((s) => !shown.has(s.id)) }
}
