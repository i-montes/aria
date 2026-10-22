// The conversation as HTML: each entry of the chat state through the part that draws it, then the
// cards waiting for an answer, the queued messages and the working line. Pure; `ui` carries what the
// controller tracks: { expanded: Set (tool groups), openRows: Set (tool rows), moreOpen: Set, now }.
import { esc } from '../onboarding/components.js'
import { renderAnswers } from './parts/answers.js'
import { renderAssistant } from './parts/assistant.js'
import { renderDivider } from './parts/divider.js'
import { messageDomId } from './parts/index.js'
import { renderJev } from './parts/jev.js'
import { renderNote } from './parts/note.js'
import { renderMarkdown } from './parts/markdown.js'
import { renderPermission } from './parts/permission.js'
import { renderPill } from './parts/pill.js'
import { renderQuota } from './parts/quota.js'
import { renderToolStep, renderTools } from './parts/tools.js'
import { renderUser } from './parts/user.js'
import { renderWorking } from './parts/working.js'

/** The catalog key a dock notice is said with: the watchdog's has a one-minute and a many-minutes wording. */
const noticeKey = (key, vars) =>
  key === 'chat.watchdog.stopped' ? `${key}.${Number(vars?.minutes) === 1 ? 'one' : 'many'}` : key

/** Dialogs whose card sits in the thread (the question card is an overlay; the rest are plain dialogs). */
export const INLINE_CARDS = ['sandbox', 'write', 'connector', 'quota']

export const renderEmptyThread = (t) =>
  `<div class="thread-empty"><span class="thread-empty-title">${esc(t('chat.empty.title'))}</span><span class="thread-empty-text">${esc(t('chat.empty.text'))}</span></div>`

function renderCard(dialog, t) {
  const card = dialog.card
  if (card.kind === 'quota') {
    return renderQuota({ id: dialog.id, percent: card.percent, freeModel: card.freeModel, secondsLeft: card.countdown, options: card.options }, {}, t)
  }
  return renderPermission({ id: dialog.id, kind: card.kind, summary: card.summary, detail: card.detail || card.path, jev: card.jev, options: card.options }, {}, t)
}

/** The group of the run going on now (design: it is open while Aria works, with its live pulse). */
export const isLive = (state, index) => state.running && index >= (state.turnFrom ?? 0)

/** Open while live unless closed by hand; folded afterwards unless opened by hand. */
export const groupOpen = (state, index, opened, closed) => (isLive(state, index) ? !closed.has(index) : opened.has(index))

/** An entry's markup, kept per entry object: the reducer never changes an entry in place (a streamed word makes a new
 * one), so an entry whose object and `key` (what else its markup depends on) are the same renders the same. Without
 * it, every streamed word rebuilt the whole conversation. */
const rendered = new WeakMap()
function cached(entry, key, t, render) {
  const hit = rendered.get(entry)
  if (hit && hit.key === key && hit.t === t) return hit.html
  const html = render()
  rendered.set(entry, { key, t, html })
  return html
}

function renderEntry(state, entry, index, ui, t) {
  switch (entry.kind) {
    case 'user':
      return cached(entry, String(index), t, () => `<div class="thread-item" id="${messageDomId(index)}">${renderUser({ text: entry.text, files: entry.files }, {}, t)}</div>`)
    case 'assistant':
      return entry.text ? cached(entry, '', t, () => `<div class="thread-item">${renderAssistant({ text: entry.text }, {}, t)}</div>`) : ''
    case 'tools': {
      const live = isLive(state, index)
      const toolUi = { open: groupOpen(state, index, ui.expanded ?? new Set(), ui.collapsed ?? new Set()), openRows: ui.openRows ?? new Set(), moreOpen: ui.moreOpen ?? new Set() }
      const key = JSON.stringify([index, live, toolUi.open, [...toolUi.openRows], [...toolUi.moreOpen]])
      return cached(entry, key, t, () => `<div class="thread-item">${renderTools({ index, calls: entry.calls, live }, toolUi, t)}</div>`)
    }
    case 'jev':
      return `<div class="thread-item">${renderJev({ mode: entry.mode, action: entry.action }, {}, t)}</div>`
    case 'note':
      return `<div class="thread-item">${renderNote({ label: entry.label, text: entry.key ? t(entry.key, entry.vars) : entry.text }, {}, t)}</div>`
    case 'divider':
      return `<div class="thread-item">${renderDivider({ text: t(entry.key, entry.vars) }, {}, t)}</div>`
    case 'earlier': {
      // The chat's divider, and the summary folded under it as a tool row (open on click, by its id). Pi's summaries are
      // markdown (## Goal, ## Progress…): no chip, and the open row shows them through the chat's markdown.
      const open = (ui.openRows ?? new Set()).has(entry.id)
      const step = { id: entry.id, think: true, label: t('chat.earlier.summary'), chip: '', html: renderMarkdown(entry.summary), status: 'done' }
      return cached(entry, String(open), t, () => `<div class="thread-item thread-earlier">${renderDivider({ text: t('chat.earlier.divider') }, {}, t)}${entry.summary.trim() ? renderToolStep({ call: step }, { openRows: open ? new Set([entry.id]) : new Set() }, t) : ''}</div>`)
    }
    case 'answers':
      return `<div class="thread-item">${renderAnswers({ answers: entry.answers }, {}, t)}</div>`
    case 'notice':
      return `<div class="notice tone-${esc(entry.tone)}">${esc(entry.key ? t(noticeKey(entry.key, entry.vars), entry.vars) : entry.text)}</div>`
    default:
      return ''
  }
}

/** The thread as one markup string per top-level element (see patch.js `patchItems`). */
export function threadItems(state, ui = {}, t) {
  const entries = state.messages.map((entry, index) => renderEntry(state, entry, index, ui, t))
  const cards = state.dialogs.filter((d) => d.card && INLINE_CARDS.includes(d.card.kind)).map((d) => `<div class="thread-item">${renderCard(d, t)}</div>`)
  const queued = state.queue.map((q, i) => `<div class="thread-item">${renderUser({ text: q.text, files: q.files, queued: true, steer: q.steer, id: String(i) }, {}, t)}</div>`)
  // The working line's markup stays the same while it runs (the view ticks its elapsed time in place): markup that
  // changed with the clock rebuilt the line on every streamed token, which restarted its animation.
  const steady = (startedAt) => ({ now: Number.isFinite(startedAt) ? startedAt : (ui.now ?? Date.now()) })
  const working = state.running ? `<div class="thread-item thread-working">${renderWorking({ phase: state.phase, startedAt: state.startedAt }, steady(state.startedAt), t)}</div>` : ''
  const blank = !state.messages.length && !state.running && !state.queue.length && !cards.length
  // A session whose history has not arrived yet is loading, not new (unless its Pi cannot start: the problem shows).
  const loading = blank && state.loaded === false && !state.problem
  const empty = blank && !loading ? renderEmptyThread(t) : ''
  // "Show earlier" above the first message while older history is left; a pill with nothing to press while it loads.
  const paging = state.earlier ?? {}
  const earlier = paging.more
    ? `<div class="thread-item thread-earlier-more">${paging.loading ? renderPill({ label: t('chat.earlier.loading'), static: true }, {}, t) : renderPill({ label: t('chat.earlier.show'), action: 'chat-earlier' }, {}, t)}</div>`
    : ''
  const loader = loading ? `<div class="thread-item thread-working">${renderWorking({ label: t('chat.loading'), startedAt: state.loadingSince }, steady(state.loadingSince), t)}</div>` : ''
  return [empty, loader, earlier, ...entries, ...cards, working, ...queued].filter(Boolean)
}

export const renderThread = (state, ui = {}, t) => threadItems(state, ui, t).join('')
