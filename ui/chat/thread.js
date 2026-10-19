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
import { renderPermission } from './parts/permission.js'
import { renderQuota } from './parts/quota.js'
import { renderTools } from './parts/tools.js'
import { renderUser } from './parts/user.js'
import { renderWorking } from './parts/working.js'

/** Dialogs whose card sits in the thread (the question card is an overlay; the rest are plain dialogs). */
export const INLINE_CARDS = ['sandbox', 'write', 'quota']

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

export function renderThread(state, ui = {}, t) {
  const opened = ui.expanded ?? new Set()
  const closed = ui.collapsed ?? new Set()
  const toolUi = (index) => ({ open: groupOpen(state, index, opened, closed), openRows: ui.openRows ?? new Set(), moreOpen: ui.moreOpen ?? new Set() })
  const entries = state.messages
    .map((entry, index) => {
      switch (entry.kind) {
        case 'user':
          return `<div class="thread-item" id="${messageDomId(index)}">${renderUser({ text: entry.text }, {}, t)}</div>`
        case 'assistant':
          return entry.text ? `<div class="thread-item">${renderAssistant({ text: entry.text }, {}, t)}</div>` : ''
        case 'tools':
          return `<div class="thread-item">${renderTools({ index, calls: entry.calls, live: isLive(state, index) }, toolUi(index), t)}</div>`
        case 'jev':
          return `<div class="thread-item">${renderJev({ mode: entry.mode, action: entry.action }, {}, t)}</div>`
        case 'note':
          return `<div class="thread-item">${renderNote({ label: entry.label, text: entry.key ? t(entry.key) : entry.text }, {}, t)}</div>`
        case 'divider':
          return `<div class="thread-item">${renderDivider({ text: t(entry.key, entry.vars) }, {}, t)}</div>`
        case 'answers':
          return `<div class="thread-item">${renderAnswers({ answers: entry.answers }, {}, t)}</div>`
        case 'notice':
          return `<div class="notice tone-${esc(entry.tone)}">${esc(entry.key ? t(entry.key, entry.vars) : entry.text)}</div>`
        default:
          return ''
      }
    })
    .join('')
  const cards = state.dialogs
    .filter((d) => d.card && INLINE_CARDS.includes(d.card.kind))
    .map((d) => `<div class="thread-item">${renderCard(d, t)}</div>`)
    .join('')
  const queued = state.queue.map((q, i) => `<div class="thread-item">${renderUser({ text: q.text, queued: true, id: String(i) }, {}, t)}</div>`).join('')
  const working = state.running ? `<div class="thread-item thread-working">${renderWorking({ phase: state.phase, startedAt: state.startedAt }, { now: ui.now ?? Date.now() }, t)}</div>` : ''
  const empty = !state.messages.length && !state.running && !state.queue.length && !cards ? renderEmptyThread(t) : ''
  return empty + entries + cards + working + queued
}
