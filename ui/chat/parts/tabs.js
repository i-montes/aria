// The session tabs (design/components/session-tabs): one tab per session with its state dot and, on hover,
// the quota card (quotabars.js: model, context, quota), an overflow menu for what does not fit, rename, close, `+` and the
// undo bar shown after closing a tab.
//
// Props (`tabs`): { sessions: [{ id, title, status, unread, model?, context?, quotaWindows?, provider?, free? }], active, closed }
//   status   'working' | 'needs' | 'done' | 'idle'; `unread` (finished while away) looks like done
//   model    display name of the model, shown in the tooltip
//   context  context window used by that session, 0..100 (percent)
//   quotaWindows  the provider's quota windows for that session's model (see quotabars.js); provider: its readable name
//   free     the session runs on a free model: the tooltip says it uses no quota
//   closed   { entry, index } of the tab just closed (from sessions.js), for the undo bar
// ui: { menuOpen, renaming, maxVisible, hideNew, hideUndo }
//   menuOpen   the overflow menu is open
//   renaming   id of the session whose title is being edited
//   maxVisible tabs shown before the rest go to the overflow (default 4, at least 2); the active tab always shows
//   hideNew    omit the `+` button
//   hideUndo   omit the undo bar
// t: translate(key, vars).
//
// Emits:
//   data-chat="select"     data-session  open that session (tab, and each row of the overflow menu)
//   data-chat="close-tab"  data-session  close it (only when there is more than one session)
//   data-chat="add"                      a new session (disabled at 8 sessions)
//   data-chat="menu"                     toggle the overflow menu
//   data-chat="undo"                     bring back the tab that was just closed
//   [data-chat-rename]     data-session  the title field (Enter commits, Escape cancels, blur commits);
//                                        the controller opens it on double click of `.p-tabs-main`
import { esc } from '../../onboarding/components.js'
import { renderQuotaCard } from './quotabars.js'
import { MAX_SESSIONS, visibleTabs } from '../sessions.js'

export const TEXTS = {
  en: {
    'chat.untitled': 'New session',
    'chat.new': 'New session',
    'chat.sessions': 'Sessions',
    'chat.tab.close': 'Close {title}',
    'chat.tab.scheduled': 'scheduled task',
    'chat.tab.rename': 'Rename session',
    'chat.tab.more': '+{n}',
    'chat.tab.menu': 'More sessions',
    'chat.tab.moreNeeds': '+{n} · a hidden session needs you',
    'chat.status.working': 'working',
    'chat.status.needs': 'needs you',
    'chat.status.done': 'done',
    'chat.status.idle': 'idle',
    'chat.undo.closed': 'Closed "{title}"',
    'chat.undo': 'Undo',
    'chat.tabs.moreLabel': '{n} more sessions',
    'chat.tabs.limit': 'Close a session to open another',
  },
  es: {
    'chat.untitled': 'Sesión nueva',
    'chat.new': 'Sesión nueva',
    'chat.sessions': 'Sesiones',
    'chat.tab.close': 'Cerrar {title}',
    'chat.tab.scheduled': 'tarea programada',
    'chat.tab.rename': 'Renombrar sesión',
    'chat.tab.more': '+{n}',
    'chat.tab.menu': 'Más sesiones',
    'chat.tab.moreNeeds': '+{n} · una sesión oculta te necesita',
    'chat.status.working': 'trabajando',
    'chat.status.needs': 'te necesita',
    'chat.status.done': 'lista',
    'chat.status.idle': 'en reposo',
    'chat.undo.closed': 'Se cerró "{title}"',
    'chat.undo': 'Deshacer',
    'chat.tabs.moreLabel': '{n} sesiones más',
    'chat.tabs.limit': 'Cierra una sesión para abrir otra',
  },
}

const STATUSES = ['working', 'needs', 'done', 'idle']
const statusOf = (s) => (STATUSES.includes(s.status) ? s.status : 'idle')
/** What the tab looks like: working > needs > done (or unread) > idle, as the design's `look`. */
const toneOf = (s) => {
  const status = statusOf(s)
  return status === 'working' || status === 'needs' ? status : status === 'done' || s.unread ? 'done' : 'idle'
}

const X =
  '<svg aria-hidden="true" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>'
const PLUS =
  '<svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>'

const CLOCK =
  '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="13" r="8"></circle><path d="M12 9v4l2.5 2"></path></svg>'

export function renderSessionTabs(tabs, ui = {}, t = (key) => key) {
  const max = Math.max(2, Math.round(Number(ui.maxVisible)) || 4)
  const { visible, overflow } = visibleTabs(tabs, max)
  const multi = tabs.sessions.length > 1
  const titleOf = (s) => s.title || t('chat.untitled')

  // The tooltip is the quota card (see quotabars.js), led by the tab's title and state.
  const tooltip = (s, i, tone) => `<div class="p-tabs-tip${i >= Math.ceil(visible.length / 2) ? ' is-right' : ''}" role="tooltip">${renderQuotaCard({
    windows: s.quotaWindows,
    provider: s.provider,
    facts: { model: s.model || '—', context: s.context ?? 0 },
    heading: { title: titleOf(s), state: t(`chat.status.${tone}`), tone },
    free: s.free === true,
  }, {}, t)}</div>`

  const tab = (s, i) => {
    const on = s.id === tabs.active
    const tone = toneOf(s)
    const status = statusOf(s)
    const cls = `p-tabs-tab status-${status} tone-${tone}${on ? ' on' : ''}${s.unread ? ' unread' : ''}${multi ? ' closable' : ''}${s.schedule ? ' scheduled' : ''}`
    if (ui.renaming === s.id) {
      return `<div class="${cls}"><input class="p-tabs-input" data-chat-rename data-session="${esc(s.id)}" value="${esc(s.title)}" aria-label="${esc(t('chat.tab.rename'))}" spellcheck="false" autocomplete="off"></div>`
    }
    const plain = status === 'idle' && tone === 'idle' ? titleOf(s) : `${titleOf(s)}, ${t(`chat.status.${tone}`)}`
    const label = s.schedule ? `${plain}, ${t('chat.tab.scheduled')}` : plain
    // A scheduled task's tab: a clock where the title starts.
    const clock = s.schedule ? `<span class="p-tabs-clock" aria-hidden="true">${CLOCK}</span>` : ''
    return `<div class="${cls}">
      <button type="button" class="p-tabs-main" role="tab" aria-selected="${on}" aria-label="${esc(label)}" data-chat="select" data-session="${esc(s.id)}">
        <span class="p-tabs-dot" aria-hidden="true"></span>${clock}<span class="p-tabs-title">${esc(titleOf(s))}</span>
      </button>${
        multi
          ? `<button type="button" class="p-tabs-x" data-chat="close-tab" data-session="${esc(s.id)}" aria-label="${esc(t('chat.tab.close', { title: titleOf(s) }))}">${X}</button>`
          : ''
      }${multi ? tooltip(s, i, tone) : ''}
    </div>`
  }

  // A session in the overflow that needs you (or finished unseen) must show on the button that hides it.
  const hiddenNeeds = overflow.some((s) => s.status === 'needs')
  const hiddenUnread = !hiddenNeeds && overflow.some((s) => s.unread)
  const menuOpen = !!ui.menuOpen && overflow.length > 0
  const moreLabel = hiddenNeeds ? t('chat.tab.moreNeeds', { n: overflow.length }) : t('chat.tabs.moreLabel', { n: overflow.length })
  const more = overflow.length
    ? `<div class="p-tabs-menu-wrap">
        <button type="button" class="p-tabs-more${hiddenNeeds ? ' needs' : hiddenUnread ? ' unread' : ''}" data-chat="menu" aria-haspopup="true" aria-expanded="${menuOpen}" aria-label="${esc(moreLabel)}">${esc(t('chat.tab.more', { n: overflow.length }))}</button>${
          menuOpen
            ? `<div class="p-tabs-menu" role="menu" aria-label="${esc(t('chat.tab.menu'))}">${overflow
                .map((s) => {
                  const tone = toneOf(s)
                  return `<button type="button" class="p-tabs-menu-row tone-${tone}" role="menuitem" data-chat="select" data-session="${esc(s.id)}">
                  <span class="p-tabs-dot" aria-hidden="true"></span><span class="p-tabs-menu-title">${esc(titleOf(s))}</span><span class="p-tabs-menu-state">${tone === 'idle' ? '' : esc(t(`chat.status.${tone}`))}</span></button>`
                })
                .join('')}</div>`
            : ''
        }
      </div>`
    : ''

  const full = tabs.sessions.length >= MAX_SESSIONS
  const addLabel = full ? t('chat.tabs.limit') : t('chat.new')
  const add = ui.hideNew
    ? ''
    : `<button type="button" class="p-tabs-add" data-chat="add" ${full ? 'disabled' : ''} aria-label="${esc(addLabel)}" title="${esc(addLabel)}">${PLUS}</button>`

  const undo =
    tabs.closed && !ui.hideUndo
      ? `<div class="p-tabs-undo" role="status"><span class="p-tabs-undo-text">${esc(t('chat.undo.closed', { title: tabs.closed.entry.title || t('chat.untitled') }))}</span><button type="button" class="p-tabs-undo-btn" data-chat="undo">${esc(t('chat.undo'))}</button></div>`
      : ''

  return `<div class="p-tabs">
    <div class="p-tabs-bar">
      <div class="p-tabs-list" role="tablist" aria-label="${esc(t('chat.sessions'))}">${visible.map(tab).join('')}</div>${more}${add}
    </div>${undo}
  </div>`
}
