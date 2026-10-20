// The memory panel: a searchable, filterable list of what Aria remembers (Engram observations), each
// row expandable into its detail with Pin and Forget. Port of design/components/memory-panel.dc.html.
// Pure: props and ui in, escaped HTML out. The design keeps its state inside the component; here it is
// `ui` (see `initialMemoryUi` and `reduceMemoryUi`), and the data and the two persistent flags (pinned,
// forgotten) come from the backend.
//
// Contract:
//   props = {
//     memories: Array<{
//       id:        string | number,
//       type:      'finding' | 'decision' | 'pattern' | 'session_summary' | 'manual' | 'learning' | 'identity',
//                                              // anything else is shown neutrally under "All"
//       title:     string,
//       key:       string | null,              // Engram topic_key, e.g. 'build/release-env'
//       content:   string,                     // free text, line breaks kept in the detail
//       rev:       number,                     // revision_count (>1 shows "rev N")
//       dup:       number,                     // duplicate_count
//       created:   number,                     // epoch ms
//       updated:   number,                     // epoch ms (drives "3d ago" and the default sort)
//       seen:      number,                     // epoch ms, last_seen_at
//       personal:  boolean,                    // scope personal (pink badge) instead of project
//       pinned:    boolean,                    // pinned rows sort first and show a pin
//       session:   string,                     // session_id, 'ses_3a5f1'
//       tool:      string,                     // tool_name: 'mem_save' | 'mem_update' | 'cli'
//       review:    number | null,              // review_after, epoch ms
//       expires:   number | null,              // expires_at, epoch ms
//       forgotten?: boolean,                   // true: left out of everything (kept while the undo bar shows)
//     }>,
//     now?:     number,                        // epoch ms for "ago" (default: Date.now())
//     project?: string,                        // 'aria' (header "engram · aria" and the project field)
//     undo?:    { id, title } | null,          // the memory just forgotten: shows the "Forgot ..." bar with Undo
//   }
//   ui = {                                     // `initialMemoryUi()`; keep it in the view and fold actions with `reduceMemoryUi`
//     query: string,                           // text search: every word must appear in title, content, key or type
//     type: 'all' | <a type id>,
//     pinnedOnly: boolean,
//     personalOnly: boolean,
//     sort: 'updated' | 'created' | 'revisions' | 'title',
//     open: string | number | null,            // the expanded row
//     enter?: boolean,                         // true: rows fade in (first render of the panel only)
//   }
//
// Emits:
//   data-chat="memory-back"                                   back to the chat
//   data-chat-field="memory-search"   (an <input>)            its value is the new query (reduce 'search')
//   data-chat="memory-clear"                                  empty the query
//   data-chat="memory-filter" data-value="<all|type id>"      a type chip
//   data-chat="memory-pinned-only" / "memory-personal-only"   toggles
//   data-chat="memory-sort" data-value="<next sort id>"       cycles the sort
//   data-chat="memory-select" data-id="<id>"                  expand / collapse a row
//   data-chat="memory-pin" data-id="<id>"                     pin / unpin  (backend call)
//   data-chat="memory-forget" data-id="<id>"                  forget it    (backend call; then props.undo)
//   data-chat="memory-undo"                                   undo the last forget (backend call)

import { esc } from '../../onboarding/components.js'
import { renderPanelHead } from './panelhead.js'

export const TEXTS = {
  en: {
    'chat.memory.title': 'Memory',
    'chat.memory.back': 'Back to chat',
    'chat.memory.active': '{n} active',
    'chat.memory.search': 'Search memories',
    'chat.memory.clear': 'Clear search',
    'chat.memory.textSearch': 'text search',
    'chat.memory.textSearchHint': 'No embeddings are stored, so search matches words, not meaning',
    'chat.memory.type.all': 'All',
    'chat.memory.type.finding': 'finding',
    'chat.memory.type.decision': 'decision',
    'chat.memory.type.pattern': 'pattern',
    'chat.memory.type.session_summary': 'session summary',
    'chat.memory.type.manual': 'manual',
    'chat.memory.type.learning': 'learning',
    'chat.memory.type.identity': 'identity',
    'chat.memory.pinned': 'Pinned',
    'chat.memory.personalOnly': 'Personal',
    'chat.memory.sort.updated': 'recently updated',
    'chat.memory.sort.created': 'recently created',
    'chat.memory.sort.revisions': 'most revised',
    'chat.memory.sort.title': 'title',
    'chat.memory.list': 'Memories',
    'chat.memory.rev': 'rev {n}',
    'chat.memory.personal': 'personal',
    'chat.memory.pin': 'Pin',
    'chat.memory.unpin': 'Unpin',
    'chat.memory.forget': 'Forget',
    'chat.memory.forgetHint': 'Marks it as deleted. Engram keeps the record.',
    'chat.memory.embeddingNone': 'none (text search only)',
    'chat.memory.ago.today': 'today',
    'chat.memory.ago.yesterday': 'yesterday',
    'chat.memory.ago.days': '{n}d ago',
    'chat.memory.ago.weeks': '{n}w ago',
    'chat.memory.empty': 'No memories here yet',
    'chat.memory.emptyQuery': 'No memory matches “{q}”',
    'chat.memory.emptyHint': 'Search matches the words you used when saving. There are no embeddings yet, so it can’t match by meaning.',
    'chat.memory.forgot': 'Forgot “{title}”',
    'chat.memory.undo': 'Undo',
    'chat.memory.keyed': '{n} with a topic key',
    'chat.memory.pinnedN': '{n} pinned',
    'chat.memory.embeddings': '{n} embeddings',
  },
  es: {
    'chat.memory.title': 'Memoria',
    'chat.memory.back': 'Volver al chat',
    'chat.memory.active': '{n} activas',
    'chat.memory.search': 'Buscar recuerdos',
    'chat.memory.clear': 'Borrar la búsqueda',
    'chat.memory.textSearch': 'búsqueda de texto',
    'chat.memory.textSearchHint': 'No se guardan embeddings, así que la búsqueda coincide con palabras, no con el significado',
    'chat.memory.type.all': 'Todos',
    'chat.memory.type.finding': 'hallazgo',
    'chat.memory.type.decision': 'decisión',
    'chat.memory.type.pattern': 'patrón',
    'chat.memory.type.session_summary': 'resumen de sesión',
    'chat.memory.type.manual': 'manual',
    'chat.memory.type.learning': 'aprendizaje',
    'chat.memory.type.identity': 'identidad',
    'chat.memory.pinned': 'Fijados',
    'chat.memory.personalOnly': 'Personales',
    'chat.memory.sort.updated': 'actualizados hace poco',
    'chat.memory.sort.created': 'creados hace poco',
    'chat.memory.sort.revisions': 'más revisados',
    'chat.memory.sort.title': 'título',
    'chat.memory.list': 'Recuerdos',
    'chat.memory.rev': 'rev {n}',
    'chat.memory.personal': 'personal',
    'chat.memory.pin': 'Fijar',
    'chat.memory.unpin': 'Desfijar',
    'chat.memory.forget': 'Olvidar',
    'chat.memory.forgetHint': 'Lo marca como borrado. Engram conserva el registro.',
    'chat.memory.embeddingNone': 'ninguno (solo búsqueda de texto)',
    'chat.memory.ago.today': 'hoy',
    'chat.memory.ago.yesterday': 'ayer',
    'chat.memory.ago.days': 'hace {n} d',
    'chat.memory.ago.weeks': 'hace {n} sem',
    'chat.memory.empty': 'Aún no hay recuerdos aquí',
    'chat.memory.emptyQuery': 'Ningún recuerdo coincide con “{q}”',
    'chat.memory.emptyHint': 'La búsqueda coincide con las palabras que usaste al guardar. Todavía no hay embeddings, así que no puede buscar por significado.',
    'chat.memory.forgot': 'Olvidaste “{title}”',
    'chat.memory.undo': 'Deshacer',
    'chat.memory.keyed': '{n} con clave de tema',
    'chat.memory.pinnedN': '{n} fijados',
    'chat.memory.embeddings': '{n} embeddings',
  },
}

export const MEMORY_TYPES = [
  { id: 'finding', color: '#7DD3FC' },
  { id: 'decision', color: '#B9A8FF' },
  { id: 'pattern', color: '#9BE7C4' },
  { id: 'session_summary', color: '#F0B44C' },
  { id: 'manual', color: '#FF9A8B' },
  { id: 'learning', color: '#F0A8D8' },
  { id: 'identity', color: '#F4F4F5' },
]
export const SORTS = ['updated', 'created', 'revisions', 'title']

const DAY = 86400000
const isType = (id) => MEMORY_TYPES.some((x) => x.id === id)
const typeLabel = (type) => String(type ?? '').replace(/_/g, ' ')

// --- pure helpers ----------------------------------------------------------------------------

export const searchWords = (query) => String(query ?? '').toLowerCase().split(/\s+/).filter(Boolean)

/** Splits `text` into `{ t, hit }` segments around the (merged) occurrences of any of `words`. */
export function highlight(text, words) {
  const value = String(text ?? '')
  const low = value.toLowerCase()
  const found = []
  for (const w of words ?? []) {
    let i = low.indexOf(w)
    while (w && i > -1) {
      found.push([i, i + w.length])
      i = low.indexOf(w, i + w.length)
    }
  }
  found.sort((a, b) => a[0] - b[0])
  const merged = []
  for (const r of found) {
    const last = merged[merged.length - 1]
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1])
    else merged.push(r.slice())
  }
  const out = []
  let pos = 0
  for (const [from, to] of merged) {
    if (from > pos) out.push({ t: value.slice(pos, from), hit: false })
    out.push({ t: value.slice(from, to), hit: true })
    pos = to
  }
  if (pos < value.length) out.push({ t: value.slice(pos), hit: false })
  if (!out.length) out.push({ t: value, hit: false })
  return out
}

/** One line of the content; a hit far into the text brings the excerpt forward ("…"). */
export function memoryExcerpt(content, words) {
  const flat = String(content ?? '').replace(/\s+/g, ' ')
  if (words?.length) {
    const hit = flat.toLowerCase().indexOf(words[0])
    if (hit > 90) return `…${flat.slice(hit - 30)}`
  }
  return flat
}

/** "today", "yesterday", "5d ago", "5w ago". */
export function agoLabel(ms, now, t) {
  const days = Math.floor((Number(now) - Number(ms)) / DAY)
  if (!Number.isFinite(days) || days <= 0) return t('chat.memory.ago.today')
  if (days === 1) return t('chat.memory.ago.yesterday')
  if (days < 30) return t('chat.memory.ago.days', { n: days })
  return t('chat.memory.ago.weeks', { n: Math.floor(days / 7) })
}

/** "2026-10-14 08:05" (UTC), or an em dash for a missing time. */
export function stamp(ms) {
  const date = new Date(Number(ms))
  return ms === null || ms === undefined || Number.isNaN(date.getTime()) ? '—' : date.toISOString().replace('T', ' ').slice(0, 16)
}

export const nextSort = (sort) => SORTS[(Math.max(0, SORTS.indexOf(sort)) + 1) % SORTS.length]

export const initialMemoryUi = () => ({ query: '', type: 'all', pinnedOnly: false, personalOnly: false, sort: 'updated', open: null })

/** Folds a panel action into `ui` (the names are the `data-chat` actions without the `memory-` prefix). */
export function reduceMemoryUi(ui, action, value) {
  const base = { ...initialMemoryUi(), ...ui }
  switch (action) {
    case 'search':
      return { ...base, query: String(value ?? ''), open: null }
    case 'clear':
      return { ...base, query: '', open: null }
    case 'filter':
      return { ...base, type: value === 'all' || isType(value) ? value : 'all', open: null }
    case 'pinned-only':
      return { ...base, pinnedOnly: !base.pinnedOnly, open: null }
    case 'personal-only':
      return { ...base, personalOnly: !base.personalOnly, open: null }
    case 'sort':
      return { ...base, sort: SORTS.includes(value) ? value : nextSort(base.sort) }
    case 'select':
      return { ...base, open: base.open !== null && String(base.open) === String(value) ? null : value }
    case 'forget':
      return { ...base, open: null }
    default:
      return base
  }
}

const SORTERS = {
  updated: (a, b) => (b.updated ?? 0) - (a.updated ?? 0),
  created: (a, b) => (b.created ?? 0) - (a.created ?? 0),
  revisions: (a, b) => (b.rev ?? 0) - (a.rev ?? 0) || (b.updated ?? 0) - (a.updated ?? 0),
  title: (a, b) => String(a.title ?? '').localeCompare(String(b.title ?? '')),
}

/** What the panel shows: the visible rows (pinned first, then sorted) and every count the chrome needs. */
export function selectMemories(memories, ui) {
  const view = { ...initialMemoryUi(), ...ui }
  const active = (Array.isArray(memories) ? memories : []).filter((m) => m && typeof m === 'object' && !m.forgotten)
  const words = searchWords(view.query)
  const matches = (m) => {
    const haystack = `${m.title ?? ''} ${m.content ?? ''} ${m.key ?? ''} ${typeLabel(m.type)}`.toLowerCase()
    return words.every((w) => haystack.includes(w))
  }
  const facetBase = active.filter((m) => matches(m) && (!view.pinnedOnly || m.pinned) && (!view.personalOnly || m.personal))
  const shown = facetBase.filter((m) => view.type === 'all' || m.type === view.type)
  const sorter = SORTERS[view.sort] ?? SORTERS.updated
  shown.sort((a, b) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0) || sorter(a, b))
  const typeCounts = { all: facetBase.length }
  for (const { id } of MEMORY_TYPES) typeCounts[id] = facetBase.filter((m) => m.type === id).length
  return {
    active,
    shown,
    words,
    typeCounts,
    pinnedCount: active.filter((m) => m.pinned).length,
    personalCount: active.filter((m) => m.personal).length,
    keyedCount: active.filter((m) => m.key).length,
  }
}

// --- rendering -------------------------------------------------------------------------------

const ICON = {
  search: '<svg aria-hidden="true" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#8A8A94" stroke-width="2" stroke-linecap="round" style="flex: 0 0 15px"><circle cx="11" cy="11" r="7"></circle><path d="M21 21l-4.3-4.3"></path></svg>',
  clear: '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"></path></svg>',
  sort: '<svg aria-hidden="true" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 4v16M3 16l4 4 4-4M17 20V4M13 8l4-4 4 4"></path></svg>',
  pin: '<svg class="p-memory-pin-icon" aria-hidden="true" width="12" height="12" viewBox="0 0 24 24" fill="#B9A8FF" stroke="#B9A8FF" stroke-width="1.6" stroke-linejoin="round"><path d="M9 3h6l-1 6 3 3v2H7v-2l3-3z"></path><path d="M12 14v7" stroke-linecap="round"></path></svg>',
}

const segs = (list) => list.map((s) => (s.hit ? `<span class="hit">${esc(s.t)}</span>` : esc(s.t))).join('')

function detail(m, o, t) {
  const fields = [
    ['project', o.project],
    ['type', m.type],
    ['scope', m.personal ? 'personal' : 'project'],
    ['topic_key', m.key || '—'],
    ['session_id', m.session],
    ['tool_name', m.tool],
    ['revision_count', String(m.rev ?? '')],
    ['duplicate_count', String(m.dup ?? '')],
    ['created_at', stamp(m.created)],
    ['updated_at', stamp(m.updated)],
    ['last_seen_at', stamp(m.seen)],
    ['pinned', m.pinned ? 'true' : 'false'],
  ]
  if (m.review) fields.push(['review_after', stamp(m.review)])
  if (m.expires) fields.push(['expires_at', stamp(m.expires)])
  fields.push(['embedding', t('chat.memory.embeddingNone')])
  const dl = fields
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd${k === 'embedding' || (k === 'topic_key' && !m.key) ? ' class="is-dim"' : ''}>${esc(v)}</dd>`)
    .join('')
  const id = esc(m.id)
  return `<div class="p-memory-detail">
      <p class="p-memory-content">${esc(m.content)}</p>
      <dl class="p-memory-fields">${dl}</dl>
      <div class="p-memory-actions">
        <button type="button" class="p-memory-btn" data-chat="memory-pin" data-id="${id}">${esc(t(m.pinned ? 'chat.memory.unpin' : 'chat.memory.pin'))}</button>
        <button type="button" class="p-memory-btn is-danger" data-chat="memory-forget" data-id="${id}" title="${esc(t('chat.memory.forgetHint'))}">${esc(t('chat.memory.forget'))}</button>
      </div>
    </div>`
}

function row(m, o, t) {
  const isOpen = o.open !== null && String(o.open) === String(m.id)
  const known = isType(m.type)
  const key = m.key ? `<span class="p-memory-key">${segs(highlight(m.key, o.words))}</span>` : ''
  const rev = m.rev > 1 ? `<span class="p-memory-rev">${esc(t('chat.memory.rev', { n: m.rev }))}</span>` : ''
  const personal = m.personal ? `<span class="p-memory-personal">${esc(t('chat.memory.personal'))}</span>` : ''
  const preview = isOpen ? '' : `<span class="p-memory-preview">${segs(highlight(memoryExcerpt(m.content, o.words), o.words))}</span>`
  return `<li class="p-memory-row type-${known ? m.type : 'other'}${isOpen ? ' is-open' : ''}${o.enter ? ' is-enter' : ''}">
    <button type="button" class="p-memory-head" data-chat="memory-select" data-id="${esc(m.id)}" aria-expanded="${isOpen}">
      <span class="p-memory-dot" aria-hidden="true"></span>
      <span class="p-memory-main">
        <span class="p-memory-title">${segs(highlight(m.title, o.words))}</span>
        ${preview}
        <span class="p-memory-tags"><span class="p-memory-type">${esc(known ? t(`chat.memory.type.${m.type}`) : typeLabel(m.type))}</span>${key}${rev}${personal}</span>
      </span>
      <span class="p-memory-when">${m.pinned ? ICON.pin : ''}${esc(agoLabel(m.updated, o.now, t))}</span>
    </button>
    ${isOpen ? detail(m, o, t) : ''}
  </li>`
}

const toggle = (action, on, label, count) =>
  `<button type="button" class="p-memory-toggle${action === 'memory-personal-only' ? ' is-personal' : ''}" aria-pressed="${on}" data-chat="${action}">${esc(label)} <span class="p-memory-n">${esc(count)}</span></button>`

export function renderMemory(props, ui, t) {
  const view = { ...initialMemoryUi(), ...ui }
  if (!SORTS.includes(view.sort)) view.sort = 'updated'
  const project = props?.project ?? 'aria'
  const now = Number.isFinite(Number(props?.now)) && props?.now !== undefined && props?.now !== null ? Number(props.now) : Date.now()
  const sel = selectMemories(props?.memories, view)
  const o = { open: view.open ?? null, words: sel.words, now, project, enter: view.enter === true }
  const query = String(view.query ?? '')

  const chips = [{ id: 'all' }, ...MEMORY_TYPES]
    .map((c) => {
      const on = view.type === c.id
      return `<button type="button" class="p-memory-chip${c.id === 'all' ? '' : ` type-${c.id}`}" aria-pressed="${on}" data-chat="memory-filter" data-value="${esc(c.id)}">${c.id === 'all' ? '' : '<span class="p-memory-chip-dot" aria-hidden="true"></span>'}<span>${esc(t(`chat.memory.type.${c.id}`))}</span><span class="p-memory-n">${esc(sel.typeCounts[c.id])}</span></button>`
    })
    .join('')

  const rows = sel.shown.map((m) => row(m, o, t)).join('')
  const empty = sel.shown.length
    ? ''
    : `<li class="p-memory-empty"><span class="p-memory-empty-title">${esc(query.trim() ? t('chat.memory.emptyQuery', { q: query.trim() }) : t('chat.memory.empty'))}</span><span class="p-memory-empty-hint">${esc(t('chat.memory.emptyHint'))}</span></li>`
  const undo = props?.undo
    ? `<div class="p-memory-undo" role="status"><span class="p-memory-undo-text">${esc(t('chat.memory.forgot', { title: props.undo.title ?? '' }))}</span><button type="button" class="p-memory-undo-btn" data-chat="memory-undo">${esc(t('chat.memory.undo'))}</button></div>`
    : ''

  return `<div class="p-memory">
  ${renderPanelHead({ back: { attrs: 'data-chat="memory-back"', label: t('chat.memory.back') }, title: t('chat.memory.title'), meta: `engram · ${project}`, aside: t('chat.memory.active', { n: sel.active.length }) })}
  <div class="p-memory-search">
    ${ICON.search}
    <label class="p-memory-sr" for="p-memory-search">${esc(t('chat.memory.search'))}</label>
    <input id="p-memory-search" class="p-memory-input" type="text" value="${esc(query)}" placeholder="${esc(t('chat.memory.search'))}" data-chat-field="memory-search" spellcheck="false" autocomplete="off">
    ${query ? `<button type="button" class="p-memory-clear" data-chat="memory-clear" aria-label="${esc(t('chat.memory.clear'))}">${ICON.clear}</button>` : ''}
    <span class="p-memory-badge" title="${esc(t('chat.memory.textSearchHint'))}">${esc(t('chat.memory.textSearch'))}</span>
  </div>
  <div class="p-memory-chips">${chips}</div>
  <div class="p-memory-filters">
    ${toggle('memory-pinned-only', !!view.pinnedOnly, t('chat.memory.pinned'), sel.pinnedCount)}
    ${toggle('memory-personal-only', !!view.personalOnly, t('chat.memory.personalOnly'), sel.personalCount)}
    <button type="button" class="p-memory-sort" data-chat="memory-sort" data-value="${nextSort(view.sort)}">${ICON.sort}${esc(t(`chat.memory.sort.${view.sort}`))}</button>
  </div>
  <ul class="p-memory-list" aria-label="${esc(t('chat.memory.list'))}">${rows}${empty}</ul>
  ${undo}
  <div class="p-memory-foot"><span>${esc(t('chat.memory.keyed', { n: sel.keyedCount }))}</span><span>${esc(t('chat.memory.pinnedN', { n: sel.pinnedCount }))}</span><span>${esc(t('chat.memory.embeddings', { n: 0 }))}</span></div>
</div>`
}
