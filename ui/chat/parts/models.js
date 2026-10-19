// The model picker (design/components/model-picker), opened by `/model` or from the meter.
//
// Props: { models: [{ provider, id, name, description?, tag?, tier? }], current, query, active, open }
//   models   what Pi can use. `description` is the second line (default `provider/id`), `tag` the
//            small pill on the right ("uses quota"), `tier: 'free'` turns that pill green
//   current  the model in use, as `provider/id` or just `id`; it gets the filled radio
//   query    text typed after `/model ` (filters by name first, then id, provider, description, tag, tier)
//   active   index in the FILTERED list of the highlighted row (clamped)
//   open     false keeps the element but hides it, so it can animate out
// ui: unused. t: translate(key, vars).
// Pure helper exported for the controller: filterModels (clampActive, stepActive and keepVisible live
// in ./commands.js and work for every picker).
//
// Emits (on the row's button / its list item):
//   data-chat="pick"          data-index  the index in the filtered list (Enter, Tab and click use it)
//   data-chat="picker-hover"  data-index  on the <li>: the mouse moved over a row, make it the active one
import { esc } from '../../onboarding/components.js'
import { clampActive, segments } from './commands.js'

export const TEXTS = {
  en: {
    'chat.models.title': 'MODELS',
    'chat.models.count.one': '{n} model',
    'chat.models.count.other': '{n} models',
    'chat.models.empty': 'No model matches “{query}”',
    'chat.models.emptyHint': 'Try part of the name or the provider.',
    'chat.models.navigate': 'navigate',
    'chat.models.use': 'use it',
    'chat.models.close': 'close',
  },
  es: {
    'chat.models.title': 'MODELOS',
    'chat.models.count.one': '{n} modelo',
    'chat.models.count.other': '{n} modelos',
    'chat.models.empty': 'Ningún modelo coincide con “{query}”',
    'chat.models.emptyHint': 'Prueba con parte del nombre o del proveedor.',
    'chat.models.navigate': 'moverse',
    'chat.models.use': 'usarlo',
    'chat.models.close': 'cerrar',
  },
}

const nameOf = (m) => String(m.name || m.id || '')
const refOf = (m) => (m.provider ? `${m.provider}/${m.id}` : String(m.id ?? ''))

/**
 * Ranks the models for `query`: the name starts with it (0), it is inside the name (1 + position),
 * it is in the id, provider, description, tag or tier (50). Ties keep the original order.
 * Returns [{ model, order, score, ranges }]; `ranges` are [start, end) in the displayed name.
 */
export function filterModels(models, query) {
  const q = String(query ?? '').trim().toLowerCase()
  const out = []
  models.forEach((model, order) => {
    const name = nameOf(model).toLowerCase()
    let score = null
    let ranges = []
    if (!q) score = 0
    else if (name.startsWith(q)) {
      score = 0
      ranges = [[0, q.length]]
    } else if (name.includes(q)) {
      const i = name.indexOf(q)
      score = 1 + i
      ranges = [[i, i + q.length]]
    } else if ([model.id, model.provider, model.description, model.tag, model.tier].map((v) => v ?? '').join(' ').toLowerCase().includes(q)) score = 50
    if (score !== null) out.push({ model, order, score, ranges })
  })
  return out.sort((a, b) => a.score - b.score || a.order - b.order)
}

const enter =
  '<svg class="p-models-enter" aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#B9A8FF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 4v7a4 4 0 0 1-4 4H4"/><path d="M9 10l-5 5 5 5"/></svg>'

export function renderModelPicker({ models = [], current = '', query = '', active = 0, open = true } = {}, ui = {}, t = (key) => key) {
  const list = filterModels(models, query)
  const n = list.length
  const on = clampActive(active, n)
  const rows = list
    .map(({ model: m, ranges }, i) => {
      const isCurrent = !!current && (current === m.id || current === refOf(m))
      const name = segments(nameOf(m), ranges)
        .map((s) => (s.hit ? `<span class="p-models-hl">${esc(s.text)}</span>` : esc(s.text)))
        .join('')
      return `<li role="option" id="p-models-opt-${i}" aria-selected="${isCurrent}" data-chat="picker-hover" data-index="${i}">
        <button type="button" class="p-models-row${i === on ? ' on' : ''}" data-chat="pick" data-index="${i}">
          <span class="p-models-ring${isCurrent ? ' is-current' : ''}" aria-hidden="true"><span class="p-models-ring-dot"></span></span>
          <span class="p-models-text">
            <span class="p-models-name">${name}</span>
            <span class="p-models-desc">${esc(m.description || refOf(m))}</span>
          </span>
          <span class="p-models-tail">${m.tag ? `<span class="p-models-tag${m.tier === 'free' ? ' is-free' : ''}">${esc(m.tag)}</span>` : ''}${enter}</span>
        </button>
      </li>`
    })
    .join('')
  const body = n
    ? `<ul role="listbox" aria-label="${esc(t('chat.models.title'))}" class="p-models-list">${rows}</ul>`
    : `<div class="p-models-empty"><span class="p-models-empty-main">${esc(t('chat.models.empty', { query: String(query ?? '').trim() }))}</span><span class="p-models-empty-hint">${esc(t('chat.models.emptyHint'))}</span></div>`
  return `<div class="p-models${open ? ' is-open' : ''}" aria-hidden="${!open}">
    <div class="p-models-panel">
      <div class="p-models-head"><span>${esc(t('chat.models.title'))}</span><span class="p-models-count">${esc(t(n === 1 ? 'chat.models.count.one' : 'chat.models.count.other', { n }))}</span></div>
      ${body}
      <div class="p-models-foot"><span><kbd>↑↓</kbd> ${esc(t('chat.models.navigate'))}</span><span><kbd>↵</kbd> ${esc(t('chat.models.use'))}</span><span><kbd>esc</kbd> ${esc(t('chat.models.close'))}</span></div>
    </div>
  </div>`
}
