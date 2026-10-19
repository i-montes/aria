// The model picker (design/components/model-picker), opened by `/model` or from the model chip under the input.
//
// Props: { models: [{ provider, id, name, description?, tag?, tier? }], current, query, active, open }
//   models   what Pi can use. `description` is the second line (default `provider/id`), `tag` the
//            small pill on the right ("uses quota"), `tier: 'free'` turns that pill green
//   current  the model in use, as `provider/id` or just `id`; it gets the filled radio
//   query    what the picker's own search field holds (filters by name first, then id, provider, description, tag, tier)
//   active   index in the ORDERED list (`orderedModels`: grouped by provider) of the highlighted row (clamped)
//   open     false keeps the element but hides it, so it can animate out
// ui: unused. t: translate(key, vars).
// Pure helpers exported for the controller: filterModels, orderedModels, providerLabel (clampActive, stepActive and keepVisible live
// in ./commands.js and work for every picker).
//
// Emits (on the row's button / its list item):
//   data-chat="pick"          data-index  the index in the ordered list (Enter, Tab and click use it)
//   data-chat="picker-hover"  data-index  on the <li>: the mouse moved over a row, make it the active one
//   data-chat-field="model-search"  the search <input>: the controller reads its value on `input`
import { esc } from '../../onboarding/components.js'
import { clampActive, segments } from './commands.js'

export const TEXTS = {
  en: {
    'chat.models.title': 'MODELS',
    'chat.models.search': 'Search models or providers…',
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
    'chat.models.search': 'Busca modelos o proveedores…',
    'chat.models.count.one': '{n} modelo',
    'chat.models.count.other': '{n} modelos',
    'chat.models.empty': 'Ningún modelo coincide con “{query}”',
    'chat.models.emptyHint': 'Prueba con parte del nombre o del proveedor.',
    'chat.models.navigate': 'moverse',
    'chat.models.use': 'usarlo',
    'chat.models.close': 'cerrar',
  },
}

const PROVIDERS = { 'opencode-go': 'OpenCode Go', opencode: 'OpenCode Zen', anthropic: 'Anthropic', 'claude-bridge': 'Claude (sign-in)' }
/** A readable provider name; an unknown provider shows as its id. */
export const providerLabel = (id) => PROVIDERS[id] ?? String(id ?? '')

const nameOf = (m) => String(m.name || m.id || '')
const refOf = (m) => (m.provider ? `${m.provider}/${m.id}` : String(m.id ?? ''))

/** OpenCode Zen (`opencode/*`) is billed per request: listed only when the settings allow it (`models.allowPayPerUse`);
 * the model in use (`current`, provider/id) always stays. */
export const withoutPayPerUse = (models, allowed, current) =>
  allowed ? models : models.filter((m) => m.provider !== 'opencode' || refOf(m) === current)

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

/**
 * The models for `query`, grouped by provider: providers in the order of their best match (without a query,
 * of first appearance), each provider's models in ranked order. Returns the flat list the picker draws and
 * the controller picks from: [{ model, ranges, index, provider, firstOfProvider }].
 */
export function orderedModels(models, query) {
  const ranked = filterModels(models, query)
  const groups = new Map()
  for (const hit of ranked) {
    const provider = hit.model.provider ?? ''
    if (!groups.has(provider)) groups.set(provider, [])
    groups.get(provider).push(hit)
  }
  const flat = []
  for (const [provider, hits] of groups) {
    hits.forEach((hit, i) => flat.push({ model: hit.model, ranges: hit.ranges, provider, firstOfProvider: i === 0 }))
  }
  return flat.map((entry, index) => ({ ...entry, index }))
}

const enter =
  '<svg class="p-models-enter" aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#B9A8FF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 4v7a4 4 0 0 1-4 4H4"/><path d="M9 10l-5 5 5 5"/></svg>'

export function renderModelPicker({ models = [], current = '', query = '', active = 0, open = true } = {}, ui = {}, t = (key) => key) {
  const list = orderedModels(models, query)
  const n = list.length
  const on = clampActive(active, n)
  const perProvider = new Map()
  for (const entry of list) perProvider.set(entry.provider, (perProvider.get(entry.provider) ?? 0) + 1)
  const rows = list
    .map(({ model: m, ranges, provider, firstOfProvider }, i) => {
      const isCurrent = !!current && (current === m.id || current === refOf(m))
      const name = segments(nameOf(m), ranges)
        .map((s) => (s.hit ? `<span class="p-models-hl">${esc(s.text)}</span>` : esc(s.text)))
        .join('')
      const header = firstOfProvider
        ? `<li role="presentation" class="p-models-group"><span>${esc(providerLabel(provider))}</span><span class="p-models-group-count">${perProvider.get(provider)}</span></li>`
        : ''
      return `${header}<li role="option" id="p-models-opt-${i}" aria-selected="${isCurrent}" data-chat="picker-hover" data-index="${i}">
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
      <div class="p-models-search"><input type="text" class="p-models-search-input" data-chat-field="model-search" value="${esc(query)}" placeholder="${esc(t('chat.models.search'))}" spellcheck="false" autocomplete="off" aria-label="${esc(t('chat.models.search'))}"></div>
      ${body}
      <div class="p-models-foot"><span><kbd>↑↓</kbd> ${esc(t('chat.models.navigate'))}</span><span><kbd>↵</kbd> ${esc(t('chat.models.use'))}</span><span><kbd>esc</kbd> ${esc(t('chat.models.close'))}</span></div>
    </div>
  </div>`
}
