// The `@` file picker (design/components/file-picker). It floats above the composer while the text
// before the caret ends in `@something`; the list is filtered and ranked by `filterFiles`.
//
// Props: { items: [path], query, active, open }
//   items   relative paths; a trailing slash marks a folder (`src/queue/`)
//   query   the text after the `@`
//   active  index in the FILTERED list of the highlighted row (clamped)
//   open    false keeps the element but hides it, so it can animate out
// ui: unused. t: translate(key, vars).
// Pure helper exported for the controller: filterFiles (clampActive, stepActive and keepVisible live
// in ./commands.js and work for every picker).
//
// Emits (on the row's button / its list item):
//   data-chat="pick"          data-index  the index in the filtered list (Enter, Tab and click attach it)
//   data-chat="picker-hover"  data-index  on the <li>: the mouse moved over a row, make it the active one
import { esc } from '../../onboarding/components.js'
import { clampActive, segments } from './commands.js'

export const TEXTS = {
  en: {
    'chat.files.title': 'FILES',
    'chat.files.count.one': '{n} match',
    'chat.files.count.other': '{n} matches',
    'chat.files.empty': 'No file matches “@{query}”',
    'chat.files.emptyHint': 'Keep typing, or add a space to leave it as text.',
    'chat.files.navigate': 'navigate',
    'chat.files.attach': 'attach',
    'chat.files.close': 'close',
  },
  es: {
    'chat.files.title': 'ARCHIVOS',
    'chat.files.count.one': '{n} resultado',
    'chat.files.count.other': '{n} resultados',
    'chat.files.empty': 'Ningún archivo coincide con “@{query}”',
    'chat.files.emptyHint': 'Sigue escribiendo, o añade un espacio para dejarlo como texto.',
    'chat.files.navigate': 'moverse',
    'chat.files.attach': 'adjuntar',
    'chat.files.close': 'cerrar',
  },
}

/**
 * Ranks the paths for `query`: the file name starts with it (0), it is inside the name (1 + position),
 * it is anywhere in the path (10 + position), its letters appear in order in the name within
 * `query length + 3` characters (40 + spread). Ties keep the original order.
 * Returns [{ path, order, score, isFolder, name, dir, nameRanges, dirRanges }]; ranges are [start, end)
 * relative to `name` (with its trailing slash for a folder) and to `dir`.
 */
export function filterFiles(paths, query) {
  const q = String(query ?? '').trim().toLowerCase()
  const out = []
  paths.forEach((path, order) => {
    const isFolder = String(path).endsWith('/')
    const clean = isFolder ? String(path).replace(/\/$/, '') : String(path)
    const dirLen = clean.lastIndexOf('/') + 1
    const hay = clean.toLowerCase()
    const nameLower = hay.slice(dirLen)
    let score = null
    let ranges = []
    if (!q) score = 0
    else if (nameLower.startsWith(q)) {
      score = 0
      ranges = [[dirLen, dirLen + q.length]]
    } else if (nameLower.includes(q)) {
      const i = nameLower.indexOf(q)
      score = 1 + i
      ranges = [[dirLen + i, dirLen + i + q.length]]
    } else if (hay.includes(q)) {
      const i = hay.indexOf(q)
      score = 10 + i
      ranges = [[i, i + q.length]]
    } else {
      let pos = 0
      const hit = []
      for (let k = dirLen; k < hay.length && pos < q.length; k++) {
        if (hay[k] === q[pos]) {
          hit.push(k)
          pos++
        }
      }
      if (pos === q.length && hit[hit.length - 1] - hit[0] <= q.length + 3) {
        score = 40 + (hit[hit.length - 1] - hit[0])
        for (const k of hit) {
          const last = ranges[ranges.length - 1]
          if (last && last[1] === k) last[1] = k + 1
          else ranges.push([k, k + 1])
        }
      }
    }
    if (score === null) return
    out.push({
      path,
      order,
      score,
      isFolder,
      name: clean.slice(dirLen) + (isFolder ? '/' : ''),
      dir: clean.slice(0, dirLen),
      nameRanges: ranges.filter((r) => r[1] > dirLen).map((r) => [Math.max(r[0], dirLen) - dirLen, r[1] - dirLen]),
      dirRanges: ranges.filter((r) => r[0] < dirLen).map((r) => [r[0], Math.min(r[1], dirLen)]),
    })
  })
  return out.sort((a, b) => a.score - b.score || a.order - b.order)
}

const hl = (parts, cls) => parts.map((s) => (s.hit ? `<span class="${cls}">${esc(s.text)}</span>` : esc(s.text))).join('')

const ICON = {
  file: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>',
  folder:
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>',
}
const enter =
  '<svg class="p-files-enter" aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#B9A8FF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 4v7a4 4 0 0 1-4 4H4"/><path d="M9 10l-5 5 5 5"/></svg>'

export function renderFilePicker({ items = [], query = '', active = 0, open = true } = {}, ui = {}, t = (key) => key) {
  const list = filterFiles(items, query)
  const n = list.length
  const on = clampActive(active, n)
  const rows = list
    .map(
      (f, i) => `<li role="option" id="p-files-opt-${i}" aria-selected="${i === on}" data-chat="picker-hover" data-index="${i}">
        <button type="button" class="p-files-row${i === on ? ' on' : ''}" data-chat="pick" data-index="${i}">
          <span class="p-files-icon ${f.isFolder ? 'is-folder' : 'is-file'}" aria-hidden="true">${f.isFolder ? ICON.folder : ICON.file}</span>
          <span class="p-files-text">
            <span class="p-files-name">${hl(segments(f.name, f.nameRanges), 'p-files-hl')}</span>
            ${f.dir ? `<span class="p-files-dir">${hl(segments(f.dir, f.dirRanges), 'p-files-dir-hl')}</span>` : ''}
          </span>
          ${enter}
        </button>
      </li>`,
    )
    .join('')
  const body = n
    ? `<ul role="listbox" aria-label="${esc(t('chat.files.title'))}" class="p-files-list">${rows}</ul>`
    : `<div class="p-files-empty"><span class="p-files-empty-main">${esc(t('chat.files.empty', { query: String(query ?? '').trim() }))}</span><span class="p-files-empty-hint">${esc(t('chat.files.emptyHint'))}</span></div>`
  return `<div class="p-files${open ? ' is-open' : ''}" aria-hidden="${!open}">
    <div class="p-files-panel">
      <div class="p-files-head"><span>${esc(t('chat.files.title'))}</span><span class="p-files-count">${esc(t(n === 1 ? 'chat.files.count.one' : 'chat.files.count.other', { n }))}</span></div>
      ${body}
      <div class="p-files-foot"><span><kbd>↑↓</kbd> ${esc(t('chat.files.navigate'))}</span><span><kbd>↵</kbd> ${esc(t('chat.files.attach'))}</span><span><kbd>esc</kbd> ${esc(t('chat.files.close'))}</span></div>
    </div>
  </div>`
}
