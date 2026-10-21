// The slash-command picker (design/components/command-picker). It floats above the composer while
// the text is `/something`; the list is filtered and ranked by `filterCommands`.
//
// Props: { items: [{ name, description, source }], query, active, open }
//   items   the commands Pi offers, and Aria's own; the skills (`source` "skill") are listed after them under
//           their own heading, by their names without Pi's `skill:` prefix (`withSkillNames`)
//   query   the text after the `/`
//   active  index in the FILTERED list of the highlighted row (clamped)
//   open    false keeps the element but hides it, so it can animate out
// ui: unused. t: translate(key, vars).
// Pure helpers exported for the controller: filterCommands, withSkillNames, toInvocation, segments, clampActive,
// stepActive, keepVisible.
//
// Emits (on the row's button / its list item):
//   data-chat="pick"          data-index  the index in the filtered list (Enter, Tab and click choose it)
//   data-chat="picker-hover"  data-index  on the <li>: the mouse moved over a row, make it the active one
import { esc } from '../../onboarding/components.js'

export const TEXTS = {
  en: {
    'chat.commands.title': 'COMMANDS',
    'chat.commands.count.one': '{n} command',
    'chat.commands.count.other': '{n} commands',
    'chat.commands.skills': 'SKILLS',
    'chat.commands.skillCount.one': '{n} skill',
    'chat.commands.skillCount.other': '{n} skills',
    'chat.commands.empty': 'No command matches “/{query}”',
    'chat.commands.emptyHint': 'Press enter to send it as a message.',
    'chat.commands.navigate': 'navigate',
    'chat.commands.select': 'select',
    'chat.commands.close': 'close',
  },
  es: {
    'chat.commands.title': 'COMANDOS',
    'chat.commands.count.one': '{n} comando',
    'chat.commands.count.other': '{n} comandos',
    'chat.commands.skills': 'SKILLS',
    'chat.commands.skillCount.one': '{n} skill',
    'chat.commands.skillCount.other': '{n} skills',
    'chat.commands.empty': 'Ningún comando coincide con “/{query}”',
    'chat.commands.emptyHint': 'Pulsa enter para enviarlo como mensaje.',
    'chat.commands.navigate': 'moverse',
    'chat.commands.select': 'elegir',
    'chat.commands.close': 'cerrar',
  },
}

/**
 * Ranks the commands for `query`: name starts with it (0), it is inside the name (1 + position),
 * its letters appear in order in the name (20 + spread), it is in the description (60). Ties keep the
 * original order. Returns [{ item, order, score, nameRanges, descRanges }]; ranges are [start, end).
 */
export function filterCommands(items, query) {
  const q = String(query ?? '').trim().toLowerCase()
  const out = []
  items.forEach((item, order) => {
    const name = String(item.name ?? '').toLowerCase()
    const desc = String(item.description ?? '')
    let score = null
    let nameRanges = []
    let descRanges = []
    if (!q) score = 0
    else if (name.startsWith(q)) {
      score = 0
      nameRanges = [[0, q.length]]
    } else if (name.includes(q)) {
      const i = name.indexOf(q)
      score = 1 + i
      nameRanges = [[i, i + q.length]]
    } else {
      let pos = 0
      const hit = []
      for (let k = 0; k < name.length && pos < q.length; k++) {
        if (name[k] === q[pos]) {
          hit.push(k)
          pos++
        }
      }
      if (pos === q.length) {
        score = 20 + (hit[hit.length - 1] - hit[0])
        for (const k of hit) {
          const last = nameRanges[nameRanges.length - 1]
          if (last && last[1] === k) last[1] = k + 1
          else nameRanges.push([k, k + 1])
        }
      } else if (desc.toLowerCase().includes(q)) {
        const i = desc.toLowerCase().indexOf(q)
        score = 60
        descRanges = [[i, i + q.length]]
      }
    }
    if (score !== null) out.push({ item, order, score, nameRanges, descRanges })
  })
  return out.sort((a, b) => isSkill(a.item) - isSkill(b.item) || a.score - b.score || a.order - b.order)
}

const isSkill = (item) => item.source === 'skill'

/** Pi names a skill `skill:<name>`; the menu shows and takes `<name>`, unless a command already has that name (then the
 * skill keeps its prefix). `invoke` is the name Pi needs. */
export function withSkillNames(items) {
  const taken = new Set(items.filter((item) => !isSkill(item)).map((item) => item.name))
  return items.map((item) => {
    if (!isSkill(item)) return item
    const short = String(item.name ?? '').replace(/^skill:/, '')
    return taken.has(short) ? item : { ...item, name: short, invoke: item.name }
  })
}

/** The text Pi needs: a skill typed by its short name (`/brainstorming …`) goes as `/skill:brainstorming …`. */
export function toInvocation(text, items) {
  const match = /^\/(\S+)/.exec(text)
  const skill = match && items.find((item) => item.invoke && item.name === match[1])
  return skill ? `/${skill.invoke}${text.slice(match[0].length)}` : text
}

/** Splits `text` into plain and highlighted pieces: [{ text, hit }]. `ranges` are sorted [start, end). */
export function segments(text, ranges) {
  const out = []
  let pos = 0
  for (const [from, to] of ranges) {
    if (from > pos) out.push({ text: text.slice(pos, from), hit: false })
    out.push({ text: text.slice(from, to), hit: true })
    pos = to
  }
  if (pos < text.length) out.push({ text: text.slice(pos), hit: false })
  if (!out.length) out.push({ text, hit: false })
  return out
}

/** The active index kept inside a list of `n` rows. */
export const clampActive = (active, n) => Math.min(Math.max(0, Number(active) || 0), Math.max(0, n - 1))

/** Arrow keys: the next (`delta` 1) or previous (-1) row, wrapping around at both ends. */
export const stepActive = (active, n, delta) => (n ? (((clampActive(active, n) + delta) % n) + n) % n : 0)

/** The scrollTop that keeps `row` ({ top, height }) inside `view` ({ scrollTop, clientHeight }) with a 4 px margin. */
export function keepVisible(row, view, margin = 4) {
  const bottom = row.top + row.height
  if (row.top < view.scrollTop) return Math.max(0, row.top - margin)
  if (bottom > view.scrollTop + view.clientHeight) return bottom - view.clientHeight + margin
  return view.scrollTop
}

const hl = (parts, cls) => parts.map((s) => (s.hit ? `<span class="${cls}">${esc(s.text)}</span>` : esc(s.text))).join('')

const enter =
  '<svg class="p-cmd-enter" aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#B9A8FF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 4v7a4 4 0 0 1-4 4H4"/><path d="M9 10l-5 5 5 5"/></svg>'

export function renderCommandPicker({ items = [], query = '', active = 0, open = true } = {}, ui = {}, t = (key) => key) {
  const list = filterCommands(items, query)
  const n = list.length
  const on = clampActive(active, n)
  const commands = list.filter(({ item }) => !isSkill(item)).length
  const heading = (skills, count) =>
    `<span>${esc(t(skills ? 'chat.commands.skills' : 'chat.commands.title'))}</span><span class="p-cmd-count">${esc(t(`chat.commands.${skills ? 'skillCount' : 'count'}.${count === 1 ? 'one' : 'other'}`, { n: count }))}</span>`
  // The panel's heading names the first group; the skills, when commands come before them, get their own.
  const skillsHeading = commands && n > commands ? `<li role="presentation" class="p-cmd-group">${heading(true, n - commands)}</li>` : ''
  const rows = list
    .map(({ item, nameRanges, descRanges }, i) => {
      return `${i === commands ? skillsHeading : ''}<li role="option" id="p-cmd-opt-${i}" aria-selected="${i === on}" data-chat="picker-hover" data-index="${i}">
        <button type="button" class="p-cmd-row${i === on ? ' on' : ''}" data-chat="pick" data-index="${i}">
          <span class="p-cmd-name"><span class="p-cmd-slash">/</span>${hl(segments(String(item.name ?? ''), nameRanges), 'p-cmd-hl')}</span>
          <span class="p-cmd-desc" title="${esc(String(item.description ?? ''))}">${hl(segments(String(item.description ?? ''), descRanges), 'p-cmd-desc-hl')}</span>
          <span class="p-cmd-tail">${enter}</span>
        </button>
      </li>`
    })
    .join('')
  const body = n
    ? `<ul role="listbox" aria-label="${esc(t('chat.commands.title'))}" class="p-cmd-list">${rows}</ul>`
    : `<div class="p-cmd-empty"><span class="p-cmd-empty-main">${esc(t('chat.commands.empty', { query: String(query ?? '').trim() }))}</span><span class="p-cmd-empty-hint">${esc(t('chat.commands.emptyHint'))}</span></div>`
  return `<div class="p-cmd${open ? ' is-open' : ''}" aria-hidden="${!open}">
    <div class="p-cmd-panel">
      <div class="p-cmd-head">${n && !commands ? heading(true, n) : heading(false, commands)}</div>
      ${body}
      <div class="p-cmd-foot"><span><kbd>↑↓</kbd> ${esc(t('chat.commands.navigate'))}</span><span><kbd>↵</kbd> ${esc(t('chat.commands.select'))}</span><span><kbd>esc</kbd> ${esc(t('chat.commands.close'))}</span></div>
    </div>
  </div>`
}
