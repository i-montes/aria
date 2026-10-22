// The tool-call group of the chat (design: design/components/tool-chips.dc.html): a header with the
// number of calls, one row per call with an icon for its kind, a chip with its target and, when the
// row is open, the result; once the group has settled, the files it changed as small diff chips.
//
// Props:  { index?: number,                     the group's position in the thread's messages
//           calls: [{ id, name, args, status: 'running'|'done'|'error', summary, result }
//                   | { id, think: true, text }],  a reasoning step: the "Thinking" row
//           live?: boolean }                    true while the run is going (any running call counts too)
//         a reasoning step may carry its own `label` instead of "Thinking", its own `chip` ('' for none) and, as its
//         detail, TRUSTED `html` (already escaped by the caller, e.g. markdown.js) instead of its text's lines
// Ui:     { open?: Set<number>|boolean,         is the group expanded (a Set holds message indexes)
//           openRows?: Set<string>,             call ids whose result is shown
//           moreOpen?: Set<number>|boolean,     all changed-file chips instead of the first three
//           animate?: boolean }                 false skips the rows' entry animation (default true)
// Emits:  data-chat="toggle-tools"       data-index="<message index>"
//         data-chat="toggle-tool-row"    data-id="<call id>"
//         data-chat="toggle-tool-diffs"  data-index="<message index>"   (the "+N more" button)
//
// Every dynamic value goes through `esc`. Static styles live in tools.css (classes `.p-tools-…`).
import { esc } from '../../onboarding/components.js'

export const TEXTS = {
  en: {
    'chat.tools.one': '1 tool call',
    'chat.tools.thinking': 'Thinking',
    'chat.tools.thought': 'Thought it through',
    'chat.tools.many': '{n} tool calls',
    'chat.tools.run': 'Run command',
    'chat.tools.read': 'Read file',
    'chat.tools.write_file': 'Write file',
    'chat.tools.write_one': 'Write 1 line',
    'chat.tools.write_n': 'Write {n} lines',
    'chat.tools.edit': 'Edit file',
    'chat.tools.grep': 'Search in files',
    'chat.tools.find': 'Find files',
    'chat.tools.ls': 'List folder',
    'chat.tools.tool': 'Tool',
    'chat.tools.running': 'Running…',
    'chat.tools.no_output': 'No output',
    'chat.tools.more_lines': '… and {n} more lines',
    'chat.tools.status_running': 'Running',
    'chat.tools.status_error': 'Failed',
    'chat.tools.diff_label': 'Show diff for {file}',
    'chat.tools.diff_preview': 'Diff preview',
    'chat.tools.more': '+{n} more',
    'chat.tools.less': 'Show less',
  },
  es: {
    'chat.tools.one': '1 llamada a herramienta',
    'chat.tools.thinking': 'Pensando',
    'chat.tools.thought': 'Lo pensó',
    'chat.tools.many': '{n} llamadas a herramientas',
    'chat.tools.run': 'Ejecutar comando',
    'chat.tools.read': 'Leer archivo',
    'chat.tools.write_file': 'Escribir archivo',
    'chat.tools.write_one': 'Escribir 1 línea',
    'chat.tools.write_n': 'Escribir {n} líneas',
    'chat.tools.edit': 'Editar archivo',
    'chat.tools.grep': 'Buscar en archivos',
    'chat.tools.find': 'Buscar archivos',
    'chat.tools.ls': 'Listar carpeta',
    'chat.tools.tool': 'Herramienta',
    'chat.tools.running': 'En ejecución…',
    'chat.tools.no_output': 'Sin salida',
    'chat.tools.more_lines': '… y {n} líneas más',
    'chat.tools.status_running': 'En ejecución',
    'chat.tools.status_error': 'Falló',
    'chat.tools.diff_label': 'Ver los cambios de {file}',
    'chat.tools.diff_preview': 'Vista previa de los cambios',
    'chat.tools.more': '+{n} más',
    'chat.tools.less': 'Mostrar menos',
  },
}

const VISIBLE_CHIPS = 3
const DETAIL_DIFF_LINES = 8
const PREVIEW_LINES = 12

// ---- tool kinds ------------------------------------------------------------------------------

const KINDS = { bash: 'run', read: 'read', write: 'write', edit: 'write', grep: 'search', find: 'search', ls: 'folder' }

/** Icon kind of a tool: run, read, write, search, folder; anything else (MCP tools too) is `tool`. */
export const toolKind = (name) => KINDS[name] ?? 'tool'

const isMcp = (name) => typeof name === 'string' && name.startsWith('mcp__')

/** `mcp__server__tool` -> { server, tool }. */
const mcpParts = (name) => {
  const parts = String(name).split('__')
  return { server: parts[1] ?? '', tool: parts.slice(2).join('__') }
}

const humanize = (raw) => {
  const text = String(raw ?? '').replace(/[_-]+/g, ' ').trim()
  return text ? text[0].toUpperCase() + text.slice(1) : ''
}

/** A step's first line, short enough for the chip. */
const firstLine = (text) => {
  const line = String(text ?? '').trim().split('\n')[0]
  return line.length > 90 ? `${line.slice(0, 89).trimEnd()}…` : line
}

/** The text split into lines; one trailing newline does not make an extra line. */
const toLines = (text) => (typeof text === 'string' && text ? text.replace(/\n$/, '').split('\n') : [])

const baseName = (path) => path.replace(/\/+$/, '').split('/').pop() || path

const argPath = (args) => {
  const path = args?.path ?? args?.file_path
  return typeof path === 'string' && path ? path : ''
}

/** The row's title: what the tool does ("Read file", "Write 3 lines"). */
export function toolLabel(call, t) {
  if (call.think) return call.label || t('chat.tools.thinking')
  const name = call.name
  switch (toolKind(name)) {
    case 'run': {
      // What the command does, in the user's words (the sandbox's bash asks the model for it), like the design's
      // "Rebuild and verify"; the command itself is the chip.
      const reason = typeof call.args?.reason === 'string' ? call.args.reason.trim() : ''
      return reason || t('chat.tools.run')
    }
    case 'read':
      return t('chat.tools.read')
    case 'search':
      return t(name === 'grep' ? 'chat.tools.grep' : 'chat.tools.find')
    case 'folder':
      return t('chat.tools.ls')
    case 'write': {
      if (name === 'edit') return t('chat.tools.edit')
      const n = toLines(call.args?.content).length
      if (n === 0) return t('chat.tools.write_file')
      return n === 1 ? t('chat.tools.write_one') : t('chat.tools.write_n', { n })
    }
    default:
      return humanize(isMcp(name) ? mcpParts(name).tool : name) || t('chat.tools.tool')
  }
}

/** The chip beside the title: the command, path or pattern (the call's summary when it has one). */
export function toolChip(call) {
  if (call.think) return firstLine(call.text)
  const { name, args = {}, summary } = call
  // A file is shown by its name, as in the design (`env.ts`); the open row and the tooltip have the rest.
  if (['read', 'write', 'edit'].includes(name) && (argPath(args) || summary)) return baseName(argPath(args) || String(summary))
  if (summary && summary !== name) return String(summary)
  const first = (value) => (typeof value === 'string' ? value.split('\n')[0] : '')
  switch (name) {
    case 'bash':
      return first(args.command)
    case 'read':
    case 'write':
    case 'edit':
      return argPath(args)
    case 'grep':
      return first(args.pattern)
    case 'find':
      return first(args.pattern) || first(args.path)
    case 'ls':
      return first(args.path) || '.'
    default:
      return isMcp(name) ? mcpParts(name).server : ''
  }
}

// ---- results and diffs -------------------------------------------------------------------------

/** The first lines of a result, bounded by line count and total characters; `hidden` counts the rest. */
export function clipResult(text, { maxLines = 12, maxChars = 1200 } = {}) {
  const all = toLines(typeof text === 'string' ? text.replace(/\s+$/, '') : '')
  const lines = []
  let used = 0
  for (let i = 0; i < all.length; i++) {
    if (lines.length >= maxLines) return { lines, hidden: all.length - i }
    const room = maxChars - used
    if (all[i].length > room) {
      lines.push(all[i].slice(0, Math.max(0, room)) + '…')
      return { lines, hidden: all.length - i - 1 }
    }
    lines.push(all[i])
    used += all[i].length
  }
  return { lines, hidden: 0 }
}

/** What a write or edit changed, from its arguments: { file, add, del, lines: [{ tone, text }] } or null. */
export function callDiff(call) {
  const args = call.args ?? {}
  const file = argPath(args)
  if (!file) return null
  const pairs = []
  if (call.name === 'write') {
    if (typeof args.content === 'string') pairs.push(['', args.content])
  } else if (call.name === 'edit') {
    const edits = Array.isArray(args.edits) ? args.edits : [args]
    for (const e of edits) {
      const before = e?.oldText ?? e?.old_string
      const after = e?.newText ?? e?.new_string
      if (typeof before === 'string' || typeof after === 'string') pairs.push([before ?? '', after ?? ''])
    }
  }
  if (!pairs.length) return null
  const lines = []
  for (const [before, after] of pairs) {
    for (const text of toLines(before)) lines.push({ tone: 'del', text })
    for (const text of toLines(after)) lines.push({ tone: 'add', text })
  }
  return { file, add: lines.filter((l) => l.tone === 'add').length, del: lines.filter((l) => l.tone === 'del').length, lines }
}

/** The settled calls' diffs, one entry per file (a file touched twice is summed). */
function changedFiles(calls) {
  const byFile = new Map()
  for (const call of calls) {
    if (call.status !== 'done') continue
    const diff = callDiff(call)
    if (!diff) continue
    const known = byFile.get(diff.file)
    if (known) {
      known.add += diff.add
      known.del += diff.del
      known.lines.push(...diff.lines)
    } else byFile.set(diff.file, { ...diff, lines: [...diff.lines] })
  }
  return [...byFile.values()]
}

// ---- markup ------------------------------------------------------------------------------------

const SVG = (body, size = 13, extra = 'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"') =>
  `<svg aria-hidden="true" width="${size}" height="${size}" viewBox="0 0 24 24" ${extra}>${body}</svg>`

const SPARK = SVG('<path d="M12 2l2.4 7.2L22 12l-7.6 2.8L12 22l-2.4-7.2L2 12l7.6-2.8z"></path>', 13, 'fill="currentColor" stroke="none"')
const ICONS = {
  tool: SPARK,
  think: SPARK,
  write: SVG('<path d="M17 3a2.8 2.8 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5z"></path>'),
  run: SVG('<path d="M4 17l6-5-6-5M12 19h8"></path>'),
  read: SVG('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><path d="M14 2v6h6"></path>'),
  search: SVG('<circle cx="11" cy="11" r="7"></circle><path d="M21 21l-4.3-4.3"></path>'),
  folder: SVG('<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>'),
}

const CHEVRON = (cls) =>
  `<svg aria-hidden="true" class="${cls}" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"></path></svg>`

const isOn = (value, index) => (value instanceof Set ? value.has(index) : value === true)

function detailLines(call, t) {
  if (call.think) return toLines(call.text).map((text) => ({ tone: '', text }))
  const out = [] // { tone, text }
  // The whole command or path, which the row's chip may cut short.
  if (call.name === 'bash' && typeof call.args?.command === 'string' && call.args.command.trim()) out.push({ tone: 'cmd', text: `$ ${call.args.command.trim()}` })
  else if (argPath(call.args) && toolChip(call) !== argPath(call.args)) out.push({ tone: 'cmd', text: argPath(call.args) })
  const diff = callDiff(call)
  if (diff) {
    const clipped = diff.lines.slice(0, DETAIL_DIFF_LINES)
    for (const l of clipped) out.push({ tone: l.tone, text: `${l.tone === 'add' ? '+' : '−'} ${l.text}` })
    if (diff.lines.length > clipped.length) out.push({ tone: 'muted', text: t('chat.tools.more_lines', { n: diff.lines.length - clipped.length }) })
  }
  const result = clipResult(call.result)
  const tone = call.status === 'error' ? 'err' : ''
  for (const text of result.lines) out.push({ tone, text })
  if (result.hidden) out.push({ tone: 'muted', text: t('chat.tools.more_lines', { n: result.hidden }) })
  if (!out.some((l) => l.tone !== 'cmd')) out.push({ tone: 'muted', text: t(call.status === 'running' ? 'chat.tools.running' : 'chat.tools.no_output') })
  return out
}

function row(call, openRows, animate, t) {
  const kind = call.think ? 'think' : toolKind(call.name)
  const open = openRows.has(call.id)
  const chip = typeof call.chip === 'string' ? call.chip : toolChip(call)
  const status = call.status === 'running' || call.status === 'error' ? call.status : 'done'
  const sr = status === 'done' ? '' : `<span class="p-tools-sr">${esc(t(`chat.tools.status_${status}`))}</span>`
  const lines = typeof call.html === 'string'
    ? ''
    : detailLines(call, t)
      .map((l) => `<span class="p-tools-line${l.tone ? ` is-${l.tone}` : ''}">${esc(l.text)}</span>`)
      .join('')
  const detail = typeof call.html === 'string' ? `<div class="p-tools-lines p-tools-md p-md">${call.html}</div>` : `<div class="p-tools-lines">${lines}</div>`
  return `
    <div class="p-tools-item${animate ? ' is-animated' : ''}">
      <button type="button" class="p-tools-row status-${status}${open ? ' is-open' : ''}" data-chat="toggle-tool-row" data-id="${esc(call.id)}" aria-expanded="${open}">
        <span class="p-tools-ico"><span class="p-tools-icon" data-icon="${kind}">${ICONS[kind]}</span>${CHEVRON('p-tools-chevron')}</span>
        <span class="p-tools-label">${esc(toolLabel(call, t))}${sr}</span>${chip ? `
        <span class="p-tools-cmd${kind === 'tool' || kind === 'think' ? ' is-sans' : ''}" title="${esc(chip)}">${esc(chip)}</span>` : ''}
      </button>
      <div class="p-tools-detail${open ? ' is-open' : ''}"><div class="p-tools-clip">${detail}</div></div>
    </div>`
}

function preview(file, t) {
  const shown = file.lines.slice(0, PREVIEW_LINES)
  const body = shown
    .map((l) => `<div class="p-tools-pline is-${l.tone}"><span class="p-tools-mark" aria-hidden="true">${l.tone === 'add' ? '+' : '−'}</span><span class="p-tools-ptext">${esc(l.text)}</span></div>`)
    .join('')
  const more = file.lines.length > shown.length ? `<div class="p-tools-pline is-ctx"><span class="p-tools-mark" aria-hidden="true"></span><span class="p-tools-ptext">${esc(t('chat.tools.more_lines', { n: file.lines.length - shown.length }))}</span></div>` : ''
  return `
        <div class="p-tools-preview" role="group" aria-label="${esc(t('chat.tools.diff_preview'))}">
          <div class="p-tools-phead"><span class="p-tools-pfile">${esc(file.file)}</span><span class="p-tools-pnum"><span class="is-add">+${file.add}</span>${file.del ? ` <span class="is-del">−${file.del}</span>` : ''}</span></div>
          <div class="p-tools-pbody">${body}${more}</div>
        </div>`
}

function diffChips(files, index, all, t) {
  if (!files.length) return ''
  const shown = all ? files : files.slice(0, VISIBLE_CHIPS)
  const hidden = files.length - VISIBLE_CHIPS
  const chips = shown
    .map((f, i) => {
      const delay = i < VISIBLE_CHIPS ? i * 80 : (i - VISIBLE_CHIPS) * 60
      return `
      <span class="p-tools-chipwrap">
        <button type="button" class="p-tools-file" data-chip="${esc(f.file)}" aria-label="${esc(t('chat.tools.diff_label', { file: f.file }))}" style="animation-delay:${delay}ms"><span class="p-tools-fname">${esc(f.file)}</span><span class="p-tools-add">+${f.add}</span>${f.del ? `<span class="p-tools-del">−${f.del}</span>` : ''}</button>${preview(f, t)}
      </span>`
    })
    .join('')
  const more =
    hidden > 0
      ? `
      <button type="button" class="p-tools-more" data-chat="toggle-tool-diffs" data-index="${esc(index)}" aria-expanded="${all}" style="animation-delay:${VISIBLE_CHIPS * 80}ms">${esc(all ? t('chat.tools.less') : t('chat.tools.more', { n: hidden }))}</button>`
      : ''
  return `<div class="p-tools-diffs">${chips}${more}</div>`
}

export function renderTools(props, ui, t) {
  const calls = props?.calls ?? []
  if (!calls.length) return ''
  const index = props.index ?? ''
  const open = isOn(ui?.open, props.index)
  const openRows = ui?.openRows ?? new Set()
  const animate = ui?.animate !== false
  const live = props.live === true || calls.some((c) => c.status === 'running')
  const files = live ? [] : changedFiles(calls)
  const tools = calls.filter((c) => !c.think).length
  const header = tools === 0 ? t('chat.tools.thought') : tools === 1 ? t('chat.tools.one') : t('chat.tools.many', { n: tools })
  const classes = ['p-tools', open ? 'is-open' : '', live ? 'is-live' : '', open && files.length ? 'has-diffs' : ''].filter(Boolean).join(' ')
  return `
    <div class="${classes}"${live ? ' aria-busy="true"' : ''}>
      <button type="button" class="p-tools-head" data-chat="toggle-tools" data-index="${esc(index)}" aria-expanded="${open}">
        ${CHEVRON('p-tools-hchev')}<span class="p-tools-title">${esc(header)}</span>${live ? '<span class="p-tools-pulse" aria-hidden="true"></span>' : ''}
      </button>
      <div class="p-tools-region"><div class="p-tools-body">
        <div class="p-tools-list">${calls.map((call) => row(call, openRows, animate, t)).join('')}
        </div>${diffChips(files, props.index, isOn(ui?.moreOpen, props.index), t)}
      </div></div>
    </div>`
}

/** One step on its own, without the group's header: the same row, folded or open (`ui.openRows`), for a text the
 * thread folds outside a turn (the earlier conversation's summary). */
export const renderToolStep = ({ call }, ui, t) =>
  `<div class="p-tools p-tools-step"><div class="p-tools-list">${row(call, ui?.openRows ?? new Set(), false, t)}</div></div>`
