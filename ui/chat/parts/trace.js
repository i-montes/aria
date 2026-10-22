// The trace panel (`/debug`): the current turn as a waterfall — every LLM call, tool call and subagent as a bar on one
// timeline, with time, tokens and cost. Aria's own take on a trace view; the dock builds the tree
// (crates/aria/src/trace.rs) and this part only draws it. Pure: props and ui in, escaped HTML out.
//
// Contract:
//   props = {
//     tree:      Span | null,   // the dock's `chatTrace`; null before the tab's first traced turn
//     now:       number,        // ms from the turn's start, for running bars (the view ticks it)
//     collapsed: Set<string>,   // agent spans folded
//     open:      string|null,   // the span whose details card is open
//     focus:     string|null,   // the row in the tab order (roving focus); the first row when null
//   }
//   Span = { id, kind: 'agent'|'llm'|'tool', name, detail?, status: 'running'|'done'|'failed', start, end?,
//            tokensIn?, tokensOut?, cacheRead?, reasoning?, cost?, input?, output?, error?, children?,
//            now?, dropped? (root only) }
//   ui = {} (nothing; the part is stateless)
//
// Emits:
//   data-chat="trace-row"   data-id   open or close that span's details (and focus it)
//   data-chat="trace-fold"  data-id   fold or unfold an agent span
//   data-chat="trace-close"           close the details card

import { esc } from '../../onboarding/components.js'

export const TEXTS = {
  en: {
    'chat.trace.title': 'Trace',
    'chat.trace.aria': 'Steps of the turn',
    'chat.trace.live': 'Live',
    'chat.trace.done': 'Done',
    'chat.trace.failed': 'Failed',
    'chat.trace.duration': 'Duration',
    'chat.trace.cost': 'Cost',
    'chat.trace.tokens': 'Tokens',
    'chat.trace.llm': 'LLM calls',
    'chat.trace.tools': 'Tool calls',
    'chat.trace.errors': 'Errors',
    'chat.trace.span': 'Step',
    'chat.trace.slowest': 'slowest',
    'chat.trace.empty': 'The next turn is traced here',
    'chat.trace.dropped': '{n} earlier steps not shown',
    'chat.trace.kind.agent': 'Agent',
    'chat.trace.kind.llm': 'LLM call',
    'chat.trace.kind.tool': 'Tool call',
    'chat.trace.started': 'Started',
    'chat.trace.tokensIn': 'Tokens in',
    'chat.trace.tokensOut': 'Tokens out',
    'chat.trace.cacheRead': 'From cache',
    'chat.trace.reasoning': 'Reasoning',
    'chat.trace.input': 'Input',
    'chat.trace.output': 'Output',
    'chat.trace.error': 'Error',
    'chat.trace.close': 'Close',
    'chat.trace.fold': 'Fold or unfold {name}',
  },
  es: {
    'chat.trace.title': 'Traza',
    'chat.trace.aria': 'Pasos del turno',
    'chat.trace.live': 'En vivo',
    'chat.trace.done': 'Listo',
    'chat.trace.failed': 'Falló',
    'chat.trace.duration': 'Duración',
    'chat.trace.cost': 'Costo',
    'chat.trace.tokens': 'Tokens',
    'chat.trace.llm': 'Llamadas al modelo',
    'chat.trace.tools': 'Herramientas',
    'chat.trace.errors': 'Errores',
    'chat.trace.span': 'Paso',
    'chat.trace.slowest': 'más lento',
    'chat.trace.empty': 'El próximo turno se traza aquí',
    'chat.trace.dropped': '{n} pasos anteriores no se muestran',
    'chat.trace.kind.agent': 'Agente',
    'chat.trace.kind.llm': 'Llamada al modelo',
    'chat.trace.kind.tool': 'Herramienta',
    'chat.trace.started': 'Empezó',
    'chat.trace.tokensIn': 'Tokens de entrada',
    'chat.trace.tokensOut': 'Tokens de salida',
    'chat.trace.cacheRead': 'Desde caché',
    'chat.trace.reasoning': 'Razonamiento',
    'chat.trace.input': 'Entrada',
    'chat.trace.output': 'Salida',
    'chat.trace.error': 'Error',
    'chat.trace.close': 'Cerrar',
    'chat.trace.fold': 'Plegar o desplegar {name}',
  },
}

const SECOND = 1000
const MINUTE = 60 * SECOND
const HOUR = 60 * MINUTE
const STEPS = [10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600].map((s) => s * SECOND)

/** The shortest round range the axis shows for a turn of `ms` (at least 10 s; whole hours past one). */
export function axisRange(ms) {
  const step = STEPS.find((s) => s >= ms)
  return step ?? Math.ceil(ms / HOUR) * HOUR
}

export function formatMs(ms) {
  const n = Math.max(0, Math.round(Number(ms) || 0))
  if (n < SECOND) return `${n}ms`
  if (n < 10 * SECOND) return `${(n / SECOND).toFixed(1)}s`
  if (n < MINUTE) return `${Math.round(n / SECOND)}s`
  if (n < HOUR) {
    const m = Math.floor(n / MINUTE)
    const s = Math.round((n % MINUTE) / SECOND)
    return s ? `${m}m ${s}s` : `${m}m`
  }
  const h = Math.floor(n / HOUR)
  const m = Math.round((n % HOUR) / MINUTE)
  return m ? `${h}h ${m}m` : `${h}h`
}

/** Dollars: three decimals under ten cents when they say something ($0.042), else two ($0.12, $1.40). */
export function formatCost(n) {
  const v = Number(n) || 0
  if (v >= 0.1) return `$${v.toFixed(2)}`
  const three = v.toFixed(3)
  return `$${three.endsWith('0') ? v.toFixed(2) : three}`
}

const number = (n) => Number(n ?? 0).toLocaleString('en-US')
const kids = (span) => (Array.isArray(span?.children) ? span.children : [])
const endOf = (span, now) => (span.end ?? Math.max(now, span.start ?? 0))

function walk(span, fn, level = 1) {
  fn(span, level)
  for (const c of kids(span)) walk(c, fn, level + 1)
}

/** The whole subtree's figures (each span's cost and tokens are its own). */
export function totals(root) {
  const sum = { cost: 0, tokens: 0, llm: 0, tools: 0, errors: 0, costKnown: false }
  if (!root) return sum
  walk(root, (s) => {
    const cost = Number(s.cost) || 0
    sum.cost += cost
    if (cost > 0) sum.costKnown = true
    sum.tokens += (Number(s.tokensIn) || 0) + (Number(s.tokensOut) || 0)
    if (s.kind === 'llm') sum.llm++
    if (s.kind === 'tool') sum.tools++
    if (s.status === 'failed' && s.kind !== 'agent') sum.errors++
  })
  return sum
}

/** The finished LLM or tool call that took longest. */
export function slowestId(root) {
  let best = null
  let most = -1
  if (!root) return null
  walk(root, (s) => {
    if ((s.kind === 'llm' || s.kind === 'tool') && s.status !== 'running' && typeof s.end === 'number') {
      const took = s.end - s.start
      if (took > most) {
        most = took
        best = s.id
      }
    }
  })
  return best
}

/** Finished subagents fold by themselves (not the root, not one folded or opened by hand). */
export function autoCollapsed(root, touched = new Set()) {
  const out = new Set()
  if (!root) return out
  walk(root, (s) => {
    if (s.kind === 'agent' && s.id !== 'root' && s.status !== 'running' && kids(s).length && !touched.has(s.id)) out.add(s.id)
  })
  return out
}

/** The rows on screen, in order, with their tree level (for the keyboard). */
export function visibleRows(root, collapsed = new Set()) {
  const rows = []
  const visit = (span, level) => {
    rows.push({ span, level })
    if (!collapsed.has(span.id)) for (const c of kids(span)) visit(c, level + 1)
  }
  if (root) visit(root, 1)
  return rows
}

const pct = (ms, range) => `${((Math.max(0, ms) / range) * 100).toFixed(2)}%`

const ICON = {
  agent: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="6" r="2.6"/><path d="M3.5 13.5c.6-2.4 2.4-3.6 4.5-3.6s3.9 1.2 4.5 3.6"/></svg>',
  llm: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2.2l1.4 3.6 3.6 1.4-3.6 1.4L8 12.2 6.6 8.6 3 7.2l3.6-1.4z"/></svg>',
  tool: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10.5 2.5a3 3 0 0 0-3 3.8L3 10.8 5.2 13l4.5-4.5a3 3 0 0 0 3.8-3L12 7 9 4z"/></svg>',
}
const CHEVRON = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 4l4 4-4 4"/></svg>'

function stateKey(status) {
  return status === 'failed' ? 'chat.trace.failed' : status === 'running' ? 'chat.trace.live' : 'chat.trace.done'
}

function figure(name, label, value) {
  return `<div class="p-trace-figure" data-figure="${name}"><span class="p-trace-flabel">${esc(label)}</span><span class="p-trace-fvalue">${esc(value)}</span></div>`
}

function row({ span, level }, ctx) {
  const { now, range, collapsed, slowest, focus, open, t } = ctx
  const kind = ['agent', 'llm', 'tool'].includes(span.kind) ? span.kind : 'tool'
  const children = kids(span)
  const folded = collapsed.has(span.id)
  const end = endOf(span, now)
  const took = end - (span.start ?? 0)
  const own = totals({ ...span, children: kind === 'agent' ? children : [] })
  const cost = kind === 'agent' ? own.cost : Number(span.cost) || 0
  const meta = `${formatMs(took)}${cost > 0 ? ` · ${formatCost(cost)}` : ''}`
  const marks = folded
    ? children.map((c) => `<span class="p-trace-mark kind-${esc(c.kind)}${c.status === 'failed' ? ' is-failed' : ''}" style="left: ${pct(c.start, range)}" aria-hidden="true"></span>`).join('')
    : ''
  const fold = children.length
    ? `<span class="p-trace-fold${folded ? '' : ' is-open'}" data-chat="trace-fold" data-id="${esc(span.id)}" title="${esc(t('chat.trace.fold', { name: span.name ?? '' }))}">${CHEVRON}</span>`
    : '<span class="p-trace-fold is-leaf"></span>'
  const count = folded ? `<span class="p-trace-count">+${children.length}</span>` : ''
  const tag = span.id === slowest ? `<span class="p-trace-tag">${esc(t('chat.trace.slowest'))}</span>` : ''
  const label = `${span.name ?? ''}, ${t(`chat.trace.kind.${kind}`)}, ${t(stateKey(span.status))}, ${meta}`
  const classes = ['p-trace-row', `kind-${kind}`, `is-${esc(span.status ?? 'done')}`, span.id === open ? 'is-open' : ''].filter(Boolean).join(' ')
  return `<div class="${classes}" role="treeitem" aria-level="${level}"${children.length ? ` aria-expanded="${!folded}"` : ''} aria-selected="${span.id === open}" data-chat="trace-row" data-id="${esc(span.id)}" tabindex="${span.id === focus ? 0 : -1}" aria-label="${esc(label)}">
  <span class="p-trace-name" style="padding-left: ${(level - 1) * 12}px">${fold}<span class="p-trace-icon">${ICON[kind]}</span><span class="p-trace-label">${esc(span.name ?? '')}</span>${span.detail ? `<span class="p-trace-detail">${esc(span.detail)}</span>` : ''}${count}${tag}</span>
  <span class="p-trace-lane"><span class="p-trace-bar" data-bar="${esc(span.id)}" style="left: ${pct(span.start ?? 0, range)}; width: ${pct(took, range)}"></span>${marks}</span>
  <span class="p-trace-meta">${esc(meta)}</span>
</div>`
}

function find(root, id) {
  let found = null
  if (root && id) walk(root, (s) => {
    if (s.id === id) found = s
  })
  return found
}

function card(span, now, t) {
  const kind = ['agent', 'llm', 'tool'].includes(span.kind) ? span.kind : 'tool'
  const took = endOf(span, now) - (span.start ?? 0)
  const facts = [
    [t('chat.trace.duration'), formatMs(took)],
    [t('chat.trace.started'), `+${formatMs(span.start ?? 0)}`],
  ]
  for (const key of ['tokensIn', 'tokensOut', 'cacheRead', 'reasoning']) {
    if (Number(span[key]) > 0) facts.push([t(`chat.trace.${key}`), number(span[key])])
  }
  if (Number(span.cost) > 0) facts.push([t('chat.trace.cost'), formatCost(span.cost)])
  const block = (key, text, cls = '') => (text ? `<div class="p-trace-block${cls}"><span class="p-trace-flabel">${esc(t(key))}</span><pre>${esc(text)}</pre></div>` : '')
  return `<div class="p-trace-card kind-${kind}" role="dialog" aria-label="${esc(span.name ?? '')}">
  <div class="p-trace-card-head"><span class="p-trace-icon">${ICON[kind]}</span><span class="p-trace-card-title">${esc(span.name ?? '')}<small>${esc(t(`chat.trace.kind.${kind}`))}${span.detail ? ` · ${esc(span.detail)}` : ''}</small></span>
    <span class="p-trace-state is-${esc(span.status ?? 'done')}">${esc(t(stateKey(span.status)))}</span>
    <button type="button" class="p-trace-close" data-chat="trace-close" aria-label="${esc(t('chat.trace.close'))}">×</button></div>
  <dl class="p-trace-facts">${facts.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('')}</dl>
  ${block('chat.trace.error', span.error, ' is-error')}${block('chat.trace.input', span.input)}${block('chat.trace.output', span.output)}
</div>`
}

export function renderTrace(props, ui, t) {
  const tree = props?.tree
  if (!tree) {
    return `<div class="p-trace is-empty"><div class="p-trace-head"><span class="p-trace-title">${esc(t('chat.trace.title'))}</span></div><p class="p-trace-empty">${esc(t('chat.trace.empty'))}</p></div>`
  }
  const now = Math.max(0, Number(props.now ?? tree.now ?? 0))
  const collapsed = props.collapsed ?? new Set()
  const rows = visibleRows(tree, collapsed)
  let last = now
  walk(tree, (s) => {
    if (typeof s.end === 'number') last = Math.max(last, s.end)
  })
  const running = tree.status === 'running'
  const range = axisRange(running ? Math.max(now, last) : last)
  const sum = totals(tree)
  const focus = rows.some((r) => r.span.id === props.focus) ? props.focus : rows[0]?.span.id
  const ctx = { now, range, collapsed, slowest: slowestId(tree), focus, open: props.open ?? null, t }
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => `<span class="p-trace-tick" style="--at: ${(f * 100).toFixed(2)}%">${esc(f ? formatMs(f * range) : '0s')}</span>`).join('')
  const nowLine = running ? `<span class="p-trace-now" style="--at: ${Math.min(1, now / range).toFixed(4)}" aria-hidden="true"></span>` : ''
  const opened = find(tree, props.open)
  const duration = (tree.end ?? now) - (tree.start ?? 0)
  return `<div class="p-trace${running ? ' is-running' : ''}">
  <div class="p-trace-head"><span class="p-trace-title">${esc(t('chat.trace.title'))}</span><span class="p-trace-input">${esc(tree.input ?? '')}</span><span class="p-trace-state is-${esc(tree.status ?? 'done')}" role="status" aria-live="polite">${esc(t(stateKey(tree.status)))}</span></div>
  <div class="p-trace-figures">${figure('duration', t('chat.trace.duration'), formatMs(duration))}${figure('cost', t('chat.trace.cost'), sum.costKnown ? formatCost(sum.cost) : '—')}${figure('tokens', t('chat.trace.tokens'), number(sum.tokens))}${figure('llm', t('chat.trace.llm'), String(sum.llm))}${figure('tools', t('chat.trace.tools'), String(sum.tools))}${figure('errors', t('chat.trace.errors'), String(sum.errors))}</div>
  <div class="p-trace-body">
    <div class="p-trace-axis" aria-hidden="true"><span class="p-trace-span">${esc(t('chat.trace.span'))}</span><span class="p-trace-ticks">${ticks}</span></div>
    <div class="p-trace-rows" role="tree" aria-label="${esc(t('chat.trace.aria'))}">${rows.map((r) => row(r, ctx)).join('')}${nowLine}</div>
    ${tree.dropped ? `<p class="p-trace-dropped">${esc(t('chat.trace.dropped', { n: tree.dropped }))}</p>` : ''}
  </div>
  ${opened ? card(opened, now, t) : ''}
</div>`
}
