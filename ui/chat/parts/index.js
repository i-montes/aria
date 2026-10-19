// The message index of the chat (design: design/components/message-index.dc.html): a rail on the
// thread's right edge with one chip per message the user sent. Hovering the edge fans the chips out
// and magnifies those near the pointer; clicking one jumps to that message. The geometry is a pure
// function (`indexLayout`) so the controller only has to feed it the pointer, the scroll and the size.
//
// Props:  { items: [{ id, label }],     derived from the thread with `indexItems(messages)`; `id` is the
//                                       DOM id of the message element (`messageDomId(index)`)
//           activeId?: string }         the item whose message is at the top of the thread
// Ui:     { expanded?: boolean,         the pointer is on the rail (chips are shown)
//           my?: number|null,           pointer y inside the rail, for the magnifying effect
//           off?: number,               scroll offset of the chips when they do not all fit
//           height?: number,            the rail's height in px (the thread's visible height; default 360)
//           settled?: boolean }         the opening fan-out has finished (no more per-chip delays)
// Emits:  data-chat="index-jump"  data-id="<item id>"
//
// Helpers: `indexItems(messages)`, `indexTargets(items)`, `messageDomId(i)`,
//          `activeIndexFor(scrollTop, offsets)`, `indexLayout(count, state)`.
// Every dynamic value goes through `esc`; numbers are coerced. Styles: index.css (`.p-index-…`).
import { esc } from '../../onboarding/components.js'

export const TEXTS = {
  en: { 'chat.index.label': 'Jump to a message you sent' },
  es: { 'chat.index.label': 'Ir a un mensaje que enviaste' },
}

const LABEL_MAX = 120

/** The DOM id the thread gives to the element of the message at `index` of `state.messages`. */
export const messageDomId = (index) => `msg-${index}`

/** One item per user message: `{ id, label }`, the label being its text on one line. */
export function indexItems(messages) {
  const items = []
  ;(messages ?? []).forEach((message, index) => {
    if (message?.kind !== 'user') return
    const text = String(message.text ?? '').replace(/\s+/g, ' ').trim()
    if (!text) return
    items.push({ id: messageDomId(index), label: text.length > LABEL_MAX ? text.slice(0, LABEL_MAX).trimEnd() + '…' : text })
  })
  return items
}

/** The DOM ids the controller has to measure (`document.getElementById`) to feed `activeIndexFor`. */
export const indexTargets = (items) => (items ?? []).map((item) => item.id)

/**
 * Which item is current. `offsets` holds, per item, its top in the thread's scroll space, or
 * `{ top, bottom }`. With bottoms it is the design's rule (the first message that still reaches below
 * the top edge); with tops only, the last message whose top has reached the edge (a jump lands 12px
 * below it). -1 when there are no items.
 */
export function activeIndexFor(scrollTop, offsets) {
  if (!offsets?.length) return -1
  const top = Number(scrollTop) || 0
  if (typeof offsets[0] === 'object') {
    const found = offsets.findIndex((o) => o.bottom > top + 8)
    return found === -1 ? offsets.length - 1 : found
  }
  let active = 0
  offsets.forEach((o, i) => {
    if (o <= top + 12) active = i
  })
  return active
}

const ROW = 30 // chip height
const GAP = 4
const PITCH = ROW + GAP
const PAD = 8
const SPREAD = 44 // px: how wide the magnifying bump is
const BOOST = 0.4 // peak extra scale

const num = (value, fallback) => (Number.isFinite(Number(value)) && value !== null && value !== '' ? Number(value) : fallback)

/**
 * Where every chip goes. `state`: { height, my, off, expanded, settled }. Returns the rows
 * (`top`, `height` of its slot, `scale`, `near`, entry `delay`) and the scroll bookkeeping.
 */
export function indexLayout(count, state = {}) {
  const H = num(state.height, 360)
  const my = state.my === null || state.my === undefined ? null : Number.isFinite(Number(state.my)) ? Number(state.my) : null
  const expanded = !!state.expanded
  const n = count
  const contentH = n * PITCH - GAP + 2 * PAD
  const overflow = contentH > H
  const maxOff = Math.max(0, contentH - H)
  const off = Math.min(maxOff, Math.max(0, num(state.off, 0)))
  const top0 = overflow ? PAD - off : (H - (n * PITCH - GAP)) / 2

  const scales = Array.from({ length: n }, (_, i) => {
    if (my === null) return 1
    const centre = top0 + i * PITCH + ROW / 2
    return 1 + BOOST * Math.exp(-((centre - my) * (centre - my)) / (2 * SPREAD * SPREAD))
  })
  const heights = scales.map((s) => ROW * s)
  const total = heights.reduce((a, b) => a + b, 0) + Math.max(0, n - 1) * GAP

  // keep the chip under the pointer where it was while the ones around it grow
  let y0 = top0
  if (my !== null && n > 0) {
    const u = Math.min(n, Math.max(0, (my - top0) / PITCH))
    const k = Math.min(n - 1, Math.floor(u))
    let before = 0
    for (let j = 0; j < k; j++) before += heights[j] + GAP
    y0 = my - (before + (u - k) * (heights[k] + GAP))
  }
  y0 = overflow ? Math.min(PAD, Math.max(y0, H - PAD - total)) : Math.min(Math.max(y0, PAD), Math.max(PAD, H - total - PAD))

  const mid = (n - 1) / 2
  let cursor = y0
  const rows = scales.map((scale, i) => {
    const row = {
      top: cursor,
      height: heights[i],
      scale,
      near: (scale - 1) / BOOST > 0.45,
      delay: state.settled || !expanded ? 0 : Math.round(Math.abs(i - mid) * 30),
    }
    cursor += heights[i] + GAP
    return row
  })
  const thumbH = Math.max(24, (H * H) / Math.max(contentH, 1))
  return { rows, overflow, maxOff, contentH, thumbH, thumbTop: maxOff ? 4 + (off / maxOff) * (H - thumbH - 8) : 4, height: H }
}

const px = (value) => `${Number(value.toFixed(2))}px`

export function renderIndex(props, ui, t) {
  const items = props?.items ?? []
  if (!items.length) return ''
  const expanded = !!ui?.expanded
  const layout = indexLayout(items.length, { height: ui?.height, my: ui?.my, off: ui?.off, expanded, settled: ui?.settled })
  const active = items.findIndex((item) => item.id === props.activeId)
  const lis = items
    .map((item, i) => {
      const r = layout.rows[i]
      const transform = expanded ? `translateX(${(-(r.scale - 1) * 60).toFixed(1)}px) scale(${r.scale.toFixed(3)})` : 'translateX(28px) scale(0.6)'
      const on = i === active
      const cls = `p-index-btn${r.near ? ' is-near' : ''}${on ? ' is-active' : ''}`
      return `
      <li class="p-index-item" style="top:${px(r.top)};height:${px(r.height)}">
        <button type="button" class="${cls}"${on ? ' aria-current="true"' : ''} data-chat="index-jump" data-id="${esc(item.id)}" title="${esc(item.label)}" style="transform:${transform};--d:${r.delay}ms">
          <span class="p-index-dot" aria-hidden="true"></span><span class="p-index-label">${esc(item.label)}</span>
        </button>
      </li>`
    })
    .join('')
  const thumb =
    expanded && layout.overflow
      ? `
    <div class="p-index-thumb" aria-hidden="true" style="top:${px(layout.thumbTop)};height:${px(layout.thumbH)}"></div>`
      : ''
  return `
    <div class="p-index">
      <div class="p-index-panel${expanded ? ' is-open' : ''}" role="navigation" aria-label="${esc(t('chat.index.label'))}" style="height:${px(layout.height)}">
        <div class="p-index-blur" aria-hidden="true"></div>
        <div class="p-index-veil" aria-hidden="true"></div>
        <ol class="p-index-list${layout.overflow ? ' is-overflow' : ''}">${lis}
        </ol>${thumb}
      </div>
    </div>`
}
