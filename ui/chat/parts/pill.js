// A pill: a small rounded control with an optional coloured dot, a label and a muted meta. One part for every pill of
// the chat: the memory panel's type filters and the background strip above the composer (reuse, never a look-alike).
// Pure: props in, escaped HTML out.
//
// Props: { label, meta?, dot?, tone?, pressed?, action?, id?, value?, extra?, title?, static? }
//   label    the text (escaped)
//   meta     a muted value after the label: a count, ':3000 · 12 min' (escaped; left out when empty)
//   dot      true shows the dot before the label; its colour comes from the tone (the user's CSS)
//   tone     adds the class `tone-<tone>` (memory types, `process` / `agent` / `queued` / `failed`)
//   pressed  true / false sets aria-pressed (a toggle); undefined leaves it out
//   action   data-chat of the pill's button
//   id       data-id; value: data-value (what the view reads for that action)
//   title    a tooltip and accessible name for the button
//   extra    TRUSTED markup (already escaped by the caller) put after the pill's button inside it, e.g. a stop button.
//            With extra the pill is a <span> holding its main button (`.p-pill-main`) and the extra, since a button
//            cannot hold another button; without it the pill is the button itself.
//   static   true: a pill with nothing to press (a fact, like a sent message's file): a <span>, same look, no action
// ui: unused. t: unused (the pill has no texts of its own).
// Emits: whatever `action` names, with data-id / data-value.
// Styles: pill.css (`.p-pill`, `.p-pill-dot`, `.p-pill-label`, `.p-pill-meta`, `.p-pill-main`).
import { esc } from '../../onboarding/components.js'

export const TEXTS = { en: {}, es: {} }

const has = (v) => v !== undefined && v !== null && v !== ''

export function renderPill({ label, meta, dot = false, tone, pressed, action, id, value, extra, title, static: still = false } = {}, ui, t) {
  const toneClass = has(tone) ? ` tone-${esc(tone)}` : ''
  const attrs =
    (pressed === undefined ? '' : ` aria-pressed="${pressed ? 'true' : 'false'}"`) +
    (has(action) ? ` data-chat="${esc(action)}"` : '') +
    (has(id) ? ` data-id="${esc(id)}"` : '') +
    (has(value) ? ` data-value="${esc(value)}"` : '') +
    (has(title) ? ` title="${esc(title)}"` : '')
  const body =
    (dot ? '<span class="p-pill-dot" aria-hidden="true"></span>' : '') +
    `<span class="p-pill-label">${esc(label)}</span>` +
    (has(meta) ? `<span class="p-pill-meta">${esc(meta)}</span>` : '')
  if (still) return `<span class="p-pill is-static${toneClass}"${has(title) ? ` title="${esc(title)}"` : ''}>${body}</span>`
  if (!extra) return `<button type="button" class="p-pill${toneClass}"${attrs}>${body}</button>`
  return `<span class="p-pill has-extra${toneClass}"><button type="button" class="p-pill-main"${attrs}>${body}</button>${extra}</span>`
}
