// The form widgets of the onboarding as small functions that return HTML. Every dynamic value is
// escaped. Interaction is declarative: a widget carries `data-action` (what it does), `data-field`
// (which setting) and `data-value`, and `view.js` handles them all in one place. `data-nav` marks
// what the arrow keys move between; `data-key` lets focus survive a re-render.

export const esc = (value) =>
  String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c])

const key = (action, field, value = '') => `data-key="${esc(`${action}:${field}:${value}`)}"`
const indentClass = (indent) => (indent ? ' indent' : '')

export const section = (label) => `<div class="section">${esc(label)}</div>`

/** tone: lavender | green | amber | neutral */
export const info = (tone, label, description, indent = false) => `
  <div class="info tone-${tone}${indentClass(indent)}">
    ${label ? `<b>${esc(label)}</b>` : ''}${description ? `<span>${esc(description)}</span>` : ''}
  </div>`

export const bullets = (items) => `<ul class="bullets">${items.map((text) => `<li>${esc(text)}</li>`).join('')}</ul>`

const label = (text, description, indent) =>
  text || description
    ? `<div class="field-label${indentClass(indent)}">${text ? `<span class="label">${esc(text)}</span>` : ''}${description ? `<span class="hint">${esc(description)}</span>` : ''}</div>`
    : ''

const option = (field, o, selected) => `
  <button type="button" class="opt${selected ? ' selected' : ''}" data-nav data-action="pick" data-field="${esc(field)}" data-value="${esc(o.value)}"
    ${key('pick', field, o.value)} aria-pressed="${selected}" ${o.disabled ? 'disabled' : ''}>
    <span class="ring"></span>
    <span class="opt-text"><span class="opt-label">${esc(o.label)}${(o.badges ?? []).map((b) => `<em class="badge">${esc(b)}</em>`).join('')}</span>${o.description ? `<span class="opt-desc">${esc(o.description)}</span>` : ''}</span>
  </button>`

/** One option picked from a list of cards. `options`: { value, label, description?, badges?, disabled? }. */
export const choice = (field, text, options, value) =>
  `<div class="group">${label(text)}<div class="choice" role="radiogroup">${options.map((o) => option(field, o, o.value === value)).join('')}</div></div>`

const segment = (field, o, selected) => `
  <button type="button" class="seg${selected ? ' selected' : ''}" data-nav data-action="pick" data-field="${esc(field)}" data-value="${esc(o.value)}"
    ${key('pick', field, o.value)} aria-pressed="${selected}" ${o.disabled ? 'disabled' : ''}>${esc(o.label)}</button>`

/** The same, as a compact row of buttons. */
export const segments = (field, text, options, value, description, indent = false) =>
  `<div class="group${indentClass(indent)}">${label(text, description)}<div class="segments" role="radiogroup">${options.map((o) => segment(field, o, o.value === value)).join('')}</div></div>`

export const toggle = (field, text, description, on, indent = false) => `
  <button type="button" class="toggle-row${indentClass(indent)}" role="switch" aria-checked="${on}" data-nav data-action="toggle" data-field="${esc(field)}" ${key('toggle', field)}>
    <span class="toggle-text"><span class="label">${esc(text)}</span>${description ? `<span class="hint">${esc(description)}</span>` : ''}</span>
    <span class="track${on ? ' on' : ''}"><span class="knob"></span></span>
  </button>`

/** `secret`: password field with Show/Hide. `revealed`, `showText` and `hideText` only matter then. */
export const textField = (field, text, value, o = {}) => `
  <div class="group${indentClass(o.indent)}">
    ${label(text, o.description, false)}
    <div class="input-row">
      <input data-nav data-action="input" data-field="${esc(field)}" ${key('input', field)} type="${o.secret && !o.revealed ? 'password' : 'text'}"
        class="${o.mono || o.secret ? 'mono' : ''}" value="${esc(value)}" placeholder="${esc(o.placeholder ?? '')}" spellcheck="false" autocomplete="off">
      ${o.secret ? `<button type="button" class="link" data-nav data-action="reveal" data-field="${esc(field)}" ${key('reveal', field)}>${esc(o.revealed ? o.hideText : o.showText)}</button>` : ''}
    </div>
  </div>`

export const slider = (field, text, description, value, min, max, step, unit) => `
  <div class="group">
    <div class="slider-head">${label(text, description)}<span class="readout" data-readout="${esc(field)}">${esc(value)}${esc(unit)}</span></div>
    <input data-nav data-action="slide" data-field="${esc(field)}" ${key('slide', field)} data-unit="${esc(unit)}" type="range" min="${min}" max="${max}" step="${step}" value="${value}">
  </div>`

export const meter = (percent, caption) => `
  <div class="meter"><div class="meter-fill" style="width:${Math.max(0, Math.min(100, percent))}%"></div></div>
  <div class="meter-caption">${esc(caption)}</div>`

export const stepper = (field, text, description, value, min, max, step, unit) => `
  <div class="group stepper-row">
    ${label(text, description)}
    <div class="stepper">
      <button type="button" data-nav data-action="step" data-field="${esc(field)}" data-value="-${step}" ${key('step', field, '-')} ${value <= min ? 'disabled' : ''} aria-label="−">−</button>
      <span class="readout">${esc(value)}${esc(unit)}</span>
      <button type="button" data-nav data-action="step" data-field="${esc(field)}" data-value="${step}" ${key('step', field, '+')} ${value >= max ? 'disabled' : ''} aria-label="+">+</button>
    </div>
  </div>`

/** A list of removable chips plus a field to add one. */
export const chips = (field, text, items, draft, placeholder, removeLabel, addLabel = '+') => `
  <div class="group">
    ${label(text)}
    ${items.length ? `<div class="chips">${items.map((item, i) => `<span class="chip">${esc(item)}<button type="button" class="chip-x" data-nav data-action="remove-chip" data-field="${esc(field)}" data-value="${i}" ${key('remove-chip', field, i)} aria-label="${esc(removeLabel)}">×</button></span>`).join('')}</div>` : ''}
    <div class="input-row">
      <input data-nav data-action="chip-draft" data-field="${esc(field)}" ${key('chip-draft', field)} type="text" class="mono" value="${esc(draft)}" placeholder="${esc(placeholder)}" spellcheck="false" autocomplete="off">
      <button type="button" class="chip-add" data-nav data-action="add-chip" data-field="${esc(field)}" ${key('add-chip', field)} aria-label="${esc(addLabel)}">+</button>
    </div>
  </div>`

/** A list of independent on/off rows. `items`: { id, label, description, on }. */
export const checks = (field, text, items) => `
  <div class="group">${label(text)}<div class="checks">${items.map((c) => `
    <button type="button" class="check-row" role="checkbox" aria-checked="${c.on}" data-nav data-action="check" data-field="${esc(field)}" data-value="${esc(c.id)}" ${key('check', field, c.id)}>
      <span class="box${c.on ? ' on' : ''}">${c.on ? '✓' : ''}</span>
      <span class="toggle-text"><span class="label">${esc(c.label)}</span>${c.description ? `<span class="hint">${esc(c.description)}</span>` : ''}</span>
    </button>`).join('')}</div></div>`

/** Experts with two switches each. `rows`: { id, label, description, enabled, readOnly }. */
export const matrix = (field, columns, rows) => `
  <div class="matrix">
    <div class="matrix-head"><span></span><span>${esc(columns[0])}</span><span>${esc(columns[1])}</span></div>
    ${rows.map((r) => `
      <div class="matrix-row">
        <span class="toggle-text"><span class="label">${esc(r.label)}</span><span class="hint">${esc(r.description)}</span></span>
        <button type="button" class="track${r.enabled ? ' on' : ''}" role="switch" aria-checked="${r.enabled}" data-nav data-action="expert-on" data-field="${esc(field)}" data-value="${esc(r.id)}" ${key('expert-on', field, r.id)}><span class="knob"></span></button>
        <button type="button" class="track${r.readOnly ? ' on' : ''}" role="switch" aria-checked="${r.readOnly}" data-nav data-action="expert-ro" data-field="${esc(field)}" data-value="${esc(r.id)}" ${key('expert-ro', field, r.id)} ${r.enabled ? '' : 'disabled'}><span class="knob"></span></button>
      </div>`).join('')}
  </div>`

/** A button with a line of state text next to it (the sign-in buttons). */
export const actionButton = (action, text, state, o = {}) => `
  <div class="group action-row">
    <button type="button" class="primary" data-nav data-action="${esc(action)}" ${key(action, '')} ${o.disabled ? 'disabled' : ''}>${esc(text)}</button>
    <span class="state ${o.tone ?? ''}" role="status">${esc(state)}</span>
  </div>`

/** The review list. `rows`: { id, label, value, editLabel }. */
export const summary = (rows) => `
  <div class="summary">${rows.map((r) => `
    <div class="summary-row">
      <span class="summary-label">${esc(r.label)}</span>
      <span class="summary-value">${esc(r.value)}</span>
      <button type="button" class="link" data-nav data-action="edit" data-value="${esc(r.id)}" ${key('edit', r.id)}>${esc(r.editLabel)}</button>
    </div>`).join('')}</div>`

/** A field that records a key combination: the keys as `<kbd>`s (`keys`, e.g. ['Super', 'A']) and `meta` beside them,
 * or `recordingText` while it listens; a line of state under it (`note`, `tone` as the action rows' states). */
export const keyCapture = (field, text, description, keys, o = {}) => `
  <div class="group">
    ${label(text, description)}
    <button type="button" class="key-capture${o.recording ? ' recording' : ''}" data-nav data-action="key-capture" data-field="${esc(field)}" ${key('key-capture', field)} aria-pressed="${!!o.recording}">
      ${o.recording ? `<span class="hint">${esc(o.recordingText)}</span>` : `${keys.map((k) => `<kbd>${esc(k)}</kbd>`).join(' ')}${o.meta ? `<span class="hint">${esc(o.meta)}</span>` : ''}`}
    </button>
    ${o.note ? `<span class="state ${o.tone ?? ''}" role="status">${esc(o.note)}</span>` : ''}
  </div>`
