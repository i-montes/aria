// The route line above an answer (design/components/jev.dc.html): "jev focused task · searches memory".
//
// Props: { mode, action?, accent? }
//   mode    the route chosen by the router; the four known modes (quick, focused task, work to split,
//           needs thinking) are translated, anything else is shown as given
//   action  what Jev does next ("searches memory"); omitted when empty
//   accent  optional hex color of the "jev" word (default #B9A8FF); anything else is ignored
// ui: unused. t(key) translates.
// Emits: nothing.
import { esc } from '../../onboarding/components.js'

const KNOWN = ['quick', 'focused_task', 'work_to_split', 'needs_thinking']

/** The text key of a known mode, or null. */
export const modeKey = (mode) => {
  const slug = String(mode ?? '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '_')
  return KNOWN.includes(slug) ? `chat.jev.mode.${slug}` : null
}

const color = (value) => (/^#[0-9a-fA-F]{3,8}$/.test(String(value ?? '')) ? value : '#B9A8FF')

export const renderJev = ({ mode, action, accent }, ui, t) => {
  const key = modeKey(mode)
  const action_ = action ? `<span aria-hidden="true">·</span><span>${esc(action)}</span>` : ''
  return `<div class="p-jev"><span style="color: ${esc(color(accent))}">jev</span><span>${esc(key ? t(key) : mode)}</span>${action_}</div>`
}

export const TEXTS = {
  en: {
    'chat.jev.mode.quick': 'quick',
    'chat.jev.mode.focused_task': 'focused task',
    'chat.jev.mode.work_to_split': 'work to split',
    'chat.jev.mode.needs_thinking': 'needs thinking',
  },
  es: {
    'chat.jev.mode.quick': 'rápido',
    'chat.jev.mode.focused_task': 'tarea concreta',
    'chat.jev.mode.work_to_split': 'trabajo por dividir',
    'chat.jev.mode.needs_thinking': 'requiere pensar',
  },
}
