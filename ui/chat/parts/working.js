// The "Aria is working" line (design/components/working.dc.html): a 3x3 pixel pulse, a shimmering
// phase label and an elapsed timer.
//
// Props: { phase?, label?, startedAt? }
//   phase      one of PHASES (translated); any other string is shown capitalised
//   label      optional custom text that wins over the phase (e.g. "Running ls")
//   startedAt  epoch ms when the run started; elapsed = ui.now - startedAt (0 when missing)
// ui: { now }  epoch ms. The timer span carries data-since="<startedAt>" so the integrator can tick it
//              without a full re-render.
// Emits: nothing.
import { esc } from '../../onboarding/components.js'

export const PHASES = [
  'thinking',
  'working',
  'searching memory',
  'searching the web',
  'checking Notion',
  'reading files',
  'writing files',
  'running command',
  'delegating',
  'planning',
  'writing the spec',
  'splitting into phases',
  'compacting context',
]

/** "3.4s", or "2m 5.4s" from a minute on. */
export const formatElapsed = (ms) => {
  const total = Math.max(0, Math.floor((Number.isFinite(ms) ? ms : 0) / 100)) / 10
  return total < 60 ? `${total.toFixed(1)}s` : `${Math.floor(total / 60)}m ${(total % 60).toFixed(1)}s`
}

const capitalise = (s) => s.charAt(0).toUpperCase() + s.slice(1)

/** The text shown for a phase: a custom label wins, a known phase is translated, anything else is capitalised as given. */
export const phaseLabel = (phase, label, t) => {
  const custom = String(label ?? '').trim()
  if (custom) return capitalise(custom)
  const name = String(phase ?? 'working')
  return PHASES.includes(name) ? t(`chat.working.phase.${name.replace(/ /g, '_')}`) : capitalise(name)
}

// Cell (row, col) of the 3x3 grid pulses after (col + |row - 1|) * 90ms: a wave from the left edge.
const CELLS = Array.from({ length: 9 }, (_, i) => `<span class="p-working-cell" style="animation-delay: ${((i % 3) + Math.abs(Math.floor(i / 3) - 1)) * 90}ms"></span>`).join('')

export const renderWorking = ({ phase, label, startedAt }, ui, t) => {
  const since = Number.isFinite(startedAt) ? startedAt : (ui?.now ?? 0)
  return `<div class="p-working"><span class="p-working-grid" aria-hidden="true">${CELLS}</span><span class="p-working-label" role="status">${esc(phaseLabel(phase, label, t))}</span><span class="p-working-time" role="timer" aria-live="off" data-since="${esc(since)}">${esc(formatElapsed((ui?.now ?? since) - since))}</span></div>`
}

export const TEXTS = {
  en: {
    'chat.working.phase.thinking': 'Thinking',
    'chat.working.phase.working': 'Working',
    'chat.working.phase.searching_memory': 'Searching memory',
    'chat.working.phase.searching_the_web': 'Searching the web',
    'chat.working.phase.checking_Notion': 'Checking Notion',
    'chat.working.phase.reading_files': 'Reading files',
    'chat.working.phase.writing_files': 'Writing files',
    'chat.working.phase.running_command': 'Running command',
    'chat.working.phase.delegating': 'Delegating',
    'chat.working.phase.planning': 'Planning',
    'chat.working.phase.writing_the_spec': 'Writing the spec',
    'chat.working.phase.splitting_into_phases': 'Splitting into phases',
    'chat.working.phase.compacting_context': 'Compacting context',
  },
  es: {
    'chat.working.phase.thinking': 'Pensando',
    'chat.working.phase.working': 'Trabajando',
    'chat.working.phase.searching_memory': 'Buscando en la memoria',
    'chat.working.phase.searching_the_web': 'Buscando en la web',
    'chat.working.phase.checking_Notion': 'Revisando Notion',
    'chat.working.phase.reading_files': 'Leyendo archivos',
    'chat.working.phase.writing_files': 'Escribiendo archivos',
    'chat.working.phase.running_command': 'Ejecutando un comando',
    'chat.working.phase.delegating': 'Delegando',
    'chat.working.phase.planning': 'Planificando',
    'chat.working.phase.writing_the_spec': 'Escribiendo la especificación',
    'chat.working.phase.splitting_into_phases': 'Dividiendo en fases',
    'chat.working.phase.compacting_context': 'Compactando el contexto',
  },
}
