// The phases of a plan, as a card in the thread. Port of design/components/phases.dc.html.
// Pure: props and ui in, escaped HTML out.
//
// Contract:
//   props = {
//     phases: Array<{
//       name:  string,
//       tasks: string[],                     // the phase's task labels, in order
//     }>,
//   }
//   ui = { animate?: boolean }               // false: no entrance animations (re-rendering a log)
//   No phases renders the header with 0 / 0 and an empty line.
//
// Emits: nothing (the card is read-only).

import { esc } from '../../onboarding/components.js'

export const TEXTS = {
  en: {
    'chat.phases.title': 'PHASES',
    'chat.phases.phases1': '{n} phase',
    'chat.phases.phasesN': '{n} phases',
    'chat.phases.tasks1': '{n} task',
    'chat.phases.tasksN': '{n} tasks',
    'chat.phases.empty': 'No phases yet',
  },
  es: {
    'chat.phases.title': 'FASES',
    'chat.phases.phases1': '{n} fase',
    'chat.phases.phasesN': '{n} fases',
    'chat.phases.tasks1': '{n} tarea',
    'chat.phases.tasksN': '{n} tareas',
    'chat.phases.empty': 'Aún no hay fases',
  },
}

const LIST = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex: 0 0 14px"><path d="M4 6h16M4 12h10M4 18h6"></path></svg>'

const phasesOf = (phases) =>
  (Array.isArray(phases) ? phases : [])
    .filter((p) => p && typeof p === 'object')
    .map((p) => ({ name: p.name, tasks: Array.isArray(p.tasks) ? p.tasks.filter((x) => x !== null && x !== undefined) : [] }))

const count = (n, one, many, t) => t(n === 1 ? one : many, { n })

/** "3 phases · 5 tasks" */
export function phasesSummary(phases, t) {
  const list = phasesOf(phases)
  const total = list.reduce((n, p) => n + p.tasks.length, 0)
  return `${count(list.length, 'chat.phases.phases1', 'chat.phases.phasesN', t)} · ${count(total, 'chat.phases.tasks1', 'chat.phases.tasksN', t)}`
}

function phase(p, i, t) {
  return `<li class="p-phases-phase" style="--p-phases-delay:${90 + i * 90}ms">
    <span class="p-phases-num" aria-hidden="true">${i + 1}</span>
    <div class="p-phases-body">
      <div class="p-phases-line"><span class="p-phases-name">${esc(p.name)}</span><span class="p-phases-count">${esc(count(p.tasks.length, 'chat.phases.tasks1', 'chat.phases.tasksN', t))}</span></div>
      <ul class="p-phases-tasks">${p.tasks.map((x) => `<li class="p-phases-task"><span class="p-phases-dot" aria-hidden="true"></span><span class="p-phases-task-text">${esc(x)}</span></li>`).join('')}</ul>
    </div>
  </li>`
}

export function renderPhases(props, ui, t) {
  const list = phasesOf(props?.phases)
  const root = ['p-phases', ui?.animate === false && 'is-static'].filter(Boolean).join(' ')
  const body = list.length
    ? `<ol class="p-phases-list">${list.map((p, i) => phase(p, i, t)).join('')}</ol>`
    : `<p class="p-phases-empty">${esc(t('chat.phases.empty'))}</p>`
  return `<div class="${root}">
  <div class="p-phases-card">
    <div class="p-phases-head"><span class="p-phases-label">${LIST}${esc(t('chat.phases.title'))}</span><span class="p-phases-summary">${esc(phasesSummary(list, t))}</span></div>
    ${body}
  </div>
</div>`
}
