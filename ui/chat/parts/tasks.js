// The task dock (above the composer) and its inline log variant. Port of
// design/components/task-dock.dc.html. Pure: props and ui in, escaped HTML out.
//
// Contract:
//   props = {
//     tasks: Array<{
//       id?:     string,                       // what `task-select` reports; defaults to `role`
//       label:   string,                       // the task (or, for agents, what the agent does)
//       status:  'todo' | 'active' | 'done',   // anything else is treated as 'todo'
//       note?:   string,                       // tasks only: shown under the label while 'active'
//       role?:   string,                       // agents only: the primary line (mono), e.g. 'backend'
//     }>,
//     kind?:    'tasks' | 'agents',            // default 'tasks'; anything else is 'tasks'
//     inline?:  boolean,                       // true: the record in the chat log (toggle on top, 14px radius all round)
//     partial?: number,                        // 0..1, how far the active task is (tasks) -- feeds the bar
//     show?:    boolean,                       // false keeps the dock hidden (collapsed) without dropping it
//   }
//   ui = {
//     open?:       boolean,                    // list expanded (default true)
//     pulse?:      boolean,                    // true for ~700 ms after a task turned done (bar flash, count pop)
//     selectable?: boolean,                    // agents rows are buttons (default true for kind 'agents')
//     animate?:    boolean,                    // false: no entrance animations (re-rendering a log)
//   }
//   An empty or missing `tasks` list renders ''.
//
// Emits:
//   data-chat="tasks-toggle"                            expand / collapse the list
//   data-chat="task-select" data-id="<id or role>"      an agent row (kind 'agents' only)

import { esc } from '../../onboarding/components.js'

export const TEXTS = {
  en: {
    'chat.tasks.kind.tasks': 'Tasks',
    'chat.tasks.kind.agents': 'Agents',
    'chat.tasks.status.done': 'done',
    'chat.tasks.status.working': 'working',
    'chat.tasks.status.queued': 'queued',
    'chat.tasks.done': 'Done',
    'chat.tasks.summary.tasks': 'Tasks: {done} of {total} done',
    'chat.tasks.summary.agents': 'Agents: {done} of {total} done',
    'chat.tasks.workingOn': ', working on {name}',
    'chat.tasks.openSession': '{role}: {label}, {status}. Open its session',
  },
  es: {
    'chat.tasks.kind.tasks': 'Tareas',
    'chat.tasks.kind.agents': 'Agentes',
    'chat.tasks.status.done': 'listo',
    'chat.tasks.status.working': 'trabajando',
    'chat.tasks.status.queued': 'en cola',
    'chat.tasks.done': 'Listo',
    'chat.tasks.summary.tasks': 'Tareas: {done} de {total} listas',
    'chat.tasks.summary.agents': 'Agentes: {done} de {total} listos',
    'chat.tasks.workingOn': ', trabajando en {name}',
    'chat.tasks.openSession': '{role}: {label}, {status}. Abrir su sesión',
  },
}

const STATUSES = ['todo', 'active', 'done']
const STATUS_KEY = { done: 'done', active: 'working', todo: 'queued' }
const CHECK = '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"></path></svg>'
const CHEVRON = '<svg class="p-tasks-chev__FLIP__" aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#B4B4BC" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 15l6-6 6 6"></path></svg>'

export const normalizeStatus = (status) => (STATUSES.includes(status) ? status : 'todo')

const itemsOf = (tasks) => (Array.isArray(tasks) ? tasks.filter((x) => x && typeof x === 'object') : [])
const clamp01 = (n) => Math.min(1, Math.max(0, Number(n) || 0))

/** Totals and the bar's fill. Agents count half a step per working agent, tasks give the active one `partial`. */
export function taskStats(tasks, kind = 'tasks', partial = 0) {
  const list = itemsOf(tasks)
  const total = list.length
  const status = list.map((x) => normalizeStatus(x.status))
  const done = status.filter((s) => s === 'done').length
  const active = status.filter((s) => s === 'active').length
  const complete = total > 0 && done === total
  const share = complete ? 0 : kind === 'agents' ? active * 0.5 : active > 0 ? clamp01(partial) : 0
  const percent = total ? Math.round(((done + share) / total) * 100) : 0
  const activeTask = list.find((x) => normalizeStatus(x.status) === 'active') ?? null
  return { total, done, active, complete, percent, activeTask }
}

function row(task, i, { kind, open, selectable }, t) {
  const status = normalizeStatus(task.status)
  const agents = kind === 'agents'
  const primary = agents ? task.role : task.label
  const second = agents ? task.label : status === 'active' ? task.note : ''
  const statusText = agents ? t(`chat.tasks.status.${STATUS_KEY[status]}`) : ''
  const icon =
    status === 'done' ? `<span class="p-tasks-check">${CHECK}</span>` : status === 'active' ? '<span class="p-tasks-spin"></span>' : '<span class="p-tasks-ring"></span>'
  const inner = `
    <span class="p-tasks-icon" aria-hidden="true">${icon}</span>
    <span class="p-tasks-text"><span class="p-tasks-primary">${esc(primary)}</span>${second ? `<span class="p-tasks-second">${esc(second)}</span>` : ''}</span>
    <span class="p-tasks-status">${esc(statusText)}</span>`
  const cls = `p-tasks-row status-${status}`
  const body =
    agents && selectable
      ? `<button type="button" class="${cls}" data-chat="task-select" data-id="${esc(task.id ?? task.role)}" aria-label="${esc(t('chat.tasks.openSession', { role: task.role ?? task.label ?? '', label: task.label ?? '', status: statusText }))}">${inner}</button>`
      : `<div class="${cls}">${inner}</div>`
  const delay = open ? 40 + i * 35 : 0
  return `<li class="p-tasks-item" aria-current="${status === 'active' ? 'step' : 'false'}" style="--p-tasks-delay:${delay}ms">${body}</li>`
}

export function renderTasks(props, ui, t) {
  const list = itemsOf(props?.tasks)
  if (!list.length) return ''
  const kind = props?.kind === 'agents' ? 'agents' : 'tasks'
  const inline = props?.inline === true
  const show = props?.show !== false
  const open = (ui?.open ?? true) !== false && show
  const selectable = ui?.selectable ?? true
  const { total, done, complete, percent, activeTask } = taskStats(list, kind, props?.partial)
  const working = !!activeTask && !complete
  const root = ['p-tasks', inline && 'is-inline', open && 'is-open', !show && 'is-hidden', ui?.pulse && 'is-pulse', ui?.animate === false && 'is-static', kind === 'agents' && 'kind-agents']
    .filter(Boolean)
    .join(' ')
  const kindLabel = t(`chat.tasks.kind.${kind}`)
  const workingOn = activeTask ? t('chat.tasks.workingOn', { name: kind === 'agents' ? (activeTask.role ?? activeTask.label ?? '') : (activeTask.label ?? '') }) : ''
  const aria = t(`chat.tasks.summary.${kind}`, { done, total }) + workingOn
  const chevron = CHEVRON.replace('__FLIP__', open !== inline ? ' chev-flip' : '')
  return `<div class="${root}">
  <div class="p-tasks-clip"><div class="p-tasks-align"><div class="p-tasks-dock">
    <div class="p-tasks-list-wrap"><div class="p-tasks-list-clip">
      <ul class="p-tasks-list" aria-label="${esc(kindLabel)}">${list.map((task, i) => row(task, i, { kind, open, selectable }, t)).join('')}</ul>
    </div></div>
    <button type="button" class="p-tasks-toggle" data-chat="tasks-toggle" aria-expanded="${open}" aria-label="${esc(aria)}">
      ${chevron}
      <span class="p-tasks-kind">${esc(kindLabel)}</span>
      <span class="p-tasks-bar" aria-hidden="true"><span class="p-tasks-fill${complete ? ' is-complete' : ''}${working ? ' is-live' : ''}" style="width:${percent}%"></span></span>
      <span class="p-tasks-count${complete ? ' is-complete' : ''}">${esc(complete ? t('chat.tasks.done') : `${done}/${total}`)}</span>
    </button>
  </div></div></div>
</div>`
}
