// The scheduled tasks panel (`/schedules`, docs/superpowers/specs/2026-10-21-schedules-design.md): every task the dock
// runs by itself, with when it runs, its next run and how its last one went, to pause, resume or delete. Tasks are made
// and changed by asking Aria in the chat.
//
// Props: { tasks: [Task + { next }], now }  (aria-core `schedules::Task`, camelCase; `next` epoch ms or null)
// Emits:
//   data-chat="schedule-toggle"  data-schedule  data-enabled ("true" to resume, "false" to pause)
//   data-chat="schedule-delete"  data-schedule
//   data-chat="panel-back"
import { esc } from '../../onboarding/components.js'

const EVERY = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
const WEEKDAYS = ['mon', 'tue', 'wed', 'thu', 'fri']

export const TEXTS = {
  en: {
    'chat.schedules.title': 'SCHEDULED',
    'chat.schedules.empty': 'Nothing scheduled. Ask Aria, for example: "every day at 7:30 give me my Notion tasks".',
    'chat.schedules.everyDay': 'Every day at {at}',
    'chat.schedules.weekdays': 'Weekdays at {at}',
    'chat.schedules.days': '{days} at {at}',
    'chat.schedules.once': 'On {date} at {at}',
    'chat.schedules.next': 'Next: {when}',
    'chat.schedules.paused': 'Paused',
    'chat.schedules.awaiting': 'Waiting for your approval: it may write ({what})',
    'chat.schedules.approve': 'Approve',
    'chat.schedules.never': 'Has not run yet',
    'chat.schedules.ok': 'Last: {summary}',
    'chat.schedules.failed': 'Last run failed: {summary}',
    'chat.schedules.pause': 'Pause',
    'chat.schedules.resume': 'Resume',
    'chat.schedules.delete': 'Delete',
    'chat.schedules.back': 'Back to the chat',
    'chat.schedules.day.mon': 'Mon',
    'chat.schedules.day.tue': 'Tue',
    'chat.schedules.day.wed': 'Wed',
    'chat.schedules.day.thu': 'Thu',
    'chat.schedules.day.fri': 'Fri',
    'chat.schedules.day.sat': 'Sat',
    'chat.schedules.day.sun': 'Sun',
  },
  es: {
    'chat.schedules.title': 'PROGRAMADAS',
    'chat.schedules.empty': 'No hay nada programado. Pídeselo a Aria, por ejemplo: "todos los días a las 7:30 dame mis tareas de Notion".',
    'chat.schedules.everyDay': 'Todos los días a las {at}',
    'chat.schedules.weekdays': 'De lunes a viernes a las {at}',
    'chat.schedules.days': '{days} a las {at}',
    'chat.schedules.once': 'El {date} a las {at}',
    'chat.schedules.next': 'Próxima: {when}',
    'chat.schedules.paused': 'En pausa',
    'chat.schedules.awaiting': 'Espera tu aprobación: puede escribir ({what})',
    'chat.schedules.approve': 'Aprobar',
    'chat.schedules.never': 'Todavía no ha corrido',
    'chat.schedules.ok': 'Última: {summary}',
    'chat.schedules.failed': 'La última falló: {summary}',
    'chat.schedules.pause': 'Pausar',
    'chat.schedules.resume': 'Reanudar',
    'chat.schedules.delete': 'Borrar',
    'chat.schedules.back': 'Volver al chat',
    'chat.schedules.day.mon': 'Lun',
    'chat.schedules.day.tue': 'Mar',
    'chat.schedules.day.wed': 'Mié',
    'chat.schedules.day.thu': 'Jue',
    'chat.schedules.day.fri': 'Vie',
    'chat.schedules.day.sat': 'Sáb',
    'chat.schedules.day.sun': 'Dom',
  },
}

/** When a task runs, in the person's words. */
export function whenLabel(when, t) {
  if (when.date) return t('chat.schedules.once', { at: when.at, date: when.date })
  const days = when.days ?? []
  if (EVERY.every((d) => days.includes(d))) return t('chat.schedules.everyDay', { at: when.at })
  if (days.length === 5 && WEEKDAYS.every((d) => days.includes(d))) return t('chat.schedules.weekdays', { at: when.at })
  const named = EVERY.filter((d) => days.includes(d)).map((d) => t(`chat.schedules.day.${d}`))
  return t('chat.schedules.days', { at: when.at, days: named.join(', ') })
}

/** "today 07:30", "tomorrow 07:30" or the date, local. */
function nextLabel(next, now) {
  const at = new Date(next)
  const time = `${String(at.getHours()).padStart(2, '0')}:${String(at.getMinutes()).padStart(2, '0')}`
  return `${at.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })} ${time}`
}

function row(task, now, t) {
  // What it may write: the connectors allowed and the folders, for the person to approve knowingly.
  const writes = [
    ...Object.entries(task.grants?.connectors ?? {}).filter(([, level]) => level === 'allow').map(([name]) => name),
    ...(task.grants?.writable ?? []),
  ].join(', ')
  const status = task.awaiting
    ? t('chat.schedules.awaiting', { what: writes })
    : !task.enabled
    ? t('chat.schedules.paused')
    : typeof task.next === 'number'
      ? t('chat.schedules.next', { when: nextLabel(task.next, now) })
      : ''
  const last = task.lastResult
    ? t(task.lastResult.ok ? 'chat.schedules.ok' : 'chat.schedules.failed', { summary: task.lastResult.summary })
    : t('chat.schedules.never')
  const resume = !task.enabled
  return `<li class="p-sched${task.enabled ? '' : ' is-paused'}${task.lastResult && !task.lastResult.ok ? ' is-failed' : ''}">
    <div class="p-sched-head"><span class="p-sched-title">${esc(task.title)}</span><span class="p-sched-when">${esc(whenLabel(task.when, t))}</span></div>
    <div class="p-sched-meta">${status ? `<span>${esc(status)}</span>` : ''}<span class="p-sched-last">${esc(last)}</span></div>
    <div class="p-sched-actions">
      <button type="button" class="p-sched-btn${task.awaiting ? ' is-approve' : ''}" data-chat="schedule-toggle" data-schedule="${esc(task.id)}" data-enabled="${resume}">${esc(t(task.awaiting ? 'chat.schedules.approve' : resume ? 'chat.schedules.resume' : 'chat.schedules.pause'))}</button>
      <button type="button" class="p-sched-btn is-danger" data-chat="schedule-delete" data-schedule="${esc(task.id)}">${esc(t('chat.schedules.delete'))}</button>
    </div>
  </li>`
}

export function renderSchedules({ tasks = [], now = Date.now() } = {}, ui = {}, t = (k) => k) {
  const body = tasks.length
    ? `<ul class="p-sched-list">${tasks.map((task) => row(task, now, t)).join('')}</ul>`
    : `<p class="chat-panel-empty">${esc(t('chat.schedules.empty'))}</p>`
  return `<div class="p-scheds"><div class="p-sched-heading">${esc(t('chat.schedules.title'))}</div>${body}</div>`
}
