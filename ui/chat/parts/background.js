// Background work (design: docs/superpowers/specs/2026-10-21-background-work-design.md, "The strip", owner's option B):
// a row of pills right above the composer, one per job of the active session that runs, waits or failed. Each pill is
// the shared pill part (pill.js): a dot, the mono title, a muted meta and a stop (■) button; a failed one has a
// dismiss (×) instead. At most four pills; a fifth job and more fold into a `+N` pill that opens the list.
// Pure: props in, escaped HTML out. The list itself (Alt+A) is drawn by the agents view.
//
// Props: { jobs, session, now? }
//   jobs     every session's jobs, as the dock sends them (`{"type":"jobs","jobs":[…]}`, crates/aria/src/jobs.rs):
//            { id, number, session, kind: 'process'|'agent', title, state: 'queued'|'running'|'exited'|'failed'|
//              'stopped'|'done'|'cut', startedAt, url, writes, dismissed, … }, plus `status` (optional: what the
//            subagent's watch says it is doing, 'writing', 'testing'…)
//   session  the active session's id: only its jobs show
//   now      epoch ms for the minutes (default Date.now())
// ui: unused. t(key, vars) translates.
// Emits:
//   data-chat="bg-open"    data-id="<job id>"   open the background list on that job
//   data-chat="bg-stop"    data-id="<job id>"   stop that job (a running or queued one)
//   data-chat="bg-dismiss" data-id="<job id>"   take a failed job off the strip
//   data-chat="bg-list"                         open the background list (the `+N` pill)
// Helpers: `stripJobs(jobs, session)` (what the strip shows), `jobMeta(job, now, t)`.
// Styles: background.css (`.p-bg-…`) on top of pill.css.
import { esc } from '../../onboarding/components.js'
import { renderPill } from './pill.js'
import { statusLabel } from './subagents.js'

export const TEXTS = {
  en: {
    'chat.bg.title': 'Background',
    'chat.bg.more': '+{n}',
    'chat.bg.stop': 'Stop',
    'chat.bg.dismiss': 'Dismiss',
    'chat.bg.queued': 'queued',
    'chat.bg.minutes': '{n} min',
    'chat.bg.note.exited': '{title} ended (exit {exit})',
    'chat.bg.note.failed': '{title} failed',
    'chat.bg.note.stopped': '{title} stopped',
    'chat.bg.note.done': '{title}: done',
    'chat.bg.note.conflict': '{title}: done, its changes could not be applied and are kept in a worktree',
    'chat.bg.note.cut': '{title}: cut when Aria stopped',
  },
  es: {
    'chat.bg.title': 'En segundo plano',
    'chat.bg.more': '+{n}',
    'chat.bg.stop': 'Detener',
    'chat.bg.dismiss': 'Descartar',
    'chat.bg.queued': 'en cola',
    'chat.bg.minutes': '{n} min',
    'chat.bg.note.exited': '{title} terminó (salida {exit})',
    'chat.bg.note.failed': '{title} falló',
    'chat.bg.note.stopped': '{title} se detuvo',
    'chat.bg.note.done': '{title}: terminado',
    'chat.bg.note.conflict': '{title}: terminado; sus cambios no se pudieron aplicar y quedan en un worktree',
    'chat.bg.note.cut': '{title}: cortado cuando Aria se detuvo',
  },
}

const MAX_PILLS = 4
const STOP = '<svg aria-hidden="true" width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="3"/></svg>'
const CROSS = '<svg aria-hidden="true" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>'

const shown = (j) => !j.dismissed && (j.state === 'running' || j.state === 'queued' || j.state === 'failed')

/** The jobs the strip shows: the session's running, queued and failed-not-dismissed ones, in the dock's order. */
export const stripJobs = (jobs, session) =>
  (Array.isArray(jobs) ? jobs : []).filter((j) => j && typeof j === 'object' && j.session === session && shown(j))

/** The port of a job's URL (`:3000`), or '' when it has none. */
function port(url) {
  try {
    const u = new URL(url)
    return `:${u.port || (u.protocol === 'https:' ? 443 : u.protocol === 'http:' ? 80 : '')}`.replace(/^:$/, '')
  } catch {
    return ''
  }
}

/** The muted meta of a job: a process `:3000 · 12 min`, a subagent what it is doing, a queued one `queued`. */
export function jobMeta(job, now = Date.now(), t) {
  if (job?.state === 'queued') return t('chat.bg.queued')
  if (job?.state === 'failed') return statusLabel('failed', t)
  if (job?.kind === 'agent') return statusLabel(job.status || (job.writes === false ? 'running' : 'writing'), t)
  const minutes = t('chat.bg.minutes', { n: Math.max(0, Math.floor((Number(now) - Number(job?.startedAt ?? now)) / 60_000)) })
  const p = job?.url ? port(job.url) : ''
  return p ? `${p} · ${minutes}` : minutes
}

const tone = (j) => (j.state === 'failed' ? 'failed' : j.state === 'queued' ? 'queued' : j.kind === 'agent' ? 'agent' : 'process')

function pill(j, now, t) {
  const failed = j.state === 'failed'
  const button = failed
    ? `<button type="button" class="p-bg-act" data-chat="bg-dismiss" data-id="${esc(j.id)}" aria-label="${esc(`${t('chat.bg.dismiss')}: ${j.title ?? ''}`)}" title="${esc(t('chat.bg.dismiss'))}">${CROSS}</button>`
    : `<button type="button" class="p-bg-act" data-chat="bg-stop" data-id="${esc(j.id)}" aria-label="${esc(`${t('chat.bg.stop')}: ${j.title ?? ''}`)}" title="${esc(t('chat.bg.stop'))}">${STOP}</button>`
  return renderPill({ label: j.title ?? '', meta: jobMeta(j, now, t), dot: true, tone: tone(j), action: 'bg-open', id: j.id, extra: button }, {}, t)
}

export function renderStrip(props, ui, t) {
  const jobs = stripJobs(props?.jobs, props?.session)
  if (!jobs.length) return ''
  const now = Number.isFinite(Number(props?.now)) && props?.now !== null && props?.now !== undefined ? Number(props.now) : Date.now()
  const pills = jobs.slice(0, MAX_PILLS).map((j) => pill(j, now, t))
  if (jobs.length > MAX_PILLS) pills.push(renderPill({ label: t('chat.bg.more', { n: jobs.length - MAX_PILLS }), tone: 'more', action: 'bg-list', title: t('chat.bg.title') }, {}, t))
  return `<div class="p-bg-strip" role="group" aria-label="${esc(t('chat.bg.title'))}">${pills.join('')}</div>`
}
