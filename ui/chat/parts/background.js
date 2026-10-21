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
// Helpers: `stripJobs(jobs, session)` (what the strip shows), `hasLiveWork(jobs, session)` (the composer's hint),
// `jobMeta(job, now, t)`, `resultAgent(job, t)` (a finished subagent's answer as an agent session).
//
// The list (Alt+A, drawn inside the subagents view): `renderJobRows({ jobs, sessions, active, open, output, now }, ui, t)`
//   jobs      every session's jobs (dismissed ones too: they still belong under Finished)
//   sessions  the tabs, [{ id, title }]: a group's heading; `active` is the active tab's id (its group comes first)
//   open      the id of the job the list was opened on (a pill): that row is outlined (`is-target`)
//   output    { [job id]: string[] }: the expanded process rows and their last lines
// Sections "Running" (running + queued, grouped by session when more than one session has jobs) and "Finished"
// (newest first). A row reuses the subagents list's row (`p-agents-row`) and adds its actions (`p-bg-act`):
//   data-chat="bg-url"     data-id   open the job's URL in the browser
//   data-chat="bg-output"  data-id   expand or fold the row's output (aria-expanded)
//   data-chat="bg-session" data-id   the subagent's session (Session), or its answer once it ended (Result)
//   data-chat="bg-stop"    data-id   stop the job
// Every row carries data-job="<id>", so the view can scroll to it.
// Styles: background.css (`.p-bg-…`) on top of pill.css and subagents.css.
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
    'chat.bg.running': 'Running',
    'chat.bg.finished': 'Finished',
    'chat.bg.kind.process': 'process',
    'chat.bg.kind.agent': 'subagent',
    'chat.bg.open': 'Open',
    'chat.bg.output': 'Output',
    'chat.bg.session': 'Session',
    'chat.bg.result': 'Result',
    'chat.bg.noOutput': 'No output yet',
    'chat.bg.worktree': 'worktree kept at {path}',
    'chat.bg.end.exited': 'exited {exit}',
    'chat.bg.end.stopped': 'stopped',
    'chat.bg.end.done': 'done',
    'chat.bg.end.failed': 'failed',
    'chat.bg.end.cut': 'cut when Aria stopped',
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
    'chat.bg.running': 'En curso',
    'chat.bg.finished': 'Terminados',
    'chat.bg.kind.process': 'proceso',
    'chat.bg.kind.agent': 'subagente',
    'chat.bg.open': 'Abrir',
    'chat.bg.output': 'Ver salida',
    'chat.bg.session': 'Ver sesión',
    'chat.bg.result': 'Ver resultado',
    'chat.bg.noOutput': 'Todavía no hay salida',
    'chat.bg.worktree': 'worktree guardado en {path}',
    'chat.bg.end.exited': 'salió con {exit}',
    'chat.bg.end.stopped': 'detenido',
    'chat.bg.end.done': 'terminado',
    'chat.bg.end.failed': 'falló',
    'chat.bg.end.cut': 'cortado cuando Aria se detuvo',
  },
}

const MAX_PILLS = 4
const STOP = '<svg aria-hidden="true" width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="3"/></svg>'
const CROSS = '<svg aria-hidden="true" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>'

const shown = (j) => !j.dismissed && (j.state === 'running' || j.state === 'queued' || j.state === 'failed')

/** The jobs the strip shows: the session's running, queued and failed-not-dismissed ones, in the dock's order. */
export const stripJobs = (jobs, session) =>
  (Array.isArray(jobs) ? jobs : []).filter((j) => j && typeof j === 'object' && j.session === session && shown(j))

/** Whether the session has live background work (running, queued, failed and not dismissed): the composer's hint. */
export const hasLiveWork = (jobs, session) => stripJobs(jobs, session).length > 0

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

// ---- the list (Alt+A) ----------------------------------------------------------------------------

const ENDS = ['exited', 'stopped', 'done', 'failed', 'cut']
const isLive = (j) => j.state === 'running' || j.state === 'queued'

/** How a finished job ended: "exited 0", "stopped", "done", "failed", "cut when Aria stopped". */
export const endLabel = (job, t) =>
  ENDS.includes(job?.state) ? t(`chat.bg.end.${job.state}`, job.state === 'exited' ? { exit: job.exit ?? '?' } : undefined) : String(job?.state ?? '')

/** A subagent job shown in the agent-session view when no watch agent is linked to it: its final answer. */
export function resultAgent(job, t) {
  const live = isLive(job ?? {})
  return {
    role: 'subagent',
    task: job?.title ?? '',
    status: live ? jobMeta(job, Date.now(), t) : endLabel(job, t),
    kind: live ? (job.state === 'queued' ? 'queued' : 'active') : 'done',
    steps: job?.result ? [{ kind: 'msg', text: job.result }] : [],
  }
}

const rowKind = (j) => (j.state === 'queued' ? 'queued' : isLive(j) ? 'active' : 'done')

function act(action, j, label, extra = '') {
  return `<button type="button" class="p-bg-act is-text" data-chat="${action}" data-id="${esc(j.id)}"${extra} aria-label="${esc(`${label}: ${j.title ?? ''}`)}">${esc(label)}</button>`
}

function row(j, props, t) {
  const live = isLive(j)
  const output = props.output && Object.hasOwn(props.output, j.id) ? props.output[j.id] : null
  const actions = []
  if (j.kind === 'agent') actions.push(act('bg-session', j, t(live ? 'chat.bg.session' : 'chat.bg.result')))
  else {
    if (live && j.url) actions.push(act('bg-url', j, t('chat.bg.open'), ` title="${esc(j.url)}"`))
    actions.push(act('bg-output', j, t('chat.bg.output'), ` aria-expanded="${output ? 'true' : 'false'}"`))
  }
  if (live) actions.push(act('bg-stop', j, t('chat.bg.stop')))
  const meta = live ? jobMeta(j, props.now, t) : endLabel(j, t)
  const lines = Array.isArray(output) ? output : []
  const block = output
    ? `<pre class="p-bg-output">${lines.length ? lines.map((l) => esc(l)).join('\n') : `<span class="p-bg-output-empty">${esc(t('chat.bg.noOutput'))}</span>`}</pre>`
    : ''
  const worktree = !live && j.worktree ? `<p class="p-bg-worktree">${esc(t('chat.bg.worktree', { path: j.worktree }))}</p>` : ''
  const target = props.open != null && j.id === props.open ? ' is-target' : ''
  return `<li class="p-bg-item${target}" data-job="${esc(j.id)}">
      <div class="p-agents-row p-bg-row kind-${rowKind(j)} tone-${esc(tone(j))}">
        <span class="p-agents-dot" aria-hidden="true"></span>
        <span class="p-agents-role">${esc(t(`chat.bg.kind.${j.kind === 'agent' ? 'agent' : 'process'}`))}</span>
        <span class="p-agents-task"${j.command ? ` title="${esc(j.command)}"` : ''}>${esc(j.title)}</span>
        <span class="p-agents-status">${esc(meta)}</span>
      </div>
      <div class="p-bg-actions">${actions.join('')}</div>${worktree}${block}
    </li>`
}

const list = (jobs, props, t, label) => `<ul class="p-agents-list p-bg-list" aria-label="${esc(label)}">${jobs.map((j) => row(j, props, t)).join('')}</ul>`

export function renderJobRows(props, ui, t) {
  const jobs = (Array.isArray(props?.jobs) ? props.jobs : []).filter((j) => j && typeof j === 'object')
  if (!jobs.length) return ''
  const now = Number.isFinite(Number(props?.now)) && props?.now != null ? Number(props.now) : Date.now()
  const p = { ...props, now }
  const sessions = Array.isArray(props?.sessions) ? props.sessions : []
  const titleOf = (id) => sessions.find((s) => s?.id === id)?.title || String(id ?? '')
  const ids = [...new Set(jobs.map((j) => j.session))]
  const grouped = ids.length > 1
  // the active session first, then the tabs in their order, then sessions without a tab
  const order = [props?.active, ...sessions.map((s) => s?.id), ...ids].filter((id, i, all) => ids.includes(id) && all.indexOf(id) === i)
  const running = jobs.filter(isLive)
  const finished = jobs.filter((j) => !isLive(j)).sort((a, b) => Number(b.endedAt ?? 0) - Number(a.endedAt ?? 0))
  const parts = []
  if (running.length) {
    parts.push(`<h3 class="p-agents-section">${esc(t('chat.bg.running'))}</h3>`)
    for (const id of order) {
      const mine = running.filter((j) => j.session === id)
      if (!mine.length) continue
      const title = titleOf(id)
      if (grouped) parts.push(`<h4 class="p-agents-section is-session">${esc(title)}</h4>`)
      parts.push(list(mine, p, t, grouped ? title : t('chat.bg.running')))
    }
  }
  if (finished.length) parts.push(`<h3 class="p-agents-section">${esc(t('chat.bg.finished'))}</h3>`, list(finished, p, t, t('chat.bg.finished')))
  return `<section class="p-bg-rows">${parts.join('')}</section>`
}
