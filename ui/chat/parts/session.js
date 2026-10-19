// One subagent's session: what it read, said, edited and is doing right now. Port of
// design/components/agent-session.dc.html (its "Trabajando" line is drawn here too, the same
// nine dots and shimmer as design/components/working.dc.html). The message box under it
// ("Message <role>…") belongs to the view. Pure: props and ui in, escaped HTML out.
//
// Contract:
//   props = {
//     agent: {
//       role:   string,                      // 'frontend' (mono title)
//       task:   string,                      // the brief, one sentence
//       status: string,                      // free text, top right: 'writing', 'done', 'queued'
//       kind:   'active' | 'done' | 'queued',// dot and status colour; anything else is 'queued'
//       steps:  Array<Step>,                 // oldest first; unknown step kinds are skipped
//     } | null,
//   }
//   Step =
//     { kind: 'tool', icon?: 'read' | 'edit' | 'run' | 'search',   // unknown icon: none
//       verb: string, target: string, meta?: string, metaColor?: '#rrggbb' | 'rgb(...)' }  // a tool call row
//     { kind: 'msg',  text: string }                                // what the agent said
//     { kind: 'code', lines: Array<{ kind: '+' | '-' | ' ', text: string }> }  // a diff; unknown line kind is context
//     { kind: 'live', text: string, since: number }                 // "working on…" with elapsed time;
//                                                                   // `since` is the epoch ms it started (the design's demo
//                                                                   // passes elapsed seconds; absolute keeps the renderer pure)
//     { kind: 'note', label: string, text: string }                 // a mono aside: "<label> text"
//   ui = { now?: number }                    // epoch ms, for the live step's elapsed time (default: 0 elapsed)
//
// Emits:
//   data-chat="agent-back"                   back to the subagents list

import { esc } from '../../onboarding/components.js'

export const TEXTS = {
  en: {
    'chat.session.back': 'Subagents',
    'chat.session.activity': 'Session activity',
    'chat.session.code': 'Code change',
    'chat.session.empty': 'Nothing yet',
    'chat.session.missing': 'This session is not available',
  },
  es: {
    'chat.session.back': 'Subagentes',
    'chat.session.activity': 'Actividad de la sesión',
    'chat.session.code': 'Cambio de código',
    'chat.session.empty': 'Todavía nada',
    'chat.session.missing': 'Esta sesión no está disponible',
  },
}

const KINDS = ['active', 'done', 'queued']
const ICONS = {
  read: '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"></path><path d="M14 3v5h5M9 13h6M9 17h6"></path></svg>',
  edit: '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20h4L19 9l-4-4L4 16z"></path><path d="M13 7l4 4"></path></svg>',
  run: '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 7l5 5-5 5M12 17h7"></path></svg>',
  search: '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="6"></circle><path d="M20 20l-4.5-4.5"></path></svg>',
}
const BACK = '<svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 6l-6 6 6 6"></path></svg>'
const LINE_CLASS = { '-': 'line-del', '+': 'line-add' }
const CELLS = Array.from({ length: 9 }, (_, i) => `<span class="p-session-cell" style="animation-delay:${((i % 3) + Math.abs(Math.floor(i / 3) - 1)) * 90}ms"></span>`).join('')

/** "5.4s", "1m 14.3s" -- the design's Trabajando timer. */
export function formatElapsed(ms) {
  const total = Math.max(0, Math.floor((Number(ms) || 0) / 100)) / 10
  return total < 60 ? `${total.toFixed(1)}s` : `${Math.floor(total / 60)}m ${(total % 60).toFixed(1)}s`
}

/** Only a plain hex or rgb()/rgba() colour reaches a style attribute. */
const safeColor = (value) => (/^(#[0-9a-f]{3,8}|rgba?\(\s*[\d.,\s%]+\))$/i.test(String(value ?? '')) ? String(value) : '')

const capitalize = (text) => String(text ?? '').charAt(0).toUpperCase() + String(text ?? '').slice(1)

function step(s, ui, t) {
  if (s.kind === 'tool') {
    const icon = ICONS[s.icon]
    const color = safeColor(s.metaColor)
    return `<li class="p-session-step"><div class="p-session-tool">
      <span class="p-session-icon${icon ? ` icon-${esc(s.icon)}` : ''}">${icon ?? ''}</span>
      <span class="p-session-call"><span class="p-session-verb">${esc(s.verb)}</span> ${esc(s.target)}</span>
      <span class="p-session-meta"${color ? ` style="color:${color}"` : ''}>${esc(s.meta)}</span>
    </div></li>`
  }
  if (s.kind === 'msg') return `<li class="p-session-step"><p class="p-session-msg">${esc(s.text)}</p></li>`
  if (s.kind === 'code') {
    const lines = (Array.isArray(s.lines) ? s.lines : []).filter((l) => l && typeof l === 'object')
    const body = lines
      .map((l) => `<div class="p-session-line ${LINE_CLASS[l.kind] ?? 'line-ctx'}"><span class="p-session-mark">${l.kind === '+' || l.kind === '-' ? l.kind : ''}</span><span class="p-session-code-text">${esc(l.text)}</span></div>`)
      .join('')
    return `<li class="p-session-step"><div class="p-session-code" role="group" aria-label="${esc(t('chat.session.code'))}">${body}</div></li>`
  }
  if (s.kind === 'live') {
    const elapsed = Number.isFinite(Number(ui?.now)) && Number.isFinite(Number(s.since)) ? Number(ui.now) - Number(s.since) : 0
    return `<li class="p-session-step"><div class="p-session-live"><span class="p-session-cells" aria-hidden="true">${CELLS}</span><span class="p-session-shimmer" role="status">${esc(capitalize(s.text))}</span><span class="p-session-timer" role="timer" aria-live="off">${esc(formatElapsed(elapsed))}</span></div></li>`
  }
  if (s.kind === 'note') return `<li class="p-session-step"><p class="p-session-note"><span class="p-session-note-label">${esc(s.label)}</span> ${esc(s.text)}</p></li>`
  return ''
}

export function renderAgentSession(props, ui, t) {
  const agent = props?.agent && typeof props.agent === 'object' ? props.agent : null
  const back = `<button type="button" class="p-session-back" data-chat="agent-back">${BACK}${esc(t('chat.session.back'))}</button>`
  if (!agent) return `<div class="p-session"><div class="p-session-top">${back}</div><p class="p-session-empty">${esc(t('chat.session.missing'))}</p></div>`
  const kind = KINDS.includes(agent.kind) ? agent.kind : 'queued'
  const steps = (Array.isArray(agent.steps) ? agent.steps : []).filter((s) => s && typeof s === 'object').map((s) => step(s, ui, t)).join('')
  const log = steps
    ? `<ol class="p-session-steps" aria-label="${esc(t('chat.session.activity'))}">${steps}</ol>`
    : `<p class="p-session-empty">${esc(t('chat.session.empty'))}</p>`
  return `<div class="p-session kind-${kind}">
  <div class="p-session-top">${back}<span class="p-session-status"><span class="p-session-dot" aria-hidden="true"></span>${esc(agent.status)}</span></div>
  <div class="p-session-head"><h2 class="p-session-role">${esc(agent.role)}</h2><p class="p-session-task">${esc(agent.task)}</p></div>
  <div class="p-session-rule" aria-hidden="true"></div>
  ${log}
</div>`
}
