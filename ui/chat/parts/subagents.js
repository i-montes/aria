// The subagents view (Alt+A): who is working on what. Port of design/components/subagents.dc.html,
// plus the header (title and summary line) that design/components/aria-on-the-desktop.dc.html puts
// above it. The "Alt A  Back to chat" button of that view belongs to the view, not to this part.
// Pure: props and ui in, escaped HTML out.
//
// Contract:
//   props = {
//     agents: Array<{
//       id?:    string,                      // what `agent-select` reports; defaults to `role`
//       role:   string,                      // 'researcher', 'frontend', ... (mono, first column)
//       task:   string,                      // what it is doing
//       status: string,                      // the free-text state shown on the right: 'writing', 'testing', 'done', 'queued'
//       kind:   'active' | 'done' | 'queued',// drives the dot and the status colour; anything else is 'queued'
//     }>,
//     footnote?: string,                     // default: texts 'chat.agents.footnote'
//   }
//   ui = {} (nothing; the part is stateless)
//
// Emits:
//   data-chat="agent-select" data-id="<id or role>"   open that agent's session

import { esc } from '../../onboarding/components.js'

export const TEXTS = {
  en: {
    'chat.agents.title': 'Subagents',
    'chat.agents.summary': '{running} running · {done} done · {queued} queued',
    'chat.agents.aria': 'Subagents at work',
    'chat.agents.open': 'Open the {role} session',
    'chat.agents.footnote': 'select an agent to open its session',
    'chat.agents.empty': 'No subagents are working',
  },
  es: {
    'chat.agents.title': 'Subagentes',
    'chat.agents.summary': '{running} trabajando · {done} listos · {queued} en cola',
    'chat.agents.aria': 'Subagentes trabajando',
    'chat.agents.open': 'Abrir la sesión de {role}',
    'chat.agents.footnote': 'elige un agente para abrir su sesión',
    'chat.agents.empty': 'No hay subagentes trabajando',
  },
}

const KINDS = ['active', 'done', 'queued']
const CHEVRON = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#8A8A94" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-top: 3px"><path d="M9 6l6 6-6 6"></path></svg>'

export const agentKind = (kind) => (KINDS.includes(kind) ? kind : 'queued')
const agentsOf = (agents) => (Array.isArray(agents) ? agents.filter((a) => a && typeof a === 'object') : [])

/** "2 running · 1 done · 1 queued" (the design's header line). */
export function agentsSummary(agents, t) {
  const kinds = agentsOf(agents).map((a) => agentKind(a.kind))
  const n = (kind) => kinds.filter((k) => k === kind).length
  return t('chat.agents.summary', { running: n('active'), done: n('done'), queued: n('queued') })
}

function item(a, t) {
  const kind = agentKind(a.kind)
  return `<li><button type="button" class="p-agents-row kind-${kind}" data-chat="agent-select" data-id="${esc(a.id ?? a.role)}" aria-label="${esc(t('chat.agents.open', { role: a.role ?? '' }))}">
      <span class="p-agents-dot" aria-hidden="true"></span>
      <span class="p-agents-role">${esc(a.role)}</span>
      <span class="p-agents-task">${esc(a.task)}</span>
      <span class="p-agents-status">${esc(a.status)}</span>
      ${CHEVRON}
    </button></li>`
}

export function renderSubagents(props, ui, t) {
  const agents = agentsOf(props?.agents)
  const footnote = props?.footnote ?? t('chat.agents.footnote')
  const rows = agents.map((a) => item(a, t)).join('')
  const list = agents.length
    ? `<ul class="p-agents-list" aria-label="${esc(t('chat.agents.aria'))}">${rows}</ul>`
    : `<p class="p-agents-empty">${esc(t('chat.agents.empty'))}</p>`
  return `<div class="p-agents">
  <div class="p-agents-head"><h2 class="p-agents-title">${esc(t('chat.agents.title'))}</h2><span class="p-agents-summary">${esc(agentsSummary(agents, t))}</span></div>
  ${list}
  <p class="p-agents-footnote">${esc(footnote)}</p>
</div>`
}
