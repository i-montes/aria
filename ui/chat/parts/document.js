// A spec or plan the assistant wrote, as a card in the thread. Port of
// design/components/document-card.dc.html. Pure: props and ui in, escaped HTML out.
//
// Contract:
//   props = {
//     doc?:    'spec' | 'plan',              // default 'spec'; anything else is 'spec'
//     status?: 'draft' | 'approved',         // default 'draft'; anything else is 'draft'
//     title:   string,
//     path:    string,                       // where the file lives, e.g. 'docs/specs/push-queue.md'
//     sections: Array<{
//       heading: string,                     // shown as given (the design's are uppercase: 'GOAL')
//       text?:   string,                     // a paragraph
//       items?:  string[],                   // a bullet list (after the paragraph if both are given)
//     }>,
//   }
//   ui = { animate?: boolean }               // false: no entrance animations (re-rendering a log)
//
// Emits: nothing (the card is read-only).

import { esc } from '../../onboarding/components.js'

export const TEXTS = {
  en: {
    'chat.doc.kind.spec': 'SPEC',
    'chat.doc.kind.plan': 'PLAN',
    'chat.doc.status.draft': 'Draft',
    'chat.doc.status.approved': 'Approved',
  },
  es: {
    'chat.doc.kind.spec': 'ESPECIFICACIÓN',
    'chat.doc.kind.plan': 'PLAN',
    'chat.doc.status.draft': 'Borrador',
    'chat.doc.status.approved': 'Aprobado',
  },
}

const FILE = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex: 0 0 14px"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><path d="M14 2v6h6M9 13h6M9 17h4"></path></svg>'
const CHECK = '<svg aria-hidden="true" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"></path></svg>'

function section(s, i) {
  const items = Array.isArray(s.items) ? s.items.filter((x) => x !== null && x !== undefined) : []
  const text = s.text ? `<p class="p-doc-text">${esc(s.text)}</p>` : ''
  const list = items.length
    ? `<ul class="p-doc-items">${items.map((it) => `<li class="p-doc-item"><span class="p-doc-dot" aria-hidden="true"></span><span class="p-doc-item-text">${esc(it)}</span></li>`).join('')}</ul>`
    : ''
  return `<div class="p-doc-section" style="--p-doc-delay:${80 + i * 70}ms"><span class="p-doc-heading">${esc(s.heading)}</span>${text}${list}</div>`
}

export function renderDocument(props, ui, t) {
  const kind = props?.doc === 'plan' ? 'plan' : 'spec'
  const approved = props?.status === 'approved'
  const sections = (Array.isArray(props?.sections) ? props.sections : []).filter((s) => s && typeof s === 'object')
  const root = ['p-doc', ui?.animate === false && 'is-static'].filter(Boolean).join(' ')
  return `<div class="${root}">
  <div class="p-doc-card">
    <div class="p-doc-top">
      <span class="p-doc-kind">${FILE}${esc(t(`chat.doc.kind.${kind}`))}</span>
      <span class="p-doc-chip ${approved ? 'is-approved' : 'is-draft'}">${approved ? CHECK : '<span class="p-doc-draft-dot" aria-hidden="true"></span>'}${esc(t(`chat.doc.status.${approved ? 'approved' : 'draft'}`))}</span>
    </div>
    <div class="p-doc-head"><h3 class="p-doc-title">${esc(props?.title)}</h3><span class="p-doc-path">${esc(props?.path)}</span></div>
    <div class="p-doc-sections">${sections.map(section).join('')}</div>
  </div>
</div>`
}
