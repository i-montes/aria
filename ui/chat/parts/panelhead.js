// The head of a full panel (the memory panel, the settings): a back button, the title, an optional mono line beside it
// and an optional figure on the right. Taken from design/components/memory-panel.dc.html so every panel opens the
// same way. Pure: props in, escaped HTML out.
//
// Contract:
//   props = {
//     back:   { attrs: string, label: string },   // attrs: the button's trusted data attributes, e.g. 'data-chat="memory-back"'
//     title:  string,
//     meta?:  string,                             // the mono line beside the title ('engram · aria', '~/.aria/settings.json')
//     aside?: string,                             // the mono figure on the right ('12 active')
//   }

import { esc } from '../../onboarding/components.js'

const BACK = '<svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 6l-6 6 6 6"></path></svg>'

export function renderPanelHead({ back, title, meta = '', aside = '' }) {
  return `<div class="p-head">
    <button type="button" class="p-head-back" ${back.attrs} aria-label="${esc(back.label)}" title="${esc(back.label)}">${BACK}</button>
    <div class="p-head-heading"><h2 class="p-head-h">${esc(title)}</h2>${meta ? `<span class="p-head-meta">${esc(meta)}</span>` : ''}</div>
    ${aside ? `<span class="p-head-aside">${esc(aside)}</span>` : ''}
  </div>`
}
