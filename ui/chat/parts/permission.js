// The permission card (design/components/sandbox-permission.dc.html): the agent wants something
// outside its sandbox (a command, a file write) and asks for a decision. Fed by a dialog title
// `{"aria":1,"kind":"sandbox"|"write",…}` (see cards.js).
//
// Props: { id, kind, summary?, detail?, jev? }
//   id       the extension dialog's id; answers go back to it
//   kind     "sandbox" (default) | "write": picks the icon and the default title
//   summary  the heading; the kind's default title when empty
//   detail   the command (sandbox) or the path (write), in the dark code box
//   jev      Jev's review: a string, or an object with `review` or `text`
// ui: unused. t(key, vars) translates.
// Emits: data-chat="permission" data-id data-choice="allow|always|deny"
import { esc } from '../../onboarding/components.js'

const lock = '<svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="10" width="16" height="11" rx="2"></rect><path d="M8 10V7a4 4 0 0 1 8 0v3"></path></svg>'
const file = '<svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z"></path><path d="M14 3v5h5"></path></svg>'

const text = (value) => (typeof value === 'string' ? value : '')
const review = (jev) => (typeof jev === 'string' ? jev : text(jev?.review) || text(jev?.text))

export function renderPermission({ id, kind, summary, detail, jev } = {}, ui, t) {
  const write = kind === 'write'
  const title = text(summary) || t(write ? 'chat.perm.write' : 'chat.perm.sandbox')
  const command = text(detail)
  const note = review(jev)
  const choice = (name, label, cls) =>
    `<button type="button" class="p-perm-btn ${cls}" data-chat="permission" data-id="${esc(id)}" data-choice="${name}">${esc(label)}</button>`
  return `<div class="p-perm" role="group" aria-label="${esc(title)}">
  <div class="p-perm-head">${write ? file : lock}<span>${esc(title)}</span></div>
  ${command ? `<div class="p-perm-cmd${write ? ' p-perm-path' : ''}">${write ? '' : '<span class="p-perm-prompt" aria-hidden="true">$</span>'}<code class="p-perm-code">${esc(command)}</code></div>` : ''}
  ${note ? `<p class="p-perm-review"><span class="p-perm-jev">jev</span> ${esc(note)}</p>` : ''}
  <div class="p-perm-actions">
    ${choice('allow', t('chat.perm.allow'), 'p-perm-primary')}
    ${choice('always', t('chat.perm.always'), 'p-perm-secondary')}
    ${choice('deny', t('chat.perm.deny'), 'p-perm-secondary')}
  </div>
</div>`
}

export const TEXTS = {
  en: {
    'chat.perm.sandbox': 'Wants to leave the sandbox',
    'chat.perm.write': 'Wants to change a file',
    'chat.perm.allow': 'Allow once',
    'chat.perm.always': 'Always here',
    'chat.perm.deny': 'Deny',
  },
  es: {
    'chat.perm.sandbox': 'Quiere salir del sandbox',
    'chat.perm.write': 'Quiere modificar un archivo',
    'chat.perm.allow': 'Permitir una vez',
    'chat.perm.always': 'Siempre aquí',
    'chat.perm.deny': 'Denegar',
  },
}
