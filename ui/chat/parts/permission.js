// The permission card (design/components/sandbox-permission.dc.html): the agent wants something
// outside its sandbox (a command, a file write) and asks for a decision. Fed by a dialog title
// `{"aria":1,"kind":"sandbox"|"write"|"connector",…}` (see cards.js). A connector card is an MCP connector that wants to
// change something (create a page, send a message).
//
// Props: { id, kind, summary?, detail?, jev?, options? }
//   id       the extension dialog's id; answers go back to it
//   kind     "sandbox" (default) | "write" | "connector": picks the icon and the default title
//   summary  the heading; the kind's default title when empty
//   detail   the command (sandbox), the path (write) or the call's arguments (connector), in the dark code box
//   jev      Jev's review: a string, or an object with `review` or `text`
//   options  [{ value, label, primary? }]: the extension's own choices (one button each, answering with `value`);
//            without them: allow / always / deny
// ui: unused. t(key, vars) translates.
// Emits: data-chat="permission" data-id data-choice="<option value>" (allow|always|deny by default)
import { esc } from '../../onboarding/components.js'

const lock = '<svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="10" width="16" height="11" rx="2"></rect><path d="M8 10V7a4 4 0 0 1 8 0v3"></path></svg>'
const file = '<svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z"></path><path d="M14 3v5h5"></path></svg>'

const plug = '<svg aria-hidden="true" data-icon="plug" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 2v6"></path><path d="M15 2v6"></path><path d="M6 8h12v4a6 6 0 0 1-12 0Z"></path><path d="M12 18v4"></path></svg>'
const ICONS = { sandbox: lock, write: file, connector: plug }

const text = (value) => (typeof value === 'string' ? value : '')
const review = (jev) => (typeof jev === 'string' ? jev : text(jev?.review) || text(jev?.text))

export function renderPermission({ id, kind, summary, detail, jev, options } = {}, ui, t) {
  const shown = kind === 'write' || kind === 'connector' ? kind : 'sandbox'
  // Only a sandbox card holds a shell command (with its "$"); a path or a call's arguments go in plain.
  const plain = shown !== 'sandbox'
  const title = text(summary) || t(`chat.perm.${shown}`)
  const command = text(detail)
  const note = review(jev)
  const choice = (name, label, cls) =>
    `<button type="button" class="p-perm-btn ${cls}" data-chat="permission" data-id="${esc(id)}" data-choice="${esc(name)}">${esc(label)}</button>`
  return `<div class="p-perm" role="group" aria-label="${esc(title)}">
  <div class="p-perm-head">${ICONS[shown]}<span>${esc(title)}</span></div>
  ${command ? `<div class="p-perm-cmd${plain ? ' p-perm-path' : ''}">${plain ? '' : '<span class="p-perm-prompt" aria-hidden="true">$</span>'}<code class="p-perm-code">${esc(command)}</code></div>` : ''}
  ${note ? `<p class="p-perm-review"><span class="p-perm-jev">jev</span> ${esc(note)}</p>` : ''}
  <div class="p-perm-actions">
    ${Array.isArray(options) && options.length
      ? options.map((o, i) => choice(o.value, o.label, o.primary || (i === 0 && !options.some((x) => x.primary)) ? 'p-perm-primary' : 'p-perm-secondary')).join('\n    ')
      : [choice('allow', t('chat.perm.allow'), 'p-perm-primary'), choice('always', t('chat.perm.always'), 'p-perm-secondary'), choice('deny', t('chat.perm.deny'), 'p-perm-secondary')].join('\n    ')}
  </div>
</div>`
}

export const TEXTS = {
  en: {
    'chat.perm.sandbox': 'Wants to leave the sandbox',
    'chat.perm.write': 'Wants to change a file',
    'chat.perm.connector': 'A connector wants to act',
    'chat.perm.allow': 'Allow once',
    'chat.perm.always': 'Always here',
    'chat.perm.deny': 'Deny',
  },
  es: {
    'chat.perm.sandbox': 'Quiere salir del sandbox',
    'chat.perm.write': 'Quiere modificar un archivo',
    'chat.perm.connector': 'Un conector quiere actuar',
    'chat.perm.allow': 'Permitir una vez',
    'chat.perm.always': 'Siempre aquí',
    'chat.perm.deny': 'Denegar',
  },
}
