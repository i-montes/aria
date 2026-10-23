// The secret card: the permission card's shell (permission.css) with a field whose value never reaches the model
// (docs/superpowers/plans/2026-10-22-secrets.md). Two uses:
// - asked: the model asked for a secret (`{"aria":1,"kind":"secret",name,purpose,env}`, see cards.js). A password
//   field; Save sends the value straight to the dock (`secretSave`), which keeps it in the keyring, and only then is
//   the dialog answered, with the name.
// - pasted: the composer caught secrets in a message (../secrets.js). One name field per secret; Keep saves them and
//   sends the message with each value replaced by its name, Send as is sends it unchanged, Edit goes back.
//
// Props: { mode: 'asked' | 'pasted', id?, name?, purpose?, env?, found?: [{ kind, name }], busy?, error? }
//   id     the dialog's id (asked)
//   error  a UI key's tail (`keyring-off`, `invalid-name`, `failed`) shown under the fields
// Emits: data-chat="secret" data-op="save"|"cancel" data-id (asked); data-chat="secret-paste" data-op="keep"|"send"|"edit"
// Fields: input[data-secret-value] (asked), input[data-secret-name][data-index] (pasted).
import { esc } from '../../onboarding/components.js'

const key = '<svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="7.5" cy="15.5" r="3.5"></circle><path d="m10 13 9-9"></path><path d="m16 7 2 2"></path><path d="m14 9 2 2"></path></svg>'

const text = (value) => (typeof value === 'string' ? value : '')

export function renderSecret({ mode = 'asked', id = '', name, purpose, env, found, busy = false, error } = {}, ui, t) {
  const disabled = busy ? ' disabled' : ''
  const problem = error ? `<p class="p-perm-review p-secret-error" role="alert">${esc(t(`chat.secret.error.${error}`))}</p>` : ''
  const button = (chat, op, label, cls) =>
    `<button type="button" class="p-perm-btn ${cls}" data-chat="${chat}" data-op="${op}"${id ? ` data-id="${esc(id)}"` : ''}${disabled}>${esc(label)}</button>`
  if (mode === 'pasted') {
    const list = Array.isArray(found) ? found : []
    const title = t(list.length === 1 ? 'chat.secret.pasted.one' : 'chat.secret.pasted.many')
    const rows = list
      .map(
        (f, i) => `<label class="p-secret-row"><span class="p-secret-kind">${esc(text(f.kind))}</span>
    <input type="text" class="p-secret-field" data-secret-name data-index="${i}" value="${esc(text(f.name))}" spellcheck="false" autocomplete="off" aria-label="${esc(t('chat.secret.name'))}"${disabled}></label>`,
      )
      .join('\n  ')
    return `<div class="p-perm p-secret" role="group" aria-label="${esc(title)}">
  <div class="p-perm-head">${key}<span>${esc(title)}</span></div>
  <p class="p-perm-review">${esc(t('chat.secret.pasted.text'))}</p>
  ${rows}
  ${problem}
  <div class="p-perm-actions">
    ${button('secret-paste', 'keep', t('chat.secret.keep'), 'p-perm-primary')}
    ${button('secret-paste', 'send', t('chat.secret.sendAsIs'), 'p-perm-secondary')}
    ${button('secret-paste', 'edit', t('chat.secret.edit'), 'p-perm-secondary')}
  </div>
</div>`
  }
  const title = t('chat.secret.asked', { name: text(name) })
  const variable = text(env)
  return `<div class="p-perm p-secret" role="group" aria-label="${esc(title)}">
  <div class="p-perm-head">${key}<span>${esc(title)}</span></div>
  ${text(purpose) ? `<p class="p-perm-review">${esc(text(purpose))}</p>` : ''}
  <input type="password" class="p-secret-field" data-secret-value data-id="${esc(id)}" placeholder="${esc(t('chat.secret.placeholder'))}" spellcheck="false" autocomplete="off" aria-label="${esc(title)}"${disabled}>
  <p class="p-secret-note">${esc(t('chat.secret.note'))}${variable ? ` <code>$${esc(variable)}</code>` : ''}</p>
  ${problem}
  <div class="p-perm-actions">
    ${button('secret', 'save', t('chat.secret.save'), 'p-perm-primary')}
    ${button('secret', 'cancel', t('chat.secret.cancel'), 'p-perm-secondary')}
  </div>
</div>`
}

export const TEXTS = {
  en: {
    'chat.secret.asked': 'Aria needs a secret: {name}',
    'chat.secret.placeholder': 'Paste it here',
    'chat.secret.note': 'It goes to the system keyring; the model never sees it, only its name. Commands read it as',
    'chat.secret.save': 'Save',
    'chat.secret.cancel': 'Cancel',
    'chat.secret.pasted.one': 'That looks like a secret',
    'chat.secret.pasted.many': 'That looks like secrets',
    'chat.secret.pasted.text': 'Keep it in the keyring instead: the message goes with its name, and the model never sees the value.',
    'chat.secret.name': 'Name',
    'chat.secret.keep': 'Keep it safe',
    'chat.secret.sendAsIs': 'Send as is',
    'chat.secret.edit': 'Edit',
    'chat.secret.error.keyring-off': 'The keyring is off in Settings: turn it on to keep secrets.',
    'chat.secret.error.invalid-name': 'A name is letters, digits and dashes (github-token).',
    'chat.secret.error.empty': 'Paste the secret first.',
    'chat.secret.error.failed': 'It could not be saved in the keyring.',
    'chat.secret.saved': 'saved as {name}',
  },
  es: {
    'chat.secret.asked': 'Aria necesita un secreto: {name}',
    'chat.secret.placeholder': 'Pégalo aquí',
    'chat.secret.note': 'Va al llavero del sistema; el modelo nunca lo ve, solo su nombre. Los comandos lo leen como',
    'chat.secret.save': 'Guardar',
    'chat.secret.cancel': 'Cancelar',
    'chat.secret.pasted.one': 'Eso parece un secreto',
    'chat.secret.pasted.many': 'Eso parecen secretos',
    'chat.secret.pasted.text': 'Guárdalo en el llavero: el mensaje va con su nombre y el modelo nunca ve el valor.',
    'chat.secret.name': 'Nombre',
    'chat.secret.keep': 'Guardarlo seguro',
    'chat.secret.sendAsIs': 'Enviar tal cual',
    'chat.secret.edit': 'Editar',
    'chat.secret.error.keyring-off': 'El llavero está desactivado en Ajustes: actívalo para guardar secretos.',
    'chat.secret.error.invalid-name': 'Un nombre lleva letras, dígitos y guiones (github-token).',
    'chat.secret.error.empty': 'Pega el secreto primero.',
    'chat.secret.error.failed': 'No se pudo guardar en el llavero.',
    'chat.secret.saved': 'guardado como {name}',
  },
}
