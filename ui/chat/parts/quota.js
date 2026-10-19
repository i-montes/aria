// The quota card (design/components/quota.dc.html): the subscription is almost used up; switch to a
// free model for the rest of the task? Fed by a dialog title `{"aria":1,"kind":"quota",…}` (cards.js).
//
// Props: { id, percent, freeModel?, secondsLeft?, options? }
//   options  [{ value, label, role?: 'free'|'keep', primary? }]: the extension's own choices; a known role shows the
//            chat's words, and the button answers with `value`
//   id           the extension dialog's id
//   percent      quota used, 0-100 (clamped, rounded)
//   freeModel    name of the free model that would be used; shown when given
//   secondsLeft  seconds left before the dialog times out; shown when given. It is only a display:
//                the real timeout is Pi's, and the card disappears when the dialog does
// ui: unused. t(key, vars) translates.
// Emits: data-chat="quota" data-id data-choice="free|keep"
import { esc } from '../../onboarding/components.js'

const clock = '<svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"></circle><path d="M12 7v5l3 2"></path></svg>'

const percentOf = (value) => {
  const n = Math.round(Number(value))
  return Number.isFinite(n) ? Math.max(0, Math.min(100, n)) : 0
}

export function renderQuota({ id, percent, freeModel, secondsLeft, options } = {}, ui, t) {
  const p = percentOf(percent)
  const hasModel = typeof freeModel === 'string' && freeModel !== ''
  const hasCountdown = secondsLeft !== undefined && secondsLeft !== null && secondsLeft !== ''
  const choice = (name, label, cls, role) =>
    `<button type="button" class="p-quota-btn ${cls}" data-chat="quota" data-id="${esc(id)}" data-choice="${esc(name)}"${role ? ` data-role="${role}"` : ''}>${esc(label)}</button>`
  return `<div class="p-quota" role="group" aria-label="${esc(t('chat.quota.title'))}">
  <div class="p-quota-top">
    <div class="p-quota-head">${clock}<span>${esc(t('chat.quota.title'))}</span></div>
    <span class="p-quota-pct">${esc(t('chat.quota.used', { n: p }))}</span>
  </div>
  <div class="p-quota-bar" role="progressbar" aria-label="${esc(t('chat.quota.bar'))}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${p}"><div class="p-quota-fill" style="width:${p}%"></div></div>
  <p class="p-quota-question">${esc(t('chat.quota.question'))}</p>
  ${hasModel || hasCountdown ? `<p class="p-quota-meta">${hasModel ? `<span>${esc(t('chat.quota.free_model', { model: freeModel }))}</span>` : ''}${hasCountdown ? `<span>${esc(t('chat.quota.countdown', { n: secondsLeft }))}</span>` : ''}</p>` : ''}
  <div class="p-quota-actions">
    ${Array.isArray(options) && options.length
      ? options
          .map((o, i) => {
            const role = o.role === 'free' || o.role === 'keep' ? o.role : null
            const label = role ? t(`chat.quota.${role}`) : o.label
            return choice(o.value, label, o.primary || (i === 0 && !options.some((x) => x.primary)) ? 'p-quota-primary' : 'p-quota-secondary', role)
          })
          .join('\n    ')
      : `${choice('free', t('chat.quota.free'), 'p-quota-primary')}\n    ${choice('keep', t('chat.quota.keep'), 'p-quota-secondary')}`}
  </div>
</div>`
}

export const TEXTS = {
  en: {
    'chat.quota.title': 'OpenCode Go quota almost used up',
    'chat.quota.used': '{n}% used',
    'chat.quota.bar': 'Quota used',
    'chat.quota.question': 'Should I switch to a free model for the rest of this task?',
    'chat.quota.free_model': 'Free model: {model}',
    'chat.quota.countdown': 'Times out in {n}s',
    'chat.quota.free': 'Switch to free',
    'chat.quota.keep': 'Keep subscription',
  },
  es: {
    'chat.quota.title': 'La cuota de OpenCode Go casi se acaba',
    'chat.quota.used': '{n}% usado',
    'chat.quota.bar': 'Cuota usada',
    'chat.quota.question': '¿Cambio a un modelo gratuito para el resto de esta tarea?',
    'chat.quota.free_model': 'Modelo gratuito: {model}',
    'chat.quota.countdown': 'Caduca en {n} s',
    'chat.quota.free': 'Cambiar a gratuito',
    'chat.quota.keep': 'Mantener la suscripción',
  },
}
