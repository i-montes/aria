// The header meter (design/components/quota-meter.dc.html): the model name, a small quota bar and its
// percentage, and a popover (on hover, focus or tap) with the quota, the context and where the
// context compacts. Clicking it opens the model picker.
//
// Props: { model, used, context, free?, hasQuota?, compactAt?, resets? }
//   model      model/provider display name (not translated)
//   used       quota used, 0-100 (clamped, rounded)
//   context    context window used, 0-100
//   free       true for a free model: "free" badge, no quota bar
//   hasQuota   false for a pay-per-use provider (default true): no quota bar
//   compactAt  where the context auto-compacts, 0-100 (default 40)
//   resets     optional, when the quota resets ("12 days"); the reset sentences are left out without it
// ui: unused. t(key, vars) translates.
// The popover shows with CSS only (:hover, :focus-within), so a re-render never loses it.
// The button's backdrop is `--p-meter-bg` (default the dock's field color); set it to the header's.
// Emits: data-chat="open-models"
import { esc } from '../../onboarding/components.js'

const clamp = (value, fallback) => {
  const n = Math.round(Number(value))
  return Number.isFinite(n) ? Math.max(0, Math.min(100, n)) : fallback
}

export function renderMeter({ model, used, context, free, hasQuota, compactAt, resets } = {}, ui, t) {
  const isFree = free === true
  const quota = hasQuota !== false
  const u = clamp(used, 0)
  const c = clamp(context, 0)
  const at = clamp(compactAt, 40)
  const showMeter = !isFree && quota
  const noQuota = !isFree && !quota
  const level = u >= 95 ? 'crit' : u >= 80 ? 'warn' : 'ok'
  const hasResets = resets !== undefined && resets !== null && resets !== ''
  const vars = { model: model ?? '', used: u, context: c, resets: resets ?? '' }
  const note = !showMeter
    ? ''
    : level === 'crit' ? t(hasResets ? 'chat.meter.note_out' : 'chat.meter.note_out_plain', { resets })
      : level === 'warn' ? t(hasResets ? 'chat.meter.note_low' : 'chat.meter.note_low_plain', { resets })
        : hasResets ? t('chat.meter.note_ok', { resets }) : ''
  const aria = noQuota ? t('chat.meter.aria_noquota', vars)
    : isFree ? t('chat.meter.aria_free', vars)
      : t(hasResets ? 'chat.meter.aria_quota_resets' : 'chat.meter.aria_quota', vars)
  const tier = isFree ? t('chat.meter.tier_free') : quota ? t('chat.meter.subscription') : t('chat.meter.tier_payg')

  return `<div class="p-meter${showMeter ? ` p-meter-${level}` : ''}">
  <button type="button" class="p-meter-btn" data-chat="open-models" aria-label="${esc(aria)}">
    <span class="p-meter-model">${esc(model)}</span>
    ${showMeter ? `<span class="p-meter-bar" aria-hidden="true"><span class="p-meter-fill" style="width:${u}%"></span></span>
    <span class="p-meter-pct">${u}%</span>` : ''}
    ${isFree ? `<span class="p-meter-free">${esc(t('chat.meter.free'))}</span>` : ''}
  </button>
  <div class="p-meter-pop" role="tooltip">
    <div class="p-meter-col">
      <div class="p-meter-row p-meter-title"><span class="p-meter-name">${esc(model)}</span><span class="p-meter-tier">${esc(tier)}</span></div>
      ${showMeter ? `<div class="p-meter-block">
        <div class="p-meter-row p-meter-label"><span class="p-meter-dim">${esc(t('chat.meter.quota'))}</span><span class="p-meter-pct">${esc(t('chat.meter.used', { n: u }))}</span></div>
        <div class="p-meter-track"><span class="p-meter-fill" style="width:${u}%"></span></div>
        ${note ? `<span class="p-meter-note">${esc(note)}</span>` : ''}
      </div>` : ''}
      ${noQuota ? `<span class="p-meter-text">${esc(t('chat.meter.no_quota'))}</span>` : ''}
      ${isFree ? `<span class="p-meter-text">${esc(t('chat.meter.free_note'))}</span>` : ''}
      <div class="p-meter-block">
        <div class="p-meter-row p-meter-label"><span class="p-meter-dim">${esc(t('chat.meter.context'))}</span><span class="p-meter-ctx">${c}%</span></div>
        <div class="p-meter-track"><span class="p-meter-ctxfill" style="width:${c}%"></span><span class="p-meter-mark" style="left:${at}%"></span></div>
        <span class="p-meter-note">${esc(t('chat.meter.compacts', { n: at }))}</span>
      </div>
      <div class="p-meter-hint"><kbd>/model</kbd><span>${esc(t('chat.meter.model_hint'))}</span></div>
    </div>
  </div>
</div>`
}

export const TEXTS = {
  en: {
    'chat.meter.free': 'free',
    'chat.meter.quota': 'QUOTA',
    'chat.meter.context': 'CONTEXT',
    'chat.meter.used': '{n}% used',
    'chat.meter.subscription': 'subscription',
    'chat.meter.tier_free': 'free',
    'chat.meter.tier_payg': 'pay per use',
    'chat.meter.note_ok': 'Resets in {resets}.',
    'chat.meter.note_low': 'Running low. Resets in {resets}.',
    'chat.meter.note_out': 'Almost out. Resets in {resets}.',
    'chat.meter.note_low_plain': 'Running low.',
    'chat.meter.note_out_plain': 'Almost out.',
    'chat.meter.no_quota': 'This provider has no quota to read. You pay per use.',
    'chat.meter.free_note': 'No quota on free models. Switch back with /model when you want the subscription.',
    'chat.meter.compacts': 'Compacts at {n}%',
    'chat.meter.model_hint': 'to change it',
    'chat.meter.aria_quota': '{model}, {used}% of the quota used. Context {context}%. Change the model with /model',
    'chat.meter.aria_quota_resets': '{model}, {used}% of the quota used, resets in {resets}. Context {context}%. Change the model with /model',
    'chat.meter.aria_free': '{model}, free, no quota. Context {context}%. Change the model with /model',
    'chat.meter.aria_noquota': '{model}, no quota to read. Context {context}%. Change the model with /model',
  },
  es: {
    'chat.meter.free': 'gratis',
    'chat.meter.quota': 'CUOTA',
    'chat.meter.context': 'CONTEXTO',
    'chat.meter.used': '{n}% usado',
    'chat.meter.subscription': 'suscripción',
    'chat.meter.tier_free': 'gratuito',
    'chat.meter.tier_payg': 'pago por uso',
    'chat.meter.note_ok': 'Se renueva en {resets}.',
    'chat.meter.note_low': 'Queda poco. Se renueva en {resets}.',
    'chat.meter.note_out': 'Casi agotada. Se renueva en {resets}.',
    'chat.meter.note_low_plain': 'Queda poco.',
    'chat.meter.note_out_plain': 'Casi agotada.',
    'chat.meter.no_quota': 'Este proveedor no tiene una cuota que leer. Pagas por uso.',
    'chat.meter.free_note': 'Los modelos gratuitos no tienen cuota. Vuelve con /model cuando quieras la suscripción.',
    'chat.meter.compacts': 'Se compacta al {n}%',
    'chat.meter.model_hint': 'para cambiarlo',
    'chat.meter.aria_quota': '{model}, {used}% de la cuota usada. Contexto {context}%. Cambia el modelo con /model',
    'chat.meter.aria_quota_resets': '{model}, {used}% de la cuota usada, se renueva en {resets}. Contexto {context}%. Cambia el modelo con /model',
    'chat.meter.aria_free': '{model}, gratuito, sin cuota. Contexto {context}%. Cambia el modelo con /model',
    'chat.meter.aria_noquota': '{model}, sin cuota que leer. Contexto {context}%. Cambia el modelo con /model',
  },
}
