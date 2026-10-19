// The provider's quota next to the assistant's name (single session): one small bar per window the provider has for the
// model in use (5h, Week, Month, and a per-model one such as Fable). Not a component of the design (its meter shows one bar);
// drawn in its spirit. The tooltip says how much is used and when the window resets.
//
// Props: { windows: [{ id, label, percent, resetsAt? }], now? }   (empty or missing: nothing is drawn)
//   id      `5h`, `week`, `month` (translated labels) or a model's name (its own `label` is shown)
//   percent 0..100 of the window spent
// ui: unused. t: translate(key, vars).
// Emits: nothing.
import { esc } from '../../onboarding/components.js'

export const TEXTS = {
  en: {
    'chat.quotabars.title': 'Quota',
    'chat.quotabars.week': 'Week',
    'chat.quotabars.month': 'Month',
    'chat.quotabars.used': '{label}: {percent}% used',
    'chat.quotabars.resets': 'resets in {when}',
  },
  es: {
    'chat.quotabars.title': 'Cuota',
    'chat.quotabars.week': 'Semana',
    'chat.quotabars.month': 'Mes',
    'chat.quotabars.used': '{label}: {percent}% usado',
    'chat.quotabars.resets': 'se renueva en {when}',
  },
}

/** "25m", "3h 20m", "5d 4h" until `iso`; nothing when it is past or not a date. */
export function formatReset(iso, now = Date.now()) {
  const at = typeof iso === 'string' ? Date.parse(iso) : NaN
  if (!Number.isFinite(at) || at <= now) return ''
  const minutes = Math.floor((at - now) / 60_000)
  if (minutes < 60) return `${Math.max(minutes, 1)}m`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return minutes % 60 ? `${hours}h ${minutes % 60}m` : `${hours}h`
  const days = Math.floor(hours / 24)
  return hours % 24 ? `${days}d ${hours % 24}h` : `${days}d`
}

const levelOf = (percent) => (percent >= 95 ? 'crit' : percent >= 80 ? 'warn' : 'ok')

export function renderQuotaBars({ windows, now } = {}, ui, t) {
  const items = (windows ?? [])
    .filter((w) => w && typeof w.percent === 'number' && Number.isFinite(w.percent) && w.id)
    .map((w) => {
      const percent = Math.min(100, Math.max(0, w.percent))
      const shown = Math.round(percent)
      const label = w.id === 'week' ? t('chat.quotabars.week') : w.id === 'month' ? t('chat.quotabars.month') : (w.label ?? w.id)
      const reset = formatReset(w.resetsAt, now ?? Date.now())
      const title = `${t('chat.quotabars.used', { label, percent: shown })}${reset ? ` · ${t('chat.quotabars.resets', { when: reset })}` : ''}`
      return `<div class="p-qb-item is-${levelOf(percent)}" title="${esc(title)}">
        <span class="p-qb-head"><span class="p-qb-label">${esc(label)}</span><span class="p-qb-num">${shown}%</span></span>
        <span class="p-qb-track" aria-hidden="true"><span class="p-qb-fill" style="width: ${percent}%"></span></span>
      </div>`
    })
  return items.length ? `<div class="p-qb" role="group" aria-label="${esc(t('chat.quotabars.title'))}">${items.join('')}</div>` : ''
}
