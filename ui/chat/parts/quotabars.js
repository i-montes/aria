// The provider's quota next to the assistant's name (single session): a compact 5-hour bar, and, on hover or focus, a card
// with every window the provider has for the model in use (5h, Week, Month, and a per-model one such as Fable) and when each
// resets. Drawn in the spirit of the design's meter (a bar with a card of details).
//
// Props: { windows: [{ id, label, percent, resetsAt? }], now? }   (empty or missing: nothing is drawn)
//   id      `5h`, `week`, `month` (translated labels) or a model's name (its own `label` is shown); the compact bar shows `5h`
//           (the first window when there is none)
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
  const stamp = now ?? Date.now()
  const list = (windows ?? [])
    .filter((w) => w && typeof w.percent === 'number' && Number.isFinite(w.percent) && w.id)
    .map((w) => {
      const percent = Math.min(100, Math.max(0, w.percent))
      const label = w.id === 'week' ? t('chat.quotabars.week') : w.id === 'month' ? t('chat.quotabars.month') : (w.label ?? w.id)
      return { label, percent, shown: Math.round(percent), level: levelOf(percent), reset: formatReset(w.resetsAt, stamp), five: w.id === '5h' }
    })
  if (!list.length) return ''
  const main = list.find((w) => w.five) ?? list[0]
  const ordered = [...list.filter((w) => w.five), ...list.filter((w) => !w.five)]
  const rows = ordered
    .map(
      (w) => `<div class="p-qb-row is-${w.level}">
        <span class="p-qb-row-head"><span class="p-qb-row-label">${esc(w.label)}</span><span class="p-qb-row-num">${w.shown}%</span></span>
        <span class="p-qb-row-track" aria-hidden="true"><span class="p-qb-fill" style="width: ${w.percent}%"></span></span>
        ${w.reset ? `<span class="p-qb-row-reset">${esc(t('chat.quotabars.resets', { when: w.reset }))}</span>` : ''}
      </div>`,
    )
    .join('')
  return `<div class="p-qb" tabindex="0" role="group" aria-label="${esc(t('chat.quotabars.title'))}">
    <div class="p-qb-inline is-${main.level}">
      <span class="p-qb-head"><span class="p-qb-label">${esc(main.label)}</span><span class="p-qb-num">${main.shown}%</span></span>
      <span class="p-qb-track" aria-hidden="true"><span class="p-qb-fill" style="width: ${main.percent}%"></span></span>
    </div>
    <div class="p-qb-pop" role="tooltip">
      <div class="p-qb-pop-title">${esc(t('chat.quotabars.title'))}</div>
      ${rows}
    </div>
  </div>`
}
