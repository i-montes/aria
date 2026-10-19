// The reasoning effort for a message (ARIA-SETUP item 19). Auto: a quick route drops to low, a think route goes at
// least to high from the session's level, and a free model runs at its top level (Pi clamps `max` to what it
// supports). A level picked by hand, or Aria's fixed setting, wins over the route.
import type { AriaConfig } from '../lib/config.ts'

export const RANK = ['off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max']

export function effortFor(o: { route?: string; free: boolean; base: string; fixed: string | null; config: AriaConfig }): string {
  const fixed = o.fixed ?? (o.config.effort.mode === 'fixed' ? o.config.effort.fixedLevel : null)
  if (fixed) return fixed
  if (o.free && o.config.effort.maxOnFree) return 'max'
  if (o.route === 'quick') return 'low'
  if (o.route === 'think') return RANK.indexOf(o.base) >= RANK.indexOf('high') ? o.base : 'high'
  return o.base
}
