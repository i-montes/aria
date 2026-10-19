// The reasoning effort for a message (ARIA-SETUP item 19). Auto: a quick route drops to low, a think route goes at
// least to high from the session's level, and a free model runs at its top level (Pi clamps `max` to what it
// supports). A level picked by hand, or Aria's fixed setting, wins over the route.
import { rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { agentDir } from '../lib/agent-dir.ts'
import { type AriaConfig, writeAriaSetting } from '../lib/config.ts'

/** The level the owner picked by hand in his own Pi, shared by every session until `/effort auto`. */
export const EFFORT_FILE = join(agentDir(), 'jev-router-effort.json')

/** `/effort <level>` (null: auto): Aria's settings under Aria, the effort file otherwise. */
export function saveEffort(level: string | null, file = EFFORT_FILE): void {
  if (writeAriaSetting(['effort', 'mode'], level ? 'fixed' : 'auto') && writeAriaSetting(['effort', 'fixedLevel'], level)) return
  try {
    if (level) writeFileSync(file, JSON.stringify({ level }) + '\n')
    else rmSync(file, { force: true })
  } catch {
    // stays for this session only
  }
}

export const RANK = ['off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max']

export function effortFor(o: { route?: string; free: boolean; base: string; fixed: string | null; config: AriaConfig }): string {
  const fixed = o.fixed ?? (o.config.effort.mode === 'fixed' ? o.config.effort.fixedLevel : null)
  if (fixed) return fixed
  if (o.free && o.config.effort.maxOnFree) return 'max'
  if (o.route === 'quick') return 'low'
  if (o.route === 'think') return RANK.indexOf(o.base) >= RANK.indexOf('high') ? o.base : 'high'
  return o.base
}
