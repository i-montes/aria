// Auto-compaction at a share of the context window (default 40%), the same for every model. Pi
// compacts when the context passes `contextWindow - reserveTokens`, so the share becomes a
// per-model `compaction.modelOverrides[...].reserveTokens` in settings.json. 0 turns automatic
// compaction off. Pure functions here; index.ts reads and writes the files.
import { ariaConfig, DEFAULTS, hasAriaConfig, writeAriaSetting } from '../lib/config.ts'
import { readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { agentDir } from '../lib/agent-dir.ts'

export const DEFAULT_PERCENT = DEFAULTS.compaction.percent
// Pi's own default reserve. reserveTokens also caps the summary's length (80% of it), so it never
// goes lower, even at 100%.
export const MIN_RESERVE = 16384
export const PERCENT_FILE = join(agentDir(), 'autocompact.json')
export const SETTINGS_FILE = join(agentDir(), 'settings.json')

export interface ModelInfo {
  provider: string
  id: string
  contextWindow?: number
}

/** A percent from the command line: an integer 0-100, or null. */
export function parsePercent(text: string): number | null {
  const t = text.trim().replace(/%$/, '')
  if (!/^\d{1,3}$/.test(t)) return null
  const n = Number(t)
  return n <= 100 ? n : null
}

export const reserveFor = (contextWindow: number, percent: number) =>
  Math.max(MIN_RESERVE, contextWindow - Math.floor((contextWindow * percent) / 100))

/** The settings with every model's reserve set for `percent`; other keys are kept. */
export function applyPercent(settings: any, models: ModelInfo[], percent: number): any {
  const compaction = { ...(settings?.compaction ?? {}) }
  if (percent === 0) return { ...settings, compaction: { ...compaction, enabled: false } }
  delete compaction.enabled
  const overrides = { ...(compaction.modelOverrides ?? {}) }
  for (const m of models) {
    if (!m.contextWindow || m.contextWindow <= 0) continue
    const key = `${m.provider}/${m.id}`
    overrides[key] = { ...overrides[key], reserveTokens: reserveFor(m.contextWindow, percent) }
  }
  return { ...settings, compaction: { ...compaction, modelOverrides: overrides } }
}

/** The share in use: Aria's setting under Aria, the owner's autocompact.json otherwise. */
export function compactPercent(file = PERCENT_FILE): number {
  return hasAriaConfig() ? ariaConfig().compaction.percent : readPercent(file)
}

export function readPercent(file = PERCENT_FILE): number {
  try {
    const p = JSON.parse(readFileSync(file, 'utf8')).percent
    return Number.isInteger(p) && p >= 0 && p <= 100 ? p : DEFAULT_PERCENT
  } catch {
    return DEFAULT_PERCENT
  }
}

/** `/autocompact <n>`: Aria's settings under Aria, autocompact.json otherwise. */
export function savePercent(percent: number, file = PERCENT_FILE): void {
  if (!writeAriaSetting(['compaction', 'percent'], percent)) writePercent(percent, file)
}

export function writePercent(percent: number, file = PERCENT_FILE): void {
  if (percent === DEFAULT_PERCENT) rmSync(file, { force: true })
  else writeAtomic(file, `${JSON.stringify({ percent })}\n`)
}

export function writeAtomic(file: string, text: string): void {
  const tmp = `${file}.${process.pid}.tmp`
  writeFileSync(tmp, text)
  renameSync(tmp, file)
}

/** Bring settings.json in line with the percent; true when it changed. */
export function syncSettings(models: ModelInfo[], percent: number, file = SETTINGS_FILE): boolean {
  const text = readFileSync(file, 'utf8')
  const settings = JSON.parse(text)
  const next = applyPercent(settings, models, percent)
  const out = `${JSON.stringify(next, null, 2)}\n`
  if (JSON.stringify(next) === JSON.stringify(settings)) return false
  writeAtomic(file, out)
  return true
}
