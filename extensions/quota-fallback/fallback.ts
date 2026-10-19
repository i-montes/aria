// When the OpenCode Go quota is nearly spent, a paid Go model is swapped for a free one, and the
// original comes back after the window resets. Pure decisions live here; index.ts applies them.
import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { exhausted, type GoUsage } from '../jev-router/usage.ts'
import { ariaConfig, DEFAULTS, type Model } from '../lib/config.ts'

// Free inside Go; not used for training, kept 30 days for safety per OpenCode's policy (checked
// 2026-10-14). Picked on 2026-10-14 from the 5-task benchmark (bugfix, feature, repo exploration,
// refactor, CLI): 10/10 and 2-4x faster than deepseek-v4.1-flash; longcat-2.5-preview-free took
// 20+ min on the feature task and step-5-preview-free kept answering 429. The Zen free tier
// (opencode/*-free) refuses any client but OpenCode itself, so Go's free models are the only ones
// Pi can use.
export const FREE_MODEL = DEFAULTS.quota.freeModel
/** The free model: Aria's setting, or FREE_MODEL. */
export const freeModel = (): Model => ariaConfig().quota.freeModel
// The user's pick (2026-10-14): use almost all the paid quota, switch just before it runs out.
export const THRESHOLD = DEFAULTS.quota.switchAt
export const switchThreshold = (): number => ariaConfig().quota.switchAt
export const CACHE_FILE = join(tmpdir(), 'pi-go-usage.json')
export const CACHE_TTL_MS = 60_000

export interface ModelLike {
  provider: string
  id: string
  cost?: { input?: number; output?: number }
}

/** A model that costs nothing: a `-free` id, or input and output both priced at zero. */
export function isFree(model: ModelLike | undefined | null): boolean {
  if (!model) return false
  if (model.id.endsWith('-free')) return true
  const c = model.cost
  return !!c && (c.input ?? 0) === 0 && (c.output ?? 0) === 0
}

/** A Go model that spends quota: anything in opencode-go that is not free. */
export function isPaidGo(model: ModelLike | undefined | null): boolean {
  return !!model && model.provider === 'opencode-go' && !isFree(model)
}

export type Decision =
  | { kind: 'switch'; window: '5-hour' | 'weekly'; percent: number; resetsAt?: string }
  | { kind: 'restore'; to: ModelLike }
  | null

export function decide(s: {
  usage: GoUsage | null
  current: ModelLike | undefined | null
  switchedFrom: ModelLike | null
  threshold: number
}): Decision {
  if (!s.usage) return null
  const spent = exhausted(s.usage, s.threshold)
  if (spent) {
    if (!isPaidGo(s.current)) return null
    return { kind: 'switch', window: spent.name, percent: spent.window.percent, resetsAt: spent.window.resetsAt }
  }
  const free = freeModel()
  const onFree = s.current?.provider === free.provider && s.current?.id === free.id
  if (s.switchedFrom && onFree) return { kind: 'restore', to: s.switchedFrom }
  return null
}

// Before switching, the TUI asks; with no answer in ASK_TIMEOUT_MS it switches so the task goes on
// (the user's pick, 2026-10-18: the free model keeps prompts 30 days, so it is their call). The
// answer is shared with every pi process (subagents follow the TUI's choice) and holds for that
// quota window; reaching 100% asks again.
export const ASK_TIMEOUT_MS = DEFAULTS.quota.countdownMs
export const askTimeoutMs = (): number => ariaConfig().quota.countdownMs
export const DECISION_FILE = join(tmpdir(), 'pi-go-switch-decision.json')
export type Choice = 'switch' | 'stay'

export const decisionKey = (d: { window: string; percent: number; resetsAt?: string }) =>
  `${d.window}|${d.resetsAt ?? ''}|${d.percent >= 100 ? 'full' : 'near'}`

export function readDecision(key: string, file = DECISION_FILE): Choice | null {
  try {
    const entry = JSON.parse(readFileSync(file, 'utf8'))
    return entry?.key === key && (entry.choice === 'switch' || entry.choice === 'stay') ? entry.choice : null
  } catch {
    return null
  }
}

export function writeDecision(key: string, choice: Choice, file = DECISION_FILE): void {
  try {
    const tmp = `${file}.${process.pid}`
    writeFileSync(tmp, JSON.stringify({ key, choice }))
    renameSync(tmp, file)
  } catch {
    // this process still acts on it
  }
}

/** What an answer from the dialog means: no pick at the timeout does `onNoAnswer` (switch by default), Esc stays. */
export function choiceFrom(pick: string | undefined, switchLabel: string, stayLabel: string, waitedMs: number, timeoutMs = ASK_TIMEOUT_MS, onNoAnswer = 'switch'): Choice {
  if (pick === switchLabel) return 'switch'
  if (pick === stayLabel) return 'stay'
  return waitedMs >= timeoutMs - 1000 && onNoAnswer !== 'stay' ? 'switch' : 'stay'
}

// One usage lookup per minute shared by every pi process (the main agent and its subagents).
export function readCachedUsage(file = CACHE_FILE, now = Date.now(), ttl = CACHE_TTL_MS): GoUsage | null {
  try {
    const entry = JSON.parse(readFileSync(file, 'utf8'))
    if (typeof entry?.at !== 'number' || now - entry.at > ttl || now < entry.at) return null
    return entry.usage ?? null
  } catch {
    return null
  }
}

export function writeCachedUsage(usage: GoUsage, file = CACHE_FILE, now = Date.now()): void {
  try {
    mkdirSync(dirname(file), { recursive: true })
    const tmp = `${file}.${process.pid}`
    writeFileSync(tmp, JSON.stringify({ at: now, usage }))
    renameSync(tmp, file)
  } catch {
    // the next process just asks again
  }
}

/** "Oct 19" from an ISO date, for the notice. */
export function shortDate(iso?: string): string {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString('en-US', { day: 'numeric', month: 'short' })
}
