// The settings Aria gives its Pi's extensions: `aria.json` in the agent directory, written by Aria from its own
// settings (crates/aria-core/src/agent_config.rs). The owner's own Pi has no such file, and then every value is the
// one the extensions have always used. A field that is missing or of the wrong type falls back on its own, so an
// older or newer Aria never breaks an extension. Read on every call: Pi restarts when the settings change anyway.
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { agentDir } from './agent-dir.ts'

export interface Model {
  provider: string
  id: string
}

export interface AriaConfig {
  version: number
  models: { plan: Model; planThinking: string; allowPayPerUse: boolean }
  quota: { switchAt: number; planSwitchAt: number; freeModel: Model; askFirst: boolean; countdownMs: number; onNoAnswer: string }
  router: { enabled: boolean }
  effort: { mode: string; fixedLevel: string | null; maxOnFree: boolean }
  compaction: { percent: number }
  style: string
  sandbox: { extraWritable: string[]; askToLeave: boolean; jevReview: boolean }
  subagents: { saveSessions: boolean; turnLimit: number | null; planExecute: boolean }
}

/** Today's values: the ones the extensions had in their code. */
export const DEFAULTS: AriaConfig = {
  version: 1,
  models: { plan: { provider: 'opencode-go', id: 'glm-5.3' }, planThinking: 'high', allowPayPerUse: false },
  quota: { switchAt: 98, planSwitchAt: 90, freeModel: { provider: 'opencode-go', id: 'cat-jam-free' }, askFirst: true, countdownMs: 5 * 60_000, onNoAnswer: 'switch' },
  router: { enabled: true },
  effort: { mode: 'auto', fixedLevel: null, maxOnFree: true },
  compaction: { percent: 40 },
  style: 'very-short',
  sandbox: { extraWritable: [], askToLeave: true, jevReview: true },
  subagents: { saveSessions: false, turnLimit: null, planExecute: true },
}

/** Fields that may also be null. */
const NULLABLE: Record<string, 'string' | 'number'> = { 'effort.fixedLevel': 'string', 'subagents.turnLimit': 'number' }

const isObject = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value)

function merge(fallback: unknown, value: unknown, path: string): unknown {
  if (path in NULLABLE) return value === null || typeof value === NULLABLE[path] ? value : fallback
  if (Array.isArray(fallback)) return Array.isArray(value) ? value.filter((item) => typeof item === 'string') : fallback
  if (isObject(fallback)) {
    const source = isObject(value) ? value : {}
    return Object.fromEntries(Object.entries(fallback).map(([key, inner]) => [key, merge(inner, source[key], path ? `${path}.${key}` : key)]))
  }
  return typeof value === typeof fallback && !(typeof value === 'number' && !Number.isFinite(value)) ? value : fallback
}

/** Is this Pi Aria's (it has `aria.json`)? The owner's own Pi keeps its own files (effort, autocompact). */
export const hasAriaConfig = (): boolean => existsSync(join(agentDir(), 'aria.json'))

export function ariaConfig(): AriaConfig {
  let found: unknown = {}
  try {
    found = JSON.parse(readFileSync(join(agentDir(), 'aria.json'), 'utf8'))
  } catch {
    // no file (the owner's own Pi) or a broken one: today's values
  }
  return merge(DEFAULTS, found, '') as AriaConfig
}
