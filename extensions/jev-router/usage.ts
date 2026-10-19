// How much of the OpenCode Go quota is spent, from GET https://opencode.ai/zen/go/v1/usage
// ({ usage: { rolling|weekly|monthly: { status, percent, resetsAt } } }). The key is the Go key
// Pi already stores in ~/.pi/agent/auth.json. Any failure returns null and nothing changes.
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { agentDir } from '../lib/agent-dir.ts'

export const GO_USAGE_URL = 'https://opencode.ai/zen/go/v1/usage'

export interface Window {
  status: string
  percent: number
  resetsAt?: string
}
export interface GoUsage {
  rolling: Window
  weekly: Window
  monthly: Window
}

export function parseGoUsage(body: unknown): GoUsage | null {
  const usage = (body as { usage?: Record<string, unknown> } | null)?.usage
  if (!usage || typeof usage !== 'object') return null
  const out: Partial<GoUsage> = {}
  for (const name of ['rolling', 'weekly', 'monthly'] as const) {
    const w = usage[name] as { status?: unknown; percent?: unknown; resetsAt?: unknown } | undefined
    if (typeof w?.percent !== 'number') return null
    out[name] = {
      status: typeof w.status === 'string' ? w.status : 'ok',
      percent: w.percent,
      ...(typeof w.resetsAt === 'string' ? { resetsAt: w.resetsAt } : {}),
    }
  }
  return out as GoUsage
}

export function goKey(authFile = join(agentDir(), 'auth.json')): string | null {
  try {
    const key = JSON.parse(readFileSync(authFile, 'utf8'))['opencode-go']?.key
    return typeof key === 'string' && key.trim() ? key.trim() : null
  } catch {
    return null
  }
}

export async function fetchGoUsage(timeoutMs = 2000, fetcher: typeof fetch = fetch): Promise<GoUsage | null> {
  const key = goKey()
  if (!key) return null
  try {
    const res = await fetcher(GO_USAGE_URL, { headers: { authorization: `Bearer ${key}` }, signal: AbortSignal.timeout(timeoutMs) })
    if (!res.ok) return null
    return parseGoUsage(await res.json())
  } catch {
    return null
  }
}

/** The window that is at or past `threshold` percent (5-hour first), or null. */
export function exhausted(usage: GoUsage, threshold: number): { name: '5-hour' | 'weekly'; window: Window } | null {
  if (usage.rolling.percent >= threshold || usage.rolling.status !== 'ok') return { name: '5-hour', window: usage.rolling }
  if (usage.weekly.percent >= threshold || usage.weekly.status !== 'ok') return { name: 'weekly', window: usage.weekly }
  return null
}
