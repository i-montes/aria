// Jev, the classifier the router and the sandbox ask, model by model. The classifiers Pi can reach (those with a key or a
// sign-in) are tried in order: the free Jev, the paid Jev, Jev through other providers, other free classifiers, the rest.
// Only the one that is spent steps aside: a 429 pauses it until the hour its retry delay names, no funds (402) pause it
// an hour, and the same question goes to the next one; each comes back at its own hour. The pauses are a file in the
// agent folder, so every session skips a spent model at once. Nothing else changes: the chat models are not touched.
// (This folder has no index.ts: Pi does not load it as an extension.)
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { agentDir } from './agent-dir.ts'

export const FREE_JEV = { provider: 'opencode', id: 'jev-1.13-free' } as const
export const PAID_JEV = { provider: 'opencode', id: 'jev-1.13' } as const

type ModelLike = { provider: string; id: string }

const key = (m: ModelLike): string => `${m.provider}/${m.id}`
const isJev = (m: ModelLike): boolean => /(^|\/|~)jev\b/.test(m.id)
const isFree = (m: ModelLike): boolean => /(-free|:free)$/.test(m.id)

/** The order classifiers are asked in. */
export function chain<M extends ModelLike>(available: readonly M[]): M[] {
  const rank = (m: M): number =>
    key(m) === key(FREE_JEV) ? 0 : key(m) === key(PAID_JEV) ? 1 : isJev(m) ? 2 : isFree(m) ? 3 : 4
  return [...available].sort((a, b) => rank(a) - rank(b))
}

/** Where the pauses are kept: `{ "<provider>/<id>": <until, ms> }`. */
export const pauseFile = (): string => join(agentDir(), 'jev-paused.json')

/** How long (s) a failed answer keeps its model aside: a 429 its retry delay (an hour when it names none), no funds
 * (402) an hour; null for an answer, or a failure that is not about quota or money. */
export function failureWait(result: any): number | null {
  if (result?.stopReason !== 'error') return null
  const text = String(result.errorMessage ?? '')
  if (/\b429\b/.test(text)) {
    const wait = text.match(/(\d+)\s*s retry delay/)
    return wait ? Number(wait[1]) : 3600
  }
  if (/\b402\b|insufficient (account )?funds/i.test(text)) return 3600
  return null
}

/** The models set aside, and until when (ms); empty when the file is missing or cannot be read. */
export function pauses(file: string = pauseFile()): Record<string, number> {
  try {
    const kept = JSON.parse(readFileSync(file, 'utf8'))
    return kept && typeof kept === 'object' ? kept : {}
  } catch {
    return {}
  }
}

/** While classifiers are set aside, until when (ms) the first of them is; null when none is. */
export function offUntil(file: string = pauseFile(), now: number = Date.now()): number | null {
  const waiting = Object.values(pauses(file)).filter((until) => until > now)
  return waiting.length ? Math.min(...waiting) : null
}

function save(file: string, kept: Record<string, number>): void {
  try {
    mkdirSync(dirname(file), { recursive: true })
    writeFileSync(file, JSON.stringify(kept))
  } catch {
    // unwritable: this question still moves on; the next one meets the same failure first
  }
}

/** The hour, as the person's clock shows it. */
export const localTime = (ms: number): string => new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

type Spent = { model: ModelLike; until: number; funds: boolean }

/** What the person is told when classifiers step aside: which, until when, and who answers now. */
export function stepNote(spent: Spent[], answered: ModelLike | null, kept: Record<string, number>): string {
  const which = spent
    .map((s) => `${s.model.id} ${s.funds ? 'has no funds' : 'is out of quota'} until ${localTime(s.until)}`)
    .join('; ')
  if (answered) return `Jev: ${which}. ${answered.id} answers meanwhile.`
  const back = Math.min(...Object.values(kept))
  return `Jev: ${which}; no classifier is left: routing is off until ${localTime(back)}.`
}

type Registry = {
  getAvailableOfType: (type: 'classifier') => Promise<readonly any[]>
  classify: (model: any, request: any, options?: any) => Promise<any>
}

/** Asks `request` of the first classifier in `chain` that is not set aside, setting aside those that fail for quota or
 * money. Null when none answers. `onNote` hears, once per step aside, what changed. */
export async function classifyWithJev(
  registry: Registry,
  request: any,
  options: any,
  { file = pauseFile(), now = Date.now, onNote }: { file?: string; now?: () => number; onNote?: (note: string) => void } = {},
): Promise<any> {
  let available: readonly any[] = []
  try {
    available = await registry.getAvailableOfType('classifier')
  } catch {
    return null
  }
  const kept = pauses(file)
  const spent: Spent[] = []
  let answer: any = null
  let answered: ModelLike | null = null
  for (const model of chain(available)) {
    if (now() < (kept[key(model)] ?? 0)) continue
    const result = await registry.classify(model, request, options)
    const wait = failureWait(result)
    if (wait !== null) {
      const until = now() + wait * 1000
      kept[key(model)] = until
      spent.push({ model, until, funds: !/\b429\b/.test(String(result.errorMessage)) })
      continue
    }
    if (result?.stopReason === 'error') continue // not about quota or money: the next one, for this question only
    answer = result
    answered = model
    break
  }
  if (spent.length) {
    for (const [name, until] of Object.entries(kept)) if (until <= now()) delete kept[name]
    save(file, kept)
    onNote?.(stepNote(spent, answered, kept))
  }
  return answer
}
