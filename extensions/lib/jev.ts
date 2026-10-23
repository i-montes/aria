// Jev, the classifier the router and the sandbox ask. The free model answers while it has quota; when it says its quota
// is spent (a 429 with the wait it asks for), the paid one (same OpenCode key, a fraction of a cent per question) takes
// over until that hour, and the question that met the 429 is asked again there. The pause is a file in the agent folder,
// so every session goes to the paid Jev at once and back to the free one at the same hour.
// (This folder has no index.ts: Pi does not load it as an extension.)
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { agentDir } from './agent-dir.ts'

export const FREE_JEV = { provider: 'opencode', id: 'jev-1.13-free' } as const
export const PAID_JEV = { provider: 'opencode', id: 'jev-1.13' } as const

/** Where the pause is kept: `{ "until": <ms> }`. */
export const pauseFile = (): string => join(agentDir(), 'jev-free-paused.json')

/** The wait (s) a quota answer asks for: its `retry delay`, or an hour when it names none; null for any other result. */
export function quotaWait(result: any): number | null {
  if (result?.stopReason !== 'error') return null
  const text = String(result.errorMessage ?? '')
  if (!/\b429\b/.test(text)) return null
  const wait = text.match(/(\d+)\s*s retry delay/)
  return wait ? Number(wait[1]) : 3600
}

/** Until when (ms) the free Jev is paused; 0 when it is not, or the file cannot be read. */
export function pausedUntil(file: string = pauseFile()): number {
  try {
    const until = JSON.parse(readFileSync(file, 'utf8'))?.until
    return typeof until === 'number' ? until : 0
  } catch {
    return 0
  }
}

function pause(file: string, until: number): void {
  try {
    mkdirSync(dirname(file), { recursive: true })
    writeFileSync(file, JSON.stringify({ until }))
  } catch {
    // unwritable: this session still asks the paid Jev now, and meets the 429 again next time
  }
}

/** The hour, as the person's clock shows it. */
export const localTime = (ms: number): string => new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

/** What the person is told once the free Jev's quota is spent: until when, and whether the paid one answered. */
export function pauseNote(until: number, paid: any): string {
  const back = localTime(until)
  if (paid?.stopReason !== 'error') return `Jev's free quota is spent: the paid Jev answers until ${back}.`
  const why = String(paid?.errorMessage ?? 'no answer').replace(/^.*"message":"([^"]+)".*$/s, '$1').slice(0, 120)
  return `Jev's free quota is spent until ${back}, and the paid Jev did not answer (${why}): routing is off until then.`
}

type Registry = {
  findOfType: (type: string, provider: string, id: string) => any
  classify: (model: any, request: any, options?: any) => Promise<any>
}

/** Asks Jev `request`: the free model, or the paid one while the free one is paused. Null when neither is there.
 * `onPause` hears the hour the free Jev comes back and the paid Jev's answer, once, when this question is the one that
 * met the free one's 429. */
export async function classifyWithJev(
  registry: Registry,
  request: any,
  options: any,
  {
    file = pauseFile(),
    now = Date.now,
    onPause,
  }: { file?: string; now?: () => number; onPause?: (until: number, paid: any) => void } = {},
): Promise<any> {
  const paid = () => registry.findOfType('classifier', PAID_JEV.provider, PAID_JEV.id)
  if (now() < pausedUntil(file)) {
    const model = paid()
    return model ? registry.classify(model, request, options) : null
  }
  const free = registry.findOfType('classifier', FREE_JEV.provider, FREE_JEV.id)
  if (!free) return null
  const result = await registry.classify(free, request, options)
  const wait = quotaWait(result)
  if (wait === null) return result
  const until = now() + wait * 1000
  pause(file, until)
  const model = paid()
  const answer = model ? await registry.classify(model, request, options) : result
  onPause?.(until, answer)
  return answer
}
