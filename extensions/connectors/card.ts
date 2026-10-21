// The chat's permission card for a connector's write: `{"aria":1,"kind":"connector",summary,detail,options}`, the
// same card the sandbox asks with (`../sandbox/card.ts`), so the dock shows it with the same part.
import { sandboxCard } from '../sandbox/card.ts'

const MAX_LINES = 12
const MAX_VALUE = 300

const cut = (text: string) => (text.length > MAX_VALUE ? `${text.slice(0, MAX_VALUE - 1)}…` : text)
const oneLine = (text: string) => text.replace(/\s*\n\s*/g, ' ')
const compact = (value: unknown) => {
  try {
    return JSON.stringify(value) ?? String(value)
  } catch {
    return String(value)
  }
}

/** What the call will send, readable: `key: value` lines, values cut to 300 characters, at most 12 lines. */
export function preview(input: unknown): string {
  if (input === undefined || input === null) return '(no arguments)'
  if (typeof input !== 'object' || Array.isArray(input)) return cut(oneLine(typeof input === 'string' ? input : compact(input)))
  const entries = Object.entries(input as Record<string, unknown>).filter(([, v]) => v !== undefined)
  if (!entries.length) return '(no arguments)'
  const lines = entries.map(([key, value]) => `${key}: ${cut(oneLine(typeof value === 'string' ? value : compact(value)))}`)
  if (lines.length <= MAX_LINES) return lines.join('\n')
  return [...lines.slice(0, MAX_LINES - 1), `… and ${lines.length - MAX_LINES + 1} more`].join('\n')
}

/** The card for `ctx.ui.select(title, options)`; the first option is the main button. */
export function connectorCard(connector: string, action: string, detail: string, options: string[]): string {
  return sandboxCard('connector', { summary: `${connector} wants to ${action}`, detail }, options)
}
