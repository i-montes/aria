// What fills the context, measured before every run (measure.ts) and published as the status `aria-context`, which
// Aria's terminal draws with `/context`.
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { measure } from './measure.ts'

export default function context(pi: ExtensionAPI) {
  pi.on('before_agent_start', (event: any, ctx: any) => {
    if (ctx?.mode === 'json') return
    try {
      const active = new Set(pi.getActiveTools())
      const tools = (pi.getAllTools() as any[]).filter((t) => active.has(t.name))
      const skills = Array.isArray(event?.systemPromptOptions?.skills) ? event.systemPromptOptions.skills : []
      const usage = ctx?.getContextUsage?.()
      const parts = measure({
        systemPrompt: String(event?.systemPrompt ?? ''),
        skills,
        tools,
        memory: '',
        usedTokens: typeof usage?.tokens === 'number' ? usage.tokens : null,
      })
      ctx?.ui?.setStatus?.('aria-context', JSON.stringify(parts))
    } catch {
      // A measure is never worth failing a run for.
    }
  })
}
