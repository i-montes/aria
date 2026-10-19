// Quota fallback: before each run (TUI, -p and subagents alike) look at the OpenCode Go quota; at
// THRESHOLD percent of the 5-hour or weekly window, a paid Go model is swapped for FREE_MODEL, and
// after the window resets the replaced model comes back. setModel from an extension is not
// persisted, so settings.json keeps the user's default and a new session starts from it.
//
// In the TUI the user is asked first (5-minute countdown; no answer switches, see fallback.ts).
//
// PI_QUOTA_FALLBACK=0 turns it off.
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { fetchGoUsage } from '../jev-router/usage.ts'
import { confirmFreeSwitch } from './ask.ts'
import { decide, freeModel, readCachedUsage, shortDate, switchThreshold, writeCachedUsage, type ModelLike } from './fallback.ts'

export default function (pi: ExtensionAPI) {
  if (process.env.PI_QUOTA_FALLBACK === '0') return
  let switchedFrom: ModelLike | null = null

  const usageNow = async () => {
    const cached = readCachedUsage()
    if (cached) return cached
    const fresh = await fetchGoUsage()
    if (fresh) writeCachedUsage(fresh)
    return fresh
  }

  const check = async (ctx: any) => {
    try {
      const FREE_MODEL = freeModel()
      const d = decide({ usage: await usageNow(), current: ctx.model, switchedFrom, threshold: switchThreshold() })
      if (!d) return
      if (d.kind === 'switch') {
        const free = ctx.modelRegistry.find(FREE_MODEL.provider, FREE_MODEL.id)
        if (!free) return
        const from = ctx.model
        if ((await confirmFreeSwitch(ctx, d, from.id)) === 'stay') return
        if (!(await pi.setModel(free))) return
        switchedFrom = from
        const until = shortDate(d.resetsAt)
        ctx.ui?.notify?.(
          `Go ${d.window} quota at ${d.percent}%: going on with ${FREE_MODEL.id} (free)${until ? ` until ${until}` : ''} instead of ${from.id}.`,
          'warning',
        )
      } else {
        const back = ctx.modelRegistry.find(d.to.provider, d.to.id)
        if (!back || !(await pi.setModel(back))) return
        switchedFrom = null
        ctx.ui?.notify?.(`The Go quota was renewed: back to ${back.id}.`, 'info')
      }
    } catch {
      // keep the current model
    }
  }

  pi.on('session_start', async (_event: any, ctx: any) => {
    await check(ctx)
  })
  pi.on('before_agent_start', async (_event: any, ctx: any) => {
    await check(ctx)
  })
}
