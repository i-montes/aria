// Auto-compaction at a share of the context window (see autocompact.ts). Each session start brings
// settings.json in line with the saved share (new models get their reserve; Pi reads it on the next
// start). `/autocompact <0-100>` changes the share and reloads Pi so it applies right away;
// `/autocompact` alone shows it.
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { compactPercent, DEFAULT_PERCENT, noteApplied, parsePercent, savePercent, syncSettings } from './autocompact.ts'

export default function (pi: ExtensionAPI) {
  const models = (ctx: any) => {
    try {
      return ctx.modelRegistry.getAvailable()
    } catch {
      return []
    }
  }

  pi.on('session_start', (_event: any, ctx: any) => {
    try {
      const percent = compactPercent()
      syncSettings(models(ctx), percent)
      noteApplied(percent)
    } catch {
      // settings.json unreadable: leave it to Pi
    }
  })

  pi.registerCommand('autocompact', {
    description: `Automatic compaction at this % of the context window (0 = off, default ${DEFAULT_PERCENT}).`,
    handler: async (args: string, ctx: any) => {
      const now = compactPercent()
      if (!args.trim()) {
        ctx.ui.notify(now === 0 ? 'Auto-compaction is off. /autocompact <1-100> turns it on.' : `Auto-compaction at ${now}% of the context window.`, 'info')
        return
      }
      const percent = parsePercent(args)
      if (percent === null) {
        ctx.ui.notify('Use /autocompact <0-100>, e.g. /autocompact 40 (0 = off).', 'error')
        return
      }
      try {
        syncSettings(models(ctx), percent)
        noteApplied(percent) // before the setting: Aria then sees the share already applied and does not restart Pi
        savePercent(percent)
      } catch (e) {
        ctx.ui.notify(`Could not update settings.json: ${e instanceof Error ? e.message : String(e)}`, 'error')
        return
      }
      ctx.ui.notify(percent === 0 ? 'Auto-compaction off.' : `Auto-compaction at ${percent}% of the context window.`, 'info')
      await ctx.reload()
    },
  })
}
