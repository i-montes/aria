// Session rotation (the spec's "Rotation"): after a compaction, Aria sends `/aria-rotate` while the tab is quiet. The
// command starts a new session file (its header names the old one as `parentSession`) whose model context is the same
// as before: first the latest compaction's summary, framed as Pi frames it, as a visible `aria-earlier` custom message
// (details: `{ from: <old session id>, summary: <the raw summary> }`), then everything Pi kept, in order (see seed.ts).
// Without a compaction it does nothing. It never starts a model turn. The extensions reload with the new session; the
// switch is marked meanwhile so they keep the session's grants (see ../lib/rotation.ts).
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { markRotating } from '../lib/rotation.ts'
import { COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX, rotationSeed } from './seed.ts'

export default function (pi: ExtensionAPI) {
  pi.registerCommand('aria-rotate', {
    description: "Aria's own: continue this conversation in a new session file that starts from the latest compaction.",
    handler: async (_args: string, ctx: any) => {
      const sm = ctx.sessionManager
      const seed = rotationSeed(sm.getBranch(), sm.getSessionId())
      if (!seed) return
      // The extensions reload with the new session; the mark lets them keep the person's session grants (lib/rotation.ts).
      markRotating(true)
      try {
        await ctx.newSession({
          parentSession: sm.getSessionFile(),
          setup: async (next: any) => {
            const framed = COMPACTION_SUMMARY_PREFIX + seed.summary + COMPACTION_SUMMARY_SUFFIX
            next.appendCustomMessageEntry('aria-earlier', framed, true, { from: seed.from, summary: seed.summary })
            for (const item of seed.items) {
              if (item.kind === 'message') next.appendMessage(item.message)
              else next.appendCustomMessageEntry(item.customType, item.content, item.display, item.details)
            }
          },
        })
      } finally {
        markRotating(false)
      }
    },
  })
}
