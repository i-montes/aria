// Session rotation (the spec's "Rotation"): after a compaction, Aria sends `/aria-rotate` while the tab is quiet. The
// command starts a new session file (its header names the old one as `parentSession`) that begins with the latest
// compaction's summary, as a visible `aria-earlier` custom message (details: `{ from: <old session id> }`), followed
// by the messages Pi kept, in order: the model's context is the same as before, in a small file. Without a compaction
// it does nothing. It never starts a model turn.
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { rotationSeed } from './seed.ts'

export default function (pi: ExtensionAPI) {
  pi.registerCommand('aria-rotate', {
    description: "Aria's own: continue this conversation in a new session file that starts from the latest compaction.",
    handler: async (_args: string, ctx: any) => {
      const sm = ctx.sessionManager
      const seed = rotationSeed(sm.getBranch(), sm.getSessionId())
      if (!seed) return
      await ctx.newSession({
        parentSession: sm.getSessionFile(),
        setup: async (next: any) => {
          next.appendCustomMessageEntry('aria-earlier', seed.summary, true, { from: seed.from })
          for (const message of seed.messages) next.appendMessage(message)
        },
      })
    },
  })
}
