// A resumed session keeps the tools of the Aria that started it: Pi restores the active tools recorded in the
// transcript, so a tool an extension of Aria registers later (a newer Aria, `/reload`) is loaded but never declared to
// the model, nor callable from codemode. At every session start, the tools of Aria's own extensions that are not active
// are made active again. A subagent (json mode) is left alone: its tools are its expert's, on purpose.
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { join, sep } from 'node:path'
import { agentDir } from '../lib/agent-dir.ts'

interface Tool {
  name: string
  exposure?: string
  sourceInfo?: { path?: string }
}

/** The tools from the extensions in `root` that registering would have activated, and are not active. */
export function missingTools(all: Tool[], active: string[], root: string): string[] {
  const on = new Set(active)
  const prefix = root.endsWith(sep) ? root : root + sep
  return all
    .filter((t) => (t.exposure ?? 'direct') === 'direct' || t.exposure === 'model-only')
    .filter((t) => !on.has(t.name) && (t.sourceInfo?.path ?? '').startsWith(prefix))
    .map((t) => t.name)
}

export default function loadout(pi: ExtensionAPI) {
  pi.on('session_start', (_event: any, ctx: any) => {
    if (ctx?.mode === 'json') return
    const active = pi.getActiveTools()
    const missing = missingTools(pi.getAllTools() as Tool[], active, join(agentDir(), 'extensions'))
    if (missing.length) pi.setActiveTools([...active, ...missing])
  })
}
