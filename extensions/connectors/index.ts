// Connectors: every call to an MCP tool that writes asks first. A read runs; a write runs, asks with the chat's
// permission card, or is refused, by the access level the person gave its connector in Aria's settings
// (`aria.json` `connectors`, see ../lib/config.ts: 'read' refuses writes, 'ask' is the default, 'allow' runs them).
// Calls from a codemode script go through the same `tool_call` hook (with `parentToolCallId`), so they are gated
// the same. Aria's memory (`engram`) is not a connector and is never gated. "Yes, and don't ask again" lasts until
// the next session_start, like the sandbox's grants.
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { ariaConfig, type AccessLevel } from '../lib/config.ts'
import { canAsk } from '../sandbox/card.ts'
import { connectorCard, preview } from './card.ts'
import { actionOf, isWrite, serverOf, type Annotations } from './classify.ts'

const ONCE = 'Yes, just this once'
const NO = 'No'
const NEVER_GATED = new Set(['engram'])

interface ToolFacts {
  annotations?: Annotations
  /** The server's own name (Pi's namespace alias), which may hold a `-` the tool name turned into `_`. */
  server?: string
  title?: string
}

/** The name a server has in a tool name: anything outside `[A-Za-z0-9_]` becomes `_`. */
const asInToolName = (server: string) => server.replace(/[^A-Za-z0-9_]/g, '_')

/** "notion" → "Notion". */
const displayName = (server: string) => server.charAt(0).toUpperCase() + server.slice(1)

/** The level set for this server: by its own name, or by a name that reads the same in a tool name. */
function levelOf(server: string): AccessLevel {
  const levels = ariaConfig().connectors
  if (levels[server]) return levels[server]
  const key = Object.keys(levels).find((k) => asInToolName(k) === asInToolName(server))
  return key ? levels[key] : 'ask'
}

export default function connectors(pi: ExtensionAPI) {
  let grants = new Set<string>()
  const facts = new Map<string, ToolFacts>()

  /** What Pi knows of a tool, looked up once per name; a name not seen yet looks again (a server connected later). */
  const factsOf = (name: string): ToolFacts => {
    if (!facts.has(name)) {
      try {
        for (const tool of (pi as any).getAllTools?.() ?? []) {
          facts.set(tool.name, { annotations: tool.annotations, server: tool.namespace?.aliases?.[0], title: tool.title ?? tool.annotations?.title })
        }
      } catch {
        // no tool list: the name decides
      }
    }
    return facts.get(name) ?? {}
  }

  pi.on('tool_call', async (event: any, ctx: any) => {
    const name = String(event?.toolName ?? '')
    const inName = serverOf(name)
    if (!inName || inName === '*') return // not an MCP tool, or a resource tool (they only read)
    const known = factsOf(name)
    const server = known.server ?? inName
    if (NEVER_GATED.has(server) || NEVER_GATED.has(inName)) return
    if (!isWrite(name, known.annotations)) return
    const level = levelOf(server)
    if (level === 'allow') return
    const connector = displayName(server)
    const action = actionOf(name, known.title)
    if (level === 'read') return { block: true, reason: `${connector} is read only (the person set it so in Aria's settings): ${action} did not run.` }
    const grant = `${server}\0${name}`
    if (grants.has(grant)) return
    if (!canAsk(ctx)) return { block: true, reason: `${connector} wants to ${action}, which needs the person's approval, and there is nobody here to ask.` }
    const forSession = `Yes, and don't ask again for ${connector} ${action} this session`
    const options = [ONCE, forSession, NO]
    const detail = preview(event.input)
    // In Aria's dock the question is the chat's permission card; in the terminal, the same words as text.
    const title = ctx.mode === 'rpc' ? connectorCard(connector, action, detail, options) : `${connector} wants to ${action}\n${detail}`
    const pick = await ctx.ui.select(title, options, { signal: ctx.signal })
    if (pick === forSession) grants.add(grant)
    if (pick === ONCE || pick === forSession) return
    return { block: true, reason: `The person said no: ${connector} ${action} did not run.` }
  })

  // A new session (startup, /new, /resume, fork) starts with no grants; a reconnected server may have new tools.
  pi.on('session_start', () => {
    grants = new Set()
    facts.clear()
  })
}
