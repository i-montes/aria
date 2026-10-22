// Connectors: every call to an MCP tool that writes asks first. A read runs; a write runs, asks with the chat's
// permission card, or is refused, by the access level the person gave its connector in Aria's settings
// (`aria.json` `connectors`, see ../lib/config.ts: 'read' refuses writes, 'ask' is the default, 'allow' runs them).
// Calls from a codemode script go through the same `tool_call` hook (with `parentToolCallId`), so they are gated
// the same. Aria's memory (`engram`) is not a connector and is never gated. "Yes, and don't ask again" lasts until
// the next session_start, like the sandbox's grants (a rotation keeps it).
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { ariaConfig, type AccessLevel } from '../lib/config.ts'
import { handOver, takeOver } from '../lib/rotation.ts'
import { canAsk } from '../sandbox/card.ts'
import { connectorCard, preview } from './card.ts'
import { actionOf, isMcpTool, isWrite, serverOf, type Annotations } from './classify.ts'

const ONCE = 'Yes, just this once'
const NO = 'No'
const NEVER_GATED = new Set(['engram'])
/** The key the session's grants are handed over under at a rotation. */
const GRANTS = 'connectors.grants'

interface ToolFacts {
  annotations?: Annotations
  /** The server's own name (Pi's namespace alias), which may hold a `-` the tool name turned into `_`. */
  server?: string
}

/** How long a look at Pi's tool list holds: a server that reconnects may bring changed annotations. */
const FACTS_TTL_MS = 60_000

/** The name a server has in a tool name: anything outside `[A-Za-z0-9_]` becomes `_`. */
const asInToolName = (server: string) => server.replace(/[^A-Za-z0-9_]/g, '_')

/** "notion" → "Notion", "my-docs" → "My Docs". */
// The same as `displayName` in ui/onboarding/view.js (the settings' access rows): change both together.
const displayName = (server: string) =>
  server
    .split(/[-_]+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ') || server

/** The level set for this server: by its exact name, else by the one key that reads the same in a tool name; none,
 * or more than one such key, is 'ask'. */
function levelOf(server: string): AccessLevel {
  const levels = ariaConfig().connectors
  if (Object.hasOwn(levels, server)) return levels[server]
  const keys = Object.keys(levels).filter((k) => asInToolName(k) === asInToolName(server))
  return keys.length === 1 ? levels[keys[0]] : 'ask'
}

export default function connectors(pi: ExtensionAPI) {
  let grants = new Set<string>()
  let facts = new Map<string, ToolFacts>()
  let readAt = 0

  /** What Pi knows of a tool. Pi's list is read again when the name is not in the last look (a server connected later)
   * or that look is older than a minute (a server that reconnected re-registers its tools). Null when Pi does not list it. */
  const factsOf = (name: string): ToolFacts | null => {
    if (!facts.has(name) || Date.now() - readAt > FACTS_TTL_MS) {
      try {
        const fresh = new Map<string, ToolFacts>()
        for (const tool of (pi as any).getAllTools?.() ?? []) {
          fresh.set(tool.name, { annotations: tool.annotations, server: tool.namespace?.aliases?.[0] })
        }
        facts = fresh
        readAt = Date.now()
      } catch {
        // no tool list: the name decides
      }
    }
    return facts.get(name) ?? null
  }

  pi.on('tool_call', async (event: any, ctx: any) => {
    const name = String(event?.toolName ?? '')
    if (!isMcpTool(name)) return
    const inName = serverOf(name)
    if (inName === '*') return // a resource tool: they only read
    const known = factsOf(name)
    // The server's own name; else the one in the tool name; else (Pi cut the name before its `__`) the part after
    // `mcp__`, gated on Ask whatever the settings say.
    const server = known?.server ?? inName ?? name.slice('mcp__'.length)
    if (NEVER_GATED.has(server) || (inName && NEVER_GATED.has(inName))) return
    // A tool Pi does not list is a write, whatever its name says.
    if (known && !isWrite(name, known.annotations)) return
    const level = known?.server || inName ? levelOf(server) : 'ask'
    if (level === 'allow') return
    const connector = known?.server || inName ? displayName(server) : server
    // A cut name has no readable tool part left: the action names the tool.
    const action = inName ? actionOf(name) : `use ${name}`
    if (level === 'read') return { block: true, reason: `${connector} is read only (the person set it so in Aria's settings): ${action} did not run.` }
    const grant = `${server}\0${name}`
    if (grants.has(grant)) return
    if (!canAsk(ctx)) return { block: true, reason: `${connector} wants to ${action}, which needs the person's approval, and there is nobody here to ask.` }
    const forSession = `Yes, and don't ask again for ${connector} ${action} this session`
    const options = [ONCE, forSession, NO]
    const detail = preview(event.input)
    // In Aria's dock the question is the chat's permission card; in the terminal, the same words as text.
    const title = ctx.mode === 'rpc' ? connectorCard(connector, action, detail, options) : `${connector} wants to ${action}\n${detail}`
    let pick: string | undefined
    try {
      pick = await ctx.ui.select(title, options, { signal: ctx.signal })
    } catch {
      pick = undefined // the question was closed (the run stopped, the dialog timed out)
    }
    if (pick === forSession) grants.add(grant)
    if (pick === ONCE || pick === forSession) return
    if (pick === NO) return { block: true, reason: `The person said no: ${connector} ${action} did not run.` }
    return { block: true, reason: `The question was closed without an answer: ${connector} ${action} did not run.` }
  })

  // A new session (startup, /new, /resume, fork) starts with no grants; Aria's rotation is the same conversation, so it
  // keeps them (../lib/rotation.ts). A reconnected server may have new tools either way.
  pi.on('session_shutdown', () => handOver(GRANTS, grants))
  pi.on('session_start', (event: any) => {
    grants = takeOver<Set<string>>(GRANTS, event) ?? new Set()
    facts = new Map()
  })
}
