// What an MCP call is: its server, whether it writes, and its action in words. Pi names a server's tools
// `mcp__<server>__<tool>`, with every character outside `[A-Za-z0-9_]` turned into `_` and, when two names collide
// or the name is longer than 64, an `_<8 hex>` hash at the end (Pi 1.3.0 `extensions/mcp/tools.js`,
// `createMcpToolName`). Its resource tools (`list_mcp_resources`, …) reach every server and only read.

export const RESOURCE_TOOLS: ReadonlySet<string> = new Set(['list_mcp_resources', 'list_mcp_resource_templates', 'read_mcp_resource'])

/** A tool name whose first word is one of these only reads. */
const READ_WORDS: ReadonlySet<string> = new Set(['search', 'fetch', 'get', 'list', 'read', 'query', 'find', 'retrieve', 'lookup', 'describe', 'view', 'show', 'count', 'check'])

export interface Annotations {
  readOnlyHint?: boolean
  destructiveHint?: boolean
}

/** Is this an MCP tool (Pi's `isMcpToolName`)? Every `mcp__` name is, even one whose `__` separator Pi cut off. */
export const isMcpTool = (toolName: string): boolean => toolName.startsWith('mcp__') || RESOURCE_TOOLS.has(toolName)

/** The server of an MCP tool (as it is in the tool name), `'*'` for a resource tool, null for any other tool and for an
 * `mcp__` name Pi cut so short that its `__` separator is gone (a server name of 50 characters or more). */
export function serverOf(toolName: string): string | null {
  if (RESOURCE_TOOLS.has(toolName)) return '*'
  const m = /^mcp__(.+?)__(.+)$/.exec(toolName)
  return m ? m[1] : null
}

/** The tool's own part of the name: no `mcp__<server>__`, no collision hash, no `<server>-`/`<server>_` prefix. */
function toolPart(toolName: string): string {
  const m = /^mcp__(.+?)__(.+)$/.exec(toolName)
  if (!m) return toolName
  const [, server, rest] = m
  let tool = rest.replace(/_[0-9a-f]{8}$/, '')
  const prefix = new RegExp(`^${server.replace(/[^A-Za-z0-9]/g, '[-_]')}[-_]`, 'i')
  if (prefix.test(tool) && tool.replace(prefix, '')) tool = tool.replace(prefix, '')
  return tool
}

/** The words of a name: split on `-`, `_`, spaces and camelCase. */
const words = (name: string): string[] =>
  name
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .split(/[-_\s]+/)
    .filter(Boolean)

/** Does this call change something? Annotations decide first; without them, the name; unknown means write. */
export function isWrite(toolName: string, annotations?: Annotations): boolean {
  if (RESOURCE_TOOLS.has(toolName)) return false
  if (annotations?.readOnlyHint === true) return false
  if (annotations?.readOnlyHint === false || annotations?.destructiveHint === true) return true
  const first = words(toolPart(toolName))[0]?.toLowerCase()
  return !first || !READ_WORDS.has(first)
}

/** The action in words: the tool's name made readable ("mcp__notion__notion_create_pages" → "create pages"). Pi's tool
 * list carries no MCP title, so the name is all there is. */
export function actionOf(toolName: string): string {
  return words(toolPart(toolName)).join(' ').toLowerCase()
}
