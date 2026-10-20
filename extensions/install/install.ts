// What the `install` tool works with, kept apart from index.ts so it can be tested without Pi: where a skill link points,
// which skills a downloaded folder holds, the MCP entry to write and how the confirmation shows it (secrets masked).
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { basename, dirname, join } from 'node:path'

/** A skill source: a git repository (with an optional ref and folder inside it), or a single `SKILL.md` to download. */
export type SkillSource = { kind: 'git'; repo: string; ref?: string; path: string } | { kind: 'file'; url: string }

/** Reads a skill link: a GitHub repository, folder (`/tree/<ref>/<path>`) or file (`/blob/<ref>/<path>`), a raw
 * `SKILL.md` URL, or any other git URL. Null when it is none of those. */
export function parseSkillSource(link: string): SkillSource | null {
  const text = link.trim()
  if (/^git@[^:]+:.+/.test(text)) return { kind: 'git', repo: text, path: '' }
  let url: URL
  try {
    url = new URL(text)
  } catch {
    return null
  }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') return null
  const parts = url.pathname.split('/').filter(Boolean)
  if (url.hostname === 'github.com' && parts.length >= 2) {
    const repo = `https://github.com/${parts[0]}/${parts[1].replace(/\.git$/, '')}`
    if (parts.length === 2) return { kind: 'git', repo, path: '' }
    if ((parts[2] === 'tree' || parts[2] === 'blob') && parts.length >= 4) {
      let path = parts.slice(4).join('/')
      // A link to the SKILL.md itself names its folder.
      if (parts[2] === 'blob' && /(^|\/)SKILL\.md$/i.test(path)) path = path.includes('/') ? dirname(path) : ''
      return { kind: 'git', repo, ref: parts[3], path }
    }
    return null
  }
  if (/\/SKILL\.md$/i.test(url.pathname)) return { kind: 'file', url: url.toString() }
  if (url.pathname.endsWith('.git')) return { kind: 'git', repo: url.toString(), path: '' }
  return null
}

/** A skill's name: the `name:` of its front matter, else its folder's name. */
export function skillName(skillMd: string, folder: string): string {
  const front = /^---\r?\n([\s\S]*?)\r?\n---/.exec(skillMd)
  const name = front && /^name:\s*["']?([^"'\r\n]+?)["']?\s*$/m.exec(front[1])
  return (name ? name[1] : basename(folder)).trim()
}

/** Pi's rule for skill names (Agent Skills): lowercase letters, digits and hyphens. */
export const validSkillName = (name: string): boolean => /^[a-z0-9][a-z0-9-]{0,63}$/.test(name)

const SKIP = new Set(['.git', 'node_modules'])

/** The skill folders under `root` (each holds a `SKILL.md`), at most `depth` levels down; a skill's own subfolders are
 * not searched. */
export function findSkills(root: string, depth = 4): string[] {
  if (existsSync(join(root, 'SKILL.md'))) return [root]
  if (depth === 0) return []
  let entries: string[]
  try {
    entries = readdirSync(root)
  } catch {
    return []
  }
  return entries
    .filter((name) => !SKIP.has(name) && statSync(join(root, name), { throwIfNoEntry: false })?.isDirectory())
    .sort()
    .flatMap((name) => findSkills(join(root, name), depth - 1))
}

/** Each skill folder with its name. */
export const namedSkills = (folders: string[]): { name: string; folder: string }[] =>
  folders.map((folder) => ({ name: skillName(readFileSync(join(folder, 'SKILL.md'), 'utf8'), folder), folder }))

export interface McpParams {
  name: string
  url?: string
  headers?: Record<string, string>
  command?: string
  args?: string[]
  env?: Record<string, string>
  description?: string
}

/** Pi's rule for server names. */
export const validServerName = (name: string): boolean => /^[A-Za-z0-9_-]{1,64}$/.test(name)

/** The `mcpServers` entry, or the reason it cannot be one. */
export function mcpEntry(p: McpParams): { entry: Record<string, unknown> } | { error: string } {
  if (!validServerName(p.name)) return { error: `"${p.name}" is not a valid server name (letters, digits, _ and - only).` }
  if (!!p.url === !!p.command) return { error: 'Give either `url` (a remote server) or `command` (a local one), not both.' }
  if (p.url) {
    if (!/^https?:\/\//.test(p.url)) return { error: `${p.url} is not an http(s) URL.` }
    if (/\/sse\/?$/.test(p.url)) return { error: 'Pi does not support SSE servers; most of them also serve streamable HTTP, often at /mcp.' }
    if (p.args?.length || (p.env && Object.keys(p.env).length)) return { error: '`args` and `env` are for a local server (`command`).' }
  } else if (p.headers && Object.keys(p.headers).length) return { error: '`headers` are for a remote server (`url`).' }
  if (p.command && /\s/.test(p.command.trim())) return { error: '`command` is one executable; put its arguments in `args`.' }
  const entry: Record<string, unknown> = p.url
    ? { url: p.url, ...(p.headers && Object.keys(p.headers).length ? { headers: p.headers } : {}) }
    : { command: p.command!.trim(), ...(p.args?.length ? { args: p.args } : {}), ...(p.env && Object.keys(p.env).length ? { env: p.env } : {}) }
  if (p.description?.trim()) entry.description = p.description.trim()
  return { entry }
}

/** A secret as the confirmation shows it: its start, never the whole value. A `${VAR}` reference is not a secret. */
export function mask(value: string): string {
  if (/^\$\{\w+\}$/.test(value)) return value
  const bearer = /^(Bearer|Basic|Token)\s+(.*)$/i.exec(value)
  if (bearer) return `${bearer[1]} ${mask(bearer[2])}`
  return value.length <= 8 ? '••••' : `${value.slice(0, 4)}…`
}

/** What the confirmation shows for an MCP entry: one line per part, secrets masked. */
export function describeMcp(name: string, entry: Record<string, any>): string {
  const pairs = (o: Record<string, string> | undefined) => Object.entries(o ?? {}).map(([k, v]) => `${k}=${mask(v)}`)
  const head = entry.url ? `${name}: ${entry.url}` : `${name}: ${[entry.command, ...(entry.args ?? [])].join(' ')}`
  const extra = [...pairs(entry.headers), ...pairs(entry.env)]
  return extra.length ? `${head}\n${extra.join('\n')}` : head
}

/** `mcp.json` with the server set (every other key and server kept). */
export function withServer(file: unknown, name: string, entry: Record<string, unknown>): Record<string, unknown> {
  const object = file && typeof file === 'object' && !Array.isArray(file) ? { ...(file as Record<string, unknown>) } : {}
  const servers = object.mcpServers && typeof object.mcpServers === 'object' && !Array.isArray(object.mcpServers) ? { ...(object.mcpServers as Record<string, unknown>) } : {}
  servers[name] = entry
  return { ...object, mcpServers: servers }
}

/** Where Aria keeps its MCP files (`~/.aria`, which `agent/` links to), or the agent folder itself outside Aria. */
export const mcpHome = (agent: string, ariaSettings = process.env.ARIA_SETTINGS): string => (ariaSettings ? dirname(ariaSettings) : agent)

/** A Pi package source `pi install` takes: `npm:`, `git:` or an https git URL. */
export const validPackage = (source: string): boolean => /^(npm:[@\w./-]+|git:[\w.-]+\/[\w./@-]+|https:\/\/\S+)$/.test(source.trim())
