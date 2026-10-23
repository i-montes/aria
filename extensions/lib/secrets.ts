// The person's secrets, as the extensions see them (docs/superpowers/plans/2026-10-22-secrets.md): the dock keeps each
// value in the keyring (`user:<name>`) and the names in `secrets.json` beside the agent folder. A command names one as
// `$ARIA_SECRET_<NAME>` and gets it in its environment (`secretEnv`); every tool output is masked (`maskText`) so that
// no value, stored or not, reaches the model. (This folder has no index.ts: Pi does not load it as an extension.)
import { execFileSync } from 'node:child_process'
import { readFileSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { agentDir } from './agent-dir.ts'

export interface SecretEntry {
  name: string
  purpose: string
}

/** Aria's home: the folder the agent folder is in. */
export const ariaHome = (): string => dirname(agentDir())

export const indexPath = (): string => join(ariaHome(), 'secrets.json')

/** `github-token` → `ARIA_SECRET_GITHUB_TOKEN` (the dock's `secrets::env_name`). */
export const envName = (name: string): string => `ARIA_SECRET_${name.toUpperCase().replace(/-/g, '_')}`

/** The dock's name folding: lower case, spaces and underscores to dashes; null when it is not a valid name. */
export function foldName(raw: string): string | null {
  const folded = raw.trim().toLowerCase().replace(/[ _]/g, '-')
  return /^[a-z0-9](?:[a-z0-9-]{0,46}[a-z0-9])?$/.test(folded) ? folded : null
}

/** The saved secrets' names (never a value), or none when there is no index yet. */
export function listSecrets(path = indexPath()): SecretEntry[] {
  try {
    const index = JSON.parse(readFileSync(path, 'utf8'))
    return (index.secrets ?? []).filter((e: any) => typeof e?.name === 'string').map((e: any) => ({ name: e.name, purpose: String(e.purpose ?? '') }))
  } catch {
    return []
  }
}

/** Reads one value from the keyring through the dock's `aria secret`; null when it cannot. */
export type ReadSecret = (name: string) => string | null

export const readFromKeyring: ReadSecret = (name) => {
  try {
    return execFileSync(process.env.ARIA_EXE || 'aria', ['secret', '--home', ariaHome(), `user:${name}`], {
      encoding: 'utf8',
      timeout: 5000,
      stdio: ['ignore', 'pipe', 'ignore'],
    })
  } catch {
    return null
  }
}

/** The environment a command gets: each stored secret it names (`$ARIA_SECRET_X`, `${ARIA_SECRET_X}`). */
export function secretEnv(command: string, entries = listSecrets(), read: ReadSecret = readFromKeyring): Record<string, string> {
  const env: Record<string, string> = {}
  for (const { name } of entries) {
    const variable = envName(name)
    if (!new RegExp(`\\$\\{?${variable}\\b`).test(command)) continue
    const value = read(name)
    if (value) env[variable] = value
  }
  return env
}

// --- Masking -----------------------------------------------------------------------------------------------------

/** Key shapes that are a secret on their own (the same list as the dock's `secrets::PREFIXES` and the page's). */
export const KEY_SHAPES: { kind: string; re: RegExp }[] = [
  { kind: 'OpenAI or Anthropic key', re: /\bsk-[A-Za-z0-9_-]{16,}/g },
  { kind: 'GitHub token', re: /\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}/g },
  { kind: 'GitLab token', re: /\bglpat-[A-Za-z0-9_-]{16,}/g },
  { kind: 'Slack token', re: /\bxox[abprs]-[A-Za-z0-9-]{10,}/g },
  { kind: 'AWS access key', re: /\bAKIA[0-9A-Z]{16}\b/g },
  { kind: 'Google API key', re: /\bAIza[0-9A-Za-z_-]{30,}/g },
  { kind: 'JWT', re: /\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/g },
]

// `NAME=value`, `export NAME="value"`, `"name": "value"`, `name: value`, where the name ends in a word that says it is a
// secret (`DB_PASSWORD`, `client_secret`; not `secrets_list`). Unquoted, a colon needs a space after it (YAML), so that
// `key:anthropic` (an account's name) or `secrets_list: TypeError` stay as they are.
const SECRET_NAME = String.raw`[A-Za-z0-9_.-]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|CREDENTIALS?|AUTH)`
const ASSIGNMENT = new RegExp(
  String.raw`^(\s*(?:export\s+)?(?:"(${SECRET_NAME})"\s*:\s*|(${SECRET_NAME})\s*=\s*|(${SECRET_NAME}):\s+))(["']?)([^"'\s#,]{4,})\5`,
  'gim',
)

export interface Masked {
  text: string
  /** What was masked and where (for the note), without the values. */
  found: { kind: string; line: number }[]
}

const lineOf = (text: string, index: number) => text.slice(0, index).split('\n').length

/**
 * `text` with every secret value replaced: the person's stored secrets by their name (`[secret github-token]`), and
 * anything that looks like a key (its shape, or a secret-named assignment) by `[masked …]`.
 */
export function maskText(text: string, stored: { name: string; value: string }[] = []): Masked {
  const found: Masked['found'] = []
  let out = text
  for (const { name, value } of stored) {
    if (value.length < 4 || !out.includes(value)) continue
    out = out.split(value).join(`[secret ${name}]`)
  }
  for (const { kind, re } of KEY_SHAPES) {
    out = out.replace(re, (match, offset: number) => {
      found.push({ kind, line: lineOf(out, offset) })
      return `[masked ${kind}]`
    })
  }
  out = out.replace(ASSIGNMENT, (whole, head: string, json: string, env: string, yaml: string, quote: string, value: string, offset: number) => {
    const name = json ?? env ?? yaml
    if (value.startsWith('[masked') || value.startsWith('[secret') || value.startsWith('$')) return whole
    found.push({ kind: `value of ${name}`, line: lineOf(out, offset) })
    return `${head}${quote}[masked]${quote}`
  })
  return { text: out, found }
}

/** The note the model gets after a masked output: where, and how to keep it without seeing it. */
export function maskNote(found: Masked['found'], path?: string): string {
  const where = found.map((f) => `${f.kind} (line ${f.line})`).join(', ')
  const file = path ? ` in \`${path}\`` : ''
  return `\n\n[aria] Masked ${found.length === 1 ? 'a secret' : `${found.length} secrets`}${file}: ${where}. You never see secret values. If the person wants to keep one, offer \`secret_import\` with the file's path and the variable that holds it; it goes to the keyring without passing through you.`
}

/** The stored secrets' values, for masking, read again only when the index changes. */
export function storedValues(read: ReadSecret = readFromKeyring): { name: string; value: string }[] {
  let stamp = 0
  try {
    stamp = statSync(indexPath()).mtimeMs
  } catch {
    return []
  }
  if (cache && cache.stamp === stamp) return cache.values
  const values = listSecrets()
    .map(({ name }) => ({ name, value: read(name) ?? '' }))
    .filter((v) => v.value.length >= 4)
  cache = { stamp, values }
  return values
}
let cache: { stamp: number; values: { name: string; value: string }[] } | null = null
