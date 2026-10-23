// The composer's check before a message goes out (docs/superpowers/plans/2026-10-22-secrets.md): a password, API key or
// token pasted in the chat is caught here, so that the person can keep it in the keyring instead of sending it to the
// model. The shapes are the same list as `extensions/lib/secrets.ts` (KEY_SHAPES) and the dock's `secrets::PREFIXES`.
// Pure: findSecrets(text) -> [{ kind, value, start, end, name }]; withSecretsSaved(text, found, names) -> the text to send.

const SHAPES = [
  { kind: 'openai', name: 'openai-key', re: /\bsk-ant-[A-Za-z0-9_-]{16,}/g, label: 'Anthropic key' },
  { kind: 'openai', name: 'openai-key', re: /\bsk-[A-Za-z0-9_-]{16,}/g, label: 'OpenAI key' },
  { kind: 'github', name: 'github-token', re: /\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}/g, label: 'GitHub token' },
  { kind: 'gitlab', name: 'gitlab-token', re: /\bglpat-[A-Za-z0-9_-]{16,}/g, label: 'GitLab token' },
  { kind: 'slack', name: 'slack-token', re: /\bxox[abprs]-[A-Za-z0-9-]{10,}/g, label: 'Slack token' },
  { kind: 'aws', name: 'aws-access-key', re: /\bAKIA[0-9A-Z]{16}\b/g, label: 'AWS access key' },
  { kind: 'google', name: 'google-api-key', re: /\bAIza[0-9A-Za-z_-]{30,}/g, label: 'Google API key' },
  { kind: 'jwt', name: 'token', re: /\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/g, label: 'token' },
]

// `DB_PASSWORD=value`, `"api_key": "value"`, `token: value`: the name ends in a word that says it is a secret (the same
// rule as extensions/lib/secrets.ts: unquoted, a colon needs a space after it).
const SECRET_NAME = String.raw`[A-Za-z0-9_.-]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)`
const ASSIGNMENT = new RegExp(
  String.raw`(?:^|\s)(?:export\s+)?(?:"(${SECRET_NAME})"\s*:\s*|(${SECRET_NAME})\s*=\s*|(${SECRET_NAME}):\s+)(["']?)([^"'\s#,]{6,})\4`,
  'gim',
)

/** `GitHub token` / `DB_PASSWORD` → `github-token` / `db-password`; '' when nothing is left. */
export function suggestName(raw) {
  return String(raw ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48)
    .replace(/-+$/, '')
}

/** The name rule the dock applies (`secrets::valid_name`), after the same folding. */
export const validName = (name) => /^[a-z0-9](?:[a-z0-9-]{0,46}[a-z0-9])?$/.test(name)

/** `github-token` → `ARIA_SECRET_GITHUB_TOKEN`. */
export const envName = (name) => `ARIA_SECRET_${name.toUpperCase().replace(/-/g, '_')}`

/** The secrets in `text`, in order, each once: its kind (for the card), the value and where it is, a name to offer. */
export function findSecrets(text) {
  if (typeof text !== 'string' || !text) return []
  const found = []
  const taken = (start, end) => found.some((f) => start < f.end && end > f.start)
  for (const shape of SHAPES) {
    for (const m of text.matchAll(shape.re)) {
      const start = m.index
      const end = start + m[0].length
      if (!taken(start, end)) found.push({ kind: shape.label, value: m[0], start, end, name: shape.name })
    }
  }
  for (const m of text.matchAll(ASSIGNMENT)) {
    const name = m[1] ?? m[2] ?? m[3]
    const value = m[5]
    if (value.startsWith('$') || value.startsWith('[')) continue
    const start = m.index + m[0].lastIndexOf(value)
    const end = start + value.length
    if (!taken(start, end)) found.push({ kind: name, value, start, end, name: suggestName(name) || 'secret' })
  }
  found.sort((a, b) => a.start - b.start)
  // The same value twice is one secret; two names alike get a number.
  const once = found.filter((f, i) => found.findIndex((g) => g.value === f.value) === i)
  const seen = new Map()
  return once.map((f) => {
    const n = (seen.get(f.name) ?? 0) + 1
    seen.set(f.name, n)
    return n > 1 ? { ...f, name: `${f.name}-${n}` } : f
  })
}

/** The message with each saved secret's value replaced by a pointer the model can use (`names[i]` for `found[i]`). */
export function withSecretsSaved(text, found, names) {
  let out = text
  found.forEach((f, i) => {
    const name = names[i]
    if (!name) return
    out = out.split(f.value).join(`[secret saved as ${name}: use $${envName(name)}]`)
  })
  return out
}
