// What a new session segment starts from (the spec's "Rotation"): the model's context after rotation is the same as
// before it. So the seed is the latest compaction's summary on the active path, the old session's id, and every entry
// Pi puts in the model's context after that summary, in order, built the way Pi 1.3.0 builds it
// (`buildContextEntries` and `buildSessionProjection` in dist/core/session-manager.js):
// - the kept entries from the compaction's `firstKeptEntryId` up to the compaction, without role-`system` messages,
//   then every entry after the compaction;
// - `context_edit` entries in that range apply to their target: `replacement: null` drops it, otherwise its content is
//   replaced (a string becomes one text block for assistant and tool results);
// - `message` entries are carried as messages, `custom_message` entries as custom messages (with their type, display
//   and details), a `branch_summary` as Pi's `branchSummary` message; an older compaction contributes nothing, and an
//   older `aria-earlier` seed is left out (the new summary covers it).
// Pure: it takes the entries as Pi's `SessionManager.getBranch()` returns them, root to leaf.

// Pi's framing of a compaction summary in the model's context (dist/core/messages.js, Pi 1.3.0). Copied: the package
// index does not export them.
export const COMPACTION_SUMMARY_PREFIX = `The conversation history before this point was compacted into the following summary:

<summary>
`
export const COMPACTION_SUMMARY_SUFFIX = `
</summary>`

type Entry = { type: string; id: string; parentId: string | null; [key: string]: any }

export type SeedItem =
  | { kind: 'message'; message: any }
  | { kind: 'custom'; customType: string; content: unknown; display: boolean; details: unknown }

export type Seed = { summary: string; from: string; items: SeedItem[] }

/** Pi's `projectContextEntry` content rule for an edited message. */
function editedContent(role: string, content: unknown) {
  return (role === 'assistant' || role === 'toolResult') && typeof content === 'string' ? [{ type: 'text', text: content }] : content
}

const EDITABLE = new Set(['user', 'assistant', 'toolResult'])

function item(entry: Entry, edit: Entry | undefined): SeedItem | null {
  if (edit && edit.replacement === null) return null
  const replaced = edit ? edit.replacement.content : undefined
  if (entry.type === 'message') {
    const message = entry.message
    if (!edit || !EDITABLE.has(message.role)) return { kind: 'message', message }
    return { kind: 'message', message: { ...message, content: editedContent(message.role, replaced) } }
  }
  if (entry.type === 'custom_message') {
    if (entry.customType === 'aria-earlier') return null
    return {
      kind: 'custom',
      customType: entry.customType,
      content: edit ? replaced : entry.content ?? [],
      display: entry.display,
      details: entry.details,
    }
  }
  if (entry.type === 'branch_summary' && entry.summary) {
    const message = { role: 'branchSummary', summary: entry.summary, fromId: entry.fromId, timestamp: new Date(entry.timestamp).getTime() }
    return { kind: 'message', message }
  }
  return null
}

/** The seed for the next segment, or null when the path has no compaction (nothing to carry). */
export function rotationSeed(path: Entry[], sessionId: string): Seed | null {
  let at = -1
  for (let i = path.length - 1; i >= 0; i--) {
    if (path[i].type === 'compaction') {
      at = i
      break
    }
  }
  if (at < 0) return null
  const compaction = path[at]

  const range: Entry[] = []
  let kept = false
  for (let i = 0; i < at; i++) {
    const entry = path[i]
    if (entry.id === compaction.firstKeptEntryId) kept = true
    if (kept && !(entry.type === 'message' && entry.message?.role === 'system')) range.push(entry)
  }
  range.push(...path.slice(at + 1))

  const edits = new Map<string, Entry>()
  for (const entry of range) if (entry.type === 'context_edit') edits.set(entry.targetId, entry)

  const items: SeedItem[] = []
  for (const entry of range) {
    const carried = item(entry, edits.get(entry.id))
    if (carried) items.push(carried)
  }
  return { summary: String(compaction.summary ?? ''), from: sessionId, items }
}
