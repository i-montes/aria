// What a new session segment starts from (the spec's "Rotation"): the latest compaction's summary on the active path,
// the old session's id, and the messages Pi kept after that compaction (from its `firstKeptEntryId` to the leaf), in
// order. Pure: it takes the entries as Pi's `SessionManager` returns them (`getBranch()`, root to leaf). Only `message`
// entries are carried: custom messages (the hidden `[now]` lines, an earlier `aria-earlier` seed) stay where they are.

type Entry = { type: string; id: string; parentId: string | null; [key: string]: any }

export type Seed = { summary: string; from: string; messages: unknown[] }

/** The entries from the root to `leafId`, following `parentId` (Pi's `getBranch`), for entries given as a whole tree. */
export function activePath(entries: Entry[], leafId: string | null): Entry[] {
  const byId = new Map(entries.map((e) => [e.id, e]))
  const path: Entry[] = []
  let current = leafId ? byId.get(leafId) : undefined
  while (current) {
    path.push(current)
    current = current.parentId ? byId.get(current.parentId) : undefined
  }
  return path.reverse()
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
  const kept = path.findIndex((e) => e.id === compaction.firstKeptEntryId)
  const start = kept >= 0 && kept < at ? kept : at + 1
  const messages = path.slice(start).filter((e) => e.type === 'message').map((e) => e.message)
  return { summary: String(compaction.summary ?? ''), from: sessionId, messages }
}
