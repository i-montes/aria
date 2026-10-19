// The parity matrix (docs/parity.md) against ARIA-SETUP.md: every setup item has a row, and every file and test a row
// names exists, so a renamed test or a moved script shows up as a failing `bun test tools/parity`.

export type Check = 'test' | 'eval' | 'script' | 'manual' | 'out-of-scope' | 'difference' | 'n/a'
export type Row = { item: number; what: string; check: Check; where: string; result: string }

const CHECKS: readonly string[] = ['test', 'eval', 'script', 'manual', 'out-of-scope', 'difference', 'n/a']
/** The items with parts the owner left out of phase 2: `meeting` (5), OpenPencil (6), rea, OpenPencil and Figma (17, 32). */
const MAY_BE_OUT_OF_SCOPE = [5, 6, 17, 32]

const cells = (line: string) => line.split('|').slice(1, -1).map((cell) => cell.trim())
const plain = (text: string) => text.replace(/`/g, '').trim()

/** The table rows whose first cell is a number. */
export function parseMatrix(markdown: string): Row[] {
  return markdown.split('\n').flatMap((line) => {
    const [item, what, check, where, result] = cells(line)
    if (!item || !/^\d+$/.test(item)) return []
    return [{ item: Number(item), what: what ?? '', check: (check ?? '') as Check, where: plain(where ?? ''), result: result ?? '' }]
  })
}

/** The numbers of ARIA-SETUP.md's `N. **…**` items. */
export function setupItems(markdown: string): number[] {
  return [...markdown.matchAll(/^(\d+)\. \*\*/gm)].map((m) => Number(m[1]))
}

/** One line per problem; empty when the matrix is sound. */
export function problems(rows: Row[], items: number[], exists: (path: string) => boolean, read: (path: string) => string): string[] {
  const found: string[] = []
  for (const item of items) if (!rows.some((r) => r.item === item)) found.push(`item ${item} has no row`)
  for (const r of rows) {
    if (!CHECKS.includes(r.check)) {
      found.push(`item ${r.item}: unknown check ${r.check}`)
      continue
    }
    if (r.check === 'test' || r.check === 'eval' || r.check === 'script') {
      const [file, name] = r.where.split('::')
      if (!file || !exists(file)) found.push(`item ${r.item}: ${file} does not exist`)
      else if (name && !read(file).includes(name)) found.push(`item ${r.item}: ${file} has no test ${name}`)
      continue
    }
    if (r.check === 'out-of-scope' && !MAY_BE_OUT_OF_SCOPE.includes(r.item)) {
      found.push(`item ${r.item} cannot be out of scope`)
      continue
    }
    if (!r.where) found.push(`item ${r.item}: no reason or steps`)
  }
  return found
}
