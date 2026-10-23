// (This folder has no index.ts: Pi does not load it as an extension.)
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'

/** The git repo containing `path` (walking up from it or its first existing parent), or null. */
export function gitRoot(path: string): string | null {
  let dir = path
  for (let i = 0; i < 64; i++) {
    if (existsSync(join(dir, '.git'))) return dir
    const up = dirname(dir)
    if (up === dir) return null
    dir = up
  }
  return null
}
