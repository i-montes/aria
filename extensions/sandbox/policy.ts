// What the sandbox allows, as pure functions: which paths a command may write, which it may not
// see, and the bwrap command line that enforces it. index.ts applies these.
//
// The profile denies host control points by default (final review, 2026-10-14): besides
// credentials, every socket or file that makes the host run code — docker, herdr, X11, the system
// bus, $XDG_RUNTIME_DIR, the repo's .git/config and hooks, project-local Pi extensions and Pi's own
// transpile cache in /tmp/jiti — is hidden or read-only, and only pure download caches are writable.
import { ariaConfig } from '../lib/config.ts'
import { existsSync, lstatSync, readFileSync, readlinkSync, realpathSync, statSync } from 'node:fs'
import { homedir } from 'node:os'
import { basename, dirname, join, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const HOME = homedir()
const RUNTIME = process.env.XDG_RUNTIME_DIR || `/run/user/${process.getuid?.() ?? 1000}`

// Download caches only: bin/ and installs/ trees hold programs the host runs later.
const CACHES = [
  '.npm',
  '.bun/install/cache',
  '.cache/pip',
  '.cache/uv',
  '.cache/pnpm',
  '.cache/go-build',
  '.cache/node-gyp',
  '.local/share/pnpm/store',
  '.cargo/registry',
  '.cargo/git',
  '.local/share/mise/downloads',
  '.gradle/caches',
]
const SECRET_DIRS = [
  '.ssh',
  '.gnupg',
  '.aws',
  '.config/gh',
  '.docker',
  '.config/herdr', // its socket runs commands in the user's terminals
  '.config/wayvnc',
  '.config/opencode',
  '.local/share/opencode',
  '.codex',
  '.kube',
  '.config/gcloud',
]
const SECRET_FILES = [
  '.pi/agent/auth.json',
  '.pi/agent/mcp.json',
  '.pi/agent/mcp-auth.json',
  '.netrc',
  '.git-credentials',
  '.npmrc',
  '.claude/.credentials.json',
  '.local/share/aria/agent/auth.json',
  // Aria's own credentials (its agent folder holds links to them).
  '.aria/auth.json',
  '.aria/mcp.json',
  '.aria/mcp-auth.json',
  '.aria/agent/auth.json',
  '.aria/agent/mcp.json',
  '.aria/agent/mcp-auth.json',
]
// Host control sockets outside $XDG_RUNTIME_DIR.
const HOST_FILES = ['/run/docker.sock', '/run/podman/podman.sock', '/run/dbus/system_bus_socket']
const HOST_DIRS = ['/tmp/.X11-unix']
// Inside /tmp but read by Pi itself: its extension transpile cache and the quota cache.
const TMP_PROTECTED = ['/tmp/jiti', '/tmp/node-compile-cache', '/tmp/pi-go-usage.json']

export const BLOCK_NOTE =
  '[sandbox: nothing outside the project, /tmp and the caches can be written, and credentials are hidden. If it really has to happen, repeat the command with outsideSandbox: true and, in reason, what it does in at most 12 words in the user\'s language; the user will be asked.]'

const inside = (path: string, root: string) => path === root || path.startsWith(root.endsWith(sep) ? root : root + sep)

/** The path with symlinks followed, for the part that exists (dangling links included). */
function realish(path: string, depth = 0): string {
  let head = path
  const tail: string[] = []
  for (let i = 0; i < 256; i++) {
    try {
      return join(realpathSync(head), ...tail.reverse())
    } catch {
      try {
        // A dangling symlink: writing through it creates its target, so follow it anyway.
        if (depth < 40 && lstatSync(head).isSymbolicLink()) {
          return realish(join(resolve(dirname(head), readlinkSync(head)), ...tail.reverse()), depth + 1)
        }
      } catch {
        // not there at all: go up
      }
      const up = dirname(head)
      if (up === head) return path
      tail.push(basename(head))
      head = up
    }
  }
  return path
}

/** Resolve a tool path the way Pi does (leading @, file: URLs, ~), plus its real location. */
export function resolvePath(path: string, cwd: string, home = HOME): { lexical: string; real: string } {
  let p = path.trim()
  if (p.startsWith('@')) p = p.slice(1)
  if (p.startsWith('file:')) {
    try {
      p = fileURLToPath(p)
    } catch {
      // keep it as written
    }
  }
  const lexical = resolve(cwd, p.replace(/^~(?=$|\/)/, home))
  return { lexical, real: realish(lexical) }
}

export function projectRoot(cwd: string, home = HOME): string | null {
  let dir = resolve(cwd)
  let root = dir
  for (let i = 0; i < 64; i++) {
    if (existsSync(join(dir, '.git'))) {
      root = dir
      break
    }
    const up = dirname(dir)
    if (up === dir) break
    dir = up
  }
  return root === resolve(home) || root === '/' ? null : root
}

/** The project's git dirs: its own, and the common one when it is a worktree or submodule. */
function gitDirs(project: string | null): { own: string | null; common: string | null } {
  if (!project) return { own: null, common: null }
  const dotgit = join(project, '.git')
  try {
    if (statSync(dotgit).isDirectory()) return { own: dotgit, common: null }
    const gitdir = readFileSync(dotgit, 'utf8').match(/^gitdir:\s*(.+)$/m)?.[1]?.trim()
    if (!gitdir) return { own: null, common: null }
    const own = resolve(project, gitdir)
    let common = own
    try {
      common = resolve(own, readFileSync(join(own, 'commondir'), 'utf8').trim())
    } catch {
      // a submodule's gitdir is its own common dir
    }
    return { own, common }
  } catch {
    return { own: null, common: null }
  }
}

export function writableRoots(cwd: string, home = HOME, exists: (p: string) => boolean = existsSync): string[] {
  const project = projectRoot(cwd, home)
  const { common } = gitDirs(project)
  const extra = common && !(project && inside(common, project)) ? [common] : []
  return [...(project ? [project] : []), ...extra, '/tmp', ...CACHES.map((c) => join(home, c)).filter(exists), ...extraWritable(home, exists)]
}

/** The folders Aria's settings add (`sandbox.extraWritable`, absolute and existing, or bwrap cannot bind them): never one
 * that holds secrets, and never one that holds or sits inside Aria's own home (its extensions, `bin/pi`, settings). */
function extraWritable(home: string, exists: (p: string) => boolean): string[] {
  const agent = process.env.PI_CODING_AGENT_DIR ?? null
  const ariaHome = agent && process.env.ARIA_SETTINGS ? dirname(agent) : agent
  return ariaConfig().sandbox.extraWritable.filter(
    (dir) =>
      dir.startsWith('/') &&
      exists(dir) &&
      !isSecret(dir, '/', home) &&
      !SECRET_DIRS.map((d) => join(home, d)).some((secret) => inside(secret, dir)) &&
      !(ariaHome && (inside(ariaHome, dir) || inside(dir, ariaHome))) &&
      !isAgentCode(dir, agent),
  )
}

/** Paths inside the writable roots that stay read-only: they make the host run code. */
export function protectedPaths(cwd: string, home = HOME): string[] {
  const project = projectRoot(cwd, home)
  const { own, common } = gitDirs(project)
  const git = [...new Set([own, common].filter((d): d is string => !!d))].flatMap((d) => [join(d, 'config'), join(d, 'hooks')])
  return [...(project ? [join(project, '.pi')] : []), ...git, ...TMP_PROTECTED]
}

export const secretDirs = (home = HOME, runtimeDir = RUNTIME) => [...SECRET_DIRS.map((d) => join(home, d)), runtimeDir]
/** Aria's credentials where its environment puts them (an Aria home outside ~/.aria): its home holds them and its
 * agent folder links to them. Nothing in the owner's own Pi, which Aria does not start. */
function ariaCredentials(): string[] {
  const agent = process.env.PI_CODING_AGENT_DIR
  const settings = process.env.ARIA_SETTINGS
  if (!agent || !settings) return []
  const names = ['auth.json', 'mcp.json', 'mcp-auth.json']
  return [...names.map((n) => join(dirname(settings), n)), ...names.map((n) => join(agent, n))]
}
export const secretFiles = (home = HOME) => [...new Set([...SECRET_FILES.map((f) => join(home, f)), ...ariaCredentials()])]

/** Aria's own Pi files: its extensions (this sandbox among them), persona, experts, settings and its record of them.
 * `agentDir` is Aria's agent folder, or null in the owner's own Pi. No grant opens them: a model that could edit them
 * could switch the sandbox off for every later session. */
export function isAgentCode(path: string, agentDir: string | null): boolean {
  if (!agentDir) return false
  return ['extensions', 'agents', 'SYSTEM.md', 'settings.json', '.aria-files'].some((name) => inside(path, join(agentDir, name)))
}

export function isWritable(path: string, cwd: string, home = HOME, exists: (p: string) => boolean = existsSync): boolean {
  const { lexical, real } = resolvePath(path, cwd, home)
  const roots = writableRoots(cwd, home, exists)
  const locked = protectedPaths(cwd, home)
  return [lexical, real].every((p) => roots.some((r) => inside(p, r)) && !locked.some((l) => inside(p, l)))
}

export function isSecret(path: string, cwd: string, home = HOME, runtimeDir = RUNTIME): boolean {
  const { lexical, real } = resolvePath(path, cwd, home)
  const dirs = secretDirs(home, runtimeDir).flatMap((d) => [d, realish(d)])
  const files = secretFiles(home).flatMap((f) => [f, realish(f)])
  return [lexical, real].some((p) => inside(p, '/proc') || dirs.some((d) => inside(p, d)) || files.includes(p))
}

/** A directory that is a secret or holds one (searching it would read credentials). */
export function containsSecret(path: string, cwd: string, home = HOME, runtimeDir = RUNTIME): boolean {
  if (isSecret(path, cwd, home, runtimeDir)) return true
  const { lexical, real } = resolvePath(path, cwd, home)
  const secrets = [...secretDirs(home, runtimeDir), ...secretFiles(home)]
  return [lexical, real].some((p) => secrets.some((s) => inside(s, p) || inside(realish(s), p)))
}

/**
 * Why a subagent call (the `parallel` tool) would leave the sandbox, or null: every `cwd` must be
 * inside the project or /tmp, and no `contextFiles` entry may be a secret.
 */
export function subagentProblem(input: unknown, cwd: string, home = HOME, runtimeDir = RUNTIME): string | null {
  const project = projectRoot(cwd, home)
  const allowed = [...(project ? [project] : []), '/tmp']
  const walk = (v: unknown): string | null => {
    if (Array.isArray(v)) {
      for (const x of v) {
        const r = walk(x)
        if (r) return r
      }
      return null
    }
    if (!v || typeof v !== 'object') return null
    for (const [k, x] of Object.entries(v as Record<string, unknown>)) {
      if (k === 'cwd' && typeof x === 'string') {
        const { lexical, real } = resolvePath(x, cwd, home)
        if (![lexical, real].every((p) => allowed.some((a) => inside(p, a)))) return `cwd ${x} is outside the project`
      } else if (k === 'contextFiles' && Array.isArray(x)) {
        const bad = x.find((f) => typeof f === 'string' && isSecret(f, cwd, home, runtimeDir))
        if (bad) return `contextFiles includes ${bad}, which holds credentials`
      } else {
        const r = walk(x)
        if (r) return r
      }
    }
    return null
  }
  return walk(input)
}

/** Where to mount over each path: bwrap cannot mount on a symlink, so a link (Aria's agent dir links its credentials
 * to the files in ~/.aria) is replaced by its target. Each mount point appears once. */
function mountPoints(paths: string[]): string[] {
  const isLink = (p: string) => {
    try {
      return lstatSync(p).isSymbolicLink()
    } catch {
      return false
    }
  }
  return [...new Set(paths.map((p) => (isLink(p) ? realish(p) : p)))]
}

export function bwrapArgs(
  cwd: string,
  opts: { home?: string; runtimeDir?: string; exists?: (p: string) => boolean } = {},
): string[] {
  const home = opts.home ?? HOME
  const exists = opts.exists ?? existsSync
  const args = ['--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc']
  for (const root of writableRoots(cwd, home, exists)) args.push('--bind', root, root)
  for (const path of protectedPaths(cwd, home).filter(exists)) args.push('--ro-bind', path, path)
  // The masks go last: a later bind of a folder that holds a secret (the project's .pi, when a credential links into
  // the project) would bring it back.
  for (const dir of mountPoints([...secretDirs(home, opts.runtimeDir ?? RUNTIME), ...HOST_DIRS]).filter(exists)) args.push('--tmpfs', dir)
  for (const file of mountPoints([...secretFiles(home), ...HOST_FILES]).filter(exists)) args.push('--ro-bind', '/dev/null', file)
  args.push('--unsetenv', 'DISPLAY', '--chdir', resolve(cwd))
  return args
}

export const shellQuote = (s: string) => `'${s.replace(/'/g, `'\\''`)}'`

export function wrapCommand(
  command: string,
  cwd: string,
  opts: { home?: string; runtimeDir?: string; exists?: (p: string) => boolean } = {},
): string {
  return ['exec', 'bwrap', ...bwrapArgs(cwd, opts).map(shellQuote), '--', 'bash', '-c', shellQuote(command)].join(' ')
}

export const looksBlocked = (output: string) => /Read-only file system/.test(output)
