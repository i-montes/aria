// Session permissions to work outside the sandbox, granted from the ask dialog: by program and
// subcommand (`git status`, `docker compose`), by exact command, by folder for Pi's write/edit, or
// everything. index.ts keeps them in memory and clears them when a session starts. A command counts
// as covered only when every program it runs is known: anything that can hide another command
// ($(…), eval, sh -c, xargs, find -exec…) is asked about every time. Programs that run arbitrary
// code (interpreters, package runners, build tools, privilege tools) are never granted as a program,
// only as one exact command (2026-10-18, the user wanted it more controlled).
import { dirname, sep } from 'node:path'

const WRAPPERS = new Set(['env', 'nohup', 'time', 'exec', 'command', 'builtin', 'nice', 'stdbuf', 'timeout', 'then', 'do', 'else'])
// Granting one of these would allow anything: they run code or commands given to them.
export const BROAD = new Set([
  'sudo', 'doas', 'pkexec', 'su', 'run0',
  'bash', 'sh', 'zsh', 'dash', 'ksh', 'fish',
  'python', 'python3', 'node', 'bun', 'deno', 'ruby', 'perl', 'php', 'lua', 'R', 'Rscript', 'osascript',
  'npx', 'bunx', 'pnpx', 'npm', 'pnpm', 'yarn', 'uv', 'uvx', 'pipx', 'poetry', 'cargo', 'go', 'java', 'dotnet',
  'make', 'just', 'task', 'cmake', 'ninja', 'gradle', 'mvn',
  'ssh', 'scp', 'rsync', 'docker-exec', 'kubectl-exec', 'watch', 'parallel', 'flock', 'script', 'expect', 'tmux', 'screen',
])
const KEYWORDS = new Set(['if', 'elif', 'fi', 'for', 'while', 'until', 'done', 'case', 'esac', 'in', 'function', '!', '{', '}', '[[', ']]'])
const HIDDEN_CODE = /\$\(|`|<\(|>\(|\beval\b|\bsource\b|^\s*\.\s|\b(ba|z|da|k|fi)?sh\s+(-\w+\s+)*-\w*c\w*\b|\bxargs\b|-exec(dir)?\b|-ok\b/m

// Tools whose first argument picks very different actions.
const SUBCOMMANDS = new Set([
  'git', 'gh', 'glab', 'docker', 'podman', 'docker-compose', 'kubectl', 'helm', 'systemctl', 'loginctl', 'timedatectl',
  'hostnamectl', 'nmcli', 'ip', 'hyprctl', 'brew', 'apt', 'apt-get', 'pacman', 'yay', 'paru', 'flatpak', 'snap', 'pip',
  'pip3', 'mise', 'terraform', 'aws', 'gcloud', 'az', 'hcloud', 'fly', 'vercel', 'wrangler', 'supabase', 'eas', 'expo',
  'op', 'tailscale', 'rclone', 'ufw', 'firewall-cmd', 'bluetoothctl', 'pactl', 'wpctl', 'herdr', 'omarchy',
])

export interface Grants {
  all: boolean
  programs: Set<string>
  exact: Set<string>
  dirs: Set<string>
}
export const emptyGrants = (): Grants => ({ all: false, programs: new Set(), exact: new Set(), dirs: new Set() })

/** The command with heredoc bodies removed; bodies of an unquoted delimiter are kept in `expanded`,
 * since the shell still runs the $(…) inside them. */
function splitHeredocs(command: string): { shell: string; expanded: string } {
  const lines = command.split('\n')
  const shell: string[] = []
  const expanded: string[] = []
  for (let i = 0; i < lines.length; i++) {
    shell.push(lines[i])
    const re = /<<(-?)\s*(['"]?)([A-Za-z_][A-Za-z0-9_]*)\2/g
    for (let m = re.exec(lines[i]); m; m = re.exec(lines[i])) {
      const [, dash, quote, word] = m
      while (++i < lines.length && (dash ? lines[i].replace(/^\t+/, '') : lines[i]) !== word) {
        if (!quote) expanded.push(lines[i])
      }
    }
  }
  return { shell: shell.join('\n'), expanded: expanded.join('\n') }
}

/** What a grant for this command would cover: each program with its subcommand (`git push`,
 * `docker compose`), as typed; null when that can't be told safely. `broad` when one of them runs
 * arbitrary code, so only the exact command may be granted. */
export function grantUnits(command: string): { units: string[]; broad: boolean } | null {
  const { shell, expanded } = splitHeredocs(command)
  if (HIDDEN_CODE.test(shell) || HIDDEN_CODE.test(expanded)) return null
  const plain = shell
    .replace(/'[^']*'|"(?:\\.|[^"\\])*"/g, 'Q')
    .replace(/\d*>&\d*-?|&>>?|\d*[<>]+/g, ' > ')
  const found = new Set<string>()
  let broad = false
  for (const segment of plain.split(/\|\||&&|[|;&\n()]/)) {
    const words = segment.trim().split(/\s+/).filter(Boolean)
    let i = 0
    // Skip keywords, VAR=value and wrappers (with their flags and numeric arguments).
    for (;;) {
      const w = words[i]
      if (w === undefined) break
      if (KEYWORDS.has(w) || /^[A-Za-z_][A-Za-z0-9_]*=/.test(w)) i++
      else if (WRAPPERS.has(w)) {
        i++
        while (words[i] !== undefined && (words[i].startsWith('-') || /^\d+[smhd]?$/.test(words[i]))) i++
      } else break
    }
    const word = words[i]
    if (!word || word === '>') continue
    if (word === 'Q') return null // a quoted program name: can't tell what runs
    const name = word.split('/').pop()!
    if (BROAD.has(name) || /^python\d/.test(name) || (word.includes('/') && !word.startsWith('/usr/'))) broad = true
    // For tools with subcommands the grant names one (`git push`, `systemctl --user restart` →
    // `systemctl restart`): granting `git status` must not allow `git push`.
    const sub = SUBCOMMANDS.has(name) ? words.slice(i + 1).find((w) => !w.startsWith('-')) : undefined
    if (sub && !/^[a-z][a-z0-9-]*$/.test(sub)) broad = true // can't tell the subcommand: exact only
    found.add(sub && !broad ? `${word} ${sub}` : word)
  }
  return found.size ? { units: [...found], broad } : null
}

/** True when the grants already cover running this command outside the sandbox. */
export function covers(grants: Grants, command: string): boolean {
  if (grants.all || grants.exact.has(command.trim())) return true
  const g = grantUnits(command)
  return !!g && !g.broad && g.units.every((u) => grants.programs.has(u))
}

const inside = (path: string, root: string) => path === root || path.startsWith(root.endsWith(sep) ? root : root + sep)

/** True when the grants cover writing this (already resolved) path. */
export const coversPath = (grants: Grants, realPath: string) => grants.all || [...grants.dirs].some((d) => inside(realPath, d))

/** The folder a write grant covers: where the file lives. */
export const grantDir = (realPath: string) => dirname(realPath)

/** A short look at the command for the dialog: its first lines, cut to a width. */
export function commandPreview(command: string, maxLines = 2, width = 100): string {
  const lines = command.trim().split('\n')
  const shown = lines.slice(0, maxLines).map((l) => (l.length > width ? `${l.slice(0, width - 1)}…` : l))
  const rest = lines.length - shown.length
  return rest > 0 ? `${shown.join('\n')}\n(+${rest} more lines)` : shown.join('\n')
}
