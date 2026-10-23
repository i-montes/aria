// The project's own checks, run after a turn that edited files: what `.aria/verify.json` names ({ lint?, types?, test? }),
// else (Linux only, where the commands run in the sandbox's box) what the project shows: package.json scripts lint,
// typecheck and test (bun when there is a bun.lock), Cargo.toml (clippy, test), pyproject.toml (ruff, pytest). Never on
// macOS: there is no box there and nobody would be asked. The model cannot write .aria/ (sandbox/policy.ts).
import { spawn } from 'node:child_process'

export interface Check {
  name: string
  command: string
}

export interface ProjectFiles {
  verifyJson?: string
  packageJson?: string
  bunLock?: boolean
  cargoToml?: boolean
  pyproject?: boolean
}

export function checksFor(platform: string, files: ProjectFiles): Check[] {
  // macOS has no box, and nothing here asks the person: no checks there (a command that asks first is for later).
  if (platform === 'darwin') return []
  if (files.verifyJson) {
    try {
      const own = JSON.parse(files.verifyJson)
      return ['lint', 'types', 'test'].filter((k) => typeof own?.[k] === 'string' && own[k].trim()).map((k) => ({ name: k, command: own[k].trim() }))
    } catch {
      return []
    }
  }
  if (files.packageJson) {
    try {
      const scripts = JSON.parse(files.packageJson)?.scripts ?? {}
      const runner = files.bunLock ? 'bun run' : 'npm run'
      // npm's own placeholder ("no test specified") always fails: it is not a check.
      const real = (k: string) => typeof scripts[k] === 'string' && !scripts[k].includes('no test specified')
      return ['lint', 'typecheck', 'test'].filter(real).map((k) => ({ name: k, command: `${runner} ${k}` }))
    } catch {
      return []
    }
  }
  if (files.cargoToml) return [{ name: 'lint', command: 'cargo clippy -q' }, { name: 'test', command: 'cargo test -q' }]
  if (files.pyproject) return [{ name: 'lint', command: 'ruff check' }, { name: 'test', command: 'pytest -q' }]
  return []
}

export interface Outcome {
  ok: boolean
  timedOut: boolean
  output: string
}

/** Runs `line` with bash in `folder`, at most `limitMs`; its whole output (stdout and stderr together). */
export function runCheck(line: string, folder: string, limitMs: number): Promise<Outcome> {
  return new Promise((done) => {
    const child = spawn('bash', ['-c', line], { cwd: folder, detached: true, stdio: ['ignore', 'pipe', 'pipe'] })
    let output = ''
    let timedOut = false
    const take = (chunk: Buffer) => {
      output += chunk.toString()
      if (output.length > 200_000) output = output.slice(-100_000)
    }
    child.stdout?.on('data', take)
    child.stderr?.on('data', take)
    const timer = setTimeout(() => {
      timedOut = true
      try {
        process.kill(-child.pid!, 'SIGKILL')
      } catch {
        child.kill('SIGKILL')
      }
    }, limitMs)
    // On exit, not on close: a child the check left behind (a server, `setsid …`) may hold its output open forever.
    child.on('exit', (code) => {
      clearTimeout(timer)
      setTimeout(() => {
        child.stdout?.destroy()
        child.stderr?.destroy()
        done({ ok: !timedOut && code === 0, timedOut, output })
      }, 50)
    })
    child.on('error', (error) => {
      clearTimeout(timer)
      done({ ok: false, timedOut, output: String(error) })
    })
  })
}

/** The last `n` lines of an output. */
export const lastLines = (output: string, n = 40): string => output.trimEnd().split('\n').slice(-n).join('\n')
