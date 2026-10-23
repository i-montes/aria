// The project's own checks, run after a turn that edited files: what `.aria/verify.json` names ({ lint?, types?, test? }),
// else (Linux only, where the commands run in the sandbox's box) what the project shows: package.json scripts lint,
// typecheck and test (bun when there is a bun.lock), Cargo.toml (clippy, test), pyproject.toml (ruff, pytest). On macOS
// there is no box, so only verify.json's commands run: the person named them.
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
  if (files.verifyJson) {
    try {
      const own = JSON.parse(files.verifyJson)
      return ['lint', 'types', 'test'].filter((k) => typeof own?.[k] === 'string' && own[k].trim()).map((k) => ({ name: k, command: own[k].trim() }))
    } catch {
      return []
    }
  }
  if (platform === 'darwin') return []
  if (files.packageJson) {
    try {
      const scripts = JSON.parse(files.packageJson)?.scripts ?? {}
      const runner = files.bunLock ? 'bun run' : 'npm run'
      return ['lint', 'typecheck', 'test'].filter((k) => typeof scripts[k] === 'string').map((k) => ({ name: k, command: `${runner} ${k}` }))
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
    child.on('close', (code) => {
      clearTimeout(timer)
      done({ ok: !timedOut && code === 0, timedOut, output })
    })
    child.on('error', (error) => {
      clearTimeout(timer)
      done({ ok: false, timedOut, output: String(error) })
    })
  })
}

/** The last `n` lines of an output. */
export const lastLines = (output: string, n = 40): string => output.trimEnd().split('\n').slice(-n).join('\n')
