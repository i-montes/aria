// After a run that edited files, the project's checks (checks.ts) run once it settles: inside the sandbox's box on Linux,
// at most 5 minutes each; the result goes to Aria's terminal footer (status `aria-verify`) and, when a check fails, to
// the model once (flow.ts). `/verify off|on` for this session. Subagents do not check.
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { wrapCommand } from '../sandbox/policy.ts'
import { checksFor, runCheck } from './checks.ts'
import { Verifier } from './flow.ts'

const LIMIT_MS = 5 * 60 * 1000

const read = (path: string) => (existsSync(path) ? readFileSync(path, 'utf8') : undefined)

const hasBwrap = () => {
  try {
    execFileSync('bwrap', ['--version'], { stdio: 'ignore' })
    return true
  } catch {
    return false
  }
}

export default function verify(pi: ExtensionAPI) {
  let ctxNow: any = null
  let subagent = false
  const folder = () => ctxNow?.cwd ?? process.cwd()
  const verifier = new Verifier({
    checks: () => {
      // On Linux the commands run boxed; without bwrap they do not run at all.
      if (process.platform === 'linux' && !hasBwrap()) return []
      const at = folder()
      return checksFor(process.platform, {
        verifyJson: read(join(at, '.aria', 'verify.json')),
        packageJson: read(join(at, 'package.json')),
        bunLock: existsSync(join(at, 'bun.lock')) || existsSync(join(at, 'bun.lockb')),
        cargoToml: existsSync(join(at, 'Cargo.toml')),
        pyproject: existsSync(join(at, 'pyproject.toml')),
      })
    },
    run: (command) => runCheck(process.platform === 'linux' ? wrapCommand(command, folder()) : command, folder(), LIMIT_MS),
    tell: (text) => void pi.sendUserMessage(text),
    status: (value) => ctxNow?.ui?.setStatus?.('aria-verify', JSON.stringify(value)),
  })

  pi.on('session_start', (_event: any, ctx: any) => {
    ctxNow = ctx
    subagent = ctx?.mode === 'json'
  })
  pi.on('tool_result', (event: any, ctx: any) => {
    ctxNow = ctx
    if (subagent || event.isError) return
    if (event.toolName === 'edit' || event.toolName === 'write') verifier.edited()
  })
  pi.on('agent_settled', async (_event: any, ctx: any) => {
    ctxNow = ctx
    if (!subagent) await verifier.runEnded()
  })
  pi.registerCommand('verify', {
    description: "The project's checks after each run that edits files: /verify off, /verify on.",
    handler: async (args: string, ctx: any) => {
      const off = args.trim() === 'off'
      verifier.setOn(!off)
      ctx.ui.notify(off ? 'Checks after editing are off for this session.' : 'Checks after editing are on.', 'info')
    },
  })
}
