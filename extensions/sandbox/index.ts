// Sandbox: every command the agent runs goes through bubblewrap (see policy.ts and
// ~/.pi/agent/docs/specs/2026-10-14-bwrap-sandbox-design.md). It may write only to the project,
// /tmp and package caches, cannot see credentials or the host sockets in $XDG_RUNTIME_DIR, and
// keeps network and processes shared, so dev servers started in the background stay alive.
// When something is blocked the model may ask to run that one command outside; in the TUI the user
// is asked, everywhere else (pi -p, subagents) it is refused. Pi's read/write/edit follow the
// same rules. The dialog shows the model's short description and only a preview of the command,
// and lets the user allow it once, allow its programs (or a folder, for write/edit) for the rest of
// the session, or turn the sandbox off for the session (see grants.ts). `/sandbox reset` forgets.
//
// PI_SANDBOX=0 turns it off.
import { execFileSync } from 'node:child_process'
import { homedir } from 'node:os'
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { createBashToolDefinition } from '@earendil-works/pi-coding-agent'
import { Type } from 'typebox'
import { commandPreview, covers, coversPath, emptyGrants, grantDir, grantUnits } from './grants.ts'
import { CLASSIFIER, parseReview, REVIEW_TIMEOUT_MS, reviewLine, reviewRequest } from './review.ts'
import { canAsk, sandboxCard } from './card.ts'
import { bashRoute } from './route.ts'
import { ariaConfig } from '../lib/config.ts'
import { BLOCK_NOTE, containsSecret, isSecret, isWritable, looksBlocked, projectRoot, resolvePath, secretDirs, secretFiles, subagentProblem, wrapCommand, writableRoots, isAgentCode } from './policy.ts'

const hasBwrap = (() => {
  try {
    execFileSync('bwrap', ['--version'], { stdio: 'ignore' })
    return true
  } catch {
    return false
  }
})()

/** macOS: no bwrap exists, so every command asks (see route.ts). */
const NO_BOX = !hasBwrap && process.platform === 'darwin'

const textOf = (r: any) => (r?.content ?? []).map((c: any) => c?.text ?? '').join('\n')
const withNote = (r: any) => ({ ...r, content: [...(r.content ?? []), { type: 'text', text: BLOCK_NOTE }] })
const refuse = (text: string) => ({ content: [{ type: 'text' as const, text }], isError: true, details: undefined })
const tilde = (p: string) => (p.startsWith(homedir()) ? `~${p.slice(homedir().length)}` : p)

const ONCE = 'Yes, just this once'
const ALL = 'Yes, no sandbox for the rest of the session'
const NO = 'No'

export default function sandbox(pi: ExtensionAPI) {
  if (process.env.PI_SANDBOX === '0') return
  const cwd = process.cwd()
  const plain = createBashToolDefinition(cwd)
  const boxed = createBashToolDefinition(cwd, {
    spawnHook: ({ command, cwd, env }) => ({ command: wrapCommand(command, cwd), cwd, env }),
  })
  let grants = emptyGrants()

  const showStatus = (ctx: any) => {
    const allowed = [...grants.programs, ...(grants.exact.size ? [`${grants.exact.size} exact command(s)`] : []), ...[...grants.dirs].map(tilde)]
    const status = !hasBwrap
      ? NO_BOX
        ? '⚠ no sandbox on this system: every command asks first'
        : '⚠ sandbox without bwrap'
      : grants.all
        ? '🔓 no sandbox (this session)'
        : allowed.length
          ? `🔒 sandbox · outside: ${allowed.join(', ')}`
          : '🔒 sandbox'
    ctx?.ui?.setStatus?.('sandbox', status)
  }

  // Jev's independent check of the command against the model's description (see review.ts).
  const review = async (ctx: any, command: string, description: string) => {
    if (!ariaConfig().sandbox.jevReview) return null // Jev's review is off in Aria's settings
    try {
      const jev = ctx.modelRegistry.findOfType('classifier', CLASSIFIER.provider, CLASSIFIER.id)
      if (!jev) return null
      const result = await ctx.modelRegistry.classify(jev, reviewRequest(command, description), {
        signal: AbortSignal.timeout(REVIEW_TIMEOUT_MS),
      })
      return parseReview(result)
    } catch {
      return null
    }
  }

  /** Ask the user (terminal or dock); true when the user allowed it (once or for the session). A command that runs
   * arbitrary code can only be allowed again as that exact command. */
  const askOutside = async (ctx: any, command: string, reason: string | undefined) => {
    const description = reason?.trim() ?? ''
    const g = grantUnits(command)
    const forUnits = g && !g.broad ? `Yes, and don't ask again for ${g.units.join(', ')} this session` : null
    const forExact = forUnits ? null : 'Yes, and allow this exact command this session'
    const verdict = reviewLine(await review(ctx, command, description))
    const options = [ONCE, forUnits ?? forExact!, ALL, NO]
    // In Aria's dock the question is the chat's permission card; in the terminal, the same text as always.
    const title =
      ctx.mode === 'rpc'
        ? sandboxCard('sandbox', { summary: description, detail: commandPreview(command), jev: verdict.replace(/^Jev: /, '') }, options)
        : [NO_BOX ? 'Run this command? (no sandbox on this system)' : 'Run outside the sandbox?', description || '(no description given)', verdict, `› ${commandPreview(command)}`].join('\n')
    const pick = await ctx.ui.select(title, options, { signal: ctx.signal })
    if (pick === forUnits && g) g.units.forEach((u) => grants.programs.add(u))
    else if (pick === forExact) grants.exact.add(command.trim())
    else if (pick === ALL) grants.all = true
    else if (pick !== ONCE) return false
    showStatus(ctx)
    return true
  }

  pi.registerTool({
    ...boxed,
    label: 'bash (sandbox)',
    parameters: Type.Object({
      command: Type.String({ description: 'Bash command to execute' }),
      timeout: Type.Optional(Type.Number({ description: 'Timeout in seconds (optional, no default timeout)' })),
      outsideSandbox: Type.Optional(
        Type.Boolean({ description: 'Only after a command failed because of the sandbox: run it outside it. The user is asked first, unless they already allowed it for this session.' }),
      ),
      reason: Type.Optional(
        Type.String({
          description:
            `${NO_BOX ? 'Always (this system has no sandbox, so the user approves every command)' : 'With outsideSandbox'}: what the command does, for the user, in the user's language and at most 12 words (e.g. 'Restarts the project's Postgres container'). The user reads only this, not the command.`,
        }),
      ),
    }),
    async execute(id: string, params: any, signal: any, onUpdate: any, ctx: any) {
      const { command, timeout, outsideSandbox, reason } = params
      // Allowed for the session: straight outside, without failing inside first. Without bwrap every command asks.
      const route = bashRoute({ covered: covers(grants, command), outside: !!outsideSandbox, hasBwrap, canAsk: canAsk(ctx), platform: process.platform, mayLeave: ariaConfig().sandbox.askToLeave })
      if (route === 'plain') return plain.execute(id, { command, timeout }, signal, onUpdate, ctx)
      if (route === 'refuse-nobody') {
        return refuse(
          hasBwrap
            ? 'Running outside the sandbox needs the user\'s permission, and there is nobody here to ask. Find another way inside the project or explain what is needed.'
            : 'There is no sandbox on this system, so every command needs the user\'s permission, and there is nobody here to ask.',
        )
      }
      if (route === 'refuse-not-allowed') return refuse('Leaving the sandbox is not allowed (the user turned it off in Aria\'s settings): the command did not run. Do it inside the project, or tell the user what is needed.')
      if (route === 'refuse-no-bwrap') return refuse('The sandbox cannot run: bwrap is missing (sudo pacman -S bubblewrap). No commands run without it.')
      if (route === 'ask') {
        if (!(await askOutside(ctx, command, reason))) return refuse('The user said no: the command did not run.')
        return plain.execute(id, { command, timeout }, signal, onUpdate, ctx)
      }
      const r = await boxed.execute(id, { command, timeout }, signal, onUpdate, ctx)
      return looksBlocked(textOf(r)) ? withNote(r) : r
    },
  } as any)

  pi.on('tool_call', async (event: any, ctx: any) => {
    const name = event.toolName
    const here = ctx?.cwd ?? cwd
    // Subagents run in Pi's process: their cwd moves the sandbox and contextFiles are read directly.
    if (name === 'parallel') {
      const problem = subagentProblem(event.input, here)
      return problem ? { block: true, reason: `The sandbox does not allow that subagent: ${problem}.` } : undefined
    }
    // grep, find and ls run in Pi's process too; with no path they search the cwd.
    if (['grep', 'find', 'ls'].includes(name)) {
      const target = String(event.input?.path ?? '.')
      const hidden = name === 'grep' ? containsSecret(target, here) : isSecret(target, here)
      return hidden ? { block: true, reason: `${target} is or contains credentials: the sandbox does not allow searching there.` } : undefined
    }
    const path = String(event.input?.path ?? '')
    if (!path || !['read', 'write', 'edit'].includes(name)) return
    if (isSecret(path, here)) return { block: true, reason: `${path} holds credentials: the sandbox does not allow reading or changing it.` }
    // Aria's own files (its extensions, persona, settings): never by the model, whatever was granted.
    const ariaAgent = process.env.PI_CODING_AGENT_DIR && process.env.ARIA_PROJECTS_DIR ? process.env.PI_CODING_AGENT_DIR : null
    if (name !== 'read' && isAgentCode(resolvePath(path, here).real, ariaAgent)) {
      return { block: true, reason: `${path} is part of Aria itself: the sandbox does not allow changing it.` }
    }
    if (name === 'read' || isWritable(path, here)) return
    const { real } = resolvePath(path, here)
    if (coversPath(grants, real)) return
    if (canAsk(ctx)) {
      const forDir = `Yes, and allow writing in ${tilde(grantDir(real))} this session`
      const options = [ONCE, forDir, ALL, NO]
      const title =
        ctx.mode === 'rpc'
          ? sandboxCard('write', { path: tilde(real), detail: `${name} ${tilde(real)}` }, options)
          : `Write outside the project?\n${name} ${tilde(real)}`
      const pick = await ctx.ui.select(title, options, { signal: ctx.signal })
      if (pick === forDir) grants.dirs.add(grantDir(real))
      else if (pick === ALL) grants.all = true
      if (pick === ONCE || pick === forDir || pick === ALL) {
        showStatus(ctx)
        return
      }
    }
    return { block: true, reason: `${path} is outside the project, /tmp and the caches: the sandbox does not allow writing it${canAsk(ctx) ? ' (the user said no)' : ''}.` }
  })

  // A new session (startup, /new, /resume, fork) starts with no permissions.
  pi.on('session_start', (_event: any, ctx: any) => {
    grants = emptyGrants()
    showStatus(ctx)
  })

  pi.registerCommand('sandbox', {
    description: 'Show what the command sandbox can write and cannot see. /sandbox reset forgets the permissions of this session.',
    handler: async (args: string, ctx: any) => {
      const here = ctx?.cwd ?? cwd
      if (args.trim() === 'reset') {
        grants = emptyGrants()
        showStatus(ctx)
        ctx.ui.notify('Session permissions cleared: everything asks again.', 'info')
        return
      }
      const allowed = grants.all ? 'everything (sandbox off this session)' : [...grants.programs, ...[...grants.exact].map((c) => `"${c}"`), ...[...grants.dirs].map(tilde)].join(', ') || 'nothing'
      ctx.ui.notify(
        [
          `Project: ${projectRoot(here) ?? '(none: you are in ~ or /)'}`,
          `Writable: ${writableRoots(here).join(', ')}`,
          `Hidden: ${[...secretDirs(), ...secretFiles()].join(', ')}`,
          `Allowed outside this session: ${allowed}`,
          hasBwrap ? 'bwrap: ok' : NO_BOX ? 'bwrap: not available on this system (every command asks first)' : 'bwrap: not installed (commands are refused)',
        ].join('\n'),
        'info',
      )
    },
  })
}
