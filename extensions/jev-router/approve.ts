// `plan_ready`: shows the owner the plan card and waits for the answer. Kept apart from index.ts (which needs Pi) so it can
// be tested with a fake dialog.
import { readFileSync } from 'node:fs'
import { isAbsolute, resolve } from 'node:path'
import type { Mode } from './modes.ts'
import { isPlanPath, parsePlan, planCard, planProblem } from './slice.ts'

export type Approval = { text: string; isError?: boolean; details?: any; approved?: string }

const fail = (text: string): Approval => ({ text, isError: true })

/** Asks and waits; Esc (the run's abort signal) makes Pi drop the dialog, which counts as "change". */
export async function askApproval(
  ctx: any,
  path: string,
  cwd: string,
  { mode, noAsk, signal }: { mode: Mode; noAsk: boolean; signal?: AbortSignal },
): Promise<Approval> {
  if (mode !== 'plan') return fail('plan_ready only works in plan mode (/plan).')
  // A scheduled task's tab runs unattended: nobody would answer, and the run would wait forever.
  if (noAsk) return fail('Nobody can approve a plan here: this is a scheduled task.')
  const full = isAbsolute(path) ? path : resolve(cwd, path)
  const problem = planProblem(path, cwd) ?? (isPlanPath(full) ? null : `\`${path}\` is not in docs/plans/: a plan lives in \`docs/plans/<YYYY-MM-DD>-<slug>.md\`.`)
  if (problem) return fail(problem)
  const card = planCard(parsePlan(readFileSync(full, 'utf8')), full)
  const { options: _options, ...plan } = card
  const pick = await ctx.ui.select(JSON.stringify(card), ['approve', 'change'], { signal })
  if (pick === 'approve') {
    return { text: 'Approved: the build starts when this answer ends. End it now with one short line.', details: { status: 'approved', plan }, approved: full }
  }
  return {
    text: 'Not approved yet: ask the owner what to change (one ask_user question or in your answer), edit the plan, commit it and call plan_ready again.',
    details: { status: 'draft', plan },
  }
}
