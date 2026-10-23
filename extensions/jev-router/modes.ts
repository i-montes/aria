// The plan mode the user turns on with a command. While it is on, the router stops routing (no Jev
// labels, no thinking changes) and sends a fixed hint instead.
//
// - plan: phase A of the slice skill, on a stronger model: one round of questions, then the plan in
//   `docs/plans/` of the repo that receives the change, approved with `plan_ready`. Only `docs/plans/`,
//   the repo root's `.gitignore`, `.git` of a git repo under ~/Projects and /tmp may be written. The
//   user starts it with /plan, or accepts the assistant's offer (the router's `plan=offer`, the
//   `start_plan` tool).
// There is no execute mode any more: approving the plan, or /execute, launches phase B in the
// background and the chat stays routed.
import { existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import { agentDir, projectsDir } from '../lib/agent-dir.ts'
import { type AriaConfig, ariaConfig, DEFAULTS, type Model } from '../lib/config.ts'

export type Mode = 'normal' | 'plan'

// The user picked this model for planning (2026-10-21, mimo-v2.6-pro after glm-5.3 and deepseek-v4-pro);
// it stays inside the OpenCode Go plan.
export const PLAN_MODEL = DEFAULTS.models.plan
/** The planning model: Aria's setting, or PLAN_MODEL. */
export const planModel = (): Model => ariaConfig().models.plan
// Used instead when the Go quota is nearly spent (see usage.ts): the same free model as the
// quota-fallback extension (cat-jam-free since 2026-10-14; longcat-2.5-preview-free before, which
// looped once in /plan and was the slowest in the coding benchmark).
export { FREE_MODEL as FREE_PLAN_MODEL } from '../quota-fallback/fallback.ts'
export const PLAN_THINKING = DEFAULTS.models.planThinking
export const planThinking = (): string => ariaConfig().models.planThinking
// At or above this percentage of the 5-hour or the weekly Go window, /plan uses the free model.
/** Modes with a person behind them: the terminal, `pi -p`, and Aria's dock (RPC). Subagents (`json`) are not routed. */
export const routedMode = (mode: string | undefined): boolean => mode === 'tui' || mode === 'print' || mode === 'rpc'

/** Jev classifies a message (and adds the hidden hint) in a routed mode, unless Aria's settings turn the router off.
 * Plan mode, the process guard and expert assignment do not depend on it. */
export const shouldClassify = (mode: string | undefined, config: AriaConfig): boolean => routedMode(mode) && config.router.enabled

export const QUOTA_THRESHOLD = DEFAULTS.quota.planSwitchAt
export const planQuotaThreshold = (): number => ariaConfig().quota.planSwitchAt
// /plan warns before starting in a session that already carries this much context.
export const LARGE_CONTEXT_TOKENS = 100_000

export const SLICE_SKILL = join(agentDir(), 'skills', 'slice', 'SKILL.md')
export const REVIEW_SKILL = join(agentDir(), 'git', 'github.com', 'DietrichGebert', 'ponytail', 'skills', 'ponytail-review', 'SKILL.md')

const PROJECTS = projectsDir()
const TMP_DIRS = [...new Set(['/tmp', tmpdir()])]

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

const within = (dir: string, p: string) => p === dir || p.startsWith(`${dir}/`)

/**
 * May plan mode write `rawPath`? Yes inside /tmp, and inside `docs/plans/`, `.git/` or the root's
 * `.gitignore` of a git repo under ~/Projects — the repo that receives the change, which need not
 * be the one Pi was opened in.
 */
export function planMayWrite(rawPath: string, cwd: string): boolean {
  const path = isAbsolute(rawPath) ? rawPath : resolve(cwd, rawPath)
  if (TMP_DIRS.some((d) => within(d, path))) return true
  if (!within(PROJECTS, path)) return false
  const root = gitRoot(path)
  if (!root || !within(PROJECTS, root)) return false
  return within(join(root, 'docs', 'plans'), path) || within(join(root, '.git'), path) || path === join(root, '.gitignore')
}

/** The directories plan mode may write, as text for the bash classifier. */
export const PLAN_DIRS_TEXT = [...TMP_DIRS, `${PROJECTS}/<any git repo>/docs/plans`, `${PROJECTS}/<any git repo>/.git`, `${PROJECTS}/<any git repo>/.gitignore`]

const RISK_QUESTION =
  'When reviewing the plan, for each risk first ask yourself: what happens to an ordinary user who repeats this in real life (for example, does it again without changing anything)? Tell me starting with the most frequent case, with a concrete example, and only then the rare or abuse cases.'

export function planPrompt(idea: string): string {
  return [
    `I want to plan this: ${idea}`,
    '',
    `Read with \`read\` the slice skill (${SLICE_SKILL}) and follow its section "Phase A: frame".`,
    'Save the plan in `docs/plans/<YYYY-MM-DD>-<slug>.md` of the repo that receives the change and tell me at the start which repo and path. Commit it there (only that file, and `.gitignore` if you changed it), then call `plan_ready` with its path: it shows me the plan and asks me to approve it.',
    RISK_QUESTION,
  ].join('\n')
}

export function executePrompt(brief: string, name: string): string {
  return [
    `Build the plan's next phase ("${name}") in the background: call \`background_agent\` with commits true, isolation "folder", title "${name}" (in my language, at most 6 words) and this task, word for word:`,
    '',
    brief,
    '',
    'Then end your answer with one short line: the phase is being built and I can keep talking to you.',
  ].join('\n')
}

export function modeHint(_mode: Mode): string {
  return `[router] plan mode (slice) on: phase A of the slice skill: one round of questions, then the plan; implement nothing. You may only write to docs/plans/ and .gitignore of a git repo under ${PROJECTS} (the one receiving the change), its .git (the commit), and /tmp. Plan mode ends when I approve the plan.`
}

export const PLAN_BLOCK_REASON =
  'Blocked by the router: plan mode only writes the plan (docs/plans/ of a repo under ~/Projects), .gitignore, the commit and /tmp. The build comes after approval.'
