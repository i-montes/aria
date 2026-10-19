// Superpowers modes the user turns on with a command. While one is on, the router stops routing
// (no Jev labels, no thinking changes) and sends a fixed hint instead.
//
// - plan: design and plan with Superpowers' brainstorming and writing-plans skills, on a stronger
//   model. Only `docs/superpowers/` and `.git` of a git repo under ~/Projects (the spec and plan
//   live in the repo that receives the change) and /tmp may be written. It ends by itself once the
//   plan is written, or when the user sends something that is not part of planning.
// - execute: carry out an approved plan with subagent-driven-development, back on the default
//   model, with normal writes. It only accepts a plan, never a spec.
import { existsSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import { agentDir, projectsDir } from '../lib/agent-dir.ts'
import { ariaConfig, DEFAULTS, type Model } from '../lib/config.ts'

export type Mode = 'normal' | 'plan' | 'execute'

// The user picked this model for planning (2026-10-13, trying glm-5.3 after deepseek-v4-pro); it
// stays inside the OpenCode Go plan.
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

export const QUOTA_THRESHOLD = DEFAULTS.quota.planSwitchAt
export const planQuotaThreshold = (): number => ariaConfig().quota.planSwitchAt
// /plan warns before starting in a session that already carries this much context.
export const LARGE_CONTEXT_TOKENS = 100_000

export const SKILLS_DIR = join(agentDir(), 'git', 'github.com', 'obra', 'superpowers', 'skills')
const skill = (name: string) => join(SKILLS_DIR, name, 'SKILL.md')

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
 * May plan mode write `rawPath`? Yes inside /tmp, and inside `docs/superpowers/` or `.git/` of
 * a git repo under ~/Projects — the repo that receives the change, which need not be the one
 * Pi was opened in.
 */
export function planMayWrite(rawPath: string, cwd: string): boolean {
  const path = isAbsolute(rawPath) ? rawPath : resolve(cwd, rawPath)
  if (TMP_DIRS.some((d) => within(d, path))) return true
  if (!within(PROJECTS, path)) return false
  const root = gitRoot(path)
  if (!root || !within(PROJECTS, root)) return false
  return within(join(root, 'docs', 'superpowers'), path) || within(join(root, '.git'), path)
}

/** The directories plan mode may write, as text for the bash classifier. */
export const PLAN_DIRS_TEXT = [...TMP_DIRS, `${PROJECTS}/<any git repo>/docs/superpowers`, `${PROJECTS}/<any git repo>/.git`]

/** A plan file written in plan mode ends planning. */
export const isPlanPath = (path: string) => /\/docs\/superpowers\/plans\/[^/]+\.md$/.test(path)

const RISK_QUESTION =
  'When reviewing the spec and the plan, for each risk first ask yourself: what happens to an ordinary user who repeats this in real life (for example, does it again without changing anything)? Tell me starting with the most frequent case, with a concrete example, and only then the rare or abuse cases.'

export function planPrompt(idea: string): string {
  return [
    `I want to plan this: ${idea}`,
    '',
    `Follow the Superpowers flow. Read with \`read\` and follow the whole \`brainstorming\` skill (${skill('brainstorming')}). Don't offer the visual companion. Once I approve the design and the written spec, read and follow \`writing-plans\` (${skill('writing-plans')}).`,
    'Ask the brainstorming questions (and the choice between approaches) with the `ask_user` tool, in batches: put in one call up to 4 questions that do not depend on each other\'s answers, each with 2-4 options and your recommended one first; a question that depends on an earlier answer goes in the next call. This replaces the skill\'s "one question at a time". To approve the design or the written spec, ask with `ask_user` too.',
    'Save the spec and the plan in `docs/superpowers/` of the repo that receives the change (if it touches several repos, the one that receives most of it; the plan says which tasks go in each repo). Tell me at the start which repo and path you will save them in.',
    RISK_QUESTION,
    'When the plan is done, don\'t offer me a choice of how to execute it: tell me to review it and run it with `/execute <plan path>`.',
  ].join('\n')
}

export function executePrompt(plan: string): string {
  return [
    `Execute the plan \`${plan}\`.`,
    '',
    `Read with \`read\` and follow the \`subagent-driven-development\` skill (${skill('subagent-driven-development')}); its templates and scripts are in that same folder. Apply the equivalences in the system prompt's "Superpowers" section.`,
    `In the final branch review, add this question to the reviewer's brief: ${RISK_QUESTION}`,
  ].join('\n')
}

/**
 * Is `path` an implementation plan (as opposed to a spec, or nothing)? Returns the reason it is
 * not, or null when it is.
 */
export function notAPlan(path: string, cwd: string): string | null {
  const full = isAbsolute(path) ? path : resolve(cwd, path)
  if (!existsSync(full)) return `Can't find \`${path}\`.`
  if (/\/specs\//.test(full)) return `\`${path}\` is a spec (design), not a plan.`
  let text = ''
  try {
    text = readFileSync(full, 'utf8')
  } catch {
    return `Can't read \`${path}\`.`
  }
  if (!/^#{2,3} Task \d+/m.test(text) && !/Implementation Plan/i.test(text)) {
    return `\`${path}\` doesn't look like an implementation plan: it has no tasks ("### Task 1…").`
  }
  return null
}

export function modeHint(mode: Mode): string {
  if (mode === 'plan') {
    return `[router] plan mode (Superpowers) on: design and plan following brainstorming and writing-plans; implement nothing. You may only write to docs/superpowers/ (spec and plan) and .git of a git repo under ${PROJECTS} (the one receiving the change), and to /tmp. Plan mode turns itself off once the plan is written.`
  }
  return '[router] execute mode (Superpowers) on: follow subagent-driven-development with the equivalences in the "Superpowers" section. The user leaves it with /execute off.'
}

export const PLAN_BLOCK_REASON =
  'Blocked by the router: plan mode only writes the spec and the plan (docs/superpowers/ of a repo under ~/Projects), their commit and /tmp. Implementation comes with /execute.'
