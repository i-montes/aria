// Slice: reading a plan (`docs/plans/<date>-<slug>.md`), the cards and briefs built from it, and the hint the router gives
// after a slice job ends. Pure functions on text; the only file reads are planProblem and latestPlan.

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { isAbsolute, join, resolve } from 'node:path'
import { normalizeTasks, type Task as ListedTask } from '../tasks/tasks.ts'
import { gitRoot } from './modes.ts'

export type Task = { title: string; status: 'todo' | 'active' | 'done' }
export type Phase = { number: number; name: string; tasks: Task[] }
// `heading`, `text` and `items` are what the chat's document card takes (ui/chat/parts/document.js).
export type Section = { heading: string; text?: string; items?: string[] }
export type Plan = { title: string; sections: Section[]; phases: Phase[] }

const MARKS: Record<string, Task['status']> = { ' ': 'todo', '~': 'active', x: 'done' }

export function parsePlan(text: string): Plan {
  const plan: Plan = { title: '', sections: [], phases: [] }
  let phase: Phase | null = null
  let section: { heading: string; lines: string[] } | null = null
  const sections: { heading: string; lines: string[] }[] = []
  for (const line of text.split('\n')) {
    const heading = /^## (.+?)\s*$/.exec(line)
    if (heading) {
      const p = /^Phase (\d+):\s*(.+)$/.exec(heading[1])
      phase = p ? { number: Number(p[1]), name: p[2], tasks: [] } : null
      if (phase) plan.phases.push(phase)
      section = phase ? null : { heading: heading[1], lines: [] }
      if (section) sections.push(section)
      continue
    }
    if (!plan.title && !phase && !section) {
      const title = /^# (.+?)\s*$/.exec(line)
      if (title) plan.title = title[1]
      continue
    }
    const task = /^- \[([ ~x])\] (.+?)\s*$/.exec(line)
    if (phase && task) phase.tasks.push({ title: task[2], status: MARKS[task[1]] })
    else section?.lines.push(line)
  }
  for (const s of sections) {
    const items = s.lines.flatMap((l) => /^(?:\d+\.|-) (.+?)\s*$/.exec(l)?.[1] ?? [])
    const paragraph = s.lines.filter((l) => l.trim() && !/^(?:\d+\.|-) /.test(l)).map((l) => l.trim()).join(' ')
    if (items.length || paragraph) plan.sections.push({ heading: s.heading, ...(paragraph && { text: paragraph }), ...(items.length && { items }) })
  }
  return plan
}

/** A slice plan lives directly in `docs/plans/`: not a subfolder, not the old Superpowers folder. */
export function isPlanPath(path: string): boolean {
  return /\/docs\/plans\/[^/]+\.md$/.test(path)
}

/** Why `path` is not a plan the build can run, or null when it is. */
export function planProblem(path: string, cwd: string): string | null {
  const full = isAbsolute(path) ? path : resolve(cwd, path)
  if (!existsSync(full)) return `Can't find \`${path}\`.`
  let text = ''
  try {
    text = readFileSync(full, 'utf8')
  } catch {
    return `Can't read \`${path}\`.`
  }
  const first = parsePlan(text).phases.find((p) => p.number === 1)
  if (!first?.tasks.length) return `\`${path}\` has no phases: it needs "## Phase 1: <story>" with at least one task ("- [ ] …").`
  return null
}

/** The newest plan in `docs/plans/` of the repo Pi runs in, or null. */
export function latestPlan(cwd: string): string | null {
  const dir = join(gitRoot(cwd) ?? cwd, 'docs', 'plans')
  if (!existsSync(dir)) return null
  const files = readdirSync(dir)
    .filter((f) => f.endsWith('.md'))
    .map((f) => join(dir, f))
    .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs)
  return files[0] ?? null
}

const count = (n: number, noun: string) => `${n} ${noun}${n === 1 ? '' : 's'}`

/** The dialog `plan_ready` shows: the chat draws `sections` and `phases`; the terminal, `summary`, `detail` and `options`. */
export function planCard(plan: Plan, path: string) {
  const tasks = plan.phases.reduce((n, p) => n + p.tasks.length, 0)
  return {
    aria: 1,
    kind: 'plan',
    title: plan.title,
    path,
    summary: `${plan.title} · ${count(plan.phases.length, 'phase')} · ${count(tasks, 'task')}`,
    detail: path,
    sections: plan.sections,
    phases: plan.phases.map((p) => ({ name: p.name, tasks: p.tasks.map((t) => t.title) })),
    options: [
      { value: 'approve', label: 'Approve and build', primary: true },
      { value: 'change', label: 'Change something' },
    ],
  }
}

const isOpen = (p: Phase) => p.tasks.some((t) => t.status !== 'done')

export const firstOpenPhase = (plan: Plan): Phase | null => plan.phases.find(isOpen) ?? null
/** The first phase after `number` that still has open tasks. */
export const nextOpenPhase = (plan: Plan, number: number): Phase | null => plan.phases.find((p) => p.number > number && isOpen(p)) ?? null

/** A phase's marks as the task list the dock and the terminal draw (`tasks_update`'s shape), each tagged with job `#job`. */
export function phaseTasks(plan: Plan, number: number, job: number): ListedTask[] {
  const tasks = plan.phases.find((p) => p.number === number)?.tasks ?? []
  return normalizeTasks({ tasks: tasks.map((t, i) => ({ id: `p${number}-t${i + 1}`, title: t.title, status: t.status, agent: `#${job}` })) })
}

type BriefBase = { plan: string; phase: number; base: string }

export function buildBrief(b: BriefBase & { name: string; skill: string; answer?: string }): string {
  return [
    `<!-- slice:build plan=${b.plan} phase=${b.phase} base=${b.base} -->`,
    `Build phase ${b.phase} ("${b.name}") of the plan ${b.plan}.`,
    `Read the plan first: the phase's story, acceptance criteria, out of scope and task list. Then read and follow the section "Phase B: build" of the slice skill (${b.skill}).`,
    'Start from the first task of the phase not marked [x].',
    ...(b.answer ? ['', `The owner answered your question: ${b.answer}`] : []),
    '',
    'End your last message with exactly one of these lines: `SLICE DONE`, `SLICE QUESTION: <the question>` (with the options as `- ` lines above it), `SLICE BLOCKED: <why>`.',
  ].join('\n')
}

export function reviewBrief(b: BriefBase & { skill: string }): string {
  return [
    `<!-- slice:review plan=${b.plan} phase=${b.phase} base=${b.base} -->`,
    `Review phase ${b.phase} of the plan ${b.plan}: the commits \`${b.base}..HEAD\` of this repository.`,
    `Read and follow the review skill (${b.skill}). Change nothing.`,
    'End with your findings, most important first; the first line says how many must be fixed (`MUST FIX: <n>`).',
  ].join('\n')
}

export type Marker = BriefBase & { kind: 'build' | 'review' }

/** The marker a brief starts with, so a job that ends can be told apart from plain background work. */
export function parseMarker(task: string): Marker | null {
  const m = /^<!-- slice:(build|review) plan=(.+?) phase=(\d+) base=(\S+) -->/.exec(task)
  return m ? { kind: m[1] as Marker['kind'], plan: m[2], phase: Number(m[3]), base: m[4] } : null
}

export type Outcome = { kind: 'done' | 'unknown' } | { kind: 'question' | 'blocked'; text: string }

/** What a build job said on its last non-empty line. */
export function outcome(result: string): Outcome {
  const last = result.split('\n').map((l) => l.trim()).filter(Boolean).pop() ?? ''
  if (last === 'SLICE DONE') return { kind: 'done' }
  const q = /^SLICE (QUESTION|BLOCKED):\s*(.*)$/.exec(last)
  return q ? { kind: q[1] === 'QUESTION' ? 'question' : 'blocked', text: q[2] } : { kind: 'unknown' }
}

// The `details.job` the background extension sends when a job ends (extensions/background/text.ts), as far as this needs.
type EndedJob = { task?: string | null; state?: string; result?: string | null }
type Paths = { skill: string; review: string; head: () => string }

/** The hidden hint the router gives after a slice job ends, or null when the job is not a slice job. */
export function nextStepHint(job: EndedJob, planText: string, paths: Paths): string | null {
  const marker = parseMarker(String(job.task ?? ''))
  if (!marker) return null
  const plan = parsePlan(planText)
  const phase = plan.phases.find((p) => p.number === marker.phase)
  const label = `Phase ${marker.phase}`
  const state = job.state ?? 'unknown'
  const result = String(job.result ?? '')
  const brief = (number: number, name: string, base: string, answer?: string) =>
    buildBrief({ plan: marker.plan, phase: number, name, base, skill: paths.skill, answer })

  if (marker.kind === 'review') {
    if (state !== 'done') return `[slice] The review of phase ${marker.phase} did not finish (${state}). Tell the owner and go on to the manual checks.`
    const review = `[slice] The review of phase ${marker.phase} is in the job's result. Tell the owner the must-fix findings in a few lines; if there are any, ask with ask_user: fix them now (a build job with commits true and a brief listing them) or move on. Go through the phase's manual checks with the owner. If a decision applies to the whole project, propose one line for CLAUDE.md or AGENTS.md.`
    const next = nextOpenPhase(plan, marker.phase)
    if (!next) return `${review} The plan is done. Offer the next story of the backlog (the owner can plan it with /plan).`
    return `${review} Then launch phase ${next.number}: background_agent with commits true, isolation "folder", title "${next.name}", and this task word for word:\n\n${brief(next.number, next.name, paths.head())}`
  }

  const said = state === 'done' ? outcome(result) : null
  if (said?.kind === 'question') {
    return `[slice] ${label} stopped with a question for the owner: ask it with ask_user (its options, your recommended one first). Then launch the phase again: background_agent with commits true, isolation "folder", and this task word for word, putting the owner's answer after "The owner answered your question:":\n\n${brief(marker.phase, phase?.name ?? '', marker.base, '<answer>')}`
  }
  if (said?.kind === 'done' && phase && !isOpen(phase)) {
    return `[slice] ${label} ("${phase.name}") is built. Launch its review now: background_agent with writes false, title "Review ${marker.phase}" in the owner's language, and this task, word for word:\n\n${reviewBrief({ plan: marker.plan, phase: marker.phase, base: marker.base, skill: paths.review })}`
  }
  const why =
    said?.kind === 'blocked' ? said.text
    : said?.kind === 'done' ? (phase ? 'the plan still has open tasks' : `the plan has no phase ${marker.phase}`)
    : state === 'done' ? 'it ended without a SLICE line'
    : `the job ${state}`
  return `[slice] ${label} did not finish (${why}). Tell the owner in two lines what happened and which tasks are done (the plan's marks); ask with ask_user whether to launch it again (/execute) or change the plan. Do not edit the code yourself.`
}
