// Run: bun test extensions/jev-router/slice.test.ts
import { afterAll, describe, expect, test } from 'bun:test'
import { mkdirSync, mkdtempSync, rmSync, utimesSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { job as endedJob } from '../background/fake-dock.test.ts'
import { normalizeTasks } from '../tasks/tasks.ts'
import {
  buildBrief,
  firstOpenPhase,
  isPlanPath,
  latestPlan,
  nextOpenPhase,
  nextStepHint,
  outcome,
  parseMarker,
  parsePlan,
  phaseTasks,
  planCard,
  planProblem,
  reviewBrief,
} from './slice.ts'

const PLAN = `# Sign-up

## Context
A shared to-do list for small teams. Stack: Deno, Hono, SQLite. Expensive decisions: none.

## Assumptions
- Email and password only.

## Backlog
1. As a visitor, I want to sign up with email and password, so that I can keep my tasks.
2. As a member, I want to create a task, so that I see it in my list. Requires: #1

## Phase 1: Sign up
As a visitor, I want to sign up with email and password, so that I can keep my tasks.

Acceptance:
- Given a valid email and an 8-character password, when I sign up, then my account exists and I am signed in.

Out of scope: password reset.

- [x] registers a new account (endpoint)
- [~] refuses a taken email (endpoint)
- [ ] the form shows which field is wrong (manual check)

## Phase 2: Create a task
- [ ] creates a task (endpoint)
`

const dir = mkdtempSync('/tmp/slice-plan-')
afterAll(() => rmSync(dir, { recursive: true, force: true }))
mkdirSync(join(dir, 'docs', 'plans'), { recursive: true })
const planFile = join(dir, 'docs', 'plans', '2026-10-23-sign-up.md')
writeFileSync(planFile, PLAN)
writeFileSync(join(dir, 'docs', 'plans', 'notes.md'), '# Notes\nnothing to build\n')

const widgetPlan = '/home/ana/Projects/acme/checkout-widget/docs/plans/x.md'

describe('parsePlan', () => {
  const plan = parsePlan(PLAN)
  test('title and phases', () => {
    expect(plan.title).toBe('Sign-up')
    expect(plan.phases).toHaveLength(2)
    expect(plan.phases[0]).toEqual({
      number: 1,
      name: 'Sign up',
      tasks: [
        { title: 'registers a new account (endpoint)', status: 'done' },
        { title: 'refuses a taken email (endpoint)', status: 'active' },
        { title: 'the form shows which field is wrong (manual check)', status: 'todo' },
      ],
    })
    expect(plan.phases[1].number).toBe(2)
    expect(plan.phases[1].tasks).toEqual([{ title: 'creates a task (endpoint)', status: 'todo' }])
  })
  test('the acceptance bullets are not tasks', () => expect(plan.phases[0].tasks).toHaveLength(3))
  test('sections: text for Context, items for Assumptions and Backlog', () => {
    expect(plan.sections).toEqual([
      { heading: 'Context', text: 'A shared to-do list for small teams. Stack: Deno, Hono, SQLite. Expensive decisions: none.' },
      { heading: 'Assumptions', items: ['Email and password only.'] },
      {
        heading: 'Backlog',
        items: [
          'As a visitor, I want to sign up with email and password, so that I can keep my tasks.',
          'As a member, I want to create a task, so that I see it in my list. Requires: #1',
        ],
      },
    ])
  })
})

describe('phaseTasks', () => {
  const plan = parsePlan(PLAN)
  test("a phase's marks become the task list, tagged with the job", () => {
    const tasks = phaseTasks(plan, 1, 4)
    expect(tasks).toEqual([
      { id: 'p1-t1', title: 'registers a new account (endpoint)', status: 'done', agent: '#4' },
      { id: 'p1-t2', title: 'refuses a taken email (endpoint)', status: 'active', agent: '#4' },
      { id: 'p1-t3', title: 'the form shows which field is wrong (manual check)', status: 'todo', agent: '#4' },
    ])
    expect(tasks).toEqual(normalizeTasks({ tasks }))
  })
  test('another phase, and one the plan does not have', () => {
    expect(phaseTasks(plan, 2, 5)).toEqual([{ id: 'p2-t1', title: 'creates a task (endpoint)', status: 'todo', agent: '#5' }])
    expect(phaseTasks(plan, 9, 5)).toEqual([])
  })
})

describe('nextOpenPhase', () => {
  test('the first phase after a number that still has open tasks', () => {
    const plan = parsePlan(PLAN)
    expect(nextOpenPhase(plan, 1)?.number).toBe(2)
    expect(nextOpenPhase(plan, 2)).toBeNull()
    expect(nextOpenPhase(parsePlan(PLAN.replace('- [ ] creates', '- [x] creates')), 1)).toBeNull()
  })
})

describe('planProblem', () => {
  test('a plan passes', () => expect(planProblem('docs/plans/2026-10-23-sign-up.md', dir)).toBeNull())
  test('a missing file is refused', () => expect(planProblem('docs/plans/zzz.md', dir)).toContain("Can't find"))
  test('a file without phases is refused', () => expect(planProblem(join(dir, 'docs', 'plans', 'notes.md'), dir)).toContain('has no phases'))
})

describe('isPlanPath', () => {
  test('a plan of docs/plans', () => expect(isPlanPath('/home/ana/Projects/acme/checkout-widget/docs/plans/2026-10-23-x.md')).toBe(true))
  test('a subfolder is not', () => expect(isPlanPath('/home/ana/Projects/acme/checkout-widget/docs/plans/sub/x.md')).toBe(false))
  test('the old Superpowers folder is not', () =>
    expect(isPlanPath('/home/ana/Projects/acme/checkout-widget/docs/superpowers/plans/x.md')).toBe(false))
})

describe('latestPlan', () => {
  test('the newest .md of docs/plans of the git root', () => {
    mkdirSync(join(dir, '.git'), { recursive: true })
    utimesSync(planFile, new Date(2026, 9, 1), new Date(2026, 9, 1))
    utimesSync(join(dir, 'docs', 'plans', 'notes.md'), new Date(2026, 9, 2), new Date(2026, 9, 2))
    expect(latestPlan(join(dir, 'docs'))).toBe(join(dir, 'docs', 'plans', 'notes.md'))
  })
  test('null without docs/plans', () => {
    const empty = mkdtempSync('/tmp/slice-empty-')
    mkdirSync(join(empty, '.git'))
    expect(latestPlan(empty)).toBeNull()
    rmSync(empty, { recursive: true, force: true })
  })
})

describe('planCard', () => {
  const card = planCard(parsePlan(PLAN), 'docs/plans/2026-10-23-sign-up.md')
  test('what the terminal shows', () => {
    expect(card.aria).toBe(1)
    expect(card.kind).toBe('plan')
    expect(card.title).toBe('Sign-up')
    expect(card.path).toBe('docs/plans/2026-10-23-sign-up.md')
    expect(card.summary).toBe('Sign-up · 2 phases · 4 tasks')
    expect(card.detail).toBe('docs/plans/2026-10-23-sign-up.md')
    expect(card.options).toEqual([
      { value: 'approve', label: 'Approve and build', primary: true },
      { value: 'change', label: 'Change something' },
    ])
  })
  test('what the chat shows', () => {
    expect(card.sections).toHaveLength(3)
    expect(card.phases).toEqual([
      {
        name: 'Sign up',
        tasks: [
          'registers a new account (endpoint)',
          'refuses a taken email (endpoint)',
          'the form shows which field is wrong (manual check)',
        ],
      },
      { name: 'Create a task', tasks: ['creates a task (endpoint)'] },
    ])
  })
})

describe('firstOpenPhase', () => {
  test('the first phase with a task not done', () => expect(firstOpenPhase(parsePlan(PLAN))?.number).toBe(1))
  test('skips a finished phase', () => {
    const done = PLAN.replace('- [~]', '- [x]').replace('- [ ] the form', '- [x] the form')
    expect(firstOpenPhase(parsePlan(done))?.number).toBe(2)
  })
  test('null when all are done', () => {
    const all = PLAN.replace(/- \[[ ~]\]/g, '- [x]')
    expect(firstOpenPhase(parsePlan(all))).toBeNull()
  })
})

describe('briefs and markers', () => {
  const build = buildBrief({ plan: widgetPlan, phase: 1, name: 'Sign up', base: 'abc123', skill: '/s/SKILL.md' })
  test('the build brief', () => {
    expect(build.split('\n')[0]).toBe(`<!-- slice:build plan=${widgetPlan} phase=1 base=abc123 -->`)
    expect(build).toContain('Build phase 1 ("Sign up")')
    expect(build).toContain(widgetPlan)
    expect(build).toContain('/s/SKILL.md')
    expect(build).toContain('"Phase B: build"')
    expect(build).toContain('`SLICE DONE`, `SLICE QUESTION: <the question>`')
    expect(build).toContain('`SLICE BLOCKED: <why>`')
    expect(build).not.toContain('The owner answered')
  })
  test('an answer goes in the brief', () => {
    const again = buildBrief({ plan: widgetPlan, phase: 1, name: 'Sign up', base: 'abc123', skill: '/s/SKILL.md', answer: 'SQLite' })
    expect(again).toContain('The owner answered your question: SQLite')
  })
  test('the review brief', () => {
    const review = reviewBrief({ plan: widgetPlan, phase: 1, base: 'abc123', skill: '/p/ponytail-review/SKILL.md' })
    expect(review.split('\n')[0]).toBe(`<!-- slice:review plan=${widgetPlan} phase=1 base=abc123 -->`)
    expect(review).toContain('`abc123..HEAD`')
    expect(review).toContain('/p/ponytail-review/SKILL.md')
    expect(review).toContain('Change nothing')
    expect(review).toContain('MUST FIX: <n>')
  })
  test('parseMarker reads both markers', () => {
    expect(parseMarker(build)).toEqual({ kind: 'build', plan: widgetPlan, phase: 1, base: 'abc123' })
    const review = reviewBrief({ plan: widgetPlan, phase: 2, base: 'def456', skill: '/p/SKILL.md' })
    expect(parseMarker(review)).toEqual({ kind: 'review', plan: widgetPlan, phase: 2, base: 'def456' })
  })
  test('no marker, or not on the first line: null', () => {
    expect(parseMarker('fix the tests')).toBeNull()
    expect(parseMarker(`Please do this:\n${build}`)).toBeNull()
  })
})

describe('outcome', () => {
  test('done', () => expect(outcome('All green.\n\nSLICE DONE\n')).toEqual({ kind: 'done' }))
  test('a question', () =>
    expect(outcome('- Postgres\n- SQLite\nSLICE QUESTION: Postgres or SQLite?')).toEqual({ kind: 'question', text: 'Postgres or SQLite?' }))
  test('blocked', () => expect(outcome('SLICE BLOCKED: tests do not run')).toEqual({ kind: 'blocked', text: 'tests do not run' }))
  test('anything else', () => {
    expect(outcome('I think it is done')).toEqual({ kind: 'unknown' })
    expect(outcome('SLICE DONE\nand one more thing')).toEqual({ kind: 'unknown' })
    expect(outcome('')).toEqual({ kind: 'unknown' })
  })
})

describe('nextStepHint', () => {
  const paths = { skill: '/s/SKILL.md', review: '/p/ponytail-review/SKILL.md', head: () => 'fff999' }
  const buildTask = buildBrief({ plan: widgetPlan, phase: 1, name: 'Sign up', base: 'abc123', skill: '/s/SKILL.md' })
  const reviewTask = reviewBrief({ plan: widgetPlan, phase: 1, base: 'abc123', skill: paths.review })
  const phase1Done = PLAN.replace('- [~]', '- [x]').replace('- [ ] the form', '- [x] the form')
  const job = (task: string, state: string, result: string) => ({ task, state, result })

  test('a build that finished starts its review', () => {
    const hint = nextStepHint(job(buildTask, 'done', 'All green.\nSLICE DONE'), phase1Done, paths)!
    expect(hint.startsWith('[slice] Phase 1 ("Sign up") is built.')).toBe(true)
    expect(hint).toContain('writes false')
    expect(hint).toContain(reviewBrief({ plan: widgetPlan, phase: 1, base: 'abc123', skill: paths.review }))
  })
  test('a question goes to the owner, then the phase launches again from the same base', () => {
    const hint = nextStepHint(job(buildTask, 'done', '- Postgres\n- SQLite\nSLICE QUESTION: Postgres or SQLite?'), PLAN, paths)!
    expect(hint.startsWith('[slice] Phase 1 stopped with a question')).toBe(true)
    expect(hint).toContain('ask_user')
    expect(hint).toContain('commits true')
    expect(hint).toContain(buildBrief({ plan: widgetPlan, phase: 1, name: 'Sign up', base: 'abc123', skill: '/s/SKILL.md', answer: '<answer>' }))
  })
  test('a failed build did not finish', () => {
    const hint = nextStepHint(job(buildTask, 'failed', ''), PLAN, paths)!
    expect(hint).toContain('Phase 1 did not finish (')
    expect(hint).toContain('failed')
    expect(hint).toContain('Do not edit the code yourself.')
  })
  test('a blocked build gives its reason', () => {
    const hint = nextStepHint(job(buildTask, 'done', 'SLICE BLOCKED: tests do not run'), PLAN, paths)!
    expect(hint).toContain('did not finish (tests do not run)')
  })
  test('SLICE DONE with open tasks is blocked', () => {
    const hint = nextStepHint(job(buildTask, 'done', 'SLICE DONE'), PLAN, paths)!
    expect(hint).toContain('did not finish (the plan still has open tasks)')
  })
  test('a review with a next phase launches it from the current head', () => {
    const hint = nextStepHint(job(reviewTask, 'done', 'MUST FIX: 0'), phase1Done, paths)!
    expect(hint.startsWith('[slice] The review of phase 1 is in the job\'s result.')).toBe(true)
    expect(hint).toContain('title "Create a task"')
    expect(hint).toContain(buildBrief({ plan: widgetPlan, phase: 2, name: 'Create a task', base: 'fff999', skill: '/s/SKILL.md' }))
  })
  test('a review on the last phase offers the next story', () => {
    const all = PLAN.replace(/- \[[ ~]\]/g, '- [x]')
    const hint = nextStepHint(job(reviewTask, 'done', 'MUST FIX: 0'), all, paths)!
    expect(hint).toContain('The plan is done.')
    expect(hint).toContain('/plan')
    expect(hint).not.toContain('slice:build')
  })
  test('a review that did not finish goes on to the manual checks', () => {
    const hint = nextStepHint(job(reviewTask, 'failed', ''), phase1Done, paths)!
    expect(hint).toContain('The review of phase 1 did not finish (')
    expect(hint).toContain('manual checks')
  })
  test('a job without a marker: null', () => expect(nextStepHint(job('fix the tests', 'done', 'SLICE DONE'), PLAN, paths)).toBeNull())
  test('the job the background extension sends carries a marker the hint can read', () => {
    const sent = endedJob({ kind: 'agent', title: 'Sign up', command: null, task: buildTask, state: 'done', isolation: 'folder', writes: true, result: 'SLICE DONE' })
    expect(parseMarker(String(sent.task ?? ''))).toEqual({ kind: 'build', plan: widgetPlan, phase: 1, base: 'abc123' })
    expect(nextStepHint(sent, phase1Done, paths)).not.toBeNull()
    expect(nextStepHint(endedJob({ kind: 'agent', task: 'fix the tests', state: 'done', result: 'SLICE DONE' }), PLAN, paths)).toBeNull()
  })
})
