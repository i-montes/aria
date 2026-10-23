// Run: bun test extensions/jev-router/modes.test.ts
import { afterAll, describe, expect, test } from 'bun:test'
import { mkdirSync, mkdtempSync, rmSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { exhausted, parseGoUsage } from './usage.ts'

// Two git repos in a projects folder of their own, outside /tmp (which plan mode may always write): modes.ts reads the
// folder (ARIA_PROJECTS_DIR) once, when it loads.
mkdirSync(join(homedir(), '.cache'), { recursive: true })
const projects = mkdtempSync(join(homedir(), '.cache', 'aria-plan-test-'))
afterAll(() => rmSync(projects, { recursive: true, force: true }))
const widget = join(projects, 'acme', 'checkout-widget')
const gateway = join(projects, 'acme', 'checkout-api')
for (const repo of [widget, gateway]) mkdirSync(join(repo, '.git'), { recursive: true })
const savedProjects = process.env.ARIA_PROJECTS_DIR
process.env.ARIA_PROJECTS_DIR = projects
const { executePrompt, modeHint, planMayWrite, routedMode, planPrompt, PLAN_MODEL, SLICE_SKILL, planModel, planQuotaThreshold, planThinking, shouldClassify } = await import('./modes.ts')
if (savedProjects === undefined) delete process.env.ARIA_PROJECTS_DIR
else process.env.ARIA_PROJECTS_DIR = savedProjects

describe('planMayWrite', () => {
  test('a plan in the repo Pi runs in', () => expect(planMayWrite('docs/plans/x.md', widget)).toBe(true))
  test('a plan in another repo under the projects folder', () => expect(planMayWrite(`${gateway}/docs/plans/x.md`, widget)).toBe(true))
  test('.git of the repo (commit)', () => expect(planMayWrite(`${widget}/.git/index`, widget)).toBe(true))
  test("the repo root's .gitignore", () => expect(planMayWrite(`${widget}/.gitignore`, widget)).toBe(true))
  test('a nested .gitignore is refused', () => expect(planMayWrite(`${widget}/src/.gitignore`, widget)).toBe(false))
  test('the old Superpowers folder is refused', () => expect(planMayWrite('docs/superpowers/specs/x.md', widget)).toBe(false))
  test('/tmp', () => expect(planMayWrite('/tmp/notes.md', widget)).toBe(true))
  test('source code is refused', () => expect(planMayWrite('src/main.ts', widget)).toBe(false))
  test('outside the projects folder is refused', () => expect(planMayWrite(join(homedir(), '.bashrc'), widget)).toBe(false))
})

describe('Go usage', () => {
  const body = (r: number, w: number) => ({
    usage: { rolling: { status: 'ok', percent: r }, weekly: { status: 'ok', percent: w }, monthly: { status: 'ok', percent: 46 } },
  })
  test('weekly at 93% is exhausted', () => expect(exhausted(parseGoUsage(body(0, 93))!, 90)?.name).toBe('weekly'))
  test('5 h first', () => expect(exhausted(parseGoUsage(body(95, 93))!, 90)?.name).toBe('5-hour'))
  test('below the threshold', () => expect(exhausted(parseGoUsage(body(10, 50))!, 90)).toBeNull())
  test('garbage parses to null', () => expect(parseGoUsage({ nope: 1 })).toBeNull())
})

describe('which modes the router works in', () => {
  test('the terminal, print and Aria (rpc) are routed; subagents (json) are not', () => {
    expect(routedMode('tui')).toBe(true)
    expect(routedMode('print')).toBe(true)
    expect(routedMode('rpc')).toBe(true)
    expect(routedMode('json')).toBe(false)
    expect(routedMode(undefined)).toBe(false)
  })
})

describe("the /plan flow is slice's phase A", () => {
  test('points at the slice skill and ends with plan_ready', () => {
    const prompt = planPrompt('a recipes app')
    expect(prompt).toContain('Phase A: frame')
    expect(prompt).toContain(SLICE_SKILL)
    expect(prompt).toContain('plan_ready')
    expect(prompt).toContain('docs/plans/')
    expect(prompt).not.toContain('brainstorming')
    expect(prompt).not.toContain('writing-plans')
  })
})

describe('the plan phase prompt', () => {
  test('launches the phase in a folder with commits, word for word', () => {
    const prompt = executePrompt('BRIEF TEXT', 'Sign up')
    expect(prompt).toContain('background_agent')
    expect(prompt).toContain('commits true')
    expect(prompt).toContain('isolation "folder"')
    expect(prompt).toContain('BRIEF TEXT')
  })
})

describe('the plan mode hint', () => {
  test('mentions slice and docs/plans/', () => {
    const hint = modeHint('plan')
    expect(hint).toContain('slice')
    expect(hint).toContain('docs/plans/')
  })
})

describe("plan mode follows Aria's settings", () => {
  const { mkdtempSync: mk, writeFileSync: wr } = require('node:fs')
  const { tmpdir: tmp } = require('node:os')
  const saved = process.env.PI_CODING_AGENT_DIR
  const withConfig = (config: unknown, run: () => void) => {
    const dir = mk(join(tmp(), 'aria-modes-'))
    if (config) wr(join(dir, 'aria.json'), JSON.stringify(config))
    process.env.PI_CODING_AGENT_DIR = dir
    try { run() } finally { if (saved === undefined) delete process.env.PI_CODING_AGENT_DIR; else process.env.PI_CODING_AGENT_DIR = saved }
  }

  test('the plan model and effort come from the config', () =>
    withConfig({ models: { plan: { provider: 'anthropic', id: 'claude-x' }, planThinking: 'max' }, quota: { planSwitchAt: 80 } }, () => {
      expect(planModel()).toEqual({ provider: 'anthropic', id: 'claude-x' })
      expect(planThinking()).toBe('max')
      expect(planQuotaThreshold()).toBe(80)
    }))

  test('without a config the plan model is mimo-v2.6-pro at high, switching at 90%', () =>
    withConfig(null, () => {
      expect(planModel()).toEqual(PLAN_MODEL)
      expect(planThinking()).toBe('high')
      expect(planQuotaThreshold()).toBe(90)
    }))
})

describe('the router switch', () => {
  const { DEFAULTS } = require('../lib/config.ts')
  test('router off means no classification (and so no hidden hint)', () => {
    expect(shouldClassify('rpc', { ...DEFAULTS, router: { enabled: false } })).toBe(false)
    expect(shouldClassify('rpc', DEFAULTS)).toBe(true)
    expect(shouldClassify('json', DEFAULTS)).toBe(false)
  })
})
