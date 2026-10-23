// Run: bun test extensions/jev-router/approve.test.ts
import { afterAll, describe, expect, test } from 'bun:test'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { askApproval, pendingBuild } from './approve.ts'
import { parsePlan, planCard } from './slice.ts'

const PLAN = `# Checkout widget

## Context
A widget for the checkout page.

## Phase 1: Pay
As a buyer, I want to pay, so that I get my order.

- [ ] charges the card
`

const root = mkdtempSync(join(tmpdir(), 'approve-'))
afterAll(() => rmSync(root, { recursive: true, force: true }))
mkdirSync(join(root, 'docs', 'plans'), { recursive: true })
const plan = join(root, 'docs', 'plans', '2026-10-23-acme-checkout-widget.md')
writeFileSync(plan, PLAN)
writeFileSync(join(root, 'notes.md'), PLAN)

const fake = (answer?: string) => {
  const calls: any[] = []
  const ctx = { ui: { select: async (...args: any[]) => (calls.push(args), answer) } }
  return { ctx, calls }
}
const plain = { mode: 'plan' as const, noAsk: false }

describe('askApproval', () => {
  test('refuses outside plan mode, without a dialog', async () => {
    const { ctx, calls } = fake('approve')
    const r = await askApproval(ctx, plan, root, { ...plain, mode: 'normal' })
    expect(r).toEqual({ text: 'plan_ready only works in plan mode (/plan).', isError: true })
    expect(calls).toHaveLength(0)
  })

  test('refuses in a scheduled task, without a dialog', async () => {
    const { ctx, calls } = fake('approve')
    const r = await askApproval(ctx, plan, root, { ...plain, noAsk: true })
    expect(r).toEqual({ text: 'Nobody can approve a plan here: this is a scheduled task.', isError: true })
    expect(calls).toHaveLength(0)
  })

  test('refuses a path that is not a plan, without a dialog', async () => {
    const { ctx, calls } = fake('approve')
    const missing = await askApproval(ctx, 'docs/plans/none.md', root, plain)
    expect(missing.isError).toBe(true)
    expect(missing.text).toContain("Can't find")
    const elsewhere = await askApproval(ctx, 'notes.md', root, plain)
    expect(elsewhere.isError).toBe(true)
    expect(elsewhere.text).toContain('docs/plans/')
    expect(calls).toHaveLength(0)
  })

  test('approved: the card is the title, and the result carries the absolute path', async () => {
    const { ctx, calls } = fake('approve')
    const signal = new AbortController().signal
    const r = await askApproval(ctx, 'docs/plans/2026-10-23-acme-checkout-widget.md', root, { ...plain, signal })
    const card = planCard(parsePlan(PLAN), plan)
    expect(calls).toEqual([[JSON.stringify(card), ['approve', 'change'], { signal }]])
    const { options: _options, ...shown } = card
    expect(r).toEqual({
      text: 'Approved: the build starts when this answer ends. End it now with one short line.',
      details: { status: 'approved', plan: shown },
      approved: plan,
    })
  })

  test('change: a draft, nothing approved', async () => {
    const { ctx } = fake('change')
    const r = await askApproval(ctx, plan, root, plain)
    expect(r.approved).toBeUndefined()
    expect(r.isError).toBeUndefined()
    expect(r.details.status).toBe('draft')
    expect(r.text).toBe(
      'Not approved yet: ask the owner what to change (one ask_user question or in your answer), edit the plan, commit it and call plan_ready again.',
    )
  })

  test('cancelled counts as change', async () => {
    const { ctx } = fake(undefined)
    const r = await askApproval(ctx, plan, root, plain)
    expect(r.approved).toBeUndefined()
    expect(r.details.status).toBe('draft')
  })
})

describe('pendingBuild', () => {
  test('approving then settling builds once', () => {
    const build = pendingBuild()
    build.queue(plan)
    expect(build.take()).toBe(plan)
    expect(build.take()).toBeNull()
  })

  test('leaving plan mode before the answer ends cancels the build (/plan off)', () => {
    const build = pendingBuild()
    build.queue(plan)
    build.cancel()
    expect(build.take()).toBeNull()
  })

  test('/execute in that window builds once: it ends plan mode, which cancels the queued one', () => {
    const build = pendingBuild()
    build.queue(plan)
    build.cancel() // endPlan -> setMode('normal')
    const builds = [plan, build.take()].filter(Boolean) // /execute's own startBuild, then agent_settled
    expect(builds).toEqual([plan])
  })
})
