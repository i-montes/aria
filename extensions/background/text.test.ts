// Run: bun test extensions
// What Aria reads about its background work: the hidden `[background]` line and the message when a job ends.
import { describe, expect, test } from 'bun:test'
import { job } from './fake-dock.ts'
import { endedMessage, listText, nowLine } from './text.ts'

const NOW = Date.UTC(2026, 9, 21, 12, 0)
const min = 60_000

describe('the [background] line', () => {
  test('nothing running: no line', () => {
    expect(nowLine([], NOW)).toBeNull()
    expect(nowLine([job({ state: 'exited', exit: 0 }), job({ id: 'j4', number: 4, kind: 'agent', state: 'done' })], NOW)).toBeNull()
  })

  test("the line names number, kind, title, folder, url, minutes and a subagent's status and isolation", () => {
    const line = nowLine(
      [
        job({ title: 'bun dev', url: 'http://localhost:3000', startedAt: NOW - 12 * min }),
        job({ id: 'j4', number: 4, kind: 'agent', title: 'fix the checkout tests', command: null, isolation: 'worktree', startedAt: NOW - 3 * min }),
        job({ id: 'j5', number: 5, kind: 'agent', title: 'review the API', command: null, writes: false, isolation: 'folder', folder: '~/Projects/api', startedAt: NOW }),
        job({ id: 'j6', number: 6, kind: 'agent', title: 'rename the models', command: null, state: 'queued', isolation: null, folder: '~/Projects/api' }),
        job({ id: 'j2', number: 2, state: 'exited' }),
      ],
      NOW,
    )
    expect(line).toBe(
      '[background] running: #3 process "bun dev" (~/Projects/shop, http://localhost:3000, 12 min); ' +
        '#4 subagent "fix the checkout tests" (~/Projects/shop, writing, worktree, 3 min); ' +
        '#5 subagent "review the API" (~/Projects/api, read-only, 0 min); ' +
        '#6 subagent "rename the models" (~/Projects/api, writing, queued)',
    )
  })

  test('a writing subagent in a folder says it holds that folder', () => {
    const line = nowLine([job({ kind: 'agent', isolation: 'folder', startedAt: NOW - min })], NOW)
    expect(line).toBe('[background] running: #3 subagent "Dev server" (~/Projects/shop, writing in the folder: do not edit it, 1 min)')
  })

  test('the list adds the finished jobs', () => {
    const text = listText(
      [
        job({ startedAt: NOW - min }),
        job({ id: 'j2', number: 2, title: 'tests', state: 'exited', exit: 1 }),
        job({ id: 'j1', number: 1, kind: 'agent', title: 'refactor', state: 'done', isolation: 'worktree', worktree: '/p/.aria-worktrees/j1', conflict: true }),
      ],
      NOW,
    )
    expect(text).toContain('#3 process "Dev server"')
    expect(text).toContain('Finished: #2 process "tests" (exited, exit 1, ~/Projects/shop)')
    expect(text).toContain('#1 subagent "refactor" (done, conflict: kept in /p/.aria-worktrees/j1, ~/Projects/shop)')
    expect(listText([], NOW)).toBe('No background work in this session.')
  })
})

describe('the message when a job ends', () => {
  test('a process that died by itself starts a turn and shows its last lines', () => {
    const m = endedMessage({ event: 'ended', seq: 7, by: 'self', job: job({ state: 'exited', exit: 1 }), lastLines: ['Error: port in use', 'exit'] })
    expect(m.triggerTurn).toBe(true)
    expect(m.text).toContain('#3')
    expect(m.text).toContain('"Dev server"')
    expect(m.text).toContain('exit 1')
    expect(m.text).toContain('Error: port in use\nexit')
  })

  test('a stopped job does not start a turn', () => {
    expect(endedMessage({ event: 'ended', seq: 8, by: 'stop', job: job({ state: 'stopped' }), lastLines: [] }).triggerTurn).toBe(false)
    const agent = endedMessage({ event: 'ended', seq: 9, by: 'stop', job: job({ kind: 'agent', state: 'stopped', isolation: 'worktree', worktree: '/p/.aria-worktrees/j3' }), lastLines: [] })
    expect(agent.triggerTurn).toBe(false)
    expect(agent.text).toContain('/p/.aria-worktrees/j3')
  })

  test('a done agent asks to tell the person and save the learnings', () => {
    const m = endedMessage({ event: 'ended', seq: 7, by: 'self', job: job({ kind: 'agent', state: 'done', isolation: 'worktree', worktree: null, result: 'All green.\n## Learnings\n- x' }), lastLines: [] })
    expect(m.triggerTurn).toBe(true)
    expect(m.text).toContain('All green.')
    expect(m.text).toContain('applied')
    expect(m.text.endsWith('Tell the person what it did, and save its Learnings to memory.')).toBe(true)
  })

  test('a conflict names the worktree and says to ask how to merge', () => {
    const m = endedMessage({ event: 'ended', seq: 7, by: 'self', job: job({ kind: 'agent', state: 'done', isolation: 'worktree', worktree: '/p/.aria-worktrees/j3', conflict: true, result: 'Done.' }), lastLines: [] })
    expect(m.text).toContain('/p/.aria-worktrees/j3')
    expect(m.text).toContain('Ask the person how to merge')
    expect(m.text.endsWith('Tell the person what it did, and save its Learnings to memory.')).toBe(true)
  })

  test('a failed agent starts a turn and names its kept worktree', () => {
    const m = endedMessage({ event: 'ended', seq: 7, by: 'self', job: job({ kind: 'agent', state: 'failed', isolation: 'worktree', worktree: '/p/.aria-worktrees/j3' }), lastLines: ['boom'] })
    expect(m.triggerTurn).toBe(true)
    expect(m.text).toContain('failed')
    expect(m.text).toContain('/p/.aria-worktrees/j3')
    expect(m.text).toContain('boom')
  })

  test('jobs cut when the dock stopped: one message, no turn', () => {
    const m = endedMessage({ event: 'cut', seq: 9, jobs: [job(), job({ id: 'j4', number: 4, kind: 'agent', title: 'refactor', isolation: 'worktree', worktree: '/p/.aria-worktrees/j4' })] })
    expect(m.triggerTurn).toBe(false)
    expect(m.text).toContain('#3 process "Dev server"')
    expect(m.text).toContain('#4 subagent "refactor"')
    expect(m.text).toContain('/p/.aria-worktrees/j4')
  })
})
