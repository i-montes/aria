// Run: bun test extensions
// background_run takes the same route as bash: the line it sends to the dock, the refusals it shares with bash, and
// the checks (subagent, dock, folder) that come before the sandbox may ask the person anything.
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { backgroundLine, refusalText, SAID_NO } from './route.ts'
import { wrapCommand } from './policy.ts'
import { prepareRun, RUN_DESCRIPTION, SUBAGENT_REFUSAL } from '../background/run.ts'
import { DOCK_REFUSAL } from '../background/client.ts'

const textOf = (r: any) => (r?.content ?? []).map((c: any) => c.text).join('\n')

describe('the line background_run sends to the dock', () => {
  test('boxed: the line is the bwrap line for the folder', () => {
    expect(backgroundLine('boxed', 'bun dev', '/p')).toBe(wrapCommand('bun dev', '/p'))
  })
  test('plain and allowed-after-asking: the command as is', () => {
    expect(backgroundLine('plain', 'bun dev', '/p')).toBe('bun dev')
    expect(backgroundLine('ask', 'bun dev', '/p', true)).toBe('bun dev')
  })
  test('ask without an explicit yes runs nothing', () => {
    expect(backgroundLine('ask', 'bun dev', '/p')).toBeNull()
    expect(backgroundLine('ask', 'bun dev', '/p', false)).toBeNull()
  })
  test('refusals stay refusals', () => {
    expect(backgroundLine('refuse-nobody', 'bun dev', '/p')).toBeNull()
    expect(backgroundLine('refuse-not-allowed', 'bun dev', '/p')).toBeNull()
  })
})

describe('the refusals bash and background_run share', () => {
  test('each refusing route has its text, the others none', () => {
    expect(refusalText('refuse-nobody', true)).toMatch(/nobody here to ask/)
    expect(refusalText('refuse-nobody', false)).toMatch(/no sandbox on this system/)
    expect(refusalText('refuse-not-allowed', true)).toMatch(/not allowed/)
    for (const r of ['plain', 'ask', 'boxed'] as const) expect(refusalText(r, true)).toBeNull()
    expect(SAID_NO).toBe('The user said no: the command did not run.')
  })
})

describe('the checks before the route', () => {
  const saved = process.env.ARIA_JOBS_SOCK
  beforeEach(() => {
    delete process.env.ARIA_JOBS_SOCK
  })
  afterEach(() => {
    if (saved === undefined) delete process.env.ARIA_JOBS_SOCK
    else process.env.ARIA_JOBS_SOCK = saved
  })

  test('a subagent is refused first', () => {
    const got = prepareRun({ command: 'bun dev', title: 'Dev' }, { mode: 'json', cwd: '/tmp' })
    expect('refused' in got && textOf(got.refused)).toBe(SUBAGENT_REFUSAL)
  })
  test('without the dock it refuses before anything is asked', () => {
    const got = prepareRun({ command: 'bun dev', title: 'Dev', folder: '/usr/share' }, { mode: 'rpc', cwd: '/tmp' })
    expect('refused' in got && textOf(got.refused)).toBe(DOCK_REFUSAL)
  })
  test('with the dock, a folder outside the projects folder is refused; one inside resolves', () => {
    process.env.ARIA_JOBS_SOCK = '/nonexistent/aria-test-jobs.sock'
    const bad = prepareRun({ command: 'bun dev', title: 'Dev', folder: '/usr/share' }, { mode: 'rpc', cwd: '/tmp' })
    expect('refused' in bad && textOf(bad.refused)).toMatch(/^Refused: .*projects folder\.$/)
    const good = prepareRun({ command: 'bun dev', title: 'Dev', folder: 'sub' }, { mode: 'rpc', cwd: '/tmp' })
    expect('folder' in good && good.folder).toBe('/tmp/sub')
  })
  test('the description keeps the deciding rules', () => {
    expect(RUN_DESCRIPTION).toContain('does not end by itself')
    expect(RUN_DESCRIPTION).toContain('at most 6 words')
  })
})
