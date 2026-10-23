// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import loadout, { missingTools } from './index.ts'

const root = '/home/ana/.aria/agent/extensions'
const all = [
  { name: 'read', exposure: 'direct', sourceInfo: { path: 'builtin:read' } },
  { name: 'secret_import', exposure: 'direct', sourceInfo: { path: `${root}/secrets/index.ts` } },
  { name: 'ask_user', exposure: 'model-only', sourceInfo: { path: `${root}/ask-user/index.ts` } },
  { name: 'later', exposure: 'deferred', sourceInfo: { path: `${root}/x/index.ts` } },
  { name: 'parallel', exposure: 'direct', sourceInfo: { path: '/home/ana/.aria/agent/npm/node_modules/p/index.ts' } },
  { name: 'evil', exposure: 'direct', sourceInfo: { path: `${root}-other/index.ts` } },
]

describe("a resumed session gets the tools Aria's extensions added since it started", () => {
  test("only Aria's own, not active yet, that registering would have activated", () => {
    expect(missingTools(all, ['read', 'ask_user'], root)).toEqual(['secret_import'])
    expect(missingTools(all, ['read'], root + '/')).toEqual(['secret_import', 'ask_user'])
    expect(missingTools(all, ['read', 'secret_import', 'ask_user'], root)).toEqual([])
  })

  const run = (mode: string, active: string[]) => {
    let handler: any
    let set: string[] | null = null
    const pi = { on: (_: string, fn: any) => (handler = fn), getActiveTools: () => active, getAllTools: () => all, setActiveTools: (names: string[]) => (set = names) }
    const before = process.env.PI_CODING_AGENT_DIR
    process.env.PI_CODING_AGENT_DIR = '/home/ana/.aria/agent'
    try {
      loadout(pi as any)
      handler({}, { mode })
    } finally {
      if (before === undefined) delete process.env.PI_CODING_AGENT_DIR
      else process.env.PI_CODING_AGENT_DIR = before
    }
    return set
  }

  test('the conversation gets them at its start; a subagent keeps its own', () => {
    expect(run('rpc', ['read', 'ask_user'])).toEqual(['read', 'ask_user', 'secret_import'])
    expect(run('rpc', ['read', 'ask_user', 'secret_import'])).toBeNull()
    expect(run('json', ['read'])).toBeNull()
  })
})
