// Run: bun test extensions
// The connectors extension with a fake `pi`: which MCP calls run, ask or are refused.
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import connectors from './index.ts'

const ONCE = 'Yes, just this once'
const NO = 'No'
const SESSION = "Yes, and don't ask again for Notion create pages this session"

const TOOLS = [
  { name: 'mcp__notion__notion_search', annotations: { readOnlyHint: true }, namespace: { name: 'mcp__notion', aliases: ['notion'] } },
  { name: 'mcp__notion__notion_create_pages', namespace: { name: 'mcp__notion', aliases: ['notion'] } },
  { name: 'mcp__notion__notion_update_page', annotations: { readOnlyHint: false }, namespace: { name: 'mcp__notion', aliases: ['notion'] } },
  { name: 'mcp__engram__mem_save', namespace: { name: 'mcp__engram', aliases: ['engram'] } },
  { name: 'mcp__my_docs__createDoc', namespace: { name: 'mcp__my_docs', aliases: ['my-docs'] } },
  { name: 'mcp__my_docs__getDoc', namespace: { name: 'mcp__my_docs', aliases: ['my-docs'] } },
  { name: 'mcp__my_doc__createDoc', namespace: { name: 'mcp__my_doc', aliases: ['my_doc'] } },
]

// Pi 1.3.0's own name for `createMcpToolName('x'.repeat(50), 'delete_page')` (extensions/mcp/tools.js, computed in a
// scratch run): longer than 64, so the name is cut before its `__` and ends in a hash.
const LONG_SERVER = 'x'.repeat(50)
const CUT_NAME = `mcp__${LONG_SERVER}_316906ee`

function fakePi(tools: any[] = TOOLS) {
  const pi = {
    tools,
    handlers: new Map<string, ((event: any, ctx: any) => any)[]>(),
    on(name: string, fn: any) {
      pi.handlers.set(name, [...(pi.handlers.get(name) ?? []), fn])
    },
    getAllTools: () => pi.tools,
    registerCommand() {},
    fire: async (name: string, event: any, ctx: any) => {
      let out: any
      for (const fn of pi.handlers.get(name) ?? []) out = (await fn(event, ctx)) ?? out
      return out
    },
  }
  return pi
}

/** A dock that answers every card with the next of `answers` and keeps what it was shown. */
function dock(...answers: (string | undefined)[]) {
  const asked: { title: string; options: string[] }[] = []
  return {
    asked,
    ctx: {
      mode: 'rpc',
      hasUI: true,
      ui: {
        select: async (title: string, options: string[]) => {
          asked.push({ title, options })
          return answers.shift()
        },
      },
    },
  }
}

const child = { mode: 'json', hasUI: false, ui: { select: async () => ONCE } }

let n = 0
const call = (toolName: string, input: any = { title: 'Weekly notes' }, extra: any = {}) => ({ type: 'tool_call', toolCallId: `call-${++n}`, toolName, input, ...extra })

const saved = process.env.PI_CODING_AGENT_DIR
let dir = ''
const setLevels = (connectors: Record<string, string>) => writeFileSync(join(dir, 'aria.json'), JSON.stringify({ version: 1, connectors }))

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'aria-agent-'))
  process.env.PI_CODING_AGENT_DIR = dir
})
afterEach(() => {
  if (saved === undefined) delete process.env.PI_CODING_AGENT_DIR
  else process.env.PI_CODING_AGENT_DIR = saved
})

async function started() {
  const pi = fakePi()
  connectors(pi as any)
  await pi.fire('session_start', { type: 'session_start' }, dock().ctx)
  return pi
}

describe('the connectors extension', () => {
  test('a_read_runs_without_asking', async () => {
    const pi = await started()
    const d = dock()
    expect(await pi.fire('tool_call', call('mcp__notion__notion_search', { query: 'notes' }), d.ctx)).toBeUndefined()
    expect(await pi.fire('tool_call', call('read_mcp_resource', { server: 'notion', uri: 'x' }), d.ctx)).toBeUndefined()
    expect(d.asked).toHaveLength(0)
  })

  test('a_write_on_ask_asks_and_runs_on_yes', async () => {
    const pi = await started()
    const d = dock(ONCE)
    expect(await pi.fire('tool_call', call('mcp__notion__notion_create_pages'), d.ctx)).toBeUndefined()
    expect(d.asked).toHaveLength(1)
    const card = JSON.parse(d.asked[0].title)
    expect(card).toMatchObject({ aria: 1, kind: 'connector', summary: 'Notion wants to create pages', detail: 'title: Weekly notes' })
    expect(d.asked[0].options).toEqual([ONCE, SESSION, NO])
    // Once is only once.
    const again = dock(ONCE)
    await pi.fire('tool_call', call('mcp__notion__notion_create_pages'), again.ctx)
    expect(again.asked).toHaveLength(1)
  })

  test('no_refuses_with_the_reason', async () => {
    const pi = await started()
    expect(await pi.fire('tool_call', call('mcp__notion__notion_create_pages'), dock(NO).ctx)).toEqual({
      block: true,
      reason: 'The person said no: Notion create pages did not run.',
    })
    // No answer (the run was stopped) is a no.
    expect((await pi.fire('tool_call', call('mcp__notion__notion_create_pages'), dock(undefined).ctx))?.block).toBe(true)
  })

  test('session_grant_skips_the_next_question', async () => {
    const pi = await started()
    expect(await pi.fire('tool_call', call('mcp__notion__notion_create_pages'), dock(SESSION).ctx)).toBeUndefined()
    const d = dock()
    expect(await pi.fire('tool_call', call('mcp__notion__notion_create_pages', { title: 'Other' }), d.ctx)).toBeUndefined()
    expect(d.asked).toHaveLength(0)
    // The grant is for that action only.
    const other = dock(ONCE)
    await pi.fire('tool_call', call('mcp__notion__notion_update_page'), other.ctx)
    expect(other.asked).toHaveLength(1)
  })

  test('read_only_refuses_without_asking', async () => {
    setLevels({ notion: 'read' })
    const pi = await started()
    const d = dock(ONCE)
    expect(await pi.fire('tool_call', call('mcp__notion__notion_create_pages'), d.ctx)).toEqual({
      block: true,
      reason: "Notion is read only (the person set it so in Aria's settings): create pages did not run.",
    })
    expect(d.asked).toHaveLength(0)
    expect(await pi.fire('tool_call', call('mcp__notion__notion_search'), d.ctx)).toBeUndefined()
    // Not even a subagent's read is refused.
    expect(await pi.fire('tool_call', call('mcp__notion__notion_search'), child)).toBeUndefined()
  })

  test('allow_runs_without_asking', async () => {
    setLevels({ notion: 'allow', 'my-docs': 'allow' })
    const pi = await started()
    const d = dock()
    expect(await pi.fire('tool_call', call('mcp__notion__notion_create_pages'), d.ctx)).toBeUndefined()
    // The level is set by the server's own name, which may hold a `-` the tool name turned into `_`.
    expect(await pi.fire('tool_call', call('mcp__my_docs__createDoc'), d.ctx)).toBeUndefined()
    expect(await pi.fire('tool_call', call('mcp__notion__notion_create_pages'), child)).toBeUndefined()
    expect(d.asked).toHaveLength(0)
  })

  test('nobody_to_ask_refuses', async () => {
    const pi = await started()
    expect(await pi.fire('tool_call', call('mcp__notion__notion_create_pages'), child)).toEqual({
      block: true,
      reason: "Notion wants to create pages, which needs the person's approval, and there is nobody here to ask.",
    })
    expect(await pi.fire('tool_call', call('mcp__notion__notion_search'), child)).toBeUndefined()
  })

  test('engram_is_never_gated', async () => {
    setLevels({ engram: 'read' })
    const pi = await started()
    const d = dock()
    expect(await pi.fire('tool_call', call('mcp__engram__mem_save', { content: 'x' }), d.ctx)).toBeUndefined()
    expect(await pi.fire('tool_call', call('mcp__engram__mem_save', { content: 'x' }), child)).toBeUndefined()
    expect(d.asked).toHaveLength(0)
  })

  test('a_nested_call_is_gated_the_same', async () => {
    const pi = await started()
    const nested = (name: string) => call(name, { title: 'From a script' }, { toolCallId: 'codemode-1/1', parentToolCallId: 'codemode-1' })
    expect(await pi.fire('tool_call', nested('mcp__notion__notion_create_pages'), dock(NO).ctx)).toEqual({
      block: true,
      reason: 'The person said no: Notion create pages did not run.',
    })
    expect((await pi.fire('tool_call', nested('mcp__notion__notion_create_pages'), child))?.block).toBe(true)
    const d = dock(ONCE)
    expect(await pi.fire('tool_call', nested('mcp__notion__notion_create_pages'), d.ctx)).toBeUndefined()
    expect(JSON.parse(d.asked[0].title).detail).toBe('title: From a script')
  })

  test('grants_clear_on_session_start', async () => {
    const pi = await started()
    await pi.fire('tool_call', call('mcp__notion__notion_create_pages'), dock(SESSION).ctx)
    await pi.fire('session_start', { type: 'session_start' }, dock().ctx)
    const d = dock(ONCE)
    await pi.fire('tool_call', call('mcp__notion__notion_create_pages'), d.ctx)
    expect(d.asked).toHaveLength(1)
  })

  test('other tools and new MCP tools', async () => {
    const pi = await started()
    const d = dock(ONCE)
    expect(await pi.fire('tool_call', call('bash', { command: 'ls' }), d.ctx)).toBeUndefined()
    expect(d.asked).toHaveLength(0)
    // A tool connected after the first look is found when it is first called.
    pi.tools = [...TOOLS, { name: 'mcp__notion__notion_delete_page', annotations: { readOnlyHint: true }, namespace: { name: 'mcp__notion', aliases: ['notion'] } }]
    expect(await pi.fire('tool_call', call('mcp__notion__notion_delete_page'), d.ctx)).toBeUndefined()
    expect(d.asked).toHaveLength(0)
  })

  test('a_cut_name_is_still_gated', async () => {
    setLevels({ [LONG_SERVER]: 'allow' })
    const pi = fakePi([])
    connectors(pi as any)
    await pi.fire('session_start', { type: 'session_start' }, dock().ctx)
    // Pi does not list it and the name holds no server: a write on Ask, whatever the settings say.
    const d = dock(NO)
    expect((await pi.fire('tool_call', call(CUT_NAME), d.ctx))?.block).toBe(true)
    expect(d.asked).toHaveLength(1)
    expect(JSON.parse(d.asked[0].title).summary).toBe(`${LONG_SERVER}_316906ee wants to use ${CUT_NAME}`)
    expect((await pi.fire('tool_call', call(CUT_NAME), child))?.block).toBe(true)
    // Listed with its namespace, it is that server's tool, at that server's level.
    pi.tools = [{ name: CUT_NAME, namespace: { name: `mcp__${LONG_SERVER}`, aliases: [LONG_SERVER] } }]
    await pi.fire('session_start', { type: 'session_start' }, dock().ctx)
    expect(await pi.fire('tool_call', call(CUT_NAME), child)).toBeUndefined()
  })

  test('a_tool_pi_does_not_list_is_a_write', async () => {
    const pi = await started()
    expect((await pi.fire('tool_call', call('mcp__notion__notion_search_all'), child))?.block).toBe(true)
    expect(await pi.fire('tool_call', call('mcp__my_docs__getDoc'), child)).toBeUndefined()
  })

  test('a_level_is_never_taken_from_another_server', async () => {
    setLevels({ 'my-docs': 'allow', 'my-doc': 'read', my_dox: 'read' })
    const pi = await started()
    expect((await pi.fire('tool_call', call('mcp__my_docs__createDoc'), child))).toBeUndefined()
    setLevels({ 'my-docs': 'allow' })
    // `my_doc` is not `my-docs`, even though one starts the other.
    expect((await pi.fire('tool_call', call('mcp__my_doc__createDoc'), child))?.block).toBe(true)
    setLevels({ 'my-doc': 'allow', my_doc_x: 'allow' })
    expect(await pi.fire('tool_call', call('mcp__my_doc__createDoc'), child)).toBeUndefined()
    // Not listed by Pi, the server comes from the name: an exact key wins, two keys that read the same are Ask.
    setLevels({ 'other-srv': 'read', other_srv: 'allow' })
    expect(await pi.fire('tool_call', call('mcp__other_srv__doThing'), child)).toBeUndefined()
    setLevels({ 'other-srv': 'allow', 'other.srv': 'allow' })
    expect((await pi.fire('tool_call', call('mcp__other_srv__doThing'), child))?.block).toBe(true)
    setLevels({ 'other-srv': 'allow' })
    expect(await pi.fire('tool_call', call('mcp__other_srv__doThing'), child)).toBeUndefined()
    setLevels({ other: 'allow' })
    expect((await pi.fire('tool_call', call('mcp__other_srv__doThing'), child))?.block).toBe(true)
  })

  test('the connector name is in words', async () => {
    const pi = await started()
    const d = dock(ONCE)
    await pi.fire('tool_call', call('mcp__my_docs__createDoc'), d.ctx)
    expect(JSON.parse(d.asked[0].title).summary).toBe('My Docs wants to create doc')
  })

  test('the terminal gets plain text', async () => {
    const pi = await started()
    const d = dock(ONCE)
    d.ctx.mode = 'tui'
    await pi.fire('tool_call', call('mcp__notion__notion_create_pages'), d.ctx)
    expect(d.asked[0].title).toBe('Notion wants to create pages\ntitle: Weekly notes')
  })
})
