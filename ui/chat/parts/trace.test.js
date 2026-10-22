// Run: bun test ui/chat/parts/trace.test.js
import { describe, expect, test } from 'bun:test'
import { TEXTS, autoCollapsed, axisRange, formatCost, formatMs, renderTrace, slowestId, totals, visibleRows } from './trace.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)

const tree = () => ({
  id: 'root', kind: 'agent', name: 'Aria', detail: 'opus', status: 'running', start: 0, input: 'Launch the pricing page', now: 20000, dropped: 0,
  children: [
    { id: 'llm-1', kind: 'llm', name: 'LLM call', detail: 'opus', status: 'done', start: 100, end: 7000, tokensIn: 1000, tokensOut: 200, cost: 0.01 },
    { id: 'tool-c1', kind: 'tool', name: 'bash', detail: 'pnpm test', status: 'failed', start: 7100, end: 15500, error: '2 of 14 tests fail', input: '{"command":"pnpm test"}' },
    {
      id: 'tool-p', kind: 'tool', name: 'parallel', status: 'done', start: 7200, end: 12000,
      children: [
        {
          id: 'agent-1', kind: 'agent', name: 'researcher', detail: 'find X', status: 'done', start: 7300, end: 11900,
          children: [
            { id: 'agent-1-llm-1', kind: 'llm', name: 'LLM call', status: 'done', start: 7400, end: 9000, tokensIn: 500, tokensOut: 50, cost: 0.03 },
            { id: 'agent-1-tool-a', kind: 'tool', name: 'web_search', detail: 'X pricing', status: 'done', start: 9100, end: 9900 },
            { id: 'agent-1-tool-b', kind: 'tool', name: 'read', status: 'done', start: 9950, end: 10100 },
          ],
        },
      ],
    },
    { id: 'llm-2', kind: 'llm', name: 'LLM call', status: 'running', start: 16000, tokensIn: 300, tokensOut: 0, cost: 0.02 },
  ],
})

describe('the figures', () => {
  test('figures_sum_the_whole_tree', () => {
    const sum = totals(tree())
    expect(sum.llm).toBe(3)
    expect(sum.tools).toBe(4) // bash and parallel, and the subagent's web_search and read
    expect(sum.errors).toBe(1)
    expect(sum.cost).toBeCloseTo(0.06)
    expect(sum.tokens).toBe(1000 + 200 + 500 + 50 + 300)
    expect(sum.costKnown).toBe(true)
  })
  test('no_cost_reads_as_a_dash', () => {
    const free = { id: 'root', kind: 'agent', name: 'Aria', status: 'done', start: 0, end: 10, now: 10, children: [{ id: 'llm-1', kind: 'llm', status: 'done', start: 0, end: 5, cost: 0 }] }
    expect(totals(free).costKnown).toBe(false)
    const html = renderTrace({ tree: free, now: 10 }, {}, t)
    expect(html).toMatch(/data-figure="cost"[\s\S]*?p-trace-fvalue">—</)
  })
  test('formats', () => {
    expect(formatMs(766)).toBe('766ms')
    expect(formatMs(6900)).toBe('6.9s')
    expect(formatMs(19000)).toBe('19s')
    expect(formatMs(82000)).toBe('1m 22s')
    expect(formatCost(0.042)).toBe('$0.042')
    expect(formatCost(0.12)).toBe('$0.12')
    expect(formatCost(1.4)).toBe('$1.40')
  })
})

describe('the timeline', () => {
  test('axis_steps_are_round', () => {
    expect(axisRange(0)).toBe(10000)
    expect(axisRange(9000)).toBe(10000)
    expect(axisRange(25000)).toBe(30000)
    expect(axisRange(61000)).toBe(120000)
    expect(axisRange(4 * 3600000)).toBe(4 * 3600000)
  })
  test('the_slowest_finished_call_is_marked', () => {
    expect(slowestId(tree())).toBe('tool-c1')
    const html = renderTrace({ tree: tree(), now: 20000 }, {}, t)
    expect(html.match(/chat\.trace\.slowest/g)).toHaveLength(1)
    expect(html).toMatch(/data-id="tool-c1"[\s\S]*?chat\.trace\.slowest/)
  })
  test('a_running_bar_reaches_now', () => {
    const html = renderTrace({ tree: tree(), now: 20000 }, {}, t)
    // llm-2 runs from 16 s to now (20 s) on a 20 s axis: 80% to 100%
    expect(html).toMatch(/data-bar="llm-2"[^>]*style="left: 80(\.0+)?%; width: 20(\.0+)?%/)
    expect(html).toContain('p-trace-now')
  })
  test('a_folded_agent_shows_marks_and_a_count', () => {
    const html = renderTrace({ tree: tree(), now: 20000, collapsed: new Set(['agent-1']) }, {}, t)
    expect(html).toContain('+3')
    expect(html.match(/p-trace-mark/g)).toHaveLength(3)
    expect(html).not.toContain('data-id="agent-1-tool-a"')
  })
  test('finished_subagents_fold_by_themselves_unless_touched', () => {
    expect([...autoCollapsed(tree(), new Set())]).toEqual(['agent-1'])
    expect([...autoCollapsed(tree(), new Set(['agent-1']))]).toEqual([])
    const running = tree()
    running.children[2].children[0].status = 'running'
    expect([...autoCollapsed(running, new Set())]).toEqual([])
  })
  test('visible rows follow the folds', () => {
    expect(visibleRows(tree(), new Set()).map((r) => r.span.id)).toEqual(['root', 'llm-1', 'tool-c1', 'tool-p', 'agent-1', 'agent-1-llm-1', 'agent-1-tool-a', 'agent-1-tool-b', 'llm-2'])
    expect(visibleRows(tree(), new Set(['agent-1'])).map((r) => r.level)).toEqual([1, 2, 2, 2, 3, 2])
  })
})

describe('the panel', () => {
  test('the_details_card_shows_tokens_cost_and_error', () => {
    const html = renderTrace({ tree: tree(), now: 20000, open: 'tool-c1' }, {}, t)
    expect(html).toContain('role="dialog"')
    expect(html).toContain('2 of 14 tests fail')
    expect(html).toContain('pnpm test')
    const llm = renderTrace({ tree: tree(), now: 20000, open: 'llm-1' }, {}, t)
    expect(llm).toContain('1,000')
    expect(llm).toContain('$0.01')
    expect(llm).toContain('chat.trace.started')
  })
  test('everything_is_escaped', () => {
    const evil = tree()
    evil.children[0].name = '<script>x</script>'
    evil.input = '<img onerror=1>'
    const html = renderTrace({ tree: evil, now: 1 }, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).not.toContain('<img')
  })
  test('the_empty_state_without_a_tree', () => {
    expect(renderTrace({ tree: null }, {}, t)).toContain('chat.trace.empty')
  })
  test('dropped_steps_are_said', () => {
    const many = tree()
    many.dropped = 7
    expect(renderTrace({ tree: many, now: 1 }, {}, t)).toContain('chat.trace.dropped n=7')
  })
  test('a tree with rows, levels and one tab stop', () => {
    const html = renderTrace({ tree: tree(), now: 20000, focus: 'llm-1' }, {}, t)
    expect(html).toContain('role="tree"')
    expect(html.match(/role="treeitem"/g)).toHaveLength(9)
    expect(html.match(/tabindex="0"/g)).toHaveLength(1)
    expect(html).toMatch(/data-id="llm-1"[^>]*tabindex="0"/)
  })
  test('texts exist in both languages', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
  })
})
