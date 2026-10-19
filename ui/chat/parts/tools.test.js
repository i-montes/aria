// Run: bun test ui/chat/parts
import { describe, expect, test } from 'bun:test'
import { TEXTS, callDiff, clipResult, renderTools, toolChip, toolKind, toolLabel } from './tools.js'

const t = (key, vars = {}) => (TEXTS.en[key] ?? key).replace(/\{(\w+)\}/g, (m, name) => (name in vars ? String(vars[name]) : m))
const call = (over = {}) => ({ id: 'c1', name: 'bash', args: { command: 'ls' }, status: 'done', summary: 'ls', result: 'a\nb', ...over })
const NASTY = '<script>alert(1)</script> "q" \'s\' & co'
const ESCAPED = '&lt;script&gt;alert(1)&lt;/script&gt; &quot;q&quot; &#39;s&#39; &amp; co'

describe('header', () => {
  test('counts the calls with the design wording', () => {
    expect(renderTools({ calls: [call()], live: false }, {}, t)).toContain('1 tool call<')
    const two = renderTools({ calls: [call(), call({ id: 'c2' })], live: false }, {}, t)
    expect(two).toContain('2 tool calls<')
    expect(two).not.toContain('1 tool call<')
  })

  test('the header button carries the message index and the open state', () => {
    const closed = renderTools({ index: 7, calls: [call()] }, { open: new Set() }, t)
    expect(closed).toContain('data-chat="toggle-tools"')
    expect(closed).toContain('data-index="7"')
    expect(closed).toContain('aria-expanded="false"')
    expect(renderTools({ index: 7, calls: [call()] }, { open: new Set([7]) }, t)).toContain('aria-expanded="true"')
    expect(renderTools({ index: 7, calls: [call()] }, { open: true }, t)).toContain('aria-expanded="true"')
    expect(renderTools({ index: 7, calls: [call()] }, { open: false }, t)).toContain('aria-expanded="false"')
  })

  test('the group is live while any call runs or live is set', () => {
    const running = renderTools({ calls: [call({ status: 'running' })] }, {}, t)
    expect(running).toContain('is-live')
    expect(running).toContain('aria-busy="true"')
    expect(renderTools({ calls: [call()], live: true }, {}, t)).toContain('is-live')
    const settled = renderTools({ calls: [call()], live: false }, {}, t)
    expect(settled).not.toContain('is-live')
    expect(settled).not.toContain('aria-busy')
  })

  test('no calls renders nothing', () => {
    expect(renderTools({ calls: [] }, {}, t)).toBe('')
  })
})

describe('rows', () => {
  test('every call is a row with its label and chip, expandable by id', () => {
    const html = renderTools({ calls: [call({ id: 'x1' }), call({ id: 'x2', name: 'read', args: { path: 'src/env.ts' }, summary: 'src/env.ts' })] }, { open: true, openRows: new Set(['x2']) }, t)
    expect(html.match(/data-chat="toggle-tool-row"/g)).toHaveLength(2)
    expect(html).toContain('data-id="x1"')
    expect(html).toContain('data-id="x2"')
    expect(html).toContain('Run command')
    expect(html).toContain('Read file')
    expect(html).toContain('src/env.ts')
    const rows = html.split('p-tools-item').slice(1)
    expect(rows[0]).toContain('aria-expanded="false"')
    expect(rows[1]).toContain('aria-expanded="true"')
    expect(rows[1]).toContain('is-open')
  })

  test('openRows may be missing', () => {
    expect(() => renderTools({ calls: [call()] }, { open: true }, t)).not.toThrow()
  })

  test('each tool kind gets its own icon', () => {
    const icon = (name, args = {}) => renderTools({ calls: [call({ name, args, summary: name })] }, { open: true }, t).match(/data-icon="(\w+)"/)[1]
    expect(icon('bash')).toBe('run')
    expect(icon('read')).toBe('read')
    expect(icon('write')).toBe('write')
    expect(icon('edit')).toBe('write')
    expect(icon('grep')).toBe('search')
    expect(icon('find')).toBe('search')
    expect(icon('ls')).toBe('folder')
    expect(icon('mcp__notion__search')).toBe('tool')
    expect(icon('something_else')).toBe('tool')
  })

  test('running rows are marked, failed rows too', () => {
    const html = renderTools(
      { calls: [call({ id: 'r', status: 'running', result: '' }), call({ id: 'e', status: 'error', result: 'boom' })] },
      { open: true, openRows: new Set(['r', 'e']) },
      t,
    )
    expect(html).toContain('status-running')
    expect(html).toContain('status-error')
    expect(html).toContain('Running…')
    expect(html).toContain('boom')
  })

  test('an expanded row shows the result, one span per line; an empty result says so', () => {
    const html = renderTools({ calls: [call({ result: 'one\ntwo' })] }, { open: true, openRows: new Set(['c1']) }, t)
    expect(html).toContain('>one</span>')
    expect(html).toContain('>two</span>')
    const empty = renderTools({ calls: [call({ result: '' })] }, { open: true, openRows: new Set(['c1']) }, t)
    expect(empty).toContain('No output')
  })

  test('long results are truncated with a count of the hidden lines', () => {
    const result = Array.from({ length: 40 }, (_, i) => `line ${i}`).join('\n')
    const html = renderTools({ calls: [call({ result })] }, { open: true, openRows: new Set(['c1']) }, t)
    expect(html).toContain('line 0')
    expect(html).not.toContain('line 39')
    expect(html).toContain('… and 28 more lines')
  })
})

describe('helpers', () => {
  test('toolKind maps Pi and MCP tools', () => {
    expect(toolKind('bash')).toBe('run')
    expect(toolKind('mcp__a__b')).toBe('tool')
    expect(toolKind('grep')).toBe('search')
    expect(toolKind(undefined)).toBe('tool')
  })

  test('toolLabel and toolChip', () => {
    expect(toolLabel(call({ name: 'write', args: { path: 'a', content: 'x\ny\nz' } }), t)).toBe('Write 3 lines')
    expect(toolLabel(call({ name: 'write', args: { path: 'a', content: 'x' } }), t)).toBe('Write 1 line')
    expect(toolLabel(call({ name: 'write', args: {} }), t)).toBe('Write file')
    expect(toolLabel(call({ name: 'edit' }), t)).toBe('Edit file')
    expect(toolLabel(call({ name: 'mcp__claude_ai_Notion__notion-search' }), t)).toBe('Notion search')
    expect(toolLabel(call({ name: 'mcp__srv__list_pages' }), t)).toBe('List pages')
    expect(toolChip(call({ name: 'mcp__claude_ai_Notion__notion-search', summary: 'mcp__claude_ai_Notion__notion-search' }))).toBe('claude_ai_Notion')
    expect(toolChip(call({ name: 'grep', args: { pattern: 'foo' }, summary: 'grep' }))).toBe('foo')
    expect(toolChip(call({ name: 'ls', args: {}, summary: 'ls' }))).toBe('.')
    expect(toolChip(call({ name: 'bash', summary: 'pnpm build' }))).toBe('pnpm build')
    expect(toolChip(call({ name: 'weird', summary: 'weird', args: {} }))).toBe('')
  })

  test('clipResult keeps the first lines and counts the rest', () => {
    expect(clipResult('a\nb\n\n')).toEqual({ lines: ['a', 'b'], hidden: 0 })
    expect(clipResult('1\n2\n3\n4', { maxLines: 2 })).toEqual({ lines: ['1', '2'], hidden: 2 })
    const long = clipResult('x'.repeat(5000), { maxChars: 100 })
    expect(long.lines[0].length).toBeLessThanOrEqual(101)
    expect(long.lines[0].endsWith('…')).toBe(true)
    expect(clipResult(undefined)).toEqual({ lines: [], hidden: 0 })
  })

  test('callDiff counts lines from write and edit arguments', () => {
    expect(callDiff(call({ name: 'bash' }))).toBeNull()
    expect(callDiff(call({ name: 'write', args: { path: 'a.txt', content: 'x\ny\n' } }))).toMatchObject({ file: 'a.txt', add: 2, del: 0 })
    const edit = callDiff(call({ name: 'edit', args: { path: 'b.ts', oldText: 'one', newText: 'two\nthree' } }))
    expect(edit).toMatchObject({ file: 'b.ts', add: 2, del: 1 })
    expect(edit.lines).toEqual([
      { tone: 'del', text: 'one' },
      { tone: 'add', text: 'two' },
      { tone: 'add', text: 'three' },
    ])
    expect(callDiff(call({ name: 'edit', args: { path: 'c', edits: [{ oldText: 'a', newText: 'b' }, { oldText: 'c', newText: 'd' }] } }))).toMatchObject({ add: 2, del: 2 })
    expect(callDiff(call({ name: 'write', args: { content: 'x' } }))).toBeNull()
  })
})

describe('changed-file chips', () => {
  const writes = (n) =>
    Array.from({ length: n }, (_, i) => call({ id: `w${i}`, name: 'write', args: { path: `f${i}.ts`, content: 'a\nb' }, summary: `f${i}.ts`, result: 'ok' }))

  test('settled groups list the changed files, three at first, with a more button', () => {
    const html = renderTools({ index: 2, calls: writes(5) }, { open: true }, t)
    expect(html.match(/data-chip="/g)).toHaveLength(3)
    expect(html).toContain('+2 more')
    expect(html).toContain('data-chat="toggle-tool-diffs"')
    expect(html).toContain('+2')
    const all = renderTools({ index: 2, calls: writes(5) }, { open: true, moreOpen: new Set([2]) }, t)
    expect(all.match(/data-chip="/g)).toHaveLength(5)
    expect(all).toContain('Show less')
  })

  test('same file edited twice is one chip with the sums', () => {
    const calls = [
      call({ id: 'a', name: 'edit', args: { path: 'x.ts', oldText: 'a', newText: 'b' } }),
      call({ id: 'b', name: 'edit', args: { path: 'x.ts', oldText: 'c', newText: 'd\ne' } }),
    ]
    const html = renderTools({ calls }, { open: true }, t)
    expect(html.match(/data-chip="/g)).toHaveLength(1)
    expect(html).toContain('+3')
    expect(html).toContain('−2')
  })

  test('no chips while live, nothing changed, or after errors', () => {
    expect(renderTools({ calls: writes(2), live: true }, { open: true }, t)).not.toContain('data-chip=')
    expect(renderTools({ calls: [call()] }, { open: true }, t)).not.toContain('data-chip=')
    expect(renderTools({ calls: [call({ name: 'write', args: { path: 'z', content: 'x' }, status: 'error' })] }, { open: true }, t)).not.toContain('data-chip=')
  })

  test('the preview shows the diff lines of the file', () => {
    const html = renderTools({ calls: [call({ name: 'edit', args: { path: 'x.ts', oldText: 'old', newText: 'new' } })] }, { open: true }, t)
    expect(html).toContain('p-tools-preview')
    expect(html).toContain('>old</span>')
    expect(html).toContain('>new</span>')
  })
})

describe('escaping', () => {
  test('names, summaries, results, paths and ids never reach the page raw', () => {
    const calls = [
      call({ id: NASTY, name: NASTY, args: { command: NASTY }, summary: NASTY, result: NASTY }),
      call({ id: 'w', name: 'write', args: { path: NASTY, content: NASTY }, summary: NASTY, result: NASTY }),
      call({ id: 'm', name: `mcp__${NASTY}__${NASTY}`, args: {}, summary: `mcp__${NASTY}__${NASTY}`, result: NASTY }),
      call({ id: 'e', name: 'edit', args: { path: NASTY, oldText: NASTY, newText: NASTY }, summary: NASTY, result: NASTY }),
    ]
    const html = renderTools({ index: 1, calls }, { open: true, openRows: new Set(calls.map((c) => c.id)), moreOpen: true }, t)
    expect(html).not.toContain('<script>')
    expect(html).not.toContain('alert(1)</')
    expect(html).toContain(ESCAPED)
    // attribute values cannot be broken out of
    expect(html).not.toMatch(/data-id="[^"]*"[^>]*"q"/)
    expect(html).toContain(`data-id="${ESCAPED}"`)
  })

  test('an unescaped label from the translator is escaped too', () => {
    const html = renderTools({ calls: [call()] }, {}, () => '<b>x</b>')
    expect(html).not.toContain('<b>x</b>')
  })
})

describe('texts', () => {
  test('English and Spanish define the same keys, all prefixed', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.tools.')).toBe(true)
    expect(TEXTS.es['chat.tools.many']).toContain('{n}')
    expect(TEXTS.es['chat.tools.read']).toBe('Leer archivo')
  })

  test('every key used by the renderer exists in both languages', () => {
    const used = new Set()
    const spy = (key, vars) => (used.add(key), t(key, vars))
    const calls = ['bash', 'read', 'write', 'edit', 'grep', 'find', 'ls', 'other'].map((name, i) =>
      call({ id: `k${i}`, name, args: { path: 'p', content: 'a', oldText: 'a', newText: 'b', pattern: 'x' }, status: i === 0 ? 'running' : i === 1 ? 'error' : 'done', result: i === 2 ? '' : Array.from({ length: 30 }, () => 'l').join('\n') }),
    )
    renderTools({ calls: calls.concat(writes()) }, { open: true, openRows: new Set(calls.map((c) => c.id)) }, spy)
    for (const key of used) {
      expect(TEXTS.en[key]).toBeDefined()
      expect(TEXTS.es[key]).toBeDefined()
    }
    function writes() {
      return Array.from({ length: 5 }, (_, i) => call({ id: `z${i}`, name: 'write', args: { path: `f${i}`, content: 'q' } }))
    }
  })
})

describe('thinking steps (design: the "Thinking" rows)', () => {
  const tt = (key, vars) => (vars?.n !== undefined ? `${key}:${vars.n}` : key)
  const think = { id: 'k1', name: 'thinking', think: true, text: 'RELEASE is required.\nI will make it optional.', status: 'done' }
  const read = { id: 'c1', name: 'read', args: { path: 'env.ts' }, status: 'done', summary: 'env.ts', result: 'x' }

  test('a thinking step is a row with the think icon, the Thinking label and its first line as the chip', () => {
    const html = renderTools({ index: 0, calls: [think, read] }, { open: true }, tt)
    expect(html).toContain('data-icon="think"')
    expect(html).toContain('chat.tools.thinking')
    expect(html).toContain('RELEASE is required.')
  })

  test('its detail is the whole text', () => {
    const html = renderTools({ index: 0, calls: [think] }, { open: true, openRows: new Set(['k1']) }, tt)
    expect(html).toContain('I will make it optional.')
  })

  test('the header counts the tool calls only', () => {
    expect(renderTools({ index: 0, calls: [think, read] }, {}, tt)).toContain('chat.tools.one')
  })

  test('only thinking: the header says so', () => {
    expect(renderTools({ index: 0, calls: [think] }, {}, tt)).toContain('chat.tools.thought')
  })
})
