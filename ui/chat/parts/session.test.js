// Run: bun test ui/chat/parts/session.test.js
import { describe, expect, test } from 'bun:test'
import { TEXTS, formatElapsed, renderAgentSession } from './session.js'
import { TEXTS as AGENT_TEXTS } from './subagents.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const EVIL = `<script>alert("x")</script> & 'q'`
const agent = {
  role: 'frontend',
  task: 'form and loading states',
  status: 'writing',
  kind: 'active',
  steps: [
    { kind: 'tool', icon: 'read', verb: 'Read', target: 'LoginForm.tsx' },
    { kind: 'msg', text: 'Submits twice.' },
    { kind: 'tool', icon: 'edit', verb: 'Edit', target: 'LoginForm.tsx', meta: '+18 −4' },
    { kind: 'code', lines: [{ kind: '-', text: 'old' }, { kind: '+', text: 'new' }, { kind: ' ', text: 'same' }] },
    { kind: 'live', text: 'Writing loading states', since: 1000 },
    { kind: 'note', label: 'Done', text: 'all good' },
  ],
}

describe('elapsed', () => {
  test('seconds with one decimal, then minutes', () => {
    expect(formatElapsed(0)).toBe('0.0s')
    expect(formatElapsed(74_300)).toBe('1m 14.3s')
    expect(formatElapsed(5_400)).toBe('5.4s')
    expect(formatElapsed(-5)).toBe('0.0s')
    expect(formatElapsed(NaN)).toBe('0.0s')
  })
})

describe('the session view', () => {
  test('has a back control, the role, task and status', () => {
    const html = renderAgentSession({ agent }, { now: 4000 }, t)
    expect(html).toContain('data-chat="agent-back"')
    expect(html).toContain('chat.session.back')
    expect(html).toContain('frontend')
    expect(html).toContain('form and loading states')
    expect(html).toContain('writing')
    expect(html).toContain('kind-active')
  })
  test('renders every step kind', () => {
    const html = renderAgentSession({ agent }, { now: 4000 }, t)
    expect(html).toContain('p-session-tool')
    expect(html).toContain('p-session-msg')
    expect(html).toContain('p-session-code')
    expect(html).toContain('p-session-live')
    expect(html).toContain('p-session-note')
    expect(html).toContain('icon-read')
    expect(html).toContain('icon-edit')
    expect(html).toContain('+18 −4')
    expect(html).toContain('line-del')
    expect(html).toContain('line-add')
    expect(html).toContain('line-ctx')
  })
  test('the live step shows elapsed time from since and now', () => {
    expect(renderAgentSession({ agent }, { now: 75_300 }, t)).toContain('1m 14.3s')
  })
  test('unknown step kinds are skipped, unknown icons and kinds are neutral', () => {
    const html = renderAgentSession({ agent: { role: 'r', kind: 'zzz', steps: [{ kind: 'weird' }, null, { kind: 'tool', icon: 'zap', verb: 'V' }] } }, {}, t)
    expect(html).toContain('kind-queued')
    expect(html.match(/<li class="p-session-step/g)).toHaveLength(1)
    expect(html).not.toContain('icon-zap')
  })
  test('a meta color is only accepted when it is a plain color', () => {
    const ok = renderAgentSession({ agent: { role: 'r', steps: [{ kind: 'tool', meta: 'm', metaColor: '#8BE0B0' }] } }, {}, t)
    expect(ok).toContain('color:#8BE0B0')
    const bad = renderAgentSession({ agent: { role: 'r', steps: [{ kind: 'tool', meta: 'm', metaColor: 'red;background:url(x)' }] } }, {}, t)
    expect(bad).not.toContain('url(x)')
  })
  test('no agent or no steps still renders the back control', () => {
    for (const props of [{}, undefined, { agent: null }, { agent: { role: 'r' } }]) {
      const html = renderAgentSession(props, undefined, t)
      expect(html).toContain('data-chat="agent-back"')
    }
    expect(renderAgentSession({ agent: { role: 'r', steps: [] } }, {}, t)).toContain('chat.session.empty')
  })
  test('every text field is escaped', () => {
    const evil = {
      role: EVIL, task: EVIL, status: EVIL, kind: 'active',
      steps: [
        { kind: 'tool', icon: 'run', verb: EVIL, target: EVIL, meta: EVIL },
        { kind: 'msg', text: EVIL },
        { kind: 'code', lines: [{ kind: '+', text: EVIL }] },
        { kind: 'live', text: EVIL, since: 0 },
        { kind: 'note', label: EVIL, text: EVIL },
      ],
    }
    const html = renderAgentSession({ agent: evil }, { now: 1 }, t)
    expect(html).not.toContain('<script')
    expect(html).not.toContain('alert("x")')
    expect(html).toContain('&lt;script&gt;')
    expect(html).toContain('&amp;')
    expect(html).toContain('&#39;q&#39;')
  })
})

test('texts: both languages have the same keys, prefixed', () => {
  expect(Object.keys(TEXTS.en).sort()).toEqual(Object.keys(TEXTS.es).sort())
  for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.session.')).toBe(true)
})

describe('what the subagent socket sends (crates/aria/src/watch.rs)', () => {
  // the app merges every part's texts (ui/chat/texts.js)
  const es = (key) => ({ ...TEXTS.en, ...AGENT_TEXTS.en, ...TEXTS.es, ...AGENT_TEXTS.es })[key] ?? key
  const agent = (steps) => ({ role: 'researcher', task: 'Count the files', status: 'reading', kind: 'active', steps })

  test('a tool step is named in the reader\'s language with its icon', () => {
    const html = renderAgentSession({ agent: agent([{ kind: 'tool', tool: 'bash', target: 'ls src' }]) }, {}, es)
    expect(html).toContain('Ejecutar')
    expect(html).toContain('icon-run')
    expect(html).toContain('ls src')
  })

  test('read, edit and search tools get their icons; an unknown tool keeps its name', () => {
    const html = renderAgentSession({ agent: agent([{ kind: 'tool', tool: 'read', target: 'a' }, { kind: 'tool', tool: 'edit', target: 'b' }, { kind: 'tool', tool: 'grep', target: 'c' }, { kind: 'tool', tool: 'codemode', target: 'd' }]) }, {}, t)
    expect(html).toContain('icon-read')
    expect(html).toContain('icon-edit')
    expect(html).toContain('icon-search')
    expect(html).toContain('Codemode')
  })

  test('the status word is translated', () => {
    expect(renderAgentSession({ agent: agent([]) }, {}, es)).toContain('leyendo')
  })

  test('what the agent says keeps its lines and code spans, without markdown heading marks', () => {
    const html = renderAgentSession({ agent: agent([{ kind: 'msg', text: 'There are 23.\n\n## Learnings\n- use `find`' }]) }, {}, t)
    expect(html).not.toContain('## ')
    expect(html).toContain('Learnings')
    expect(html).toContain('<code')
  })
})
