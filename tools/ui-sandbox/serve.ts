// The UI sandbox: serves ui/ with a stubbed bridge and the design's "simulate events" buttons, so every
// state of the chat can be seen and tried without Pi (the same events the dock would send).
//
//   bun tools/ui-sandbox/serve.ts [port]        then open http://127.0.0.1:8770/
//
// Query: ?lang=es   ?first-run   (the onboarding instead of the chat)
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

const root = join(import.meta.dir, '../..')
const ui = join(root, 'ui')
const port = Number(process.argv[2] ?? 8770)
const golden = JSON.parse(readFileSync(join(root, 'crates/aria-core/tests/golden/settings-provider-opencode-go.json'), 'utf8'))
golden.setup = { completed: true, mode: 'full' }
golden.assistant.name = 'Aria'

const rewrite = (text: string) => text.replaceAll('aria://app/', '/')

const sandbox = `<script>
(() => {
  const params = new URLSearchParams(location.search)
  const settings = params.has('first-run') ? null : ${JSON.stringify(golden)}
  if (settings && params.get('lang')) settings.assistant.language = params.get('lang')
  window.__sent = []
  const COMMANDS = [
    { name: 'effort', description: 'Set the effort level', source: 'extension' },
    { name: 'autocompact', description: 'Set the compaction threshold', source: 'extension' },
    { name: 'plan', description: 'Plan with the planning model', source: 'extension' },
    { name: 'execute', description: 'Execute the approved plan', source: 'extension' },
    { name: 'skill:systematic-debugging', description: 'Debug by finding the cause first', source: 'skill' },
  ]
  const FILES = ['crates/aria/src/chat.rs', 'crates/aria/src/dock.rs', 'ui/chat/view.js', 'ui/chat/state.js', 'README.md', 'docs/', 'design/Aria.html']
  const MODELS = [
    { provider: 'opencode-go', id: 'deepseek-v4.1-flash', name: 'DeepSeek V4.1 Flash', tag: 'daily' },
    { provider: 'opencode-go', id: 'glm-5.3', name: 'GLM 5.3', tag: 'strong' },
    { provider: 'opencode-go', id: 'cat-jam-free', name: 'Cat Jam', tag: 'free', tier: 'free' },
  ]
  window.webkit = { messageHandlers: { aria: { postMessage(m) {
    window.__sent.push(m)
    setTimeout(() => {
      const r = (x) => window.aria.receive({ id: m.id, ...x })
      if (m.type === 'loadSettings') r({ type: 'settings', settings, exists: !!settings, problem: null, connected: ['opencode-go:api_key'] })
      else if (m.type === 'sessionsLoad') r({ type: 'sessions', sessions: [{ id: 's1', title: 'Fix the failing build' }], active: 's1' })
      else if (m.type === 'saveSettings') r({ type: 'saved' })
      else if (m.type === 'saveKey') r({ type: 'keyResult', ok: m.key !== 'bad', error: m.key === 'bad' ? 'key-rejected' : null })
      else if (m.type === 'startOauth') { r({ type: 'oauthUrl', url: 'https://claude.ai/oauth/authorize?x=1' }); setTimeout(() => r({ type: 'oauthResult', ok: true, error: null }), 300) }
      else if (m.type === 'installClaudeBridge') setTimeout(() => (window.sim.installError ? r({ type: 'error', code: window.sim.installError }) : r({ type: 'saved' })), 300)
      else if (m.type === 'closeRequest') document.body.classList.remove('open')
      else if (m.type === 'session') {
        const inner = m.message
        const w = (x) => window.aria.receive({ type: 'session', session: m.session, message: { id: inner.id, ...x } })
        if (inner.type === 'listCommands') w({ type: 'commands', commands: COMMANDS })
        else if (inner.type === 'listFiles') w({ type: 'files', files: FILES.filter((f) => f.includes(inner.query)) })
        else if (inner.type === 'listModels') w({ type: 'models', models: MODELS })
        else if (inner.type === 'setModel') { w({ type: 'saved' }); sim.sess(m.session, { type: 'chatState', workdir: '/home/ana/Projects/aria', running: false, model: inner.model, thinking: 'medium', context: 22, problem: null }) }
        else if (inner.type === 'chatLoad') { sim.sess(m.session, { type: 'chatQuota', windows: [{ id: '5h', label: '5h', percent: 42, resetsAt: new Date(Date.now() + 3 * 3600e3 + 20 * 60e3).toISOString() }, { id: 'week', label: 'Week', percent: 17, resetsAt: new Date(Date.now() + 5 * 86400e3).toISOString() }, { id: 'month', label: 'Month', percent: 83, resetsAt: null }, { id: 'fable', label: 'Fable', percent: 96, resetsAt: new Date(Date.now() + 2 * 86400e3).toISOString() }] }) }
        if (inner.type === 'chatLoad') sim.sess(m.session, { type: 'chatState', workdir: '/home/ana/Projects/aria', running: false, model: sim.model, thinking: 'medium', context: sim.context, problem: null })
        else if (inner.type === 'chatSend' && inner.text === 'REFUSE') sim.sess(m.session, { type: 'chatEvent', event: { type: 'command_failed', command: 'prompt', error: 'Cannot submit a prompt while compaction is in progress', text: 'REFUSE' } })
        else if (inner.type === 'chatSend' && inner.behavior === 'followUp') { sim.queue.push(inner.text); sim.ev({ type: 'queue_update', steering: [], followUp: [...sim.queue] }, m.session) }
        else if (inner.type === 'chatClearQueue') { sim.queue = [...(inner.keep ?? [])]; sim.ev({ type: 'queue_update', steering: [], followUp: [...sim.queue] }, m.session) }
        else if (inner.type === 'chatSend') sim.reply(m.session, inner.text)
        else if (inner.type === 'chatAbort') { sim.stopRun(m.session) }
      }
    }, 15)
  } } } }

  const card = (c) => JSON.stringify({ aria: 1, ...c })
  const sim = {
    model: 'opencode-go/deepseek-v4.1-flash', context: 18, n: 0, timer: null, active: 's1', queue: [],
    sess: (session, message) => window.aria.receive({ type: 'session', session, message }),
    ev: (event, session) => sim.sess(session ?? sim.active, { type: 'chatEvent', event }),
    state: (extra = {}, session) => sim.sess(session ?? sim.active, { type: 'chatState', workdir: '/home/ana/Projects/aria', running: false, model: sim.model, thinking: 'medium', context: sim.context, problem: null, ...extra }),
    say(text, session) {
      sim.ev({ type: 'message_start', message: { role: 'assistant' } }, session)
      sim.ev({ type: 'message_end', message: { role: 'assistant', content: [{ type: 'text', text }] } }, session)
    },
    tools(session) {
      const calls = [['bash', { command: 'pnpm build 2>&1 | tail -20' }], ['read', { path: 'vite.config.ts' }], ['grep', { pattern: 'RELEASE' }], ['edit', { path: 'env.ts', old: 'x', new: 'y' }]]
      calls.forEach(([name, args], i) => setTimeout(() => {
        sim.ev({ type: 'tool_execution_start', toolCallId: 'c' + sim.n + i, toolName: name, args }, session)
        setTimeout(() => sim.ev({ type: 'tool_execution_end', toolCallId: 'c' + sim.n + i, toolName: name, isError: false, result: { content: [{ type: 'text', text: name + ' ok\\nsecond line' }] } }, session), 500)
      }, i * 700))
      sim.n++
    },
    reply(session, text) {
      sim.stopRun(session)
      sim.ev({ type: 'agent_start' }, session)
      sim.ev({ type: 'aria_jev', mode: 'focused task', action: 'searches memory' }, session)
      sim.tools(session)
      sim.timer = setTimeout(() => {
        sim.say('Commit \`a41c9e2\` removed \`RELEASE\` from \`.env\`, but \`env.ts\` still requires it.\\n\\nWe solved this before: we made it optional, with the commit hash as a fallback.\\n\\n\`\`\`diff\\n- const required = ["API_URL", "RELEASE"]\\n+ const required = ["API_URL"]\\n+ RELEASE: process.env.RELEASE ?? gitSha(),\\n\`\`\`', session)
        sim.ev({ type: 'agent_settled', aborted: false }, session)
      }, 3600)
    },
    stopRun(session) { clearTimeout(sim.timer); if (sim.timer) { sim.ev({ type: 'agent_settled', aborted: true }, session); sim.timer = null } },
  }
  window.sim = sim

  const BUTTONS = {
    'Permission request': () => sim.sess(sim.active, { type: 'chatDialog', id: 'p' + ++sim.n, method: 'select', request: { title: card({ kind: 'sandbox', summary: 'Wants to leave the sandbox', detail: 'rm -rf ../old-build', jev: 'Deletes a folder outside the project. Looks like leftover build output.' }), options: ['allow', 'always', 'deny'] }, hidden: false }),
    'Write outside': () => sim.sess(sim.active, { type: 'chatDialog', id: 'w' + ++sim.n, method: 'select', request: { title: card({ kind: 'write', path: '~/Documents/notes.md', summary: 'Wants to change a file' }), options: ['allow', 'always', 'deny'] }, hidden: false }),
    'Quota almost out': () => sim.sess(sim.active, { type: 'chatDialog', id: 'q' + ++sim.n, method: 'select', request: { title: card({ kind: 'quota', percent: 98, freeModel: 'opencode-go/cat-jam-free', countdown: 300 }), options: ['free', 'keep'], timeout: 300000 }, hidden: false }),
    'Compact context': () => { sim.ev({ type: 'compaction_start', reason: 'threshold' }); setTimeout(() => { sim.ev({ type: 'compaction_end', reason: 'threshold', result: { tokensBefore: 150000 }, aborted: false }); sim.sess(sim.active, { type: 'chatEvent', event: { type: 'ui_notice', request: { method: 'notify', message: 'engram: saved the decisions from this session', notifyType: 'info' } } }) }, 1500) },
    'Run /plan': () => { sim.sess(sim.active, { type: 'chatEvent', event: { type: 'aria_jev', mode: 'needs thinking', action: 'strong model · Superpowers' } }) },
    'Toggle working': () => sim.ev({ type: sim.working ? 'agent_settled' : 'agent_start', aborted: false }) || (sim.working = !sim.working),
    'Tool calls': () => { sim.ev({ type: 'agent_start' }); sim.tools(); setTimeout(() => sim.ev({ type: 'agent_settled' }), 3600) },
    'Run all tasks': () => { window.aria.chat.setTasks([{ id: '1', label: 'Reproduce the build error', status: 'done' }, { id: '2', label: 'Make RELEASE optional', status: 'active' }, { id: '3', label: 'Run the tests', status: 'todo' }]) },
    'Ask a question': () => sim.sess(sim.active, { type: 'chatDialog', id: 'a' + ++sim.n, method: 'input', request: { title: card({ kind: 'question', questions: [{ question: 'Which fallback should RELEASE use?', header: 'Fallback', options: [{ label: 'Git commit hash', description: 'Always available in CI', recommended: true }, { label: 'Package version' }, { label: 'A fixed string' }] }, { question: 'Which checks should run?', header: 'Checks', multi: true, options: [{ label: 'Unit tests' }, { label: 'Type check' }, { label: 'Lint' }] }] }) }, hidden: false }),
    'Send a message': () => { sim.sess(sim.active, { type: 'chatEvent', event: { type: 'message_start', message: { role: 'user', content: 'x' } } }); sim.say('Done. The build passes now.') },
    'Subagents (Alt A)': () => { window.aria.chat.setAgents([{ id: 'a1', role: 'frontend', task: 'Adjust the settings screen', status: 'working · 3 steps', kind: 'active' }, { id: 'a2', role: 'security', task: 'Review the sandbox policy', status: 'done', kind: 'done' }, { id: 'a3', role: 'reviewer', task: 'Review the diff', status: 'queued', kind: 'queued' }]); window.aria.chat.toggleAgents() },
    'Other tab needs you': () => { sim.sess('s2', { type: 'chatDialog', id: 'x' + ++sim.n, method: 'select', request: { title: card({ kind: 'sandbox', summary: 'Wants to leave the sandbox', detail: 'ls /etc' }), options: ['allow', 'always', 'deny'] }, hidden: false }) },
    'Many messages': () => { for (let i = 0; i < 8; i++) { sim.sess(sim.active, { type: 'chatEvent', event: { type: 'message_start', message: { role: 'user', content: 'x' } } }); sim.say('Answer number ' + (i + 1) + ' with enough text to take some room in the thread.\\n\\nA second paragraph.') } },
  }
  const nav = document.createElement('nav')
  nav.style.cssText = 'position:fixed;left:8px;top:8px;width:230px;z-index:99;display:flex;flex-direction:column;gap:4px;font:12px system-ui;color:#ccc'
  nav.innerHTML = '<b style="letter-spacing:.08em">SIMULATE</b>'
  for (const [label, fn] of Object.entries(BUTTONS)) {
    const b = document.createElement('button'); b.textContent = label; b.style.cssText = 'text-align:left;padding:5px 8px;border-radius:6px;border:1px solid #333;background:#16161a;color:#ddd;cursor:pointer'
    b.onclick = fn; nav.appendChild(b)
  }
  addEventListener('DOMContentLoaded', () => { document.body.appendChild(nav); document.documentElement.style.background = '#0b0b0d'; setTimeout(() => window.aria.receive({ type: 'open', monitor: 'DP-1' }), 600) })
})()
</script>`

Bun.serve({
  port,
  fetch(req) {
    const path = new URL(req.url).pathname
    if (path === '/') {
      const html = rewrite(readFileSync(join(ui, 'index.html'), 'utf8')).replace('<link', `${sandbox}<link`)
      return new Response(html, { headers: { 'content-type': 'text/html' } })
    }
    const file = Bun.file(join(ui, path))
    const type = path.endsWith('.js') ? 'text/javascript' : path.endsWith('.css') ? 'text/css' : 'application/json'
    return file.text().then((text) => new Response(path.endsWith('.json') ? text : rewrite(text), { headers: { 'content-type': type } }), () => new Response('not found', { status: 404 }))
  },
})
console.log(`UI sandbox: http://127.0.0.1:${port}/  (?lang=es, ?first-run)`)
