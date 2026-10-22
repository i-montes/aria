// Drives the chat's controller in headless Chrome against the sandbox's stubbed bridge and checks the
// behaviours the unit tests cannot see (focus, redraws, key order, races). Needs Chrome and the sandbox:
//
//   bun tools/ui-sandbox/serve.ts 8770 &      then      bun tools/ui-sandbox/check.ts
//
// Exit code 1 when any check fails.
const url = process.env.URL ?? 'http://127.0.0.1:8770/'
const port = 9338
const browser = process.env.CHROME ?? (process.platform === 'darwin' ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome-stable')
const chrome = Bun.spawn([browser, '--headless=new', '--disable-gpu', '--no-sandbox', `--user-data-dir=/tmp/aria-check-${process.pid}`, `--remote-debugging-port=${port}`, '--window-size=940,940', 'about:blank'], { stderr: 'ignore', stdout: 'ignore' })
let target: any
for (let i = 0; i < 50 && !target; i++) {
  await Bun.sleep(200)
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t: any) => t.type === 'page') } catch {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl)
await new Promise((r) => (ws.onopen = r))
let id = 0
const pending = new Map<number, (v: any) => void>()
ws.onmessage = (e) => {
  const m = JSON.parse(String(e.data))
  if (m.id && pending.has(m.id)) { pending.get(m.id)!(m.result); pending.delete(m.id) }
  else if (m.method === 'Runtime.exceptionThrown') console.log('PAGE ERROR', m.params.exceptionDetails.exception?.description?.split('\n')[0] ?? m.params.exceptionDetails.text)
}
const send = (method: string, params: any = {}) => new Promise<any>((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })) })
await send('Runtime.enable'); await send('Page.enable')
await send('Page.navigate', { url })
await Bun.sleep(2200)

// Everything runs inside the page: each check returns [name, ok, detail].
const script = `(async () => {
  const wait = (ms) => new Promise((r) => setTimeout(r, ms))
  const $ = (s) => document.querySelector(s)
  const typeIn = (value) => { const i = $('#chat-input'); i.focus(); i.value = value; i.setSelectionRange(value.length, value.length); i.dispatchEvent(new Event('input', { bubbles: true })) }
  const key = (el, k, extra = {}) => el.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...extra }))
  const sess = (message, session = 's1') => window.aria.receive({ type: 'session', session, message })
  const ev = (event) => sess({ type: 'chatEvent', event })
  const sent = () => window.__sent.filter((m) => m.type === 'session').map((m) => m.message)
  const results = []
  const check = (name, ok, detail = '') => results.push([name, !!ok, String(detail)])
  const settle = () => wait(30) // the page applies session messages one microtask later
  const reset = async () => { window.sim.queue = []; window.sim.steering = []; ev({ type: 'agent_settled', aborted: false }); ev({ type: 'session_reset' }); const i = $('#chat-input'); if (i) { i.value = ''; i.dispatchEvent(new Event('input', { bubbles: true })) }; await wait(50) }

  await wait(300)

  // 0. the header with a single session: the name, and no model button
  check('the single-session header has the assistant name and no model button', $('#chat-header .chat-name') && !$('#chat-header .p-meter') && !$('#chat-header [data-chat="open-models"]'), $('#chat-header').innerHTML.slice(0, 200))

  await wait(200)
  check('beside the name only the 5h bar shows', document.querySelectorAll('#chat-header .p-qb-inline').length === 1 && $('#chat-header .p-qb-inline').textContent.includes('5h') && $('#chat-header .p-qb-inline').textContent.includes('42%'), $('#chat-header .p-qb-inline')?.textContent)
  const popBox = () => $('#chat-header .p-qb-pop').getBoundingClientRect()
  check('the card with every window is hidden until hover', getComputedStyle($('#chat-header .p-qb-pop')).visibility === 'hidden')
  $('#chat-header .p-qb').dispatchEvent(new MouseEvent('mouseover', { bubbles: true }))
  $('#chat-header .p-qb').focus() // hover cannot be simulated from here; focus opens the same card
  await wait(250)
  const pop = $('#chat-header .p-qb-pop')
  check('the card lists all four windows (5h, Week, Month, Fable) when it opens', getComputedStyle(pop).visibility === 'visible' && pop.querySelectorAll('.p-qb-hero, .p-qb-row').length === 4, pop.querySelectorAll('.p-qb-hero, .p-qb-row').length)
  check('a nearly full window turns amber or red in the card', pop.querySelector('.p-qb-row.is-warn') && pop.querySelector('.p-qb-row.is-crit'))
  check('the card names the provider and leads with the 5-hour window', pop.textContent.includes('OpenCode Go') && pop.querySelector('.p-qb-hero')?.textContent.includes('42'), pop.querySelector('.p-qb-hero')?.textContent.replace(/\\s+/g, ' '))
  check('the card is fully inside the dock', popBox().right <= window.innerWidth && popBox().left >= 0, JSON.stringify(popBox()))
  document.activeElement.blur()

  // 8c. an agent's session keeps the reader's place while the agent writes, and follows it when at the bottom
  const longSteps = Array.from({ length: 40 }, (_, i) => ({ kind: 'tool', tool: 'read', target: 'src/file' + i + '.ts' }))
  const agentWith = (steps) => [{ id: 'w1', role: 'researcher', task: 'Count the files', status: 'reading', kind: 'active', steps }]
  window.aria.chat.setAgents(agentWith(longSteps)); if ($('#chat-body').dataset.view !== 'agents') window.aria.chat.toggleAgents(); await wait(60)
  $('#view-agents [data-chat="agent-select"]').click(); await wait(80)
  const pane = () => $('#view-session .chat-panel-scroll')
  pane().scrollTop = 120; await wait(20)
  window.aria.chat.setAgents(agentWith([...longSteps, { kind: 'msg', text: 'more' }])); await wait(60)
  check('a new step does not send the agent session back to the top', pane().scrollTop >= 100, pane().scrollTop)
  pane().scrollTop = pane().scrollHeight; await wait(20)
  window.aria.chat.setAgents(agentWith([...longSteps, { kind: 'msg', text: 'more' }, { kind: 'msg', text: 'and more' }])); await wait(60)
  check('at the bottom, the session follows what the agent writes', pane().scrollHeight - pane().scrollTop - pane().clientHeight < 4, [pane().scrollTop, pane().scrollHeight, pane().clientHeight].join(' '))
  window.aria.chat.toggleAgents(); await wait(40)

  // 8c. background work: the strip above the composer, Stop on a pill, a pill opening the list, a job's output
  {
    window.sim.outputs = 0
    ;[...document.querySelectorAll('nav button')].find((b) => b.textContent === 'Background work').click(); await wait(80)
    const pills = () => [...document.querySelectorAll('#chat-background [data-chat="bg-open"]')].map((b) => b.dataset.id)
    check('the strip shows the session’s background work', $('.chat-bg')?.classList.contains('is-on') && pills().join() === 'j1,j2,j3', pills().join())
    $('#chat-background [data-chat="bg-stop"][data-id="j1"]').click(); await wait(80)
    const stops = window.__sent.filter((m) => m.type === 'jobStop')
    check('stop sends jobStop with the job id', stops.length === 1 && stops[0].job === 'j1' && pills().join() === 'j2,j3', JSON.stringify(stops) + ' ' + pills().join())
    $('#chat-background [data-chat="bg-open"][data-id="j2"]').click(); await wait(80)
    const list = $('#view-agents')
    check('a pill opens the background list on that job', !list.hidden && list.querySelector('[data-job="j2"]')?.classList.contains('is-target'), list.innerHTML.slice(0, 120))
    list.querySelector('[data-chat="bg-output"][data-id="j1"]').click(); await wait(120)
    const out = list.querySelector('[data-job="j1"] .p-bg-output')?.textContent ?? ''
    check('Output shows the job’s lines', ['$ bun dev', 'ready on http://localhost:3000', 'GET / 200 in 4 ms'].every((l) => out.includes(l)), out)
    list.querySelector('[data-chat="bg-output"][data-id="j1"]').click(); await wait(40)
    window.aria.chat.toggleAgents(); await wait(40)
    window.aria.receive({ type: 'jobs', jobs: [] }); window.sim.jobs = []; await wait(40)
  }

  // 8d. the background list (Alt+A): every session's jobs, the output polled while a row is open, a failed pill dismissed
  {
    const now = Date.now()
    const job = (x) => ({ number: 1, session: 's1', kind: 'process', command: null, task: null, folder: '~/p', state: 'running', startedAt: now - 60000, endedAt: null, url: null, exit: null, isolation: null, worktree: null, writes: false, result: null, conflict: false, dismissed: false, ...x })
    const jobs = [job({ id: 'b1', title: 'bun dev', url: 'http://localhost:3000' }), job({ id: 'b2', kind: 'agent', title: 'tests', task: 'Fix the tests' }), job({ id: 'b3', state: 'failed', endedAt: now - 1000, title: 'broken' }), job({ id: 'b4', kind: 'agent', state: 'done', endedAt: now - 2000, title: 'research', task: 'Research', result: 'The answer <b>is</b> 42' })]
    window.aria.chat.setAgents([{ id: 'w1', role: 'tests', task: 'Fix the tests', status: 'writing', kind: 'active', steps: [{ kind: 'msg', text: 'on it' }] }, { id: 'w2', role: 'reviewer', task: 'Review', status: 'queued', kind: 'queued' }, { id: 'w3', role: 'research', task: 'Research', status: 'done', kind: 'done', steps: [{ kind: 'msg', text: 'watch steps' }] }])
    window.aria.receive({ type: 'jobs', jobs }); await wait(60)
    check('the hint says alt+a background while a job runs', $('.p-composer-hints')?.textContent.includes('background') || $('.p-composer-hints')?.textContent.includes('segundo plano'))
    const before = window.__sent.length
    $('#chat-background [data-chat="bg-open"][data-id="b2"]').click(); await wait(80)
    const dismissed = window.__sent.slice(before).filter((m) => m.type === 'jobDismiss').map((m) => m.job)
    check('opening the list dismisses the session’s failed job from the strip', dismissed.join() === 'b3', dismissed.join())
    const list = $('#view-agents')
    check('the list shows the jobs and the parallel agents, not the job’s own agent', list.querySelectorAll('[data-job]').length === 4 && !!list.querySelector('[data-chat="agent-select"][data-id="w2"]') && !list.querySelector('[data-chat="agent-select"][data-id="w1"]'), list.querySelectorAll('[data-job]').length)
    check('opened from a pill, that row is outlined and holds the focus', list.querySelector('[data-job="b2"]')?.classList.contains('is-target') && document.activeElement?.closest('[data-job]')?.dataset.job === 'b2', document.activeElement?.outerHTML?.slice(0, 80))
    const outputsBefore = window.__sent.filter((m) => m.type === 'jobOutput').length
    list.querySelector('[data-chat="bg-output"][data-id="b1"]').click(); await wait(80)
    check('Output shows the last lines', !!list.querySelector('[data-job="b1"] .p-bg-output')?.textContent.includes('ready on'), list.querySelector('[data-job="b1"]')?.textContent.replace(/\\s+/g, ' '))
    const stopB1 = () => list.querySelector('[data-chat="bg-stop"][data-id="b1"]')
    stopB1().focus()
    const outBlock = () => list.querySelector('[data-job="b1"] .p-bg-output')
    const linesBefore = outBlock()?.textContent.length ?? 0
    await wait(2100)
    check('a focused Stop keeps the focus while the output is polled', document.activeElement === stopB1() && document.activeElement?.dataset.id === 'b1', document.activeElement?.outerHTML?.slice(0, 80))
    const ob = outBlock()
    check('the output block shows its newest line after new lines arrive', ob && ob.textContent.length > linesBefore && ob.scrollHeight > ob.clientHeight && ob.scrollHeight - ob.scrollTop - ob.clientHeight < 4, ob && [ob.scrollTop, ob.scrollHeight, ob.clientHeight].join(' '))
    const polled = window.__sent.filter((m) => m.type === 'jobOutput').slice(outputsBefore)
    check('the output is asked again every second while the row is open, top level, 80 lines', polled.length >= 3 && polled.every((m) => m.job === 'b1' && m.lines === 80 && typeof m.id === 'number'), polled.length)
    list.querySelector('[data-chat="bg-output"][data-id="b1"]').click(); await wait(40)
    const folded = window.__sent.filter((m) => m.type === 'jobOutput').length
    await wait(1200)
    check('folding the row stops the polling', window.__sent.filter((m) => m.type === 'jobOutput').length === folded && !list.querySelector('.p-bg-output'))
    list.querySelector('[data-chat="bg-url"][data-id="b1"]').click(); await wait(20)
    check('Open sends the job’s URL to the browser', window.__sent.at(-1)?.type === 'openUrl' && window.__sent.at(-1)?.url === 'http://localhost:3000', JSON.stringify(window.__sent.at(-1)))
    list.querySelector('[data-chat="bg-session"][data-id="b2"]').click(); await wait(60)
    check('Session opens the watch agent the job runs', $('#view-session').textContent.includes('on it'), $('#view-session').textContent.slice(0, 80))
    $('#view-session [data-chat="agent-back"]').click(); await wait(60)
    $('#view-agents [data-chat="bg-session"][data-id="b4"]').click(); await wait(60)
    check('Result shows a finished agent’s answer, escaped, even while its watch agent is listed', $('#view-session').textContent.includes('The answer <b>is</b> 42') && !$('#view-session').textContent.includes('watch steps') && !$('#view-session b'), $('#view-session .p-session-steps')?.innerHTML.slice(0, 120))
    window.aria.chat.toggleAgents(); await wait(40)
    window.aria.receive({ type: 'jobs', jobs: [job({ id: 'b5', state: 'exited', exit: 0, endedAt: now })] }); await wait(40)
    check('a finished job alone does not change the hint', !$('.p-composer-hints')?.textContent.includes('background'), $('.p-composer-hints')?.textContent)
    window.aria.receive({ type: 'jobs', jobs: [] }); window.aria.chat.setAgents([]); await wait(40)
  }


  // 0b. a settings change from outside the page (/effort, /autocompact) makes the page read the settings again
  const loadsBefore = window.__sent.filter((m) => m.type === 'loadSettings').length
  window.aria.receive({ type: 'settingsChanged' }); await wait(80)
  check('the page reads the settings again when they change outside it', window.__sent.filter((m) => m.type === 'loadSettings').length === loadsBefore + 1)

  // 0c. the settings screen: the gear opens its overview (not the setup); a change shows the save bar, saving keeps the
  // reader there, and the dock's own change notice does not take the screen away
  $('[data-chat="settings"]').click(); await wait(250)
  check('the gear opens the settings overview', !!$('#flow [data-action="settings-open"][data-value="agents"]') && !$('#flow [data-action="next"]'))
  $('#flow [data-action="settings-open"][data-value="safety"]')?.click(); await wait(120)
  $('#flow [data-action="toggle"][data-field="commit"]')?.click(); await wait(80)
  check('a change shows the save bar', !!$('#flow [data-action="settings-save"]:not(:disabled)'))
  $('#flow [data-action="settings-save"]')?.click(); await wait(250)
  const savedShown = () => !!$('#flow .settings-bar .state.ok') && !!$('#flow [data-action="settings-home"]')
  check('saving keeps the section and says it was saved', savedShown(), $('#flow').textContent.slice(0, 80))
  window.aria.receive({ type: 'settingsChanged' }); await wait(200)
  check('the change notice that follows the save keeps the section', !!$('#flow [data-action="settings-home"]'), $('#flow').textContent.slice(0, 80))
  // each connected server (not the memory) has its access level: Ask until one is chosen; the choice is saved
  $('#flow [data-action="settings-home"]')?.click(); await wait(80)
  $('#flow [data-action="settings-open"][data-value="integrations"]')?.click(); await wait(120)
  const accessSeg = (level) => $('#flow [data-action="pick"][data-field="access:notion"][data-value="' + level + '"]')
  check('the integrations list each configured server with Read only / Ask / Allow, Ask by default', accessSeg('read') && accessSeg('ask')?.classList.contains('selected') && accessSeg('allow') && !$('#flow [data-field="access:engram"]'), $('#flow').textContent.slice(0, 200))
  check('a server configured without a credential has its row too', !!$('#flow [data-field="access:rea"][data-value="ask"].selected') && !!$('#flow [data-field="access:tavily"]'), [...document.querySelectorAll('#flow [data-field^="access:"][data-value="ask"]')].map((b) => b.dataset.field).join())
  accessSeg('allow')?.click(); await wait(80)
  $('#flow [data-action="settings-save"]')?.click(); await wait(250)
  check('choosing a level saves it', window.__sent.filter((m) => m.type === 'saveSettings').at(-1)?.settings?.integrations?.access?.notion === 'allow' && accessSeg('allow')?.classList.contains('selected'), JSON.stringify(window.__sent.filter((m) => m.type === 'saveSettings').at(-1)?.settings?.integrations?.access))
  // the MCP servers the other apps have: listed with where they came from, SSE without a checkbox; importing one
  // adds its access row and takes it off the list
  {
    const box = (id) => $('#flow [data-action="check"][data-field="mcpImport"][data-value="' + id + '"]')
    const importButton = () => $('#flow [data-action="mcp-import"]')
    check('the found servers are listed, SSE without a checkbox, Import closed until one is checked', box('claude-code:files') && box('cursor:docs') && !box('cursor:feed') && $('#flow').textContent.includes('Not supported (SSE)') && $('#flow').textContent.includes('my docs → my-docs') && importButton()?.disabled, $('#flow').textContent.slice(-600))
    check('a warning about the server is said in plain words', $('#flow').textContent.includes('an argument starts with ~, which Aria reads as your home folder'), $('#flow').textContent.slice(-600))
    box('claude-code:files')?.click(); await wait(80)
    check('checking a server opens Import and is not a change to save', importButton() && !importButton().disabled && !$('#flow [data-action="settings-save"]'))
    importButton()?.click(); await wait(300)
    const sentImport = window.__sent.filter((m) => m.type === 'mcpImport').at(-1)
    check('importing a found server', JSON.stringify(sentImport?.ids) === '["claude-code:files"]' && !box('claude-code:files') && box('cursor:docs') && $('#flow').textContent.includes('Imported 1: files') && !!$('#flow [data-field="access:files"][data-value="ask"].selected'), $('#flow').textContent.slice(-400))
  }
  // recording a shortcut: the row in the first section listens, saves through the bar, and says which one the dock
  // really uses when the chosen one is taken (the sandbox's dock holds SUPER + T)
  {
    $('#flow [data-action="settings-home"]')?.click(); await wait(80)
    $('#flow [data-action="settings-open"][data-value="assistant"]')?.click(); await wait(120)
    const capture = () => $('#flow [data-action="key-capture"]')
    const keysShown = () => [...(capture()?.querySelectorAll('kbd') ?? [])].map((k) => k.textContent).join(' + ')
    check('recording a shortcut: the row shows the default', keysShown() === 'Super + A' && capture().textContent.includes('Default'), capture()?.textContent)
    const press = (init) => { const e = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init }); document.activeElement.dispatchEvent(e); return e }
    const record = async (init) => { capture().click(); await wait(40); capture().focus(); const e = press(init); await wait(40); return e }
    await record({ key: 'K', code: 'KeyK', shiftKey: true, metaKey: true })
    check('recording a shortcut: a press becomes its keys, and the save bar shows', keysShown() === 'Super + Shift + K' && !!$('#flow [data-action="settings-save"]:not(:disabled)'), keysShown())
    $('#flow [data-action="settings-save"]')?.click(); await wait(250)
    check('recording a shortcut: it is saved in the form of the setting', window.__sent.filter((m) => m.type === 'saveSettings').at(-1)?.settings?.shortcut === 'SUPER + SHIFT + K')
    await record({ key: 't', code: 'KeyT', metaKey: true })
    $('#flow [data-action="settings-save"]')?.click(); await wait(300)
    check('recording a shortcut: a taken one says which is in use', $('#flow').textContent.includes('In use: Super + A (Super + T was taken)'), $('#flow .key-capture')?.parentElement?.textContent.replace(/\\s+/g, ' '))
    await record({ key: 'k', code: 'KeyK' })
    check('recording a shortcut: a key without a modifier is refused with a note', capture().classList.contains('recording') && $('#flow').textContent.includes('Add a modifier'), capture()?.parentElement?.textContent.replace(/\\s+/g, ' '))
    const esc = press({ key: 'Escape', code: 'Escape' }); await wait(40)
    check('recording a shortcut: Esc cancels and leaves the dock open', !capture().classList.contains('recording') && document.body.classList.contains('open') && esc.defaultPrevented && keysShown() === 'Super + T')
    await record({ key: 'Backspace', code: 'Backspace' })
    check('recording a shortcut: Backspace goes back to the default', keysShown() === 'Super + A' && capture().textContent.includes('Default'), keysShown())
    const typed = $('#flow input[data-field="shortcutText"]')
    typed.focus(); typed.value = 'super + 1'; typed.dispatchEvent(new Event('input', { bubbles: true }))
    $('#flow [data-action="shortcut-apply"]')?.click(); await wait(60)
    check('typing a shortcut: one the desktop would take can be typed', keysShown() === 'Super + 1' && !!$('#flow [data-action="settings-save"]:not(:disabled)'), keysShown())
    window.aria.receive({ type: 'platform', hotkey: '', taken: ['Super + A', 'Super + Shift + A', 'Super + Alt + A'], os: 'linux' }); await wait(60)
    check('no shortcut available: the row says so', $('#flow').textContent.includes('None available: Super + A, Super + Shift + A, Super + Alt + A are all taken'), $('#flow').textContent.replace(/\\s+/g, ' ').slice(0, 1200))
    window.aria.receive({ type: 'platform', hotkey: 'Super + A', taken: [], os: 'linux' }); await wait(40)
    const typedAgain = $('#flow input[data-field="shortcutText"]')
    typedAgain.focus(); typedAgain.value = 'super + return'; typedAgain.dispatchEvent(new Event('input', { bubbles: true }))
    key(typedAgain, 'Enter'); await wait(60)
    check('typing a shortcut: Enter applies it, and a wrong one is refused with a note', keysShown() === 'Super + 1' && $('#flow').textContent.includes('That is not a shortcut'), capture()?.parentElement?.textContent.replace(/\\s+/g, ' '))
    const typedMore = $('#flow input[data-field="shortcutText"]')
    typedMore.value = 'super + r'; typedMore.dispatchEvent(new Event('input', { bubbles: true })); await wait(40)
    check('typing a shortcut: typing again takes the note away and keeps the field', !$('#flow').textContent.includes('That is not a shortcut') && document.activeElement === typedMore, capture()?.parentElement?.textContent.replace(/\\s+/g, ' '))
    typedMore.value = ''; typedMore.dispatchEvent(new Event('input', { bubbles: true }))
    // The first-run notice goes up before the dock knows which shortcut it registered: it follows the real one.
    window.aria.receive({ type: 'notice' }); await wait(60)
    window.aria.receive({ type: 'platform', hotkey: 'Super + Shift + A', taken: ['Super + A'], os: 'linux' }); await wait(60)
    const noticeKeys = () => [...document.querySelectorAll('#notice kbd')].map((k) => k.textContent).join(' + ')
    check('the first-run notice says the shortcut really registered', document.body.classList.contains('show-notice') && noticeKeys() === 'Super + Shift + A', noticeKeys())
    window.aria.receive({ type: 'platform', hotkey: '', taken: ['Super + A', 'Super + Shift + A', 'Super + Alt + A'], os: 'linux' }); await wait(60)
    check('the first-run notice says when none could be registered', !document.querySelector('#notice kbd') && $('#notice').textContent.trim().length > 0, $('#notice').textContent)
    window.aria.receive({ type: 'platform', hotkey: '', taken: [], os: 'linux' }); await wait(60)
    check('the first-run notice says when the shortcut could not be set up at all', $('#notice').textContent.includes('Aria keeps trying'), $('#notice').textContent)
    window.aria.receive({ type: 'platform', hotkey: 'Super + A', taken: [], os: 'linux' }); await wait(40)
    document.body.classList.remove('show-notice')
    $('#flow [data-action="settings-save"]')?.click(); await wait(250)
  }
  $('#flow [data-action="settings-home"]')?.click(); await wait(80)
  $('#flow [data-action="settings-exit"]')?.click(); await wait(150)
  check('leaving the settings goes back to the chat', document.body.classList.contains('mode-chat'))

  // 1. the textarea survives a run starting, text appearing and the model changing
  typeIn('half-typed')
  const mark = $('#chat-input'); mark.__mark = 1
  ev({ type: 'agent_start' }); sess({ type: 'chatState', workdir: '/w', running: true, model: 'a/b', thinking: null, context: 5, problem: null })
  ev({ type: 'message_start', message: { role: 'assistant' } }); ev({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', contentIndex: 0, delta: 'hi' } })
  await wait(60)
  check('the textarea element survives a run and keeps its text', $('#chat-input') === mark && mark.value === 'half-typed', mark.value)
  check('the model chip sits under the input', $('.p-composer-hints .p-composer-model')?.textContent === 'b', $('.p-composer-hints')?.textContent)
  check('no "shift+enter" reminder is shown', !document.body.textContent.includes('shift+enter'))
  await reset()

  // 2. queue: Alt+Enter during a run queues it for the end, and delivery (Pi's order) moves it into the thread
  ev({ type: 'agent_start' }); await settle()
  typeIn('do this later'); key($('#chat-input'), 'Enter', { altKey: true }); await wait(60)
  check('Alt+Enter during a run queues with followUp', sent().some((m) => m.type === 'chatSend' && m.behavior === 'followUp'))
  check('the queued bubble shows', $('.p-user.queued, .p-user-queued, [data-chat="cancel-queued"]'))
  ev({ type: 'queue_update', steering: [], followUp: [] }); ev({ type: 'message_start', message: { role: 'user', content: 'do this later' } }); await wait(80)
  check('a delivered message lands in the thread and leaves the queue', !$('[data-chat="cancel-queued"]') && $('#chat-list').textContent.includes('do this later'))
  await reset()

  // 2b. Enter during a run puts the message in at the run's next step, and cancelling another keeps it there
  const wasChatMode = document.body.classList.contains('mode-chat')
  document.body.classList.add('mode-chat') // the hints line needs its real width
  ev({ type: 'agent_start' }); await settle(); window.__sent.length = 0
  const hints = $('.p-composer-hints')
  check('while Aria works the hints say what Enter and Alt+Enter do, and fit in their line', hints?.textContent.includes('alt+enter') && hints.scrollWidth <= hints.clientWidth + 1 && hints.textContent.indexOf('next step') < hints.textContent.indexOf('alt+enter'), hints?.textContent.replace(/\s+/g, ' ') + ' ' + hints?.scrollWidth + '/' + hints?.clientWidth)
  document.body.classList.toggle('mode-chat', wasChatMode)
  typeIn('look at the logs first'); key($('#chat-input'), 'Enter'); await wait(60)
  check('Enter during a run sends it as steering', sent().some((m) => m.type === 'chatSend' && m.behavior === 'steer' && m.text === 'look at the logs first'), JSON.stringify(sent().slice(-1)))
  check('its bubble says it goes in at the next step', $('.p-user.is-steer')?.textContent.includes('step'), $('.p-user.is-queued')?.textContent)
  typeIn('and then the tests'); key($('#chat-input'), 'Enter', { altKey: true }); await wait(60)
  window.__sent.length = 0
  const cancelSecond = $('[data-chat="cancel-queued"][data-id="1"]'); check('both queued messages show', cancelSecond, document.querySelectorAll('[data-chat="cancel-queued"]').length); cancelSecond?.click(); await wait(60)
  check('cancelling the other keeps the steering one as steering', sent().some((m) => m.type === 'chatClearQueue' && m.steer?.[0] === 'look at the logs first' && m.keep?.length === 0), JSON.stringify(sent().slice(-1)))
  await reset()

  // 3. cancelling one of two queued messages re-queues the other
  ev({ type: 'agent_start' }); await settle(); typeIn('first'); key($('#chat-input'), 'Enter'); await wait(60); typeIn('second'); key($('#chat-input'), 'Enter'); await wait(80)
  window.__sent.length = 0
  const cancelButton = $('[data-chat="cancel-queued"][data-id="0"]')
  check('each queued bubble has its own cancel', cancelButton, $('#chat-list').innerHTML.slice(-600))
  cancelButton?.click(); await wait(60)
  const cancel = sent().find((m) => m.type === 'chatClearQueue')
  check('cancelling the first keeps the second', cancel && JSON.stringify(cancel.steer) === '["second"]' && cancel.keep.length === 0, JSON.stringify(cancel))
  await reset()

  // 4. the question card: the free-text field keeps focus, enables Continue, Enter submits, the answers card has its header
  sim_ask()
  await wait(450)
  // design: the card takes the composer's place; the thread stays in view above it
  const qBox = $('#chat-question').getBoundingClientRect()
  const qList = $('#chat-list').getBoundingClientRect()
  check('the question card sits where the composer was, below the thread', qBox.height > 80 && qBox.top >= qList.bottom - 1 && qList.height > 40, JSON.stringify({ q: [qBox.top, qBox.height], list: [qList.top, qList.bottom] }))
  const wrapStyle = () => ($('#chat-composer-wrap') ? getComputedStyle($('#chat-composer-wrap')).visibility : 'missing')
  check('the composer is folded away while the card is up', wrapStyle() === 'hidden', wrapStyle())
  let other = $('[data-op="other"]')
  other.click(); await wait(40)
  other = $('[data-op="other"]'); other.focus()
  other.value = 'my own fallback'; other.dispatchEvent(new Event('input', { bubbles: true })); await wait(60)
  check('typing in "Other" keeps focus in that field', document.activeElement?.dataset?.op === 'other' && document.activeElement.value === 'my own fallback', document.activeElement?.outerHTML?.slice(0, 80))
  const primary = $('.p-q-primary, .p-q-continue, [data-op="next"], [data-op="submit"]')
  check('Continue is enabled once something is typed', primary && !primary.disabled)
  document.activeElement.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true })); await wait(80)
  check('Enter in Other moves to the next question', $('#chat-question').textContent.includes('Which checks'), $('#chat-question').textContent.slice(0, 120))
  other = $('[data-op="other"]')
  if (other) { other.focus(); other.value = 'unit tests'; other.dispatchEvent(new Event('input', { bubbles: true })); await wait(40); document.activeElement.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true })); await wait(100) }
  const answered = sent().find((m) => m.type === 'uiAnswer')
  check('the answers go out as JSON', answered && answered.value && JSON.parse(answered.value).length === 2, answered?.value)
  const text = $('#chat-list').textContent
  check('the answers card shows the question headers and what was said', text.includes('Fallback') && text.includes('my own fallback'), text.slice(0, 160))
  await wait(450)
  check('after answering the composer comes back and takes the keyboard', wrapStyle() === 'visible' && document.activeElement === $('#chat-input'), wrapStyle() + ' ' + document.activeElement?.id)
  await reset()

  // 4b. Scrolling a picker: the row under the moving pointer is marked, and the list stays where it was scrolled
  typeIn('/'); await wait(120)
  const cmdList = $('.p-cmd-list')
  check('the command list is long enough to scroll', cmdList && cmdList.scrollHeight > cmdList.clientHeight + 40, cmdList && [cmdList.scrollHeight, cmdList.clientHeight].join(' '))
  cmdList.scrollTop = 40; await wait(60)
  const listTop = cmdList.getBoundingClientRect().top
  const under = [...cmdList.querySelectorAll('li[data-chat="picker-hover"]')].find((li) => li.getBoundingClientRect().top > listTop + 20)
  under.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 300, clientY: under.getBoundingClientRect().top + 5 }))
  await wait(80)
  check('hovering a row while scrolled keeps the scroll', $('.p-cmd-list') === cmdList && Math.abs(cmdList.scrollTop - 40) < 2, $('.p-cmd-list')?.scrollTop)
  check('the hovered row is the marked one', under.querySelector('.p-cmd-row')?.classList.contains('on') && under.getAttribute('aria-selected') === 'true')
  window.aria.chat.escape(); await reset()
  $('[data-chat="open-models"]')?.click(); await wait(200)
  const modelList = $('.p-models-list')
  check('the model list is long enough to scroll', modelList && modelList.scrollHeight > modelList.clientHeight + 40, modelList && [modelList.scrollHeight, modelList.clientHeight].join(' '))
  modelList.scrollTop = 60; await wait(60)
  const modelTop = modelList.getBoundingClientRect().top
  const overModel = [...modelList.querySelectorAll('li[data-chat="picker-hover"]')].find((li) => li.getBoundingClientRect().top > modelTop + 20)
  overModel.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 310, clientY: overModel.getBoundingClientRect().top + 5 }))
  await wait(80)
  check('hovering a model while scrolled keeps the scroll and marks it', $('.p-models-list') === modelList && Math.abs(modelList.scrollTop - 60) < 2 && overModel.querySelector('.p-models-row')?.classList.contains('on'), $('.p-models-list')?.scrollTop)
  window.aria.chat.escape(); await reset()

  // 5. Esc order
  typeIn('/'); await wait(120)
  check('the command picker opens', $('.p-cmd.is-open'))
  const pickerText = $('.p-cmd').textContent
  check('the skills have their own heading, after the commands', $('.p-cmd-group')?.textContent.includes('SKILLS') && $('.p-cmd-group').compareDocumentPosition($('.p-cmd-row')) === Node.DOCUMENT_POSITION_PRECEDING)
  check("a skill shows without Pi's skill: prefix", pickerText.includes('/systematic-debugging') && !pickerText.includes('skill:'), pickerText.slice(0, 200))
  check("Pi's own commands are in the menu", ['compact', 'session', 'copy', 'export', 'reload'].every((name) => pickerText.includes('/' + name)))
  window.aria.chat.escape()
  window.__sent.length = 0
  typeIn('/systematic-debugging the flaky test'); key($('#chat-input'), 'Enter'); await settle()
  check('a skill typed by its short name reaches Pi by its real name', sent().some((m) => m.type === 'chatSend' && m.text === '/skill:systematic-debugging the flaky test'), JSON.stringify(sent().slice(-1)))
  check('the thread shows it as it was typed', $('#chat-list').textContent.includes('/systematic-debugging the flaky test') && !$('#chat-list').textContent.includes('skill:'))
  await reset()
  typeIn('/'); await wait(120)
  check('Esc closes the picker first', window.aria.chat.escape() === true && !$('.p-cmd.is-open'))
  ev({ type: 'agent_start' }); await settle(); window.__sent.length = 0
  check('Esc during a run does not stop it: it is left to the dock, which closes', window.aria.chat.escape() === false && !sent().some((m) => m.type === 'chatAbort'))
  check('the composer no longer offers Esc to stop', !$('#chat-composer').textContent.includes('esc'), $('#chat-composer').textContent.slice(-80))
  await reset()
  window.aria.chat.toggleAgents(); await wait(40)
  check('Esc in a panel goes back to the chat', window.aria.chat.escape() === true && $('#view-chat') && !$('#view-chat').hidden)
  check('Esc on an idle chat is left to the dock', window.aria.chat.escape() === false)

  // 6. a picker reply that arrives after the text changed does not open it
  typeIn('/'); typeIn(''); await wait(150)
  check('a stale command reply opens nothing', !$('.p-cmd.is-open'))

  // 7. memory search keeps focus while typing
  window.aria.chat.setMemory([{ id: 'm1', type: 'decision', title: 'Use Rust', key: null, content: 'We use Rust.', rev: 1, dup: 0, created: Date.now(), updated: Date.now(), seen: 0, personal: false, pinned: false, session: null, tool: 'mem_save', review: null, expires: null }])
  typeIn('/memory'); await wait(120); key($('#chat-input'), 'Enter'); await wait(40); key($('#chat-input'), 'Enter'); await wait(80) // the first Enter picks the command, the second runs it
  const search = $('[data-chat-field="memory-search"]')
  if (search) { search.focus(); search.value = 'ru'; search.dispatchEvent(new Event('input', { bubbles: true })); await wait(60) }
  check('typing in the memory search keeps its focus', document.activeElement?.dataset?.chatField === 'memory-search' && document.activeElement.value === 'ru')
  window.aria.chat.escape(); await wait(40)

  // 10. the model picker: its own search, grouped by provider, Esc closes only it
  window.aria.chat.setSettings(null)
  $('.p-composer-model').click(); await wait(20)
  check('Esc right after opening the model picker (before the list arrives) closes only the picker', window.aria.chat.escape() === true && !$('.p-models.is-open'))
  $('.p-composer-model').click(); await wait(150)
  check('the model picker has a search field with focus', document.activeElement?.dataset?.chatField === 'model-search', document.activeElement?.tagName)
  check('the models are grouped by provider', document.querySelectorAll('.p-models-group').length >= 1, document.querySelector('.p-models')?.textContent.slice(0, 120))
  const modelSearch = document.activeElement
  modelSearch.value = 'glm'; modelSearch.dispatchEvent(new Event('input', { bubbles: true })); await wait(60)
  check('typing in the search narrows the list and keeps focus', document.activeElement?.dataset?.chatField === 'model-search' && document.querySelectorAll('.p-models-row').length === 1, document.querySelectorAll('.p-models-row').length)
  window.__sent.length = 0
  document.activeElement.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true })); await wait(80)
  check('Enter picks the model', sent().some((m) => m.type === 'setModel' && m.model === 'opencode-go/glm-5.3'), JSON.stringify(sent()))
  check('the picker closes after picking', !$('.p-models.is-open'))
  $('.p-composer-model').click(); await wait(150)
  check('Esc inside the search closes only the picker and returns the keyboard to the composer', window.aria.chat.escape() === true && !$('.p-models.is-open') && document.activeElement === $('#chat-input'))
  // 11. the message index: opens at the thread's right edge, follows the pointer, leaves the scrollbar free
  await reset()
  for (const text of ['first question', 'second one', 'third one', 'fourth one']) { typeIn(text); key($('#chat-input'), 'Enter'); await wait(60); ev({ type: 'agent_settled', aborted: false }); await wait(30) }
  const listBox = $('#chat-list').getBoundingClientRect()
  const move = (x, y) => $('#chat-list').dispatchEvent(new MouseEvent('mousemove', { clientX: x, clientY: y, bubbles: true }))
  move(listBox.right - 200, listBox.top + 100); await wait(60)
  check('the index stays closed away from the edge', !$('.p-index-panel.is-open'))
  move(listBox.right - 30, listBox.top + 100); await wait(80)
  check('the index stays closed a little inside the edge (it is for the scrollbar only)', !$('.p-index-panel.is-open'))
  move(listBox.right - 4, listBox.top + 100); await wait(80)
  check('the index opens when the pointer is over the scrollbar', $('.p-index-panel.is-open'))
  const chipTops = () => [...document.querySelectorAll('.p-index-item')].map((li) => li.style.top).join()
  const tops0 = chipTops()
  move(listBox.right - 120, listBox.bottom - 60); await wait(120)
  check('the chips follow the pointer', chipTops() !== tops0, tops0 + ' -> ' + chipTops())
  const hit = document.elementFromPoint(listBox.right - 3, listBox.top + 200)
  check('the thread scrollbar is not covered by the index', hit && !hit.closest('.chat-index') && (hit.id === 'chat-list' || hit.closest('#chat-list')), hit?.className)
  move(listBox.left + 40, listBox.top + 100); await wait(300)
  check('the index closes after the pointer leaves', !$('.p-index-panel.is-open'))
  // 12. /connect: Anthropic by API key, by Claude sign-in, and the failures
  await reset()
  window.__sent.length = 0
  typeIn('/connect'); await wait(120); key($('#chat-input'), 'Enter'); await wait(40); key($('#chat-input'), 'Enter'); await wait(80) // pick the command, then run it
  check('/connect opens the card at the provider choice', $('#chat-connect .p-connect') && $('[data-op="provider"][data-value="anthropic"]'), $('#chat-connect')?.textContent.slice(0, 60))
  $('[data-op="provider"][data-value="anthropic"]').click(); await wait(40)
  check('Anthropic offers Claude sign-in or a key', $('[data-op="method"][data-value="oauth"]') && $('[data-op="method"][data-value="key"]'))
  $('[data-op="method"][data-value="key"]').click(); await wait(40)
  const keyField = $('[data-chat-field="connect-key"]')
  keyField.focus(); keyField.value = 'bad'; keyField.dispatchEvent(new Event('input', { bubbles: true })); await wait(30)
  $('[data-op="save"]').click(); await wait(120)
  check('a rejected key shows its reason and stays on the key step', $('.p-connect-error') && $('[data-chat-field="connect-key"]'), $('#chat-connect')?.textContent.slice(0, 120))
  const field2 = $('[data-chat-field="connect-key"]')
  field2.focus(); field2.value = 'sk-ant-good'; field2.dispatchEvent(new Event('input', { bubbles: true })); await wait(30)
  window.__sent.length = 0
  field2.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true })); await wait(200)
  check('the key is saved as the Anthropic API key', window.__sent.some((m) => m.type === 'saveKey' && m.provider === 'anthropic-api' && m.key === 'sk-ant-good'), JSON.stringify(window.__sent.map((m) => m.type)))
  check('every Pi is restarted after connecting', window.__sent.some((m) => m.type === 'enginesRestart'))
  check('the card says it is connected and offers the models', $('[data-op="models"]') && $('#chat-list').textContent.includes('Anthropic is connected'), $('#chat-connect')?.textContent.slice(0, 80))
  $('[data-op="models"]').click(); await wait(200)
  check('"Choose a model" opens the model picker filtered to Claude', $('.p-models.is-open') && document.querySelector('[data-chat-field="model-search"]')?.value === 'claude')
  window.aria.chat.escape(); await wait(40)
  // by Claude sign-in
  typeIn('/connect anthropic'); await wait(100); key($('#chat-input'), 'Enter'); await wait(60)
  window.__sent.length = 0
  $('[data-op="method"][data-value="oauth"]').click(); await wait(1100)
  check('Claude sign-in starts the browser flow, then installs the Claude provider', window.__sent.map((m) => m.type).join() === 'startOauth,installClaudeBridge,enginesRestart,session' || window.__sent.map((m) => m.type).slice(0, 3).join() === 'startOauth,installClaudeBridge,enginesRestart', window.__sent.map((m) => m.type).join())
  check('and ends connected', $('[data-op="models"]'))
  $('[data-op="close"]').click(); await wait(40)
  check('the X closes the card', !$('#chat-connect'))
  window.sim.installError = 'claude-code-missing'
  typeIn('/connect anthropic'); await wait(100); key($('#chat-input'), 'Enter'); await wait(60)
  $('[data-op="method"][data-value="oauth"]').click(); await wait(1100)
  check('a missing Claude Code is explained and the card goes back to the choice', $('.p-connect-error')?.textContent.includes('Claude Code') && $('[data-op="method"][data-value="key"]'), $('#chat-connect')?.textContent.slice(0, 120))
  window.sim.installError = null
  check('Esc closes the card first', window.aria.chat.escape() === true && !$('#chat-connect'))
  // 13. the keyboard's choice is not undone by a pointer that rests on a row
  await reset(); typeIn('/'); await wait(160)
  const rowsNow = () => [...document.querySelectorAll('.p-cmd-row')]
  const box = rowsNow()[3].getBoundingClientRect()
  const px = box.left + 20, py = box.top + 10
  rowsNow()[3].dispatchEvent(new MouseEvent('mousemove', { clientX: px, clientY: py, bubbles: true })); await wait(40)
  check('moving the pointer onto a row selects it', rowsNow()[3].classList.contains('on'))
  key($('#chat-input'), 'ArrowUp'); await wait(40)
  const refire = () => { const under = document.elementFromPoint(px, py); under?.dispatchEvent(new MouseEvent('mouseover', { clientX: px, clientY: py, bubbles: true })); under?.dispatchEvent(new MouseEvent('mousemove', { clientX: px, clientY: py, bubbles: true })) }
  refire(); await wait(40) // the browser re-sends hover for the new row under a pointer that did not move
  check('the keyboard choice stays while the pointer rests on another row', rowsNow()[2].classList.contains('on'), rowsNow().findIndex((r) => r.classList.contains('on')))
  key($('#chat-input'), 'ArrowUp'); await wait(40); refire(); await wait(40)
  key($('#chat-input'), 'ArrowUp'); await wait(40); refire(); await wait(40)
  check('and keeps moving with the keyboard', rowsNow().findIndex((r) => r.classList.contains('on')) === 0, rowsNow().findIndex((r) => r.classList.contains('on')))
  const box4 = rowsNow()[4].getBoundingClientRect()
  rowsNow()[4].dispatchEvent(new MouseEvent('mousemove', { clientX: box4.left + 25, clientY: box4.top + 12, bubbles: true })); await wait(40)
  check('a real pointer move selects again', rowsNow()[4].classList.contains('on'), rowsNow().findIndex((r) => r.classList.contains('on')))
  window.aria.chat.escape(); await wait(30)
  // 14. review fixes: a refused message comes back to the composer; settings open means "not seen"
  await reset(); window.__sent.length = 0
  typeIn('REFUSE'); key($('#chat-input'), 'Enter'); await wait(120)
  check('a refused message leaves the thread and comes back to the composer', ![...document.querySelectorAll('#chat-list .p-user')].some((el) => el.textContent.includes('REFUSE')) && $('#chat-input').value === 'REFUSE' && $('#chat-list').textContent.includes('compaction'), $('#chat-input').value)
  $('#chat-input').value = ''
  $('[data-chat="settings"]').click(); await wait(120)
  check('opening the settings tells the dock the chat is not shown', window.__sent.some((m) => m.type === 'chatShown' && m.shown === false))
  // 15. the sandbox's card shows its four choices and answers with the chosen string
  await reset(); window.__sent.length = 0
  ;[...document.querySelectorAll('nav button')].find((b) => b.textContent === 'Permission request').click(); await wait(80)
  const permButtons = [...document.querySelectorAll('[data-chat="permission"]')]
  check('the sandbox card has its four choices', permButtons.length === 4, permButtons.map((b) => b.textContent).join(' | '))
  permButtons[3].click(); await wait(60)
  check('choosing sends that exact option back', sent().some((m) => m.type === 'uiAnswer' && m.value === 'No'), JSON.stringify(sent().slice(-1)))
  check('and leaves its label as the note', $('#chat-list').textContent.includes('No') && !$('[data-chat="permission"]'))
  // 15b. a connector's card: its plug, its three choices, and the chosen string goes back
  await reset(); window.__sent.length = 0
  ;[...document.querySelectorAll('nav button')].find((b) => b.textContent === 'Connector wants to write').click(); await wait(80)
  const connButtons = [...document.querySelectorAll('[data-chat="permission"]')]
  check('the connector card shows its summary, its arguments and three choices', connButtons.length === 3 && $('.p-perm [data-icon="plug"]') && $('.p-perm').textContent.includes('Notion wants to create pages') && $('.p-perm-code')?.textContent.includes('title: Weekly notes') && !$('.p-perm-prompt'), connButtons.map((b) => b.textContent).join(' | '))
  connButtons[0].click(); await wait(60)
  check('the connector card sends the chosen option', sent().some((m) => m.type === 'uiAnswer' && m.value === 'Yes, just this once') && !$('[data-chat="permission"]'), JSON.stringify(sent().slice(-1)))
  // 16. the quota card answers with the extension's string and notes the decision in the chat's words
  await reset(); window.__sent.length = 0
  ;[...document.querySelectorAll('nav button')].find((b) => b.textContent === 'Quota almost out').click(); await wait(80)
  const quotaButtons = [...document.querySelectorAll('[data-chat="quota"]')]
  check("the quota card has its two choices in the chat's words", quotaButtons.length === 2 && !quotaButtons[0].textContent.includes('cat-jam'), quotaButtons.map((b) => b.textContent).join(' | '))
  quotaButtons[0].click(); await wait(60)
  check("choosing free sends the extension's own string", sent().some((m) => m.type === 'uiAnswer' && m.value === 'Switch to cat-jam-free now'))
  check('the note says it switched to the free model', $('#chat-list').textContent.includes('switched to the free model'))
  // 17. the router's route line
  await reset()
  ;[...document.querySelectorAll('nav button')].find((b) => b.textContent === 'Run /plan').click(); await wait(60)
  check('the route line from the router shows in the thread', $('#chat-list .p-jev')?.textContent.includes('strong model'), $('#chat-list').textContent.slice(0, 120))
  // 17b. paged history: the earlier divider with its folded summary, "Show earlier" twice, the reader's place kept
  {
    await reset()
    const wasChatMode = document.body.classList.contains('mode-chat')
    document.body.classList.add('mode-chat') // the thread needs its real height to scroll
    ;[...document.querySelectorAll('nav button')].find((b) => b.textContent === 'Earlier conversation').click(); await wait(80)
    const list = $('#chat-list')
    const divider = () => $('#chat-list .thread-earlier .p-divider')
    check('the earlier divider shows, the summary folded under it', divider()?.textContent.includes('Earlier in this conversation') && $('#chat-list .thread-earlier [data-chat="toggle-tool-row"]')?.getAttribute('aria-expanded') === 'false' && !list.textContent.includes('<summary>'), list.textContent.slice(0, 160))
    $('#chat-list .thread-earlier [data-chat="toggle-tool-row"]').click(); await wait(60)
    const opened = () => $('#chat-list .thread-earlier .p-tools-detail.is-open')
    check('a click opens the summary, drawn as markdown: headings, no literal ## or **', opened()?.querySelector('.p-md-h')?.textContent === 'Goal' && opened().textContent.includes('Update the docs') && !$('#chat-list .thread-earlier').innerText.includes('##') && !$('#chat-list .thread-earlier').innerText.includes('**'), $('#chat-list .thread-earlier')?.innerText.slice(0, 160))
    check('the folded summary has no chip', !$('#chat-list .thread-earlier .p-tools-cmd'))
    const more = () => $('#chat-list [data-chat="chat-earlier"]')
    check('"Show earlier" sits above the first message', more()?.textContent.includes('Show earlier') && list.querySelector('.thread-item') === more().closest('.thread-item'))
    list.scrollTop = 0; await wait(40)
    const at = divider().getBoundingClientRect().top
    more().click()
    check('asking says it is loading and posts chatEarlier without a cursor', $('#chat-list .thread-earlier-more')?.textContent.includes('Loading') && !more() && JSON.stringify(sent().filter((m) => m.type === 'chatEarlier').at(-1)) === '{"type":"chatEarlier"}', JSON.stringify(sent().at(-1)))
    await wait(120)
    const text = () => list.textContent
    check('the first page goes above, in order, before the divider', text().indexOf('Older question 1') >= 0 && text().indexOf('Older question 1') < text().indexOf('Older question 20') && text().indexOf('Older question 20') < text().indexOf('Earlier in this conversation') && text().includes('Recent question 10'), text().slice(0, 120))
    check('the reader stays where they were', Math.abs(divider().getBoundingClientRect().top - at) < 2 && list.scrollTop > 0, [at, divider().getBoundingClientRect().top, list.scrollTop].join(' '))
    check('the summary stays open', !!$('#chat-list .thread-earlier .p-tools-detail.is-open'))
    more().click(); await wait(120)
    check('the second ask sends the cursor it got', JSON.stringify(sent().filter((m) => m.type === 'chatEarlier').at(-1)?.cursor) === '{"segment":0,"index":40}', JSON.stringify(sent().filter((m) => m.type === 'chatEarlier')))
    check('the last page goes on top and the button goes away', text().indexOf('Oldest question 3') >= 0 && text().indexOf('Oldest question 3') < text().indexOf('Older question 1') && !more() && !$('#chat-list .thread-earlier-more'), text().slice(0, 120))
    document.body.classList.toggle('mode-chat', wasChatMode)
  }
  // 9. no model button next to the working folder
  document.querySelector('[data-chat="add"]').click(); await wait(60)
  check('no model button beside the working folder, with one session or several', !$('.chat-path-row .p-meter') && !$('.chat-path-row .p-composer-model'))
  check('no native tooltip says the session name: not on the tabs, not on the plus', ![...document.querySelectorAll('#chat-header .p-tabs-main, #chat-header [data-chat="add"]')].some((el) => el.hasAttribute('title')))
  check('no model button in the header, with one session or several', !$('#chat-header .p-meter') && !$('#chat-header .p-composer-model') && !$('#chat-header [data-chat="open-models"]'), $('#chat-header').innerHTML.slice(0, 160))
  // the quota card is reused as the hover card of each tab
  const tabCards = document.querySelectorAll('#chat-header .p-tabs-tip .p-qb-card')
  check('each session tab opens the same quota card on hover', tabCards.length === 2, tabCards.length)
  check('the tab card has the quota hero, the tab title and the facts', tabCards[0].querySelector('.p-qb-hero') && tabCards[0].querySelector('.p-qb-heading-title') && tabCards[0].querySelector('.p-qb-facts'), tabCards[0].innerHTML.slice(0, 160))
  document.querySelector('#chat-header .p-tabs-tab').dispatchEvent(new MouseEvent('mouseover', { bubbles: true }))
  check('with several sessions the header has tabs and no quota bar', !$('#chat-header .p-qb'), $('#chat-header').innerHTML.slice(0, 120))
  // 8. the textarea is not replaced just because text appeared
  const before = $('#chat-input'); typeIn('abc'); typeIn('abc\\ndef'); await wait(40)
  check('typing and adding a line does not replace the textarea', $('#chat-input') === before && before.rows === 2, before.rows)

  // 8b. streaming redraws only what changed: earlier messages keep their nodes, a tool call's JSON never shows
  await reset()
  const activeTab = $('#chat-header [aria-selected="true"]')?.dataset.session ?? 's1'
  const on = (event) => sess({ type: 'chatEvent', event }, activeTab)
  on({ type: 'agent_start' })
  on({ type: 'message_start', message: { role: 'assistant' } })
  on({ type: 'message_end', message: { role: 'assistant', content: [{ type: 'text', text: 'An earlier answer.' }] } })
  on({ type: 'message_start', message: { role: 'assistant' } })
  on({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', contentIndex: 0, delta: 'Part one.' } })
  await settle()
  const firstItem = $('#chat-list .thread-item')
  on({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', contentIndex: 0, delta: ' Part two.' } })
  on({ type: 'message_update', assistantMessageEvent: { type: 'toolcall_delta', contentIndex: 1, delta: '{"command":"ls"}' } })
  await settle()
  check('a streamed delta keeps the earlier messages as the same elements', firstItem?.textContent.includes('An earlier answer') && firstItem.isConnected && $('#chat-list .thread-item') === firstItem, firstItem?.textContent)
  check('a tool call being written never shows its JSON', !$('#chat-list').textContent.includes('"command"'), $('#chat-list').textContent.slice(-120))
  on({ type: 'agent_settled', aborted: false })

  // 8c. while Aria thinks, the working line is not redrawn per streamed token and the reader can scroll up
  await reset()
  const wasChat = document.body.classList.contains('mode-chat')
  document.body.classList.add('mode-chat') // the thread needs its real height to scroll
  for (let i = 0; i < 30; i++) on({ type: 'message_end', message: { role: 'assistant', content: [{ type: 'text', text: 'Line ' + i + ' of an earlier answer.' }] } })
  on({ type: 'agent_start' })
  on({ type: 'message_start', message: { role: 'assistant', content: [] } })
  on({ type: 'message_update', assistantMessageEvent: { type: 'thinking_start', contentIndex: 0 } })
  await settle(); await wait(40)
  const grid = $('#chat-list .p-working-grid')
  const list = $('#chat-list')
  list.scrollTop = list.scrollHeight
  await wait(40)
  list.scrollTop -= 30 // the first frames of a wheel or trackpad scroll move only a little
  await wait(40)
  const scrolled = list.scrollTop
  for (let i = 0; i < 5; i++) {
    on({ type: 'message_update', assistantMessageEvent: { type: 'thinking_delta', contentIndex: 0, delta: ' token' + i } })
    await wait(250) // longer than a tenth of a second: the elapsed time has changed
  }
  check('a thinking token keeps the working line (its animation does not restart)', grid && grid.isConnected && $('#chat-list .p-working-grid') === grid)
  check('a thinking token does not pull a reader who scrolled up back to the bottom', scrolled > 0 && Math.abs(list.scrollTop - scrolled) < 2, scrolled + ' -> ' + list.scrollTop + ' of ' + list.scrollHeight + '/' + list.clientHeight)
  list.scrollTop = list.scrollHeight // back at the end: it follows again
  await wait(40)
  for (let i = 0; i < 30; i++) on({ type: 'message_update', assistantMessageEvent: { type: 'text_delta', contentIndex: 1, delta: ' more words of the answer that make the thread grow' } })
  await wait(80)
  check('a reader back at the end follows the answer again', list.scrollHeight - list.scrollTop - list.clientHeight < 2, list.scrollHeight - list.scrollTop - list.clientHeight)
  const shown = parseFloat($('#chat-list .p-working-time')?.textContent ?? '0')
  check('the working line still counts the time', shown >= 1, $('#chat-list .p-working-time')?.textContent)
  on({ type: 'agent_settled', aborted: false })
  document.body.classList.toggle('mode-chat', wasChat)

  // 8e. attachments: the paperclip, the chips, sending them, a drop, a paste, the limits, a queued and a refused message
  {
    // Earlier checks leave another tab active: everything here goes to the tab on screen.
    const here = $('#chat-header [aria-selected="true"]')?.dataset.session ?? 's1'
    const onHere = (event) => sess({ type: 'chatEvent', event }, here)
    const resetHere = async () => { window.sim.stopRun(here); window.sim.queue = []; window.sim.steering = []; onHere({ type: 'agent_settled', aborted: false }); onHere({ type: 'session_reset' }); typeIn(''); await wait(60) }
    const wasChat = document.body.classList.contains('mode-chat')
    document.body.classList.add('mode-chat') // focus needs the chat on screen
    await resetHere()
    const filesIn = (...files) => { const dt = new DataTransfer(); for (const f of files) dt.items.add(f); return dt }
    const chips = () => [...document.querySelectorAll('#chat-composer .p-composer-files .p-pill-label')].map((e) => e.textContent)
    const fileInput = $('#chat-file')
    let chooser = 0
    const stopChooser = (e) => { chooser++; e.preventDefault() }
    fileInput.addEventListener('click', stopChooser)
    $('#chat-composer [data-chat="attach"]').click()
    fileInput.removeEventListener('click', stopChooser)
    check('the paperclip opens the file chooser', chooser === 1, chooser)
    const typedBefore = 'look at this'
    typeIn(typedBefore)
    fileInput.files = filesIn(new File(['hello world'], 'notes.txt', { type: 'text/plain' }), new File([new Uint8Array(2048)], 'shot.png', { type: 'image/png' })).files
    fileInput.dispatchEvent(new Event('change', { bubbles: true }))
    await wait(80)
    check('attach adds a chip', chips().join() === 'notes.txt,shot.png' && $('#chat-input').value === typedBefore, chips().join() + ' | ' + $('#chat-input').value)
    check('the chips show the sizes', $('#chat-composer .p-composer-files')?.textContent.includes('2 KB'), $('#chat-composer .p-composer-files')?.textContent)
    const focusable = [...document.querySelectorAll('#chat-composer .p-composer-files button, #chat-composer .p-composer-files [tabindex]')]
    check('in the composer only the chips’ remove buttons take the focus', focusable.length === 2 && focusable.every((b) => b.matches('.p-attach-remove')), focusable.map((b) => b.className).join())
    $('#chat-composer [data-chat="attach-remove"]').click(); await wait(40)
    check('remove drops it', chips().join() === 'shot.png' && document.activeElement === $('#chat-input'), chips().join() + ' ' + document.activeElement?.outerHTML.slice(0, 80))
    const before = sent().length
    $('#chat-send').click(); await wait(80)
    const out = sent().slice(before).filter((m) => m.type === 'chatSend')
    const a = out[0]?.attachments ?? []
    check('send carries the attachments', out.length === 1 && out[0].text === typedBefore && a.length === 1 && a[0].name === 'shot.png' && a[0].mime === 'image/png' && atob(a[0].data).length === 2048, JSON.stringify(out.map((m) => ({ ...m, attachments: (m.attachments ?? []).map((x) => x.name) }))))
    const bubbles = [...document.querySelectorAll('#chat-list .p-user')]
    check('a sent message’s chips are not buttons', !document.querySelector('#chat-list .p-user button.p-pill') && !!document.querySelector('#chat-list .p-user span.p-pill.is-static'), document.querySelector('#chat-list .p-user')?.innerHTML.slice(0, 160))
    check('the sent bubble shows the chip once, without the raw lines', bubbles.length === 1 && bubbles[0].textContent.includes('shot.png') && !$('#chat-list').textContent.includes('[attached'), bubbles.length + ' ' + $('#chat-list').textContent.slice(0, 160))
    check('the composer is empty after sending', !$('#chat-composer .p-composer-files'), chips().join())
    await resetHere()

    // a drop on the chat view
    const view = $('#view-chat')
    const drag = filesIn(new File(['a,b'], 'data.csv', { type: 'text/csv' }))
    $('#chat-list').dispatchEvent(new DragEvent('dragenter', { dataTransfer: drag, bubbles: true, cancelable: true }))
    check('dragging files over the chat outlines it', view.classList.contains('is-dropping'))
    $('#chat-list').dispatchEvent(new DragEvent('drop', { dataTransfer: drag, bubbles: true, cancelable: true }))
    await wait(80)
    check('a dropped file is attached', chips().join() === 'data.csv' && !view.classList.contains('is-dropping'), chips().join())

    // a drag coming from the header into the list keeps the outline (the header's dragleave comes after the list's dragenter)
    const header = $('#chat-header')
    header.dispatchEvent(new DragEvent('dragenter', { dataTransfer: drag, bubbles: true, cancelable: true }))
    check('dragging files over the header does not outline the chat', !view.classList.contains('is-dropping'))
    $('#chat-list').dispatchEvent(new DragEvent('dragenter', { dataTransfer: drag, bubbles: true, cancelable: true, relatedTarget: header }))
    header.dispatchEvent(new DragEvent('dragleave', { dataTransfer: drag, bubbles: true, cancelable: true, relatedTarget: $('#chat-list') }))
    check('a drag from the header into the list keeps the outline', view.classList.contains('is-dropping'))
    $('#chat-list').dispatchEvent(new DragEvent('dragleave', { dataTransfer: drag, bubbles: true, cancelable: true, relatedTarget: header }))
    check('leaving the view drops the outline', !view.classList.contains('is-dropping'))
    // a file dropped outside the chat view does not open in the dock, and is not attached
    const overHeader = new DragEvent('dragover', { dataTransfer: drag, bubbles: true, cancelable: true })
    header.dispatchEvent(overHeader)
    const onHeader = new DragEvent('drop', { dataTransfer: filesIn(new File(['h'], 'header.txt', { type: 'text/plain' })), bubbles: true, cancelable: true })
    header.dispatchEvent(onHeader); await wait(60)
    check('a file dropped on the header is stopped and not attached', overHeader.defaultPrevented && onHeader.defaultPrevented && chips().join() === 'data.csv', chips().join())

    // a pasted image is attached; a pasted text is the browser's own paste
    const pasted = new ClipboardEvent('paste', { clipboardData: filesIn(new File([new Uint8Array(10)], 'image.png', { type: 'image/png' })), bubbles: true, cancelable: true })
    $('#chat-input').dispatchEvent(pasted); await wait(80)
    check('a pasted image is attached', chips().join() === 'data.csv,image.png' && pasted.defaultPrevented, chips().join())
    const textData = new DataTransfer(); textData.setData('text/plain', 'words')
    const pastedText = new ClipboardEvent('paste', { clipboardData: textData, bubbles: true, cancelable: true })
    $('#chat-input').dispatchEvent(pastedText)
    check('a pasted text is left to the browser', !pastedText.defaultPrevented)

    // the limits: too many files are refused with a note before they are read
    fileInput.files = filesIn(...Array.from({ length: 9 }, (_, i) => new File(['x'], 'f' + i + '.txt', { type: 'text/plain' }))).files
    fileInput.dispatchEvent(new Event('change', { bubbles: true })); await wait(80)
    check('more than ten files are refused with a note', chips().length === 2 && /10/.test([...document.querySelectorAll('#chat-list .notice')].at(-1)?.textContent ?? ''), chips().length + ' ' + $('#chat-list').textContent.slice(-120))
    // (one file over 20 MB is left out alone: attachments.test.js, without building such a file here)
    const bigDrop = filesIn(new File(['y'], 'ok.txt', { type: 'text/plain' }))
    $('#chat-list').dispatchEvent(new DragEvent('drop', { dataTransfer: bigDrop, bubbles: true, cancelable: true })); await wait(80)
    check('a file within the limits still joins', chips().join() === 'data.csv,image.png,ok.txt', chips().join())

    // a queued message keeps its files (Pi's queue holds the lines), and so does a cancel that re-queues the rest
    window.sim.queue = []; window.sim.steering = []
    onHere({ type: 'agent_start' }); await settle()
    typeIn('queued one')
    key($('#chat-input'), 'Enter', { altKey: true }); await wait(80)
    const queued = $('#chat-list .p-user.is-queued')
    check('a queued message keeps its attachments as chips', queued?.textContent.includes('data.csv') && queued.textContent.includes('ok.txt') && !queued.textContent.includes('[attached'), queued?.textContent)
    typeIn('second')
    key($('#chat-input'), 'Enter', { altKey: true }); await wait(80)
    $('#chat-list .p-user.is-queued [data-chat="cancel-queued"][data-id="1"]').click(); await wait(80)
    const clear = sent().filter((m) => m.type === 'chatClearQueue').at(-1)
    check('cancelling another queued message sends this one back with its lines', clear?.keep?.length === 1 && clear.keep[0].startsWith('queued one\\n\\n[attached: data.csv'), JSON.stringify(clear))
    check('after the cancel the queued message still shows its chips', $('#chat-list .p-user.is-queued')?.textContent.includes('data.csv'), $('#chat-list .p-user.is-queued')?.textContent)
    // cancelled before Pi reported the queue: the message without its lines yet goes again with its files
    await wait(40)
    fileInput.files = filesIn(new File(['e'], 'early.txt', { type: 'text/plain' })).files
    fileInput.dispatchEvent(new Event('change', { bubbles: true })); await wait(80)
    typeIn('early one')
    const mark = sent().length
    key($('#chat-input'), 'Enter', { altKey: true })
    $('#chat-list .p-user.is-queued [data-chat="cancel-queued"][data-id="0"]').click() // before the stub's queue_update
    const early = sent().slice(mark)
    const resent = early.filter((m) => m.type === 'chatSend' && m.text === 'early one')
    const cleared = early.find((m) => m.type === 'chatClearQueue')
    check('a queued message cancelled before Pi reported it keeps its files', resent.length === 2 && resent[1].attachments?.[0]?.name === 'early.txt' && cleared && !cleared.keep.includes('early one'), JSON.stringify(early.map((m) => [m.type, m.text ?? m.keep, (m.attachments ?? []).length])))
    await wait(80) // the stub's queue updates land before the reset
    await resetHere()

    // a refused message gives its text and its files back to the composer
    fileInput.files = filesIn(new File(['z'], 'back.txt', { type: 'text/plain' })).files
    fileInput.dispatchEvent(new Event('change', { bubbles: true })); await wait(80)
    typeIn('REFUSE')
    $('#chat-send')?.click(); await wait(120)
    check('a refused message gives its text and its files back', $('#chat-input').value === 'REFUSE' && chips().join() === 'back.txt' && !$('#chat-list .p-user'), $('#chat-input').value + ' | ' + chips().join())
    $('#chat-composer [data-chat="attach-remove"]')?.click(); await wait(40)

    // history replays the lines as chips
    onHere({ type: 'history', messages: [{ role: 'user', content: [{ type: 'text', text: 'Why?\\n\\n[attached: build.log (text/plain, 12 KB) at /x/1-build.log]\\n[attached image: screenshot.png at /x/2-screenshot.png]' }] }] }); await wait(80)
    const old = $('#chat-list .p-user')
    check('a message from history shows its files as chips', old?.textContent.includes('build.log') && old.textContent.includes('screenshot.png') && !$('#chat-list').textContent.includes('[attached'), old?.textContent)
    await resetHere()
    document.body.classList.toggle('mode-chat', wasChat)
  }

  // 8d. Cmd+Q asks first: one press shows the question and sends nothing, a second one inside two seconds quits. Off a
  // Mac the same keys are Super+Q, the compositor's: the page leaves them alone.
  window.__sent.length = 0
  const cmdQ = () => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'q', metaKey: true, bubbles: true, cancelable: true }))
  if (/Mac/.test(navigator.userAgent)) {
    cmdQ(); await wait(60)
    check('one Cmd+Q only asks', $('#quit-hint').classList.contains('is-on') && $('#quit-hint').textContent.includes('⌘Q') && !window.__sent.some((m) => m.type === 'quitRequest'), $('#quit-hint').textContent)
    cmdQ(); await wait(60)
    check('a second Cmd+Q quits', window.__sent.some((m) => m.type === 'quitRequest') && !$('#quit-hint').classList.contains('is-on'))
  } else {
    cmdQ(); await wait(60); cmdQ(); await wait(60)
    check('Super+Q off a Mac is left alone', !$('#quit-hint').classList.contains('is-on') && !window.__sent.some((m) => m.type === 'quitRequest'))
  }

  // 8e. a scheduled task's tab: made hidden by the dock, shown with its run (a clock, the time), and /schedules lists it
  await reset()
  window.aria.receive({ type: 'scheduleTab', id: 's-notion0000', title: 'Notion tasks', schedule: 's-notion0000' }); await settle()
  const tabIds = () => [...document.querySelectorAll('#chat-header [data-chat="select"]')].map((b) => b.dataset.session)
  check('a scheduled tab is not in the bar before it has news', !tabIds().includes('s-notion0000'), tabIds().join(','))
  const onTask = (event) => window.aria.receive({ type: 'session', session: 's-notion0000', message: { type: 'chatEvent', event } })
  onTask({ type: 'scheduled_run', text: 'Give me my Notion tasks', at: '07:30', late: false })
  onTask({ type: 'agent_start' })
  onTask({ type: 'message_end', message: { role: 'assistant', content: [{ type: 'text', text: '3 tasks for today' }] } })
  onTask({ type: 'agent_settled', aborted: false }); await wait(120)
  check('after its run the scheduled tab shows, with its clock', tabIds().includes('s-notion0000') && $('#chat-header .scheduled .p-tabs-clock'), $('#chat-header').innerHTML.slice(0, 200))
  $('#chat-header [data-session="s-notion0000"][data-chat="select"]')?.click(); await wait(120)
  check("its run is the person's message with the scheduled time", $('#chat-list .p-user.is-scheduled')?.textContent.includes('07:30'), $('#chat-list').textContent.slice(0, 160))
  $('#chat-header [data-session="s-notion0000"][data-chat="close-tab"]')?.click(); await wait(120)
  check('closing it only hides it', !tabIds().includes('s-notion0000') && !window.__sent.some((m) => m.type === 'sessionClose' && m.session === 's-notion0000'), tabIds().join(','))
  typeIn('/schedules'); await wait(120); key($('#chat-input'), 'Enter'); await wait(40); key($('#chat-input'), 'Enter'); await wait(200) // pick, then run
  check('/schedules lists the task', $('#view-schedules')?.textContent.includes('Notion tasks') && !$('#view-schedules').hidden, $('#view-schedules')?.textContent.slice(0, 120))
  $('[data-chat="schedule-toggle"]')?.click(); await wait(150)
  check('pausing it shows it paused, with a resume button', $('#view-schedules .p-sched.is-paused') && $('[data-chat="schedule-toggle"]')?.dataset.enabled === 'true')
  $('[data-chat="panel-back"]')?.click(); await wait(80)

  // 9. the settings: a control redraws the section through its components, keeping the scroll and the focus
  $('[data-chat="settings"]').click(); await wait(200)
  $('#flow [data-action="settings-open"][data-value="agents"]')?.click(); await wait(120)
  const content = $('#content')
  content.scrollTop = content.scrollHeight
  const kept = content.scrollTop
  const sw = $('#flow [data-action="expert-on"]')
  const was = sw?.getAttribute('aria-checked')
  sw?.click(); await wait(80)
  const after = $('#flow [data-action="expert-on"]')
  check('a setup switch flips through a redraw of its component', after && after !== sw && after.getAttribute('aria-checked') !== was, was + ' -> ' + after?.getAttribute('aria-checked'))
  check('the redraw keeps the scroll position', kept > 0 && Math.abs(content.scrollTop - kept) < 2, kept + ' vs ' + content.scrollTop)
  // a file dropped on the settings (the setup screens) never loads in the page's place
  {
    const dt = new DataTransfer(); dt.items.add(new File(['x'], 'photo.png', { type: 'image/png' }))
    const over = new DragEvent('dragover', { dataTransfer: dt, bubbles: true, cancelable: true })
    const drop = new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true })
    content.dispatchEvent(over); content.dispatchEvent(drop)
    check('a file dropped on the setup screens is stopped', over.defaultPrevented && drop.defaultPrevented && over.dataTransfer.dropEffect === 'none', [over.defaultPrevented, drop.defaultPrevented, over.dataTransfer.dropEffect].join())
    const text = new DataTransfer(); text.setData('text/plain', 'words')
    const plain = new DragEvent('dragover', { dataTransfer: text, bubbles: true, cancelable: true })
    content.dispatchEvent(plain)
    check('a text drag on the setup screens is left to the browser', !plain.defaultPrevented)
  }
  check('the redraw gives the focus back to the switch that was clicked', document.activeElement === after, document.activeElement?.outerHTML?.slice(0, 80))

  return results
  function sim_ask() { window.sim.sess('s1', { type: 'chatDialog', id: 'ask1', method: 'input', request: { title: JSON.stringify({ aria: 1, kind: 'question', questions: [{ question: 'Which fallback?', header: 'Fallback', options: [{ label: 'Git hash' }, { label: 'Version' }] }, { question: 'Which checks?', header: 'Checks', options: [{ label: 'Unit' }, { label: 'Lint' }] }] }) }, hidden: false }) }
})()`
const reply = await send('Runtime.evaluate', { expression: script, awaitPromise: true, returnByValue: true })

// Real key presses (the browser's own button activation, which page events cannot reach): Enter starts recording the
// shortcut, Alt + Space records it without its release clicking the row again, Esc cancels without closing the dock.
const evaluate = async (expression: string) => (await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }))?.result?.value
const KEYS: Record<string, [string, number]> = { Enter: ['Enter', 13], Escape: ['Escape', 27], ' ': ['Space', 32], Alt: ['AltLeft', 18] }
const realKey = async (type: 'keyDown' | 'keyUp', key: string, modifiers = 0) => {
  const [code, vk] = KEYS[key]
  const text = type === 'keyDown' && (key === 'Enter' ? '\r' : key === ' ' ? ' ' : '')
  await send('Input.dispatchKeyEvent', { type: text ? 'keyDown' : type === 'keyDown' ? 'rawKeyDown' : 'keyUp', key, code, modifiers, windowsVirtualKeyCode: vk, ...(text ? { text, unmodifiedText: text } : {}) })
}
const tap = async (key: string, modifiers = 0) => { await realKey('keyDown', key, modifiers); await realKey('keyUp', key, modifiers); await Bun.sleep(60) }
const row = () => evaluate(`(() => { const b = document.querySelector('#flow [data-action="key-capture"]'); return b && { recording: b.classList.contains('recording'), keys: [...b.querySelectorAll('kbd')].map((k) => k.textContent).join(' + '), open: document.body.classList.contains('open') } })()`)
await evaluate(`(async () => { const w = (ms) => new Promise((r) => setTimeout(r, ms)); document.querySelector('#flow [data-action="settings-home"]')?.click(); await w(80); document.querySelector('#flow [data-action="settings-open"][data-value="assistant"]')?.click(); await w(120); document.querySelector('#flow [data-action="key-capture"]')?.focus() })()`)
const realChecks: [string, boolean, string][] = []
await tap('Enter')
const started = await row()
realChecks.push(['real keys: Enter on the row starts recording', !!started?.recording, JSON.stringify(started)])
await realKey('keyDown', 'Alt', 1); await realKey('keyDown', ' ', 1); await realKey('keyUp', ' ', 1); await realKey('keyUp', 'Alt'); await Bun.sleep(80)
const recorded = await row()
realChecks.push(['real keys: Alt + Space is recorded, and its release does not start recording again', recorded?.keys === 'Alt + Space' && !recorded.recording, JSON.stringify(recorded)])
await tap('Enter'); await tap('Escape')
const cancelled = await row()
realChecks.push(['real keys: Esc cancels the recording and leaves the dock open', !!cancelled && !cancelled.recording && cancelled.open && cancelled.keys === 'Alt + Space', JSON.stringify(cancelled)])
chrome.kill()
if (!Array.isArray(reply?.result?.value)) {
  console.error('the checks did not run:', JSON.stringify(reply.exceptionDetails ?? reply).slice(0, 600))
  process.exit(1)
}
let failed = 0
for (const [name, ok, detail] of [...(reply.result.value as [string, boolean, string][]), ...realChecks]) {
  console.log(`${ok ? 'ok:  ' : 'FAIL:'} ${name}${ok || !detail ? '' : `  (${detail})`}`)
  if (!ok) failed++
}
console.log(failed ? `${failed} check(s) failed` : 'all checks passed')
process.exit(failed ? 1 : 0)
