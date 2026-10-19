// Drives the chat's controller in headless Chrome against the sandbox's stubbed bridge and checks the
// behaviours the unit tests cannot see (focus, redraws, key order, races). Needs Chrome and the sandbox:
//
//   bun tools/ui-sandbox/serve.ts 8770 &      then      bun tools/ui-sandbox/check.ts
//
// Exit code 1 when any check fails.
const url = process.env.URL ?? 'http://127.0.0.1:8770/'
const port = 9338
const chrome = Bun.spawn(['google-chrome-stable', '--headless=new', '--disable-gpu', '--no-sandbox', `--user-data-dir=/tmp/aria-check-${process.pid}`, `--remote-debugging-port=${port}`, '--window-size=940,940', 'about:blank'], { stderr: 'ignore', stdout: 'ignore' })
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
  const reset = async () => { window.sim.queue = []; ev({ type: 'agent_settled', aborted: false }); ev({ type: 'session_reset' }); const i = $('#chat-input'); if (i) { i.value = ''; i.dispatchEvent(new Event('input', { bubbles: true })) }; await wait(50) }

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

  // 2. queue: Enter during a run queues, and delivery (Pi's order) moves it into the thread
  ev({ type: 'agent_start' }); await settle()
  typeIn('do this later'); key($('#chat-input'), 'Enter'); await wait(60)
  check('Enter during a run queues with followUp', sent().some((m) => m.type === 'chatSend' && m.behavior === 'followUp'))
  check('the queued bubble shows', $('.p-user.queued, .p-user-queued, [data-chat="cancel-queued"]'))
  ev({ type: 'queue_update', steering: [], followUp: [] }); ev({ type: 'message_start', message: { role: 'user', content: 'do this later' } }); await wait(80)
  check('a delivered message lands in the thread and leaves the queue', !$('[data-chat="cancel-queued"]') && $('#chat-list').textContent.includes('do this later'))
  await reset()

  // 3. cancelling one of two queued messages re-queues the other
  ev({ type: 'agent_start' }); await settle(); typeIn('first'); key($('#chat-input'), 'Enter'); await wait(60); typeIn('second'); key($('#chat-input'), 'Enter'); await wait(80)
  window.__sent.length = 0
  const cancelButton = $('[data-chat="cancel-queued"][data-id="0"]')
  check('each queued bubble has its own cancel', cancelButton, $('#chat-list').innerHTML.slice(-600))
  cancelButton?.click(); await wait(60)
  const cancel = sent().find((m) => m.type === 'chatClearQueue')
  check('cancelling the first keeps the second', cancel && JSON.stringify(cancel.keep) === '["second"]', JSON.stringify(cancel))
  await reset()

  // 4. the question card: the free-text field keeps focus, enables Continue, Enter submits, the answers card has its header
  sim_ask()
  await wait(80)
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
  await reset()

  // 5. Esc order
  typeIn('/'); await wait(120)
  check('the command picker opens', $('.p-cmd.is-open'))
  check('Esc closes the picker first', window.aria.chat.escape() === true && !$('.p-cmd.is-open'))
  ev({ type: 'agent_start' }); await settle(); window.__sent.length = 0
  check('Esc stops a run', window.aria.chat.escape() === true && sent().some((m) => m.type === 'chatAbort'))
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

  // 9. the setup (settings): a control redraws the step through its components, keeping the scroll and the focus
  $('[data-chat="settings"]').click(); await wait(200)
  $('#flow [data-action="edit"][data-value="agents"]')?.click(); await wait(120)
  const content = $('#content')
  content.scrollTop = content.scrollHeight
  const kept = content.scrollTop
  const sw = $('#flow [data-action="expert-on"]')
  const was = sw?.getAttribute('aria-checked')
  sw?.click(); await wait(80)
  const after = $('#flow [data-action="expert-on"]')
  check('a setup switch flips through a redraw of its component', after && after !== sw && after.getAttribute('aria-checked') !== was, was + ' -> ' + after?.getAttribute('aria-checked'))
  check('the redraw keeps the scroll position', kept > 0 && Math.abs(content.scrollTop - kept) < 2, kept + ' vs ' + content.scrollTop)
  check('the redraw gives the focus back to the switch that was clicked', document.activeElement === after, document.activeElement?.outerHTML?.slice(0, 80))

  return results
  function sim_ask() { window.sim.sess('s1', { type: 'chatDialog', id: 'ask1', method: 'input', request: { title: JSON.stringify({ aria: 1, kind: 'question', questions: [{ question: 'Which fallback?', header: 'Fallback', options: [{ label: 'Git hash' }, { label: 'Version' }] }, { question: 'Which checks?', header: 'Checks', options: [{ label: 'Unit' }, { label: 'Lint' }] }] }) }, hidden: false }) }
})()`
const reply = await send('Runtime.evaluate', { expression: script, awaitPromise: true, returnByValue: true })
chrome.kill()
if (!reply?.result?.value) {
  console.error('the checks did not run:', JSON.stringify(reply).slice(0, 600))
  process.exit(1)
}
let failed = 0
for (const [name, ok, detail] of reply.result.value as [string, boolean, string][]) {
  console.log(`${ok ? 'ok:  ' : 'FAIL:'} ${name}${ok || !detail ? '' : `  (${detail})`}`)
  if (!ok) failed++
}
console.log(failed ? `${failed} check(s) failed` : 'all checks passed')
process.exit(failed ? 1 : 0)
