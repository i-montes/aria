// Composes the README scene in the UI sandbox page (fake content only); regenerate.sh runs it through agent-browser eval.
(async () => {
  const wait = (ms) => new Promise((r) => setTimeout(r, ms))
  const chat = window.aria.chat
  const sim = window.sim
  document.querySelectorAll('nav').forEach((n) => n.remove())
  chat.setSessions({ sessions: [{ id: 's1', title: 'Checkout widget' }, { id: 's2', title: 'Weekly notes' }], active: 's1' })
  await wait(200)
  const WD = '/home/ana/Projects/checkout-widget'
  sim.state({ workdir: WD }, 's1')
  const ev = (e) => sim.ev(e, 's1')
  const user = (text) => ev({ type: 'message_start', message: { role: 'user', content: [{ type: 'text', text }] } })
  const tool = async (id, toolName, args, text) => {
    ev({ type: 'tool_execution_start', toolCallId: id, toolName, args })
    await wait(80)
    ev({ type: 'tool_execution_end', toolCallId: id, toolName, isError: false, result: { content: [{ type: 'text', text }] } })
  }
  user('Add input validation to the checkout form and run the tests.')
  ev({ type: 'agent_start' })
  await tool('t1', 'read', { path: 'src/checkout/form.ts' }, 'export function submit(form) { return api.post("/orders", form) }')
  await tool('t2', 'edit', { path: 'src/checkout/form.ts', edits: [{ oldText: 'export function submit(form) {\n  return api.post("/orders", form)\n}', newText: 'export function submit(form) {\n  const errors = validate(form)\n  if (errors.length) return { ok: false, errors }\n  return api.post("/orders", form)\n}' }] }, 'edited')
  await tool('t3', 'edit', { path: 'src/checkout/validate.ts', edits: [{ oldText: '', newText: 'export const validate = (f) => {\n  const errors = []\n  if (!/^\\S+@\\S+$/.test(f.email)) errors.push("email")\n  if (f.quantity < 1) errors.push("quantity")\n  return errors\n}\n' }] }, 'edited')
  await tool('t4', 'bash', { command: 'bun test src/checkout' }, '14 pass\n0 fail')
  sim.say('Done. `submit` now validates the form before it calls the API; the new `validate` helper checks the email and the quantity.\n\n`bun test src/checkout`: **14 passed, 0 failed**.', 's1')
  ev({ type: 'agent_settled', aborted: false })
  await wait(400)
  chat.setTasks([{ id: '1', label: 'Add the validate helper', status: 'done' }, { id: '2', label: 'Call it from submit', status: 'done' }, { id: '3', label: 'Run the checkout tests', status: 'done' }])
  await wait(300)
  document.querySelector('[data-chat=toggle-tools]')?.click()
  document.activeElement?.blur?.()
  await wait(700)
  return 'ok'
})()
