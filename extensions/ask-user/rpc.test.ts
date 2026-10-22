// Run: bun test extensions
// The tool's path in Aria's dock with a fake UI: what it asks, what it does with the answer, and Esc.
import { describe, expect, test } from 'bun:test'
import { askInDock } from './dock.ts'
import { normalize } from './questions.ts'

const tool = () => ({ execute: (_id: string, p: any, signal: any, _u: any, ctx: any) => askInDock(ctx, normalize(p), signal) })

const params = { questions: [{ header: 'Indent', question: 'Tabs or spaces?', options: [{ label: 'Spaces', recommended: true }, { label: 'Tabs' }] }] }

describe('ask_user in the dock', () => {
  test('asks with the question card, passing the abort signal, and reports the choice', async () => {
    const seen: any[] = []
    const signal = new AbortController().signal
    const ctx = { mode: 'rpc', hasUI: true, ui: { input: async (title: string, _p: unknown, opts: any) => { seen.push({ title, opts }); return JSON.stringify([{ question: 'Tabs or spaces?', answers: ['Tabs'] }]) } } }
    const out = await tool().execute('1', params, signal, () => {}, ctx)
    expect(JSON.parse(seen[0].title).kind).toBe('question')
    expect(seen[0].opts.signal).toBe(signal)
    expect(out.content[0].text).toContain('The user chose: Tabs')
  })

  test('Esc (the signal aborts and Pi drops the dialog) ends the tool as cancelled', async () => {
    const controller = new AbortController()
    const ctx = { mode: 'rpc', hasUI: true, ui: { input: (_t: string, _p: unknown, opts: any) => new Promise((resolve) => opts.signal.addEventListener('abort', () => resolve(undefined))) } }
    const running = tool().execute('1', params, controller.signal, () => {}, ctx)
    controller.abort()
    const out = await running
    expect(out.details.cancelled).toBe(true)
  })

  test('a garbage answer is "no answer", never a made-up choice', async () => {
    const ctx = { mode: 'rpc', hasUI: true, ui: { input: async () => 'Tabs' } }
    const out = await tool().execute('1', params, undefined, () => {}, ctx)
    expect(out.details.cancelled).toBe(true)
  })

  test('"talk about this" tells the model to talk, not that the user wrote something', async () => {
    const ctx = { mode: 'rpc', hasUI: true, ui: { input: async () => JSON.stringify([{ question: 'Tabs or spaces?', answers: [], talk: true }]) } }
    const out = await tool().execute('1', params, undefined, () => {}, ctx)
    expect(out.content[0].text).toContain('wants to talk about this')
    expect(out.content[0].text).not.toContain('The user wrote')
  })
})

describe('ask_user in a scheduled tab', () => {
  test('asks nobody and tells the model to decide', async () => {
    const saved = process.env.ARIA_GRANTS
    process.env.ARIA_GRANTS = JSON.stringify({ noAsk: true })
    let asked = false
    const ctx = { mode: 'rpc', hasUI: true, ui: { input: async () => { asked = true; return '[]' } } }
    const out = await tool().execute('1', params, new AbortController().signal, () => {}, ctx)
    if (saved === undefined) delete process.env.ARIA_GRANTS
    else process.env.ARIA_GRANTS = saved
    expect(asked).toBe(false)
    expect(out.content[0].text).toContain('scheduled')
  })
})
