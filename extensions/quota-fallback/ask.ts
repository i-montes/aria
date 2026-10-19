// The one question before work moves to the free model, shared by quota-fallback and jev-router's
// /plan: both spend the same OpenCode Go account, so one answer per quota window covers every
// switch (and every pi process, through DECISION_FILE). Without a TUI and without an answer yet it
// switches, as before.
import { ASK_TIMEOUT_MS, type Choice, choiceFrom, decisionKey, FREE_MODEL, readDecision, writeDecision } from './fallback.ts'

export interface Spent {
  window: string
  percent: number
  resetsAt?: string
}

/** The question and its choices: the chat's quota card in Aria's dock, plain text in the terminal, nothing without a UI. */
export function quotaQuestion(ctx: any, spent: Spent, fromId: string): { title: string; options: string[] } | null {
  if (!ctx?.hasUI || (ctx.mode !== 'tui' && ctx.mode !== 'rpc')) return null
  const SWITCH = `Switch to ${FREE_MODEL.id} now`
  const STAY = `Stay on ${fromId}`
  const options = [SWITCH, STAY]
  if (ctx.mode === 'rpc') {
    const title = JSON.stringify({
      aria: 1,
      kind: 'quota',
      percent: spent.percent,
      window: spent.window,
      freeModel: `${FREE_MODEL.provider}/${FREE_MODEL.id}`,
      countdown: ASK_TIMEOUT_MS / 1000,
      options: [
        { value: SWITCH, label: SWITCH, role: 'free', primary: true },
        { value: STAY, label: STAY, role: 'keep' },
      ],
    })
    return { title, options }
  }
  const title = [
    `OpenCode Go ${spent.window} quota at ${spent.percent}%.`,
    `Switch to ${FREE_MODEL.id}? It is free, but the provider keeps prompts for 30 days.`,
    'With no answer it switches when the countdown ends and the task goes on.',
  ].join('\n')
  return { title, options }
}

export async function confirmFreeSwitch(ctx: any, spent: Spent, fromId: string): Promise<Choice> {
  const key = decisionKey(spent)
  const saved = readDecision(key)
  if (saved) return saved
  const question = quotaQuestion(ctx, spent, fromId)
  if (!question) return 'switch'
  const [SWITCH, STAY] = question.options
  const title = question.title
  const started = Date.now()
  let pick: string | undefined
  try {
    pick = await ctx.ui.select(title, [SWITCH, STAY], { timeout: ASK_TIMEOUT_MS })
  } catch {
    pick = undefined
  }
  const choice = choiceFrom(pick, SWITCH, STAY, Date.now() - started)
  writeDecision(key, choice)
  return choice
}
