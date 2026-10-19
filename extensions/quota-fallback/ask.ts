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

export async function confirmFreeSwitch(ctx: any, spent: Spent, fromId: string): Promise<Choice> {
  const key = decisionKey(spent)
  const saved = readDecision(key)
  if (saved) return saved
  if (ctx?.mode !== 'tui' || !ctx.hasUI) return 'switch'
  const SWITCH = `Switch to ${FREE_MODEL.id} now`
  const STAY = `Stay on ${fromId}`
  const title = [
    `OpenCode Go ${spent.window} quota at ${spent.percent}%.`,
    `Switch to ${FREE_MODEL.id}? It is free, but the provider keeps prompts for 30 days.`,
    'With no answer it switches when the countdown ends and the task goes on.',
  ].join('\n')
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
