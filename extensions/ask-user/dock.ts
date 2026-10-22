// ask_user in Aria's dock: the chat's question card through a dialog, whose answers come back as JSON. Kept apart from
// index.ts (which needs the terminal's UI library) so it can be tested on its own.
import { ariaConfig } from '../lib/config.ts'
import { type Answer, CANCELLED_TEXT, formatAnswers, parseCardAnswers, type Question, questionCard } from './questions.ts'

export interface Details {
  questions: Question[]
  answers: Answer[]
  cancelled: boolean
}

const NOBODY_TEXT =
  'Nobody can answer here: this is a scheduled task running by itself. Choose what seems best, and say what you chose and why in your answer.'

const result = (text: string, details: Details) => ({ content: [{ type: 'text' as const, text }], details })

/** Asks and waits; Esc (the run's abort signal) makes Pi drop the dialog, and the tool ends as cancelled. */
export async function askInDock(ctx: any, questions: Question[], signal?: AbortSignal) {
  // A scheduled task's tab runs unattended: nobody would answer, and the run would wait forever.
  if (ariaConfig().noAsk) {
    return result(NOBODY_TEXT, { questions, answers: [], cancelled: true })
  }
  const reply = await ctx.ui.input(questionCard(questions), undefined, { signal })
  const answers = parseCardAnswers(reply, questions)
  if (!answers || answers.length === 0) return result(CANCELLED_TEXT, { questions, answers: [], cancelled: true })
  return result(formatAnswers(questions, answers), { questions, answers, cancelled: false })
}
