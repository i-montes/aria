// The words `ask_user` shows the model. They name the conversation's language, never a fixed one: when Aria's config
// (`aria.json`, `assistant.language`) carries it, the language is named; otherwise the model is told to use the
// conversation's. No Pi imports here, so the tests can load it.
import { ariaConfig } from '../lib/config.ts'

/** "French, the conversation's language", or just "the conversation's language" when none is named. */
export const inLanguage = (language: string): string => (language.trim() ? `${language.trim()}, the conversation's language` : "the conversation's language")

export function toolTexts(language: string = ariaConfig().assistant.language) {
  const where = inLanguage(language)
  return {
    header: 'Very short tab label, at most 12 characters, e.g. "Scope", "Repo".',
    question: `The full question, in ${where}, ending with a question mark.`,
    description:
      'Ask the user 1-4 questions, each with 2-4 options; the user can always write their own answer instead. Use it only when you are blocked on a decision that is genuinely the user\'s: one you cannot settle from the request, the code or a sensible default (design choices while planning, scope, what to do with uncommitted changes you did not make, a process that is not yours). Do not use it to ask "should I continue?" or for facts you can check yourself. Ask everything you need to decide now in ONE call: put up to 4 questions that do not depend on each other\'s answers together, even when a skill says to ask one question at a time; only a question whose options depend on an earlier answer waits for the next call. Put the option you recommend first with `recommended: true`. ' +
      `Write questions and options in ${where}.`,
  }
}
