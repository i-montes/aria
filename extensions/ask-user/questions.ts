// The pure part of `ask_user`: normalising what the model sends, and the text
// the model gets back. The TUI lives in index.ts.
export const MAX_QUESTIONS = 4
export const MIN_OPTIONS = 2
export const MAX_OPTIONS = 4
export const OTHER_LABEL = 'Other (type your own)'
export const RECOMMENDED_SUFFIX = ' (recommended)'

export interface Option {
  label: string
  description?: string
}
export interface Question {
  id: string
  header: string
  question: string
  options: Option[]
}
export interface Answer {
  id: string
  label: string
  wasCustom: boolean
  /** 1-based position of the chosen option, when it was not written. */
  index?: number
}

type Raw = { header?: unknown; question?: unknown; options?: unknown }

/**
 * Clean up what the model sent: trim, drop empty or duplicate options and any "other" the model
 * added itself, put the recommended option first with a marker, cap the counts. Throws with a
 * message the model can act on when what is left cannot be asked.
 */
export function normalize(params: { questions?: unknown }): Question[] {
  const raw = Array.isArray(params?.questions) ? (params.questions as Raw[]) : []
  if (raw.length === 0) throw new Error('ask_user needs at least one question.')
  if (raw.length > MAX_QUESTIONS) throw new Error(`ask_user takes at most ${MAX_QUESTIONS} questions; ask the rest later.`)
  return raw.map((q, i) => {
    const question = String(q?.question ?? '').trim()
    if (!question) throw new Error(`Question ${i + 1} has no text.`)
    const seen = new Set<string>()
    let recommended: Option | null = null
    const options: Option[] = []
    for (const o of Array.isArray(q?.options) ? (q.options as any[]) : []) {
      const label = String(o?.label ?? '').trim()
      const key = label.toLowerCase()
      if (!label || seen.has(key) || /^(otra|otro|other)\b/i.test(label)) continue
      seen.add(key)
      const option: Option = { label, ...(o?.description ? { description: String(o.description).trim() } : {}) }
      if (o?.recommended === true && !recommended) recommended = option
      else options.push(option)
    }
    if (recommended) options.unshift({ ...recommended, label: recommended.label + RECOMMENDED_SUFFIX })
    if (options.length < MIN_OPTIONS) throw new Error(`Question ${i + 1} needs at least ${MIN_OPTIONS} distinct options.`)
    const header = String(q?.header ?? '').trim().slice(0, 12) || `P${i + 1}`
    return { id: `q${i + 1}`, header, question, options: options.slice(0, MAX_OPTIONS) }
  })
}

/** What the model reads after the user answered. */
export function formatAnswers(questions: Question[], answers: Answer[]): string {
  return questions
    .map((q) => {
      const a = answers.find((x) => x.id === q.id)
      if (!a) return `${q.question}\n→ (no answer)`
      const label = a.label.endsWith(RECOMMENDED_SUFFIX) ? a.label.slice(0, -RECOMMENDED_SUFFIX.length) : a.label
      return a.wasCustom ? `${q.question}\n→ The user wrote: ${label}` : `${q.question}\n→ The user chose: ${label}`
    })
    .join('\n\n')
}

export const CANCELLED_TEXT =
  'The user closed the question without answering. Do not assume an answer: tell them in one sentence what you need to know and wait for their message.'

/** What the model reads when nobody can answer (a subagent, or `pi -p`). */
export function noUserText(mode: string): string {
  if (mode === 'json') {
    return 'There is no user in this session (you are a subagent). Do not ask: if the decision blocks the task, finish with NEEDS_CONTEXT and the question; otherwise go on with the recommended option and say so in your report.'
  }
  return 'This session is not interactive: nobody can answer. Go on with the recommended option and say in your answer what you decided and why.'
}
