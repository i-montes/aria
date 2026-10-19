// Structured cards from Pi's extension dialogs. A dialog only carries a string `title`, so an
// extension that wants a designed card puts a JSON object in it:
//
//   {"aria":1,"kind":"sandbox"|"quota"|"question"|"write", ...fields}
//
// `aria` must be the number 1 (a version/marker); anything else is not ours. Fields per kind
// (the contract the extensions follow; missing or mistyped fields get the defaults shown):
//   sandbox   summary: string ("")   heading; the card's default title when empty
//             detail:  string ("")   the command to run outside the sandbox
//             jev:     string | { review?: string, text?: string } ("")   Jev's review
//   quota     percent:   number 0-100 (0)   quota used
//             freeModel: string ("")        the free model to switch to
//             countdown: number >= 0 | null (null)   seconds before the dialog times out
//   question  questions: [{ question, header?, options: [{ label, description?, recommended? }],
//                           multi? }]  (non-object entries are dropped; see question.js)
//   write     path:    string ("")   the file to be changed
//             summary: string ("")   what changes
// Extra fields are kept untouched. The dialog's id and its answer channel stay Pi's: the card is
// shown while the dialog exists and answered through it.
//
// parseCardTitle(title) -> { kind, ...fields } | null. It never throws: non-strings, non-JSON,
// non-objects, a wrong marker and unknown kinds all give null.
const KINDS = new Set(['sandbox', 'quota', 'question', 'write'])

const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value)
const str = (value) => (typeof value === 'string' ? value : '')

const FIELDS = {
  sandbox: (c) => ({ summary: str(c.summary), detail: str(c.detail), jev: typeof c.jev === 'string' || isObject(c.jev) ? c.jev : '' }),
  quota: (c) => {
    const percent = Math.round(Number(c.percent))
    const seconds = typeof c.countdown === 'number' ? Math.floor(c.countdown) : NaN
    return {
      percent: Number.isFinite(percent) ? Math.max(0, Math.min(100, percent)) : 0,
      freeModel: str(c.freeModel),
      countdown: Number.isFinite(seconds) && seconds >= 0 ? seconds : null,
    }
  },
  question: (c) => ({ questions: Array.isArray(c.questions) ? c.questions.filter(isObject) : [] }),
  write: (c) => ({ path: str(c.path), summary: str(c.summary) }),
}

export function parseCardTitle(title) {
  if (typeof title !== 'string') return null
  const text = title.trim()
  if (!text.startsWith('{')) return null
  let card
  try {
    card = JSON.parse(text)
  } catch {
    return null
  }
  if (!isObject(card) || card.aria !== 1) return null
  const kind = card.kind
  if (typeof kind !== 'string' || !KINDS.has(kind)) return null
  const { aria, ...rest } = card
  return { ...rest, kind, ...FIELDS[kind](card) }
}
