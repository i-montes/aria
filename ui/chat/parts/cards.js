// Structured cards from Pi's extension dialogs. A dialog only carries a string `title`, so an
// extension that wants a designed card puts a JSON object in it:
//
//   {"aria":1,"kind":"sandbox"|"quota"|"question"|"write"|"connector"|"secret"|"plan", ...fields}
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
//             detail:  string ("")   the tool and path, in the code box
//   connector summary: string ("")   "Notion wants to create pages"; the card's default title when empty
//             detail:  string ("")   the call's arguments, a few `key: value` lines
//   secret    name: string ("")   the name it is kept under   purpose: string ("")   what it is for
//             env:  string ("")   the variable commands read it from; the page answers `saved:<name>` (secret.js)
//   plan      title:    string ("")   the plan's name        path:    string ("")   where the file lives
//             summary:  string ("")   one line for the terminal's card   detail: string ("")   the terminal's code box
//             sections: [{ heading, text?, items? }] ([])   the document card's sections (document.js)
//             phases:   [{ name, tasks: string[] }] ([])    the phases card (phases.js)
//             The page draws the plan as those two cards, in the thread, with its options as buttons (`approve`, `change`);
//             answering leaves no note: the approved plan comes back as the `plan_ready` result (state.js).
//   sandbox, write, connector, quota and plan may carry `options: [{ value, label, primary? }]`: the strings the extension's
//   `ctx.ui.select` accepts back; the card shows one button per option and answers with its `value`.
// Extra fields are kept untouched. The dialog's id and its answer channel stay Pi's: the card is
// shown while the dialog exists and answered through it.
//
// parseCardTitle(title) -> { kind, ...fields } | null. It never throws: non-strings, non-JSON,
// non-objects, a wrong marker and unknown kinds all give null.
const KINDS = new Set(['sandbox', 'quota', 'question', 'write', 'connector', 'secret', 'plan'])

const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value)
const str = (value) => (typeof value === 'string' ? value : '')

/** The options the extension accepts back (`ctx.ui.select`'s strings): `{ value, label, primary? }`, bad entries dropped. */
const options = (list) =>
  Array.isArray(list)
    ? list
        .filter((o) => isObject(o) && typeof o.value === 'string' && typeof o.label === 'string')
        .map((o) => ({ value: o.value, label: o.label, ...(o.primary === true ? { primary: true } : {}), ...(o.role === 'free' || o.role === 'keep' ? { role: o.role } : {}) }))
    : []

const FIELDS = {
  sandbox: (c) => ({ summary: str(c.summary), detail: str(c.detail), jev: typeof c.jev === 'string' || isObject(c.jev) ? c.jev : '', options: options(c.options) }),
  quota: (c) => {
    const percent = Math.round(Number(c.percent))
    const seconds = typeof c.countdown === 'number' ? Math.floor(c.countdown) : NaN
    return {
      percent: Number.isFinite(percent) ? Math.max(0, Math.min(100, percent)) : 0,
      freeModel: str(c.freeModel),
      countdown: Number.isFinite(seconds) && seconds >= 0 ? seconds : null,
      options: options(c.options),
    }
  },
  question: (c) => ({ questions: Array.isArray(c.questions) ? c.questions.filter(isObject) : [] }),
  connector: (c) => ({ summary: str(c.summary), detail: str(c.detail), options: options(c.options) }),
  write: (c) => ({ path: str(c.path), summary: str(c.summary), detail: str(c.detail), options: options(c.options) }),
  plan: (c) => ({
    title: str(c.title), path: str(c.path), summary: str(c.summary), detail: str(c.detail),
    sections: Array.isArray(c.sections) ? c.sections.filter(isObject) : [],
    phases: Array.isArray(c.phases) ? c.phases.filter(isObject) : [],
    options: options(c.options),
  }),
  secret: (c) => ({ name: str(c.name), purpose: str(c.purpose), env: str(c.env) }),
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
