// The question wizard (design/components/question-card.dc.html): 1-4 questions, one step each, with
// tabs, a recommended option first, an "other" free-text answer, multi-select where the question
// allows it, back/continue/send, skip, "talk about this" and dismiss. Fed by a dialog title
// `{"aria":1,"kind":"question",…}` (see cards.js).
//
// Props: { id, questions: [{ question, header?, options: [{ label, description?, recommended? }], multi? }] }
//   id         the extension dialog's id
//   questions  up to 4; `type: "check"` is read as `multi`; options are capped at 4 per question and the
//              recommended ones are moved first (stable). `normalizeQuestions` does this for everything
//              below, so indexes always refer to the normalized order.
// ui (the wizard state, from questionInit/questionReduce): { step, picked, other, talk }
//   step    index of the visible question
//   picked  number[][]: option indexes picked per question
//   other   string[]: the free-text answer per question
//   talk    boolean[]: "talk about this" chosen for the question
// Emits: data-chat="question" data-id data-op="pick|other|next|back|submit|skip|goto|talk|dismiss"
//   data-q (question index) and data-value (option index for pick; the text for other).
//   `other` comes from an <input>: listen to its `input` event and read the element's `value`
//   (data-value is only the value at render time). Everything else is a click.
//   pick/other/next/back/goto/skip/talk/dismiss/submit all go through questionReduce(state, op, {q, value});
//   after it, `state.outcome` is "submit" (send questionAnswers) or "dismiss" (cancel the dialog), else null.
//   The design advances by itself 480 ms after a single-choice pick; that timing is the integrator's
//   (dispatch `next`, or `submit` on the last question, once the pick settled).
//
// Answers sent back: [{ question: string, answers: string[], talk?: true }], only for answered
// questions; `answers` has the picked option labels in option order, then the "other" text.
// `talk: true` (with `answers: []`) means the user wants to discuss that question instead of answering.
import { esc } from '../../onboarding/components.js'

const MAX_QUESTIONS = 4
const MAX_OPTIONS = 4
const MAX_OTHER = 500

const tick = (size, width) => `<svg aria-hidden="true" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"></path></svg>`
const chat = (size, width) => `<svg aria-hidden="true" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round"><path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"></path></svg>`
const closeIcon = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18M6 6l12 12"></path></svg>'
const enter = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" class="p-q-enter"><path d="M20 4v7a4 4 0 0 1-4 4H4"></path><path d="M9 10l-5 5 5 5"></path></svg>'

const isObject = (value) => value !== null && typeof value === 'object'

export function normalizeQuestions(questions) {
  const list = Array.isArray(questions) ? questions.filter(isObject) : []
  return list.slice(0, MAX_QUESTIONS).map((q) => {
    const options = (Array.isArray(q.options) ? q.options : [])
      .filter((o) => typeof o === 'string' || isObject(o))
      .map((o) => (typeof o === 'string' ? { label: o } : o))
      .map((o) => ({ label: String(o.label ?? ''), description: String(o.description ?? ''), recommended: o.recommended === true }))
    const ordered = [...options.filter((o) => o.recommended), ...options.filter((o) => !o.recommended)]
    return {
      question: String(q.question ?? ''),
      header: String(q.header ?? ''),
      multi: q.multi === true || q.type === 'check',
      options: ordered.slice(0, MAX_OPTIONS),
    }
  })
}

export function questionInit(questions) {
  const list = normalizeQuestions(questions)
  return {
    step: 0,
    picked: list.map(() => []),
    other: list.map(() => ''),
    talk: list.map(() => false),
    multi: list.map((q) => q.multi),
    sizes: list.map((q) => q.options.length),
    outcome: null,
  }
}

const answered = (state, i) =>
  (state.picked?.[i]?.length ?? 0) > 0 || String(state.other?.[i] ?? '').trim() !== '' || state.talk?.[i] === true

const toIndex = (value) => {
  const n = typeof value === 'number' ? value : typeof value === 'string' && value.trim() !== '' ? Number(value) : NaN
  return Number.isInteger(n) ? n : -1
}

const put = (list, i, value) => list.map((x, k) => (k === i ? value : x))

/** Pure transition of the wizard. Returns the same object when nothing changes. See the header for the ops. */
export function questionReduce(state, op, payload = {}) {
  if (!state || !Array.isArray(state.picked)) return state
  const n = state.picked.length
  const step = Math.min(Math.max(state.step ?? 0, 0), Math.max(n - 1, 0))
  const q = payload?.q === undefined ? step : toIndex(payload.q)
  const validQ = q >= 0 && q < n
  switch (op) {
    case 'pick': {
      const v = toIndex(payload?.value)
      if (!validQ || v < 0 || v >= (state.sizes?.[q] ?? 0)) return state
      const current = state.picked[q]
      const next = state.multi?.[q] ? (current.includes(v) ? current.filter((x) => x !== v) : [...current, v].sort((a, b) => a - b)) : [v]
      return {
        ...state,
        picked: put(state.picked, q, next),
        other: state.multi?.[q] ? state.other : put(state.other, q, ''),
        talk: put(state.talk, q, false),
      }
    }
    case 'other': {
      if (!validQ) return state
      const text = String(payload?.value ?? '').slice(0, MAX_OTHER)
      const filled = text.trim() !== ''
      return {
        ...state,
        other: put(state.other, q, text),
        picked: filled && !state.multi?.[q] ? put(state.picked, q, []) : state.picked,
        talk: filled ? put(state.talk, q, false) : state.talk,
      }
    }
    case 'next':
      return step < n - 1 && answered(state, step) ? { ...state, step: step + 1 } : state
    case 'back':
      return step > 0 ? { ...state, step: step - 1 } : state
    case 'goto':
      return validQ && q !== state.step ? { ...state, step: q } : state
    case 'submit':
      return answered(state, step) ? { ...state, outcome: 'submit' } : state
    case 'skip':
      if (step < n - 1) return { ...state, step: step + 1 }
      return { ...state, outcome: state.picked.some((_, i) => answered(state, i)) ? 'submit' : 'dismiss' }
    case 'talk':
      if (n === 0) return state
      return { ...state, picked: put(state.picked, step, []), other: put(state.other, step, ''), talk: put(state.talk, step, true), outcome: 'submit' }
    case 'dismiss':
      return { ...state, outcome: 'dismiss' }
    default:
      return state
  }
}

/** The answers to send back: [{ question, answers: string[], talk?: true }], answered questions only. */
export function questionAnswers(questions, state) {
  const list = normalizeQuestions(questions)
  const out = []
  list.forEach((q, i) => {
    if (state?.talk?.[i] === true) {
      out.push({ question: q.question, answers: [], talk: true })
      return
    }
    const picks = [...new Set(state?.picked?.[i] ?? [])]
      .filter((k) => Number.isInteger(k) && k >= 0 && k < q.options.length)
      .sort((a, b) => a - b)
      .map((k) => q.options[k].label)
    const other = String(state?.other?.[i] ?? '').trim()
    const answers = other ? [...picks, other] : picks
    if (answers.length) out.push({ question: q.question, answers })
  })
  return out
}

export function renderQuestion({ id, questions } = {}, ui, t) {
  const list = normalizeQuestions(questions)
  const n = list.length
  const step = Math.min(Math.max(Number(ui?.step) || 0, 0), Math.max(n - 1, 0))
  const picked = (i) => (Array.isArray(ui?.picked?.[i]) ? ui.picked[i] : [])
  const otherText = (i) => String(ui?.other?.[i] ?? '')
  const talked = (i) => ui?.talk?.[i] === true
  const done = (i) => picked(i).length > 0 || otherText(i).trim() !== '' || talked(i)
  const q = list[step]
  const last = step >= n - 1
  const key = esc(id)
  const attrs = (op, qi) => `data-chat="question" data-id="${key}" data-op="${op}"${qi === undefined ? '' : ` data-q="${qi}"`}`
  const panelId = `p-q-panel-${key}`

  const tabs = list.map((item, i) => {
    const active = i === step
    return `<button type="button" role="tab" class="p-q-tab${active ? ' active' : ''}${talked(i) ? ' p-q-tab-talk' : done(i) ? ' p-q-tab-done' : ''}" id="p-q-tab-${key}-${i}" aria-selected="${active}"${active ? ` aria-controls="${panelId}"` : ''} ${attrs('goto', i)}>${talked(i) ? chat(11, 2.4) : done(i) ? tick(11, 3) : ''}<span>${esc(item.header || t('chat.question.header_default'))}</span></button>`
  }).join('')

  const options = q ? q.options.map((o, k) => {
    const on = picked(step).includes(k)
    return `<button type="button" class="p-q-opt${on ? ' selected' : ''}" aria-pressed="${on}" ${attrs('pick', step)} data-value="${k}">
        <span class="p-q-ind ${q.multi ? 'p-q-check' : 'p-q-radio'}${on ? ' on' : ''}" aria-hidden="true">${q.multi ? tick(12, 3) : '<span class="p-q-dot"></span>'}</span>
        <span class="p-q-optbody"><span class="p-q-optline"><span class="p-q-label">${esc(o.label)}</span>${o.recommended ? `<span class="p-q-rec">${esc(t('chat.question.recommended'))}</span>` : ''}</span>${o.description ? `<span class="p-q-desc">${esc(o.description)}</span>` : ''}</span>
      </button>`
  }).join('') : ''

  const otherOn = q ? otherText(step).trim() !== '' : false
  const other = q ? `<div class="p-q-opt p-q-other${otherOn ? ' selected' : ''}">
        <span class="p-q-ind ${q.multi ? 'p-q-check' : 'p-q-radio'}${otherOn ? ' on' : ''}" aria-hidden="true">${q.multi ? tick(12, 3) : '<span class="p-q-dot"></span>'}</span>
        <input type="text" class="p-q-other-input" ${attrs('other', step)} data-value="${esc(otherText(step))}" value="${esc(otherText(step))}" maxlength="${MAX_OTHER}" placeholder="${esc(t('chat.question.other'))}" aria-label="${esc(t('chat.question.other'))}" autocomplete="off" spellcheck="false">
      </div>` : ''

  return `<div class="p-q" role="group" aria-label="${esc(t('chat.question.label'))}">
  <div class="p-q-card">
    <button type="button" class="p-q-x" aria-label="${esc(t('chat.question.dismiss'))}" ${attrs('dismiss')}>${closeIcon}</button>
    <div class="p-q-top"><div class="p-q-tabs" role="tablist" aria-label="${esc(t('chat.question.tabs'))}">${tabs}</div></div>
    <div class="p-q-body" aria-live="polite">
      ${q ? `<div class="p-q-panel" role="tabpanel" id="${panelId}"${n ? ` aria-labelledby="p-q-tab-${key}-${step}"` : ''}>
        <div class="p-q-question">${esc(q.question)}</div>
        <div class="p-q-options">
      ${options}
      ${other}
        </div>
      </div>` : ''}
    </div>
    <div class="p-q-foot">
      <button type="button" class="p-q-ghost p-q-talk" ${attrs('talk')}>${chat(15, 2)}<span>${esc(t('chat.question.talk'))}</span></button>
      <div class="p-q-right">
        ${step > 0 ? `<button type="button" class="p-q-ghost" ${attrs('back', step)}>${esc(t('chat.question.back'))}</button>` : ''}
        <button type="button" class="p-q-ghost" ${attrs('skip')}>${esc(t('chat.question.skip'))}</button>
        <button type="button" class="p-q-primary" ${attrs(last ? 'submit' : 'next')}${q && done(step) ? '' : ' disabled'}><span>${esc(t(last ? 'chat.question.send' : 'chat.question.continue'))}</span>${enter}</button>
      </div>
    </div>
  </div>
</div>`
}

export const TEXTS = {
  en: {
    'chat.question.label': 'Questions from Aria',
    'chat.question.tabs': 'Questions',
    'chat.question.header_default': 'Question',
    'chat.question.recommended': 'Recommended',
    'chat.question.other': 'Other…',
    'chat.question.talk': 'Talk about this',
    'chat.question.back': 'Back',
    'chat.question.skip': 'Skip',
    'chat.question.continue': 'Continue',
    'chat.question.send': 'Send',
    'chat.question.dismiss': 'Dismiss questions',
  },
  es: {
    'chat.question.label': 'Preguntas de Aria',
    'chat.question.tabs': 'Preguntas',
    'chat.question.header_default': 'Pregunta',
    'chat.question.recommended': 'Recomendada',
    'chat.question.other': 'Otra…',
    'chat.question.talk': 'Hablar de esto',
    'chat.question.back': 'Atrás',
    'chat.question.skip': 'Omitir',
    'chat.question.continue': 'Continuar',
    'chat.question.send': 'Enviar',
    'chat.question.dismiss': 'Descartar las preguntas',
  },
}
