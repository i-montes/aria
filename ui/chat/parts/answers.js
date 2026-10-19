// The user's answers to Aria's questions, shown as a bubble (design/components/answer-card.dc.html).
//
// Props: { answers: [{ header, selected: [string], talk? }], title?, talkText? }
//   header    the question's short header (default "Question")
//   selected  the picked option labels, one line each
//   talk      true when the user chose "let's talk about this first" instead of an option
//   title     the card's title (default "Answers")
//   talkText  the line shown for a talk answer (default "Let's talk about this first")
// ui: unused. t(key) translates.
// Emits: nothing.
import { esc } from '../../onboarding/components.js'

const CHECK =
  '<svg aria-hidden="true" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#8BE0B0" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"></path></svg>'
const CHAT =
  '<svg aria-hidden="true" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"></path></svg>'

export const renderAnswers = ({ answers, title, talkText } = {}, ui, t) => {
  const heading = esc(title || t('chat.answers.title'))
  const rows = (answers ?? [])
    .map((a, i) => {
      const body = a.talk
        ? `<span class="p-answers-talk">${CHAT}<span>${esc(talkText || t('chat.answers.talk'))}</span></span>`
        : (a.selected ?? []).map((v) => `<span class="p-answers-value">${esc(v)}</span>`).join('')
      return `<div class="p-answers-row" style="animation-delay: ${70 + i * 70}ms"><span class="p-answers-header">${esc(a.header || t('chat.answers.question'))}</span><div class="p-answers-values">${body}</div></div>`
    })
    .join('')
  return `<div class="p-answers"><div class="p-answers-card" role="group" aria-label="${heading}"><div class="p-answers-title">${CHECK}<span>${heading}</span></div><div class="p-answers-rows">${rows}</div></div></div>`
}

export const TEXTS = {
  en: {
    'chat.answers.title': 'Answers',
    'chat.answers.question': 'Question',
    'chat.answers.talk': 'Let’s talk about this first',
  },
  es: {
    'chat.answers.title': 'Respuestas',
    'chat.answers.question': 'Pregunta',
    'chat.answers.talk': 'Hablemos primero de esto',
  },
}
