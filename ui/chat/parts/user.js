// The user's message (design/components/user-message.dc.html).
//
// Props: { text, files?, command?, queued?, steer?, id? }
//   text     the message, in markdown (markdown.js); "@path" mentions become chips. The dock's attachment lines at its
//            end (`[attached: …]`, `[attached image: …]`) are not shown: they become chips (attachments.js)
//   files    the message's files when the page already knows them ([{ name, size?, image?, path? }]); else the lines'
//   command  true when the text is a slash command (mono font, smaller)
//   queued   true while the message waits for the run to finish (dashed, "Queued" line + Cancel)
//   steer    a queued message that goes in at the run's next step, not when it ends (its own line, accent border)
//   id       the queued message's id, sent back by Cancel
// ui: unused. t(key) translates.
// Emits: data-chat="cancel-queued" data-id="<id>", data-chat="open-link" on links (markdown.js)
import { esc } from '../../onboarding/components.js'
import { renderAttachments, splitAttached } from './attachments.js'
import { renderMarkdown } from './markdown.js'

/** An @mention starts at the beginning or after whitespace and stops before trailing punctuation. */
export const splitMentions = (text) => {
  const source = String(text ?? '')
  const out = []
  const re = /@[^\s@]+/g
  let last = 0
  let m
  while ((m = re.exec(source)) !== null) {
    if (m.index > 0 && !/\s/.test(source[m.index - 1])) continue
    let tok = m[0]
    const trail = (tok.match(/[.,;:!?)]+$/) || [''])[0]
    if (trail) tok = tok.slice(0, tok.length - trail.length)
    if (tok.length < 2) continue
    if (m.index > last) out.push({ t: source.slice(last, m.index), chip: false })
    out.push({ t: tok, chip: true })
    last = m.index + tok.length
    re.lastIndex = last
  }
  if (last < source.length) out.push({ t: source.slice(last), chip: false })
  return out.length ? out : [{ t: source, chip: false }]
}

const CLOCK =
  '<svg aria-hidden="true" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="13" r="7"></circle><path d="M12 10v3l2 1.5M9.5 3h5"></path></svg>'

export const renderUser = ({ text: raw, files: given, command, queued, steer, id }, ui, t) => {
  const split = splitAttached(raw)
  const text = split.text
  const files = given?.length ? given : split.files
  // A slash command is shown as typed; a message is markdown with its mentions as chips.
  const parts = command
    ? splitMentions(text)
        .map((s) => (s.chip ? `<span class="p-user-chip">${esc(s.t)}</span>` : `<span>${esc(s.t)}</span>`))
        .join('')
    : renderMarkdown(text, { mentions: true })
  const queue = queued
    ? `<div class="p-user-queue">${CLOCK}<span>${esc(t(steer ? 'chat.user.steer' : 'chat.user.queued'))}</span><button type="button" class="p-user-cancel" data-chat="cancel-queued" data-id="${esc(id)}">${esc(t('chat.user.cancel'))}</button></div>`
    : ''
  const bubble = text || !files.length ? `<div class="p-user-bubble p-md${command ? ' is-command' : ''}">${parts}</div>` : ''
  const chips = files.length ? `<div class="p-user-files">${renderAttachments({ files }, {}, t)}</div>` : ''
  return `<div class="p-user${queued ? ' is-queued' : ''}${queued && steer ? ' is-steer' : ''}">${bubble}${chips}${queue}</div>`
}

export const TEXTS = {
  en: {
    'chat.user.queued': 'Queued · sends when Aria finishes',
    'chat.user.steer': 'Next · goes in when the current step ends',
    'chat.user.cancel': 'Cancel',
  },
  es: {
    'chat.user.queued': 'En cola · se envía cuando Aria termine',
    'chat.user.steer': 'Siguiente · entra cuando termine el paso en curso',
    'chat.user.cancel': 'Cancelar',
  },
}
