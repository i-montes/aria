// Attachments: the chips of the files a message carries (above the composer's input before sending, in the user's
// bubble after), the attachment lines the dock adds to a message, and the page's limits. Each chip is a pill
// (pill.js; reuse, never a look-alike).
//
// Props: { files, removable? }
//   files      [{ id?, name, size?, image?, path? }]  size: bytes (a number) or the dock's text ('12 KB')
//   removable  true adds a remove button to each chip (the composer); the thread's chips have none
// ui: unused. t(key, vars) translates.
// Emits: data-chat="attach-remove" data-id="<id>" (removable chips only).
// Styles: attachments.css (`.p-attach`, `.p-attach-remove`) on top of pill.css.
//
// The dock writes, after the message's text and a blank line, one line per file (crates/aria/src/attachments.rs):
//   [attached: <name> (<mime>, <size>) at <path>]      a file, or an image the model cannot see
//   [attached image: <name> at <path>]                 an image also sent to the model as an image
// A name never holds `[` or `]` (the dock makes them parentheses), so a line ends at its last `]`.
import { esc } from '../../onboarding/components.js'
import { renderPill } from './pill.js'

export const TEXTS = {
  en: {
    'chat.attach.remove': 'Remove {name}',
    'chat.attach.image': 'image',
    'chat.attach.tooMany': 'At most {max} files per message',
    'chat.attach.tooBig': '{name} is larger than {max}, so it was not attached',
    'chat.attach.tooMuch': 'The files of one message can take at most {max} together',
    'chat.attach.unreadable': 'Could not read {name}, so it was not attached',
    'chat.attach.notBack': '{count} file(s) of the refused message did not fit next to the ones attached now',
  },
  es: {
    'chat.attach.remove': 'Quitar {name}',
    'chat.attach.image': 'imagen',
    'chat.attach.tooMany': 'Como máximo {max} archivos por mensaje',
    'chat.attach.tooBig': '{name} pesa más de {max}, así que no se adjuntó',
    'chat.attach.tooMuch': 'Los archivos de un mensaje pueden sumar como máximo {max}',
    'chat.attach.unreadable': 'No se pudo leer {name}, así que no se adjuntó',
    'chat.attach.notBack': '{count} archivo(s) del mensaje rechazado no cupieron junto a los que ya están adjuntos',
  },
}

const MB = 1024 * 1024
/** The same limits the dock enforces (it refuses beyond them too). */
export const LIMITS = { files: 10, each: 20 * MB, total: 50 * MB }

/** `7 B`, `12 KB`, `1.3 MB` (base 1024), as the dock writes sizes in its lines. */
export function humanSize(bytes) {
  const n = Math.max(0, Math.floor(Number(bytes) || 0))
  if (n < 1024) return `${n} B`
  if (n < MB) return `${Math.floor((n + 512) / 1024)} KB`
  return `${(n / MB).toFixed(1)} MB`
}

const FILE_LINE = /^\[attached: (.*) \(([^,()]*), ([^()]*)\) at (.+)\]$/
const IMAGE_LINE = /^\[attached image: (.*) at (.+)\]$/

function parseLine(line) {
  let m = FILE_LINE.exec(line)
  if (m) return { name: m[1], mime: m[2], size: m[3], path: m[4], image: m[2].startsWith('image/') }
  m = IMAGE_LINE.exec(line)
  if (m) return { name: m[1], mime: '', size: '', path: m[2], image: true }
  return null
}

/** A message as the person wrote it and the files the dock's lines name: `{ text, files }`. Only the lines at the end
 * count; one in the middle of the text is the person's own words. */
export function splitAttached(text) {
  const lines = String(text ?? '').split('\n')
  const files = []
  while (lines.length) {
    const file = parseLine(lines.at(-1))
    if (!file) break
    files.unshift(file)
    lines.pop()
  }
  return { text: files.length ? lines.join('\n').trimEnd() : String(text ?? ''), files }
}

/** The message without its attachment lines: what the page compares when Pi echoes or queues a message. */
export const plainText = (text) => splitAttached(text).text

/** Which of `adding` ([{ name, size }]) can join `have` before anything is read: a file over the size limit is left
 * out alone; more files, or more bytes together, than a message may carry refuses the whole addition. */
export function admit(have, adding) {
  const notes = []
  const fit = []
  for (const file of adding) {
    if (file.size > LIMITS.each) notes.push({ key: 'chat.attach.tooBig', vars: { name: file.name, max: humanSize(LIMITS.each).replace('.0', '') } })
    else fit.push(file)
  }
  if (!fit.length) return { take: [], notes }
  const bytes = (list) => list.reduce((sum, f) => sum + (Number(f.size) || 0), 0)
  if (have.length + fit.length > LIMITS.files) return { take: [], notes: [...notes, { key: 'chat.attach.tooMany', vars: { max: LIMITS.files } }] }
  if (bytes(have) + bytes(fit) > LIMITS.total) return { take: [], notes: [...notes, { key: 'chat.attach.tooMuch', vars: { max: humanSize(LIMITS.total).replace('.0', '') } }] }
  return { take: fit, notes }
}

/** How many sent messages keep their files' data at most (the oldest goes first): one that is never echoed (a skill
 * Pi expands, a command that runs at once) must not keep its files in the page for good. */
export const SENT_KEPT = 3

/** `list` with `entry` (a message sent with files) added, the oldest dropped beyond `SENT_KEPT`. */
export const remember = (list = [], entry) => [...list, entry].slice(-SENT_KEPT)

/** The sent messages still holding their files' data (`[{ text, … }]`, `text` as sent), less the ones this dock message
 * shows Pi has taken: its echo of the message, or its queue holding it, each releasing the first entry with that text
 * only (two sends of the same text are two entries); a settled run releases them all, as nothing can refuse them then.
 * Data is kept only for a message that can still be refused (its files go back to the composer then). */
export function releaseAccepted(list = [], message) {
  const event = message?.type === 'chatEvent' ? message.event ?? {} : {}
  const textOf = (content) => (typeof content === 'string' ? content : (content ?? []).filter((b) => b.type === 'text').map((b) => b.text).join(''))
  if (event.type === 'agent_settled') return list.length ? [] : list
  let taken = null
  if (event.type === 'message_start' && event.message?.role === 'user') taken = [plainText(textOf(event.message.content))]
  else if (event.type === 'queue_update') taken = [...(event.steering ?? []), ...(event.followUp ?? [])].map(plainText)
  if (!taken) return list
  const left = [...list]
  for (const text of taken) {
    const at = left.findIndex((m) => m.text === text)
    if (at >= 0) left.splice(at, 1)
  }
  return left.length === list.length ? list : left
}

const X =
  '<svg aria-hidden="true" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>'

export function renderAttachments({ files, removable = false } = {}, ui, t = (k) => k) {
  if (!Array.isArray(files) || !files.length) return ''
  const chips = files.map((file) => {
    const meta = typeof file.size === 'number' ? humanSize(file.size) : file.size || (file.image ? t('chat.attach.image') : '')
    const label = t('chat.attach.remove', { name: file.name })
    const extra = removable
      ? `<button type="button" class="p-attach-remove" data-chat="attach-remove" data-id="${esc(file.id)}" aria-label="${esc(label)}" title="${esc(label)}">${X}</button>`
      : ''
    // Static either way: a chip itself does nothing, so only its remove button (the composer's) can take the focus.
    return renderPill({ label: file.name, meta, tone: file.image ? 'image' : 'file', title: file.path || file.name, extra, static: true }, {}, t)
  })
  return `<div class="p-attach">${chips.join('')}</div>`
}
