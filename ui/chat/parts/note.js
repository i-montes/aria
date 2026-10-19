// A one-line system note (design/components/note.dc.html): "engram saved: you prefer pnpm over npm".
//
// Props: { label?, text }
//   label  the source's name (engram, notion, tavily, sandbox, model…), shown brighter; not translated
//   text   what happened
// ui: unused. t(key) translates (this part has no fixed texts of its own).
// Emits: nothing.
import { esc } from '../../onboarding/components.js'

export const renderNote = ({ label, text }, ui, t) =>
  `<p class="p-note">${label ? `<span class="p-note-label">${esc(label)}</span> ` : ''}${esc(text)}</p>`

export const TEXTS = { en: {}, es: {} }
