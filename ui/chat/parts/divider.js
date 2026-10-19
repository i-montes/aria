// A separator with a caption (design/components/divider.dc.html): "—— context at 40% · compacted ——".
//
// Props: { text }
// ui: unused. t(key) translates (this part has no fixed texts of its own).
// Emits: nothing.
import { esc } from '../../onboarding/components.js'

export const renderDivider = ({ text }, ui, t) =>
  `<div class="p-divider" role="separator"><span class="p-divider-rule" aria-hidden="true"></span><span class="p-divider-text">${esc(text)}</span><span class="p-divider-rule" aria-hidden="true"></span></div>`

export const TEXTS = { en: {}, es: {} }
