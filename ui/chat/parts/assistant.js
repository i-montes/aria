// The assistant's message (design/components/aria-message.dc.html, plus the "Proposed change"
// diff group of design/components/aria-on-the-desktop.dc.html).
//
// Props: { text, detail?, source? }
//   text    the answer, in markdown (markdown.js); a fenced ```diff block is drawn as the design's diff group
//   detail  a second, muted paragraph (same rendering)
//   source  where the answer came from ("source · engram")
// ui: unused. t(key) translates.
// Emits: data-chat="open-link" on the answer's links (markdown.js).
import { esc } from '../../onboarding/components.js'
import { renderMarkdown } from './markdown.js'

/** Escape, then turn `code` spans into <code>. An unclosed backtick stays text. */
export const renderInline = (text) =>
  String(text ?? '')
    .split(/(`[^`\n]+`)/)
    .map((piece) => (/^`[^`\n]+`$/.test(piece) ? `<code class="p-assistant-code">${esc(piece.slice(1, -1))}</code>` : esc(piece)))
    .join('')

/**
 * Lines of a unified diff as { kind: 'add' | 'del' | 'ctx' | 'meta', text }. File headers (---, +++,
 * diff, index) count only before the first change; hunk headers (@@) are always meta.
 */
export const parseDiff = (source) => {
  const raw = String(source ?? '').replace(/\r/g, '').split('\n')
  while (raw.length && raw[raw.length - 1] === '') raw.pop()
  let changed = false
  return raw.map((line) => {
    if (line.startsWith('@@')) return { kind: 'meta', text: line }
    if (!changed && (line.startsWith('+++ ') || line.startsWith('--- ') || line.startsWith('diff ') || line.startsWith('index '))) return { kind: 'meta', text: line }
    if (line.startsWith('+')) return ((changed = true), { kind: 'add', text: line.slice(1) })
    if (line.startsWith('-')) return ((changed = true), { kind: 'del', text: line.slice(1) })
    return { kind: 'ctx', text: line.startsWith(' ') ? line.slice(1) : line }
  })
}

const MARK = { add: '+', del: '-', ctx: ' ', meta: '' }

const renderDiff = (lines, t) =>
  `<div class="p-assistant-diff" role="group" aria-label="${esc(t('chat.assistant.change'))}">${lines
    .map((l) => `<div class="p-assistant-line is-${l.kind}"><span class="p-assistant-mark">${MARK[l.kind]}</span><span class="p-assistant-code-text">${esc(l.text)}</span></div>`)
    .join('')}</div>`

const renderBlocks = (source, cls, t) => {
  const html = renderMarkdown(source, { code: (lang, text) => (lang === 'diff' ? renderDiff(parseDiff(text), t) : null) })
  return html ? `<div class="p-md ${cls}">${html}</div>` : ''
}

export const renderAssistant = ({ text, detail, source }, ui, t) => {
  const main = renderBlocks(text, 'p-assistant-text', t) || '<div class="p-md p-assistant-text"></div>'
  const more = detail ? renderBlocks(detail, 'p-assistant-detail', t) : ''
  const from = source ? `<p class="p-assistant-source">${esc(t('chat.assistant.source'))} · ${esc(source)}</p>` : ''
  return `<div class="p-assistant"><div class="p-assistant-card">${main}${more}${from}</div></div>`
}

export const TEXTS = {
  en: {
    'chat.assistant.source': 'source',
    'chat.assistant.change': 'Proposed change',
  },
  es: {
    'chat.assistant.source': 'fuente',
    'chat.assistant.change': 'Cambio propuesto',
  },
}
