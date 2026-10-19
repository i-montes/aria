// The assistant's message (design/components/aria-message.dc.html, plus the "Proposed change"
// diff group of design/components/aria-on-the-desktop.dc.html).
//
// Props: { text, detail?, source? }
//   text    the answer; `code` spans and fenced ```diff blocks are rendered, the rest is text
//   detail  a second, muted paragraph (same rendering)
//   source  where the answer came from ("source · engram")
// ui: unused. t(key) translates.
// Emits: nothing.
import { esc } from '../../onboarding/components.js'

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

/** Splits a message into { type: 'text', text } and { type: 'diff', lines } blocks, in order. An unterminated diff (a streaming message) runs to the end. */
export const parseBlocks = (source) => {
  const blocks = []
  let text = []
  let diff = null
  const flushText = () => {
    const joined = text.join('\n').replace(/^\n+|\n+$/g, '')
    if (joined.trim()) blocks.push({ type: 'text', text: joined })
    text = []
  }
  for (const line of String(source ?? '').replace(/\r/g, '').split('\n')) {
    if (diff) {
      if (/^```\s*$/.test(line)) {
        blocks.push({ type: 'diff', lines: parseDiff(diff.join('\n')) })
        diff = null
      } else diff.push(line)
    } else if (/^```diff\s*$/.test(line)) {
      flushText()
      diff = []
    } else text.push(line)
  }
  if (diff) blocks.push({ type: 'diff', lines: parseDiff(diff.join('\n')) })
  else flushText()
  return blocks
}

const MARK = { add: '+', del: '-', ctx: ' ', meta: '' }

const renderDiff = (lines, t) =>
  `<div class="p-assistant-diff" role="group" aria-label="${esc(t('chat.assistant.change'))}">${lines
    .map((l) => `<div class="p-assistant-line is-${l.kind}"><span class="p-assistant-mark">${MARK[l.kind]}</span><span class="p-assistant-code-text">${esc(l.text)}</span></div>`)
    .join('')}</div>`

const renderBlocks = (source, cls, t) =>
  parseBlocks(source)
    .map((b) => (b.type === 'diff' ? renderDiff(b.lines, t) : `<p class="${cls}">${renderInline(b.text)}</p>`))
    .join('')

export const renderAssistant = ({ text, detail, source }, ui, t) => {
  const main = renderBlocks(text, 'p-assistant-text', t) || '<p class="p-assistant-text"></p>'
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
