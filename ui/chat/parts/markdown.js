// Markdown for the chat's bubbles (the assistant's answers and the user's messages). Pure: text in, HTML out. Every
// piece of the source is escaped before any tag is added, so a message can never bring its own HTML.
//
// Blocks: paragraphs (a line break stays a line break), # headings, - * + and 1. lists (nested by indentation),
// > quotes, ``` fenced code, | tables |, --- rules. Inline: `code`, **bold**, *italic*, ~~strike~~, [links](https://…)
// and bare https:// links, and with `mentions` the @path chips of the user's messages.
//
// renderMarkdown(source, options?)
//   options.code(lang, text)  HTML for a fenced block, or null for the default <pre> (the assistant draws ```diff)
//   options.mentions          true: @path becomes a chip (the user's messages)
//
// Emits: data-chat="open-link" data-url="<url>" on links (the page opens them in the browser; never the dock itself).

import { esc } from '../../onboarding/components.js'

const SAFE_URL = /^(https?:\/\/|mailto:)/i

const FENCE = /^ {0,3}(`{3,}|~{3,})\s*([\w+#.-]*)/
const HEADING = /^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/
const RULE = /^ {0,3}([-*_])(\s*\1){2,}\s*$/
const QUOTE = /^ {0,3}>\s?/
const ITEM = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/
const TABLE_SEPARATOR = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/

const link = (url, label) => `<a class="p-md-link" href="${esc(url)}" data-chat="open-link" data-url="${esc(url)}">${label}</a>`

/** Inline markdown of one paragraph's text (no block syntax). */
export function renderInline(text, options = {}) {
  const kept = []
  let out = inline(text, options, (html) => `\u0000${kept.push(html) - 1}\u0000`)
  // Kept pieces can hold others (a link's label with code): restore until none is left.
  while (/\u0000\d+\u0000/.test(out)) out = out.replace(/\u0000(\d+)\u0000/g, (_, i) => kept[Number(i)])
  return out
}

/** `text` with its markdown as HTML; the pieces nothing else may touch (code, links, chips) are set aside with `keep`. */
function inline(text, options, keep) {
  let out = String(text ?? '')
  // Code spans first: nothing inside them is markdown.
  out = out.replace(/(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/g, (_, ticks, code) => keep(`<code class="p-md-code">${esc(code)}</code>`))
  out = out.replace(/\[([^\]\n]+)\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)/g, (whole, label, url) =>
    keep(SAFE_URL.test(url) ? link(url, inline(label, {}, keep)) : esc(whole)),
  )
  out = out.replace(/\bhttps?:\/\/[^\s<>\u0000]*[^\s<>.,;:!?)\]'"*_~\u0000]/g, (url) => keep(link(url, esc(url))))
  if (options.mentions) {
    // A mention starts at the beginning or after whitespace and stops before trailing punctuation.
    out = out.replace(/(^|\s)(@[^\s@]*[^\s@.,;:!?)])/g, (_, before, mention) => `${before}${keep(`<span class="p-user-chip">${esc(mention)}</span>`)}`)
  }
  return esc(out)
    .replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^\w])__(?=\S)([\s\S]*?\S)__(?!\w)/g, '$1<strong>$2</strong>')
    .replace(/(^|[^*\w])\*(?=[^\s*])([^*\n]*?[^\s*])\*(?!\*)/g, '$1<em>$2</em>')
    .replace(/(^|[^\w])_(?=[^\s_])([^_\n]*?[^\s_])_(?!\w)/g, '$1<em>$2</em>')
    .replace(/~~(?=\S)([\s\S]*?\S)~~/g, '<del>$1</del>')
}

const cells = (line) =>
  line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split(/(?<!\\)\|/)
    .map((cell) => cell.trim().replace(/\\\|/g, '|'))

const startsBlock = (line, next) =>
  FENCE.test(line) || HEADING.test(line) || RULE.test(line) || QUOTE.test(line) || ITEM.test(line) || (line.includes('|') && TABLE_SEPARATOR.test(next ?? '') && (next ?? '').includes('-'))

const indentOf = (line) => line.match(/^\s*/)[0].replace(/\t/g, '    ').length

/** The lines of one list (from `start`), and where it ends. Items, their continuation and nested lists all belong. */
function takeList(lines, start) {
  const base = indentOf(lines[start])
  const items = []
  let i = start
  while (i < lines.length) {
    const line = lines[i]
    const item = line.match(ITEM)
    if (item && indentOf(line) <= base + 1 && indentOf(line) >= base) {
      items.push({ ordered: /\d/.test(item[2]), start: parseInt(item[2], 10), lines: [item[3]], width: item[1].length + item[2].length + 1 })
      i++
      continue
    }
    if (!line.trim()) {
      // A blank line ends the list unless the list (or this item) goes on after it.
      const next = lines[i + 1]
      if (next !== undefined && next.trim() && (indentOf(next) > base || (ITEM.test(next) && indentOf(next) === base))) {
        items.at(-1).lines.push('')
        i++
        continue
      }
      break
    }
    if (indentOf(line) > base) {
      items.at(-1).lines.push(line.slice(Math.min(indentOf(line), items.at(-1).width + 1)))
      i++
      continue
    }
    if (!startsBlock(line, lines[i + 1])) {
      items.at(-1).lines.push(line.trim()) // a lazy continuation of the item's text
      i++
      continue
    }
    break
  }
  return { items, end: i }
}

function renderList(items, options) {
  const ordered = items[0].ordered
  const body = items
    .map((item) => {
      const inner = renderBlocks(item.lines, options)
      // A one-paragraph item is drawn without its <p>, so tight lists stay tight.
      const tight = inner.match(/^<p class="p-md-p">([\s\S]*)<\/p>$/)
      return `<li>${tight ? tight[1] : inner}</li>`
    })
    .join('')
  const start = ordered && items[0].start !== 1 ? ` start="${items[0].start}"` : ''
  return ordered ? `<ol class="p-md-list"${start}>${body}</ol>` : `<ul class="p-md-list">${body}</ul>`
}

function renderBlocks(lines, options) {
  const out = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (!line.trim()) {
      i++
      continue
    }
    const fence = line.match(FENCE)
    if (fence) {
      const code = []
      i++
      // An unclosed fence (an answer still streaming) runs to the end.
      while (i < lines.length && !new RegExp(`^ {0,3}${fence[1][0]}{${fence[1].length},}\\s*$`).test(lines[i])) code.push(lines[i++])
      i++
      const text = code.join('\n')
      out.push(options.code?.(fence[2].toLowerCase(), text) ?? `<pre class="p-md-pre"><code>${esc(text)}</code></pre>`)
      continue
    }
    const heading = line.match(HEADING)
    if (heading) {
      out.push(`<p class="p-md-h p-md-h${heading[1].length}">${renderInline(heading[2], options)}</p>`)
      i++
      continue
    }
    if (RULE.test(line)) {
      out.push('<hr class="p-md-hr">')
      i++
      continue
    }
    if (QUOTE.test(line)) {
      const quoted = []
      while (i < lines.length && lines[i].trim() && (QUOTE.test(lines[i]) || !startsBlock(lines[i], lines[i + 1]))) quoted.push(lines[i++].replace(QUOTE, ''))
      out.push(`<blockquote class="p-md-quote">${renderBlocks(quoted, options)}</blockquote>`)
      continue
    }
    if (ITEM.test(line)) {
      const { items, end } = takeList(lines, i)
      out.push(renderList(items, options))
      i = end
      continue
    }
    if (line.includes('|') && lines[i + 1] && TABLE_SEPARATOR.test(lines[i + 1]) && lines[i + 1].includes('-')) {
      const head = cells(line)
      const align = cells(lines[i + 1]).map((c) => (c.startsWith(':') && c.endsWith(':') ? 'center' : c.endsWith(':') ? 'right' : ''))
      const rows = []
      i += 2
      while (i < lines.length && lines[i].trim() && lines[i].includes('|')) rows.push(cells(lines[i++]))
      const cell = (tag, text, at) => `<${tag}${align[at] ? ` style="text-align:${align[at]}"` : ''}>${renderInline(text, options)}</${tag}>`
      out.push(
        `<div class="p-md-table"><table><thead><tr>${head.map((c, at) => cell('th', c, at)).join('')}</tr></thead><tbody>${rows
          .map((row) => `<tr>${head.map((_, at) => cell('td', row[at] ?? '', at)).join('')}</tr>`)
          .join('')}</tbody></table></div>`,
      )
      continue
    }
    const paragraph = []
    while (i < lines.length && lines[i].trim() && (paragraph.length === 0 || !startsBlock(lines[i], lines[i + 1]))) paragraph.push(lines[i++].trim())
    out.push(`<p class="p-md-p">${paragraph.map((l) => renderInline(l, options)).join('<br>')}</p>`)
  }
  return out.join('')
}

export function renderMarkdown(source, options = {}) {
  return renderBlocks(String(source ?? '').replace(/\r\n?/g, '\n').split('\n'), options)
}
