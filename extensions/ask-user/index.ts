// `ask_user`: the agent asks the user 1-4 questions, each with 2-4 options plus a free-text answer,
// in a TUI form (based on Pi's examples/extensions/questionnaire.ts). One question answers on
// Enter; several show tabs and a submit tab. The answer comes back as the tool result, so it never
// goes through the router as a new user message.
//
// Subagents (`--mode json`) do not get the tool. Without a TUI (`pi -p`, or a subagent that calls it
// anyway) nobody can answer: the tool says so and tells the model to go on with the recommended
// option, or to report NEEDS_CONTEXT as a subagent.
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { Editor, type EditorTheme, Key, matchesKey, Text, visibleWidth, wrapTextWithAnsi } from '@earendil-works/pi-tui'
import { Type } from 'typebox'
import { type Answer, CANCELLED_TEXT, formatAnswers, MAX_OPTIONS, MAX_QUESTIONS, MIN_OPTIONS, normalize, noUserText, OTHER_LABEL, type Question } from './questions.ts'
import { askInDock } from './dock.ts'
import { toolTexts } from './texts.ts'

const askParams = (words: ReturnType<typeof toolTexts>) => Type.Object({
  questions: Type.Array(
    Type.Object({
      header: Type.String({ description: words.header }),
      question: Type.String({ description: words.question }),
      options: Type.Array(
        Type.Object({
          label: Type.String({ description: 'The choice, 1-5 words.' }),
          description: Type.Optional(Type.String({ description: 'What choosing it means: trade-off or consequence.' })),
          recommended: Type.Optional(Type.Boolean({ description: 'Mark the one option you recommend.' })),
        }),
        { minItems: MIN_OPTIONS, maxItems: MAX_OPTIONS, description: '2-4 distinct choices. Do not add an "Other" option: the user can always write their own.' },
      ),
    }),
    { minItems: 1, maxItems: MAX_QUESTIONS, description: '1-4 independent questions shown at once.' },
  ),
})

interface Details {
  questions: Question[]
  answers: Answer[]
  cancelled: boolean
}

const result = (text: string, details: Details) => ({ content: [{ type: 'text' as const, text }], details })

export default function askUser(pi: ExtensionAPI) {
  // Subagents (`--mode json`) have nobody to ask: keep the tool out of their declared tools.
  pi.on('session_start', (_event: any, ctx: any) => {
    if (ctx.mode === 'json') pi.setActiveTools(pi.getActiveTools().filter((n) => n !== 'ask_user'))
  })

  // Read when Pi starts: a change of the conversation language restarts it, as it does the persona.
  const words = toolTexts()
  pi.registerTool({
    name: 'ask_user',
    label: 'Question',
    description: words.description,
    promptSnippet: 'ask_user: ask the user multiple-choice questions when a decision is theirs',
    parameters: askParams(words),
    exposure: 'model-only',
    executionMode: 'sequential',

    async execute(_id, params, signal, _onUpdate, ctx) {
      const questions = normalize(params)
      // Aria's dock: the chat's question card, through a dialog; its answers come back as JSON.
      if (ctx.mode === 'rpc' && ctx.hasUI) return askInDock(ctx, questions, signal)
      if (ctx.mode !== 'tui') return result(noUserText(ctx.mode), { questions, answers: [], cancelled: true })

      const isMulti = questions.length > 1
      const totalTabs = questions.length + 1

      const details = await ctx.ui.custom<Details>((tui, theme, _kb, done) => {
        let tab = 0
        let optionIndex = 0
        let inputMode = false
        let cached: string[] | undefined
        const answers = new Map<string, Answer>()
        let finished = false

        const editorTheme: EditorTheme = {
          borderColor: (s) => theme.fg('accent', s),
          selectList: {
            selectedPrefix: (t) => theme.fg('accent', t),
            selectedText: (t) => theme.fg('accent', t),
            description: (t) => theme.fg('muted', t),
            scrollInfo: (t) => theme.fg('dim', t),
            noMatch: (t) => theme.fg('warning', t),
          },
        }
        const editor = new Editor(tui, editorTheme)

        const refresh = () => {
          cached = undefined
          tui.requestRender()
        }
        const finish = (cancelled: boolean) => {
          if (finished) return
          finished = true
          done({ questions, answers: [...answers.values()], cancelled })
        }
        // Esc on the agent (or any abort) closes the form as cancelled.
        signal?.addEventListener('abort', () => finish(true), { once: true })

        const current = () => questions[tab]
        const options = () => {
          const q = current()
          return q ? [...q.options.map((o) => ({ ...o, isOther: false })), { label: OTHER_LABEL, isOther: true }] : []
        }
        const allAnswered = () => questions.every((q) => answers.has(q.id))

        const advance = () => {
          if (!isMulti) return finish(false)
          tab = tab < questions.length - 1 ? tab + 1 : questions.length
          optionIndex = 0
          refresh()
        }

        editor.onSubmit = (value) => {
          const q = current()
          const text = value.trim()
          if (!q || !text) return
          answers.set(q.id, { id: q.id, label: text, wasCustom: true })
          inputMode = false
          editor.setText('')
          advance()
        }

        function handleInput(data: string) {
          if (inputMode) {
            if (matchesKey(data, Key.escape)) {
              inputMode = false
              editor.setText('')
              refresh()
              return
            }
            editor.handleInput(data)
            refresh()
            return
          }
          if (isMulti) {
            if (matchesKey(data, Key.tab) || matchesKey(data, Key.right)) {
              tab = (tab + 1) % totalTabs
              optionIndex = 0
              return refresh()
            }
            if (matchesKey(data, Key.shift('tab')) || matchesKey(data, Key.left)) {
              tab = (tab - 1 + totalTabs) % totalTabs
              optionIndex = 0
              return refresh()
            }
          }
          if (tab === questions.length) {
            if (matchesKey(data, Key.enter) && allAnswered()) finish(false)
            else if (matchesKey(data, Key.escape)) finish(true)
            return
          }
          const opts = options()
          if (matchesKey(data, Key.up)) {
            optionIndex = Math.max(0, optionIndex - 1)
            return refresh()
          }
          if (matchesKey(data, Key.down)) {
            optionIndex = Math.min(opts.length - 1, optionIndex + 1)
            return refresh()
          }
          // Number keys pick an option directly.
          const n = Number.parseInt(data, 10)
          if (/^[1-9]$/.test(data) && n <= opts.length) optionIndex = n - 1
          if ((matchesKey(data, Key.enter) || /^[1-9]$/.test(data)) && opts[optionIndex]) {
            const q = current()!
            const opt = opts[optionIndex]
            if (opt.isOther) {
              inputMode = true
              editor.setText('')
              return refresh()
            }
            answers.set(q.id, { id: q.id, label: opt.label, wasCustom: false, index: optionIndex + 1 })
            return advance()
          }
          if (matchesKey(data, Key.escape)) finish(true)
        }

        function render(width: number): string[] {
          if (cached) return cached
          const w = Math.max(1, width)
          const lines: string[] = []
          const wrap = (prefix: string, text: string) => {
            const pw = visibleWidth(prefix)
            if (pw >= w) return void lines.push(...wrapTextWithAnsi(prefix + text, w))
            const pad = ' '.repeat(pw)
            wrapTextWithAnsi(text, w - pw).forEach((l, i) => lines.push(`${i === 0 ? prefix : pad}${l}`))
          }
          const q = current()
          const opts = options()

          lines.push(theme.fg('accent', '─'.repeat(w)))
          if (isMulti) {
            const tabs = ['← ']
            questions.forEach((qq, i) => {
              const done = answers.has(qq.id)
              const text = ` ${done ? '■' : '□'} ${qq.header} `
              tabs.push(`${i === tab ? theme.bg('selectedBg', theme.fg('text', text)) : theme.fg(done ? 'success' : 'muted', text)} `)
            })
            const send = ' ✓ Send '
            tabs.push(`${tab === questions.length ? theme.bg('selectedBg', theme.fg('text', send)) : theme.fg(allAnswered() ? 'success' : 'dim', send)} →`)
            wrap(' ', tabs.join(''))
            lines.push('')
          }

          const renderOptions = () => {
            opts.forEach((o, i) => {
              const selected = i === optionIndex
              const label = `${i + 1}. ${o.label}${o.isOther && inputMode ? ' ✎' : ''}`
              wrap(selected ? theme.fg('accent', '> ') : '  ', theme.fg(selected || (o.isOther && inputMode) ? 'accent' : 'text', label))
              if (o.description) wrap('     ', theme.fg('muted', o.description))
            })
          }

          if (tab === questions.length) {
            wrap(' ', theme.fg('accent', theme.bold('Ready to send')))
            lines.push('')
            for (const qq of questions) {
              const a = answers.get(qq.id)
              if (a) wrap(' ', `${theme.fg('muted', `${qq.header}: `)}${theme.fg('text', (a.wasCustom ? '(typed) ' : '') + a.label)}`)
            }
            lines.push('')
            if (allAnswered()) wrap(' ', theme.fg('success', 'Enter to send'))
            else wrap(' ', theme.fg('warning', `Unanswered: ${questions.filter((x) => !answers.has(x.id)).map((x) => x.header).join(', ')}`))
          } else if (q) {
            wrap(' ', theme.fg('text', theme.bold(q.question)))
            lines.push('')
            renderOptions()
            if (inputMode) {
              lines.push('')
              wrap(' ', theme.fg('muted', 'Your answer:'))
              for (const l of editor.render(Math.max(1, w - 2))) lines.push(` ${l}`)
            }
          }

          lines.push('')
          const help = inputMode
            ? 'Enter send • Esc back to the options'
            : isMulti
              ? 'Tab/←→ question • ↑↓ or 1-9 choose • Enter confirm • Esc cancel'
              : '↑↓ or 1-9 choose • Enter confirm • Esc cancel'
          wrap(' ', theme.fg('dim', help))
          lines.push(theme.fg('accent', '─'.repeat(w)))
          cached = lines
          return lines
        }

        return { render, invalidate: () => void (cached = undefined), handleInput }
      })

      if (!details || details.cancelled) return result(CANCELLED_TEXT, details ?? { questions, answers: [], cancelled: true })
      return result(formatAnswers(questions, details.answers), details)
    },

    renderCall(args, theme) {
      const qs = Array.isArray((args as any)?.questions) ? (args as any).questions : []
      const first = String(qs[0]?.question ?? '')
      let text = theme.fg('toolTitle', theme.bold('question '))
      text += theme.fg('muted', qs.length > 1 ? `${qs.length} questions` : first)
      return new Text(text, 0, 0)
    },

    renderResult(res, _options, theme) {
      const d = res.details as Details | undefined
      if (!d || d.cancelled) {
        const t = res.content[0]
        return new Text(theme.fg('warning', d?.answers ? 'No answer' : t?.type === 'text' ? t.text : ''), 0, 0)
      }
      const lines = d.questions.map((q) => {
        const a = d.answers.find((x) => x.id === q.id)
        return `${theme.fg('success', '✓ ')}${theme.fg('accent', q.header)}: ${a ? (a.wasCustom ? theme.fg('muted', '(typed) ') : '') + a.label : '—'}`
      })
      return new Text(lines.join('\n'), 0, 0)
    },
  })
}
