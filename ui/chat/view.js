// The chat's page: pure functions that build its HTML (tested without a browser) and a small
// controller that wires them to the bridge. The state lives in `state.js`; everything dynamic is
// escaped with `esc`.
import { esc } from '../onboarding/components.js'
import { addUser, answerDialog, createChat, reduce } from './state.js'

/** `/home/ana/Projects` -> `~/Projects` (Linux and Mac homes); anything else stays as it is. */
export const shortPath = (path) => String(path ?? '').replace(/^\/(?:home|Users)\/[^/]+(?=\/|$)/, '~')

export function renderPath(state, t = (key) => key) {
  const path = state.workdir
  return path
    ? `<span class="path-icon" aria-hidden="true">▸</span><span class="path-text" title="${esc(path)}">${esc(shortPath(path))}</span>`
    : `<span class="path-text muted">${esc(t('chat.path.unknown'))}</span>`
}

const GLYPH = { running: '', done: '✓', error: '!' }

function toolGroup(entry, index, t, open) {
  const rows = open
    ? `<div class="tool-rows">${entry.calls
        .map(
          (call) => `
      <div class="tool-row status-${call.status}" title="${esc(call.result)}">
        <span class="tool-dot">${GLYPH[call.status]}</span><span class="tool-name">${esc(call.name)}</span><span class="tool-summary">${esc(call.summary === call.name ? '' : call.summary)}</span>
      </div>`,
        )
        .join('')}</div>`
    : ''
  return `
    <div class="tools">
      <button type="button" class="tools-head" data-chat="toggle-tools" data-index="${index}" aria-expanded="${open}">
        <span class="chev${open ? ' open' : ''}">›</span><span>${esc(t('chat.tools', { n: entry.calls.length }))}</span>
      </button>${rows}
    </div>`
}

const WORKING_CELLS = Array.from({ length: 9 }, (_, i) => `<i style="animation-delay:${(i % 3) * 90 + Math.floor(i / 3) * 90}ms"></i>`).join('')

/** The conversation. `expanded` holds the indexes of the tool groups that are open. */
export function renderMessages(state, t, expanded = new Set()) {
  const entries = state.messages
    .map((entry, index) => {
      if (entry.kind === 'user') return `<div class="msg user"><p>${esc(entry.text)}</p></div>`
      if (entry.kind === 'assistant') {
        if (!entry.text && !entry.thinking) return ''
        const thinking = entry.thinking && !entry.text ? `<p class="thinking">${esc(entry.thinking)}</p>` : ''
        return `<div class="msg assistant">${thinking}${entry.text ? `<p>${esc(entry.text)}</p>` : ''}</div>`
      }
      if (entry.kind === 'tools') return toolGroup(entry, index, t, expanded.has(index))
      return `<div class="notice tone-${esc(entry.tone)}">${esc(entry.key ? t(entry.key, entry.vars) : entry.text)}</div>`
    })
    .join('')
  const working = state.running ? `<div class="working"><span class="cells">${WORKING_CELLS}</span><span class="working-label">${esc(t('chat.working'))}</span></div>` : ''
  return entries + working
}

export const renderEmpty = (state, t) => (state.messages.length || state.running ? '' : `<p class="empty">${esc(t('chat.hint.empty'))}</p>`)

export function renderProblem(state, t) {
  if (!state.problem) return ''
  return `<div class="problem"><b>${esc(t('chat.problem.title'))}</b>${esc(t(`chat.problem.${state.problem}`))}</div>`
}

/** One extension dialog. Its controls carry `data-chat` actions and the dialog's `data-id`. */
export function renderDialog(dialog, t) {
  const { id, method, request } = dialog
  const idAttr = `data-id="${esc(id)}"`
  const title = request.title ? `<b class="dialog-title">${esc(request.title)}</b>` : ''
  const message = request.message ? `<p class="dialog-message">${esc(request.message)}</p>` : ''
  const cancel = `<button type="button" class="secondary" data-chat="cancel" ${idAttr}>${esc(t('chat.cancel'))}</button>`
  let body = ''
  if (method === 'select') {
    body = `<div class="dialog-options">${(request.options ?? []).map((option) => `<button type="button" class="opt-btn" data-chat="answer" ${idAttr} data-value="${esc(option)}">${esc(option)}</button>`).join('')}</div><div class="dialog-actions">${cancel}</div>`
  } else if (method === 'confirm') {
    body = `<div class="dialog-actions"><button type="button" class="primary" data-chat="confirm" ${idAttr} data-value="true">${esc(t('chat.yes'))}</button><button type="button" class="secondary" data-chat="confirm" ${idAttr} data-value="false">${esc(t('chat.no'))}</button>${cancel}</div>`
  } else {
    const field =
      method === 'editor'
        ? `<textarea class="dialog-field" rows="6" data-chat-field ${idAttr} spellcheck="false">${esc(request.prefill ?? '')}</textarea>`
        : `<input type="text" class="dialog-field" data-chat-field ${idAttr} placeholder="${esc(request.placeholder ?? '')}" spellcheck="false" autocomplete="off">`
    body = `${field}<div class="dialog-actions"><button type="button" class="primary" data-chat="submit" ${idAttr}>${esc(t('chat.ok'))}</button>${cancel}</div>`
  }
  return `<div class="dialog" role="dialog" aria-label="${esc(request.title ?? '')}">${title}${message}${body}</div>`
}

/** What the text before the caret asks for: a slash command at the very start, or an `@` file. */
export function pickerQuery(text, caret) {
  const before = text.slice(0, caret)
  const command = /^\/([^\s/]*)$/.exec(before)
  if (command) return { kind: 'command', query: command[1], start: 0 }
  const file = /(?:^|\s)@([^\s]*)$/.exec(before)
  if (file) return { kind: 'file', query: file[1], start: caret - file[1].length - 1 }
  return null
}

/** Puts `path` where the `@` token was (from `start` to `caret`); returns the new text and caret. */
export function insertFile(text, caret, start, path) {
  const after = text.slice(caret)
  const space = after.startsWith(' ') ? '' : ' '
  const inserted = `@${path}${space}`
  return { text: text.slice(0, start) + inserted + after, caret: start + inserted.length + (space ? 0 : 1) }
}

const modelLabel = (model) => (model ? model.split('/').pop() : '')

function renderPicker(picker, t) {
  if (!picker) return ''
  const titleKey = { command: 'chat.picker.commands', file: 'chat.picker.files', model: 'chat.picker.models' }[picker.kind]
  const rows = picker.items.length
    ? picker.items
        .map(
          (item, i) => `
      <button type="button" class="picker-row${i === picker.index ? ' active' : ''}" data-chat="pick" data-index="${i}">
        <span class="picker-main">${esc(item.label)}</span>${item.hint ? `<span class="picker-hint">${esc(item.hint)}</span>` : ''}
      </button>`,
        )
        .join('')
    : `<div class="picker-empty">${esc(t('chat.picker.empty'))}</div>`
  return `<div class="picker-title">${esc(t(titleKey))}</div>${rows}`
}

const icon = {
  send: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7"/></svg>',
  stop: '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="5" y="5" width="14" height="14" rx="3"/></svg>',
  plus: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>',
  gear: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>',
}

/**
 * `root` is the chat's container. `send(message)` posts to the dock; `request(message)` posts and
 * resolves with the reply; `onSettings()` opens the settings.
 */
export function createChatView({ root, t, send, request, onSettings }) {
  let state = createChat()
  const expanded = new Set()
  let picker = null // { kind, items: [{label, hint, value}], index, start, token }
  let commands = null
  let fileToken = 0
  let fileTimer = null

  root.innerHTML = `
    <header class="chat-head">
      <div class="chat-path" id="chat-path"></div>
      <button type="button" class="icon-btn" data-chat="new" data-nav aria-label="${esc(t('chat.new'))}" title="${esc(t('chat.new'))}">${icon.plus}</button>
      <button type="button" class="icon-btn" data-chat="settings" data-nav aria-label="${esc(t('chat.settings'))}" title="${esc(t('chat.settings'))}">${icon.gear}</button>
    </header>
    <div id="chat-banner"></div>
    <div id="chat-problem"></div>
    <div class="chat-list" id="chat-list"></div>
    <div id="chat-dialogs"></div>
    <div class="picker" id="chat-picker" hidden></div>
    <div class="composer">
      <textarea id="chat-input" rows="1" placeholder="${esc(t('chat.placeholder'))}" aria-label="${esc(t('chat.placeholder'))}" spellcheck="false"></textarea>
      <button type="button" class="stop-btn" id="chat-stop" data-chat="stop" aria-label="${esc(t('chat.stop'))}" title="${esc(t('chat.stop'))}" hidden>${icon.stop}</button>
      <button type="button" class="send-btn" id="chat-send" data-chat="send" aria-label="${esc(t('chat.send'))}">${icon.send}</button>
    </div>
    <div class="chat-foot">
      <button type="button" class="model-chip" id="chat-model" data-chat="model"></button>
      <span class="hints"><kbd>/</kbd> ${esc(t('chat.hint.commands'))} · <kbd>@</kbd> ${esc(t('chat.hint.files'))}</span>
    </div>`
  const $ = (id) => root.querySelector(`#${id}`)
  const input = $('chat-input')
  let dialogsKey = ''

  function render() {
    $('chat-path').innerHTML = renderPath(state, t)
    $('chat-problem').innerHTML = renderProblem(state, t)
    const list = $('chat-list')
    const nearBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 80
    list.innerHTML = renderEmpty(state, t) + renderMessages(state, t, expanded)
    if (nearBottom) list.scrollTop = list.scrollHeight
    const key = state.dialogs.map((d) => d.id).join('|')
    if (key !== dialogsKey) {
      dialogsKey = key
      $('chat-dialogs').innerHTML = state.dialogs.map((d) => renderDialog(d, t)).join('')
      $('chat-dialogs').querySelector('button, input, textarea')?.focus()
    }
    $('chat-stop').hidden = !state.running
    $('chat-model').textContent = state.model ? modelLabel(state.model) : t('chat.model.none')
    $('chat-model').title = state.model ?? ''
    const pick = $('chat-picker')
    pick.hidden = !picker
    pick.innerHTML = renderPicker(picker, t)
  }

  // ---- composer -------------------------------------------------------------------------------

  function submit() {
    const text = input.value.trim()
    if (!text) return
    send({ type: 'chatSend', text, behavior: state.running ? 'steer' : null })
    state = addUser(state, text)
    input.value = ''
    autosize()
    closePicker()
    render()
    const list = $('chat-list')
    list.scrollTop = list.scrollHeight
  }

  const autosize = () => {
    input.style.height = 'auto'
    input.style.height = `${Math.min(input.scrollHeight, 154)}px`
  }

  function closePicker() {
    picker = null
    fileToken++
    render()
  }

  async function updatePicker() {
    const ask = pickerQuery(input.value, input.selectionStart ?? input.value.length)
    if (!ask) return picker?.kind === 'model' ? undefined : closePicker()
    if (ask.kind === 'command') {
      if (!commands) {
        const reply = await request({ type: 'listCommands' })
        commands = (reply.commands ?? []).map((c) => ({ label: `/${c.name}`, hint: c.description ?? '', value: c.name }))
      }
      const items = commands.filter((c) => c.value.toLowerCase().includes(ask.query.toLowerCase())).slice(0, 8)
      picker = { kind: 'command', items, index: 0, start: ask.start }
      return render()
    }
    clearTimeout(fileTimer)
    const token = ++fileToken
    fileTimer = setTimeout(async () => {
      const reply = await request({ type: 'listFiles', query: ask.query })
      if (token !== fileToken) return // a newer keystroke or a closed picker
      picker = { kind: 'file', items: (reply.files ?? []).slice(0, 8).map((f) => ({ label: f, hint: '', value: f })), index: 0, start: ask.start }
      render()
    }, 120)
  }

  async function openModels() {
    if (picker?.kind === 'model') return closePicker()
    const reply = await request({ type: 'listModels' })
    const items = (reply.models ?? []).map((m) => ({ label: m.name ?? m.id, hint: `${m.provider}/${m.id}`, value: `${m.provider}/${m.id}` }))
    picker = { kind: 'model', items, index: Math.max(0, items.findIndex((i) => i.value === state.model)), start: 0 }
    render()
  }

  function choose(index) {
    const item = picker?.items[index]
    if (!item) return
    if (picker.kind === 'model') {
      closePicker()
      request({ type: 'setModel', model: item.value })
      return
    }
    const caret = input.selectionStart ?? input.value.length
    if (picker.kind === 'command') {
      input.value = `/${item.value} ${input.value.slice(caret)}`
      input.selectionStart = input.selectionEnd = item.value.length + 2
    } else {
      const next = insertFile(input.value, caret, picker.start, item.value)
      input.value = next.text
      input.selectionStart = input.selectionEnd = next.caret
    }
    autosize()
    closePicker()
    input.focus()
  }

  input.addEventListener('input', () => {
    autosize()
    updatePicker()
  })

  input.addEventListener('keydown', (event) => {
    if (picker && ['ArrowDown', 'ArrowUp'].includes(event.key)) {
      event.preventDefault()
      event.stopPropagation()
      const size = picker.items.length
      if (size) picker = { ...picker, index: (picker.index + (event.key === 'ArrowDown' ? 1 : -1) + size) % size }
      return render()
    }
    if (picker && (event.key === 'Enter' || event.key === 'Tab') && !event.shiftKey && picker.items.length) {
      event.preventDefault()
      return choose(picker.index)
    }
    if (picker && event.key === 'Escape') {
      event.stopPropagation()
      return closePicker()
    }
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault()
      submit()
    } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.stopPropagation() // the arrows move the caret here, not focus between controls
    }
  })

  // ---- clicks ---------------------------------------------------------------------------------

  function answer(id, fields) {
    send({ type: 'uiAnswer', id, ...fields })
    state = answerDialog(state, id)
    render()
    input.focus()
  }

  root.addEventListener('click', (event) => {
    const el = event.target.closest('[data-chat]')
    if (!el) return
    const { chat, id, value, index } = el.dataset
    if (chat === 'send') return submit()
    if (chat === 'stop') return send({ type: 'chatAbort' })
    if (chat === 'new') return send({ type: 'chatNew' })
    if (chat === 'settings') return onSettings?.()
    if (chat === 'model') return openModels()
    if (chat === 'pick') return choose(Number(index))
    if (chat === 'toggle-tools') {
      const at = Number(index)
      if (!expanded.delete(at)) expanded.add(at)
      return render()
    }
    if (chat === 'answer') return answer(id, { value })
    if (chat === 'confirm') return answer(id, { confirmed: value === 'true' })
    if (chat === 'cancel') return answer(id, { cancelled: true })
    if (chat === 'submit') {
      const field = [...root.querySelectorAll('[data-chat-field]')].find((f) => f.dataset.id === id)
      return answer(id, { value: field?.value ?? '' })
    }
  })

  return {
    /** A dock message about the chat (`chatState`, `chatEvent`, `chatDialog`, …). */
    receive(message) {
      if (message.type === 'chatState' && message.model !== state.model) commands = null
      state = reduce(state, message)
      render()
    },
    show() {
      send({ type: 'chatLoad' })
      render()
      input.focus()
    },
    render,
    setBanner(html) {
      $('chat-banner').innerHTML = html
    },
  }
}
