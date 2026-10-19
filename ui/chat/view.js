// The chat's page: pure functions that build its HTML (tested without a browser) and a small
// controller that wires them to the bridge. The state lives in `state.js`; everything dynamic is
// escaped with `esc`.
import { esc } from '../onboarding/components.js'
import {
  addSession, applyStatus, closeSession, createTabs, renameSession, selectSession, setTitleIfEmpty, toSaved, undoClose, visibleTabs,
} from './sessions.js'
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

/** The tab bar: one tab per session with its state, an overflow menu, and the `+`. `ui`: { menuOpen, renaming }. */
export function renderTabs(tabs, t, ui = {}) {
  const { visible, overflow } = visibleTabs(tabs)
  const closable = tabs.sessions.length > 1
  const label = (s) => esc(s.title || t('chat.untitled'))
  const tab = (s) => {
    const on = s.id === tabs.active
    const body =
      ui.renaming === s.id
        ? `<input class="tab-input" data-chat-rename data-session="${esc(s.id)}" value="${esc(s.title)}" aria-label="${esc(t('chat.tab.rename'))}" spellcheck="false" autocomplete="off">`
        : `<button type="button" class="tab-main" role="tab" aria-selected="${on}" data-chat="select" data-session="${esc(s.id)}" title="${esc(`${s.title || t('chat.untitled')} · ${t(`chat.status.${s.status}`)}`)}">
             <span class="tab-dot" aria-hidden="true"></span><span class="tab-title">${label(s)}</span>
           </button>
           ${closable ? `<button type="button" class="tab-x" data-chat="close-tab" data-session="${esc(s.id)}" aria-label="${esc(t('chat.tab.close', { title: s.title || t('chat.untitled') }))}">×</button>` : ''}`
    return `<div class="tab status-${esc(s.status)}${on ? ' on' : ''}${s.unread ? ' unread' : ''}">${body}</div>`
  }
  const menu = ui.menuOpen
    ? `<div class="tab-menu" role="menu" aria-label="${esc(t('chat.tab.menu'))}">${overflow
        .map(
          (s) => `<button type="button" role="menuitem" class="tab-menu-row status-${esc(s.status)}${s.unread ? ' unread' : ''}" data-chat="select" data-session="${esc(s.id)}">
        <span class="tab-dot" aria-hidden="true"></span><span class="tab-title">${label(s)}</span><span class="tab-state">${esc(t(`chat.status.${s.status}`))}</span></button>`,
        )
        .join('')}</div>`
    : ''
  const more = overflow.length
    ? `<button type="button" class="tab-more" data-chat="menu" aria-haspopup="true" aria-expanded="${!!ui.menuOpen}">${esc(t('chat.tab.more', { n: overflow.length }))}</button>`
    : ''
  return `<div class="tabs" role="tablist" aria-label="${esc(t('chat.sessions'))}">${visible.map(tab).join('')}${more}<button type="button" class="tab-add" data-chat="add" aria-label="${esc(t('chat.new'))}" title="${esc(t('chat.new'))}">+</button></div>${menu}`
}

/** "Closed <title> · Undo", shown for a few seconds after a tab is closed. */
export function renderUndo(tabs, t) {
  if (!tabs.closed) return ''
  const title = tabs.closed.entry.title || t('chat.untitled')
  return `<div class="undo"><span>${esc(t('chat.undo.closed', { title }))}</span><button type="button" class="link" data-chat="undo">${esc(t('chat.undo'))}</button></div>`
}

const icon = {
  send: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7"/></svg>',
  stop: '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="5" y="5" width="14" height="14" rx="3"/></svg>',
  plus: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>',
  gear: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>',
}

const UNDO_MS = 8000

/**
 * `root` is the chat's container. `send(message)` posts to the dock; `request(message)` posts and
 * resolves with the reply; `onSettings()` opens the settings. Every chat message goes out wrapped
 * in `{ type: 'session', session, message }`: each tab is its own session and its own Pi.
 */
export function createChatView({ root, t, send, request, onSettings, makeId = () => crypto.randomUUID() }) {
  let tabs = createTabs(null, makeId)
  const chats = new Map() // session id -> its conversation (state.js)
  const drafts = new Map() // session id -> what was typed and not sent
  let picker = null // { kind, items: [{label, hint, value}], index, start }
  let commands = null
  let fileToken = 0
  let fileTimer = null
  let undoTimer = null
  const ui = { menuOpen: false, renaming: null }

  const chatOf = (id) => chats.get(id) ?? createChat()
  const current = () => chatOf(tabs.active)
  const post = (message, session = tabs.active) => send({ type: 'session', session, message })
  const ask = (message, session = tabs.active) => request({ type: 'session', session, message })
  const save = () => send({ type: 'sessionsSave', ...toSaved(tabs) })

  let input = null
  let banner = ''
  let dialogsKey = ''
  const $ = (id) => root.querySelector(`#${id}`)

  /** Builds the chat's HTML. Called once the texts are loaded (and again when the language changes). */
  function mount() {
    root.innerHTML = `
    <div id="chat-tabs"></div>
    <div id="chat-undo"></div>
    <header class="chat-head">
      <div class="chat-path" id="chat-path"></div>
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
    input = $('chat-input')
    input.value = drafts.get(tabs.active) ?? ''
    dialogsKey = ''
    bindInput()
    render()
  }

  const expandedOf = new Map() // session id -> open tool groups
  const expanded = () => {
    if (!expandedOf.has(tabs.active)) expandedOf.set(tabs.active, new Set())
    return expandedOf.get(tabs.active)
  }

  function render() {
    if (!input) return
    const state = current()
    $('chat-tabs').innerHTML = renderTabs(tabs, t, ui)
    $('chat-undo').innerHTML = renderUndo(tabs, t)
    $('chat-path').innerHTML = renderPath(state, t)
    $('chat-problem').innerHTML = renderProblem(state, t)
    const list = $('chat-list')
    const nearBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 80
    list.innerHTML = renderEmpty(state, t) + renderMessages(state, t, expanded())
    if (nearBottom) list.scrollTop = list.scrollHeight
    const key = `${tabs.active}:${state.dialogs.map((d) => d.id).join('|')}`
    if (key !== dialogsKey) {
      dialogsKey = key
      $('chat-dialogs').innerHTML = state.dialogs.map((d) => renderDialog(d, t)).join('')
      // Never take the keyboard from someone who is typing: the next Space or Enter would answer the dialog.
      if (!input.value) $('chat-dialogs').querySelector('button, input, textarea')?.focus()
    }
    $('chat-stop').hidden = !state.running
    $('chat-model').textContent = state.model ? modelLabel(state.model) : t('chat.model.none')
    $('chat-model').title = state.model ?? ''
    const pick = $('chat-picker')
    pick.hidden = !picker
    pick.innerHTML = renderPicker(picker, t)
    $('chat-banner').innerHTML = banner
    if (ui.renaming) root.querySelector('[data-chat-rename]')?.focus()
  }

  // ---- tabs -----------------------------------------------------------------------------------

  /** Shows another session: its draft, its conversation, and a `chatLoad` so its Pi (and any dialog) wakes up. */
  function switchTo(next) {
    if (next === tabs.active) return
    drafts.set(tabs.active, input.value)
    closePickerSilently()
    tabs = selectSession(tabs, next)
    input.value = drafts.get(next) ?? ''
    autosize()
    commands = null
    ui.menuOpen = false
    save()
    post({ type: 'chatLoad' })
    render()
    input.focus()
  }

  function addTab() {
    const id = makeId()
    drafts.set(tabs.active, input.value)
    tabs = addSession(tabs, id)
    if (tabs.active !== id) return // already at the limit
    closePickerSilently()
    input.value = ''
    autosize()
    commands = null
    ui.menuOpen = false
    save()
    post({ type: 'chatLoad' })
    render()
    input.focus()
  }

  function closeTab(id) {
    const wasActive = id === tabs.active
    const before = tabs
    tabs = closeSession(tabs, id)
    if (tabs === before) return
    ui.menuOpen = false
    send({ type: 'sessionClose', session: id })
    clearTimeout(undoTimer)
    undoTimer = setTimeout(() => {
      if (tabs.closed) chats.delete(tabs.closed.entry.id)
      tabs = { ...tabs, closed: null }
      render()
    }, UNDO_MS)
    if (wasActive) {
      closePickerSilently()
      input.value = drafts.get(tabs.active) ?? ''
      autosize()
      post({ type: 'chatLoad' })
    }
    save()
    render()
  }

  function undo() {
    clearTimeout(undoTimer)
    tabs = undoClose(tabs)
    save()
    render()
  }

  function commitRename(id, title) {
    ui.renaming = null
    tabs = renameSession(tabs, id, title)
    save()
    render()
    input.focus()
  }

  // ---- composer -------------------------------------------------------------------------------

  function submit() {
    const text = input.value.trim()
    if (!text) return
    const session = tabs.active
    post({ type: 'chatSend', text, behavior: current().running ? 'steer' : null }, session)
    chats.set(session, addUser(chatOf(session), text))
    const before = tabs
    tabs = setTitleIfEmpty(tabs, session, text)
    if (tabs !== before) save()
    input.value = ''
    drafts.delete(session)
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

  function closePickerSilently() {
    picker = null
    fileToken++
  }

  function closePicker() {
    closePickerSilently()
    render()
  }

  async function updatePicker() {
    const ask_ = pickerQuery(input.value, input.selectionStart ?? input.value.length)
    if (!ask_) return picker?.kind === 'model' ? undefined : closePicker()
    const session = tabs.active
    if (ask_.kind === 'command') {
      if (!commands) {
        const reply = await ask({ type: 'listCommands' }, session)
        commands = (reply.commands ?? []).map((c) => ({ label: `/${c.name}`, hint: c.description ?? '', value: c.name }))
      }
      const items = commands.filter((c) => c.value.toLowerCase().includes(ask_.query.toLowerCase())).slice(0, 8)
      picker = { kind: 'command', items, index: 0, start: ask_.start }
      return render()
    }
    clearTimeout(fileTimer)
    const token = ++fileToken
    fileTimer = setTimeout(async () => {
      const reply = await ask({ type: 'listFiles', query: ask_.query }, session)
      if (token !== fileToken || session !== tabs.active) return // a newer keystroke, a closed picker or another tab
      picker = { kind: 'file', items: (reply.files ?? []).slice(0, 8).map((f) => ({ label: f, hint: '', value: f })), index: 0, start: ask_.start }
      render()
    }, 120)
  }

  async function openModels() {
    if (picker?.kind === 'model') return closePicker()
    const session = tabs.active
    const reply = await ask({ type: 'listModels' }, session)
    if (session !== tabs.active) return
    const items = (reply.models ?? []).map((m) => ({ label: m.name ?? m.id, hint: `${m.provider}/${m.id}`, value: `${m.provider}/${m.id}` }))
    picker = { kind: 'model', items, index: Math.max(0, items.findIndex((i) => i.value === current().model)), start: 0 }
    render()
  }

  function choose(index) {
    const item = picker?.items[index]
    if (!item) return
    if (picker.kind === 'model') {
      closePicker()
      ask({ type: 'setModel', model: item.value })
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

  function bindInput() {
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
  }

  // ---- clicks ---------------------------------------------------------------------------------

  function answer(id, fields) {
    post({ type: 'uiAnswer', id, ...fields })
    chats.set(tabs.active, answerDialog(current(), id))
    tabs = applyStatus(tabs, tabs.active, current(), true)
    render()
    input.focus()
  }

  root.addEventListener('click', (event) => {
    const el = event.target.closest('[data-chat]')
    if (!el) {
      if (ui.menuOpen) {
        ui.menuOpen = false
        render()
      }
      return
    }
    const { chat, id, value, index, session } = el.dataset
    if (chat === 'send') return submit()
    if (chat === 'stop') return post({ type: 'chatAbort' })
    if (chat === 'settings') return onSettings?.()
    if (chat === 'model') return openModels()
    if (chat === 'pick') return choose(Number(index))
    if (chat === 'select') return switchTo(session)
    if (chat === 'add') return addTab()
    if (chat === 'close-tab') return closeTab(session)
    if (chat === 'undo') return undo()
    if (chat === 'menu') {
      ui.menuOpen = !ui.menuOpen
      return render()
    }
    if (chat === 'toggle-tools') {
      const at = Number(index)
      if (!expanded().delete(at)) expanded().add(at)
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

  root.addEventListener('dblclick', (event) => {
    const main = event.target.closest?.('.tab-main')
    if (!main) return
    ui.renaming = main.dataset.session
    render()
  })

  root.addEventListener('keydown', (event) => {
    const field = event.target?.dataset?.chatRename !== undefined ? event.target : null
    if (!field) return
    if (event.key === 'Enter') commitRename(field.dataset.session, field.value)
    else if (event.key === 'Escape') {
      event.stopPropagation()
      ui.renaming = null
      render()
    }
  })

  root.addEventListener('focusout', (event) => {
    const field = event.target?.dataset?.chatRename !== undefined ? event.target : null
    if (field && ui.renaming === field.dataset.session) commitRename(field.dataset.session, field.value)
  })

  return {
    /** The tabs saved last time (`sessionsLoad`'s reply); creates and saves a first session when there are none. */
    setSessions(saved) {
      tabs = createTabs(saved, makeId)
      if (!saved?.sessions?.length) save()
      render()
    },
    /** What a session said: its chat messages (`chatState`, `chatEvent`, `chatDialog`, …). */
    receive(session, message) {
      if (message.type === 'chatState' && session === tabs.active && message.model !== chatOf(session).model) commands = null
      chats.set(session, reduce(chatOf(session), message))
      tabs = applyStatus(tabs, session, chatOf(session), session === tabs.active)
      render()
    },
    mount,
    show() {
      post({ type: 'chatLoad' })
      render()
      input?.focus()
    },
    render,
    setBanner(html) {
      banner = html
      if (input) $('chat-banner').innerHTML = html
    },
  }
}
