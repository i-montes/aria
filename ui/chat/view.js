// The chat's page: the controller that composes the design's parts (header, thread, composer, pickers,
// cards, panels) and wires them to the bridge, plus the few pure helpers it keeps (tested without a
// browser). Conversations live in `state.js`, the tabs in `sessions.js`, the HTML of each element in
// `parts/` and `thread.js`. Everything dynamic is escaped with `esc`.
import { esc } from '../onboarding/components.js'
import { renderAgentSession } from './parts/session.js'
import { renderCommandPicker, filterCommands, stepActive, clampActive } from './parts/commands.js'
import { renderComposer } from './parts/composer.js'
import { PROVIDERS, renderConnect } from './parts/connect.js'
import { renderFilePicker, filterFiles } from './parts/files.js'
import { indexItems, indexLayout, renderIndex } from './parts/index.js'
import { renderMemory, initialMemoryUi, reduceMemoryUi } from './parts/memory.js'
import { renderModelPicker, orderedModels, providerLabel } from './parts/models.js'
import { questionAnswers, questionInit, questionReduce, normalizeQuestions, renderQuestion } from './parts/question.js'
import { renderQuotaBars } from './parts/quotabars.js'
import { renderSubagents } from './parts/subagents.js'
import { renderSessionTabs } from './parts/tabs.js'
import { renderTasks } from './parts/tasks.js'
import { formatElapsed } from './parts/working.js'
import {
  MAX_SESSIONS, addSession, applyStatus, closeSession, createTabs, renameSession, selectSession, setTitleIfEmpty, toSaved, undoClose,
} from './sessions.js'
import { addNote, addQueued, addUser, answerDialog, tookBack, answersForCard, clearQueue, createChat, reduce } from './state.js'
import { renderThread } from './thread.js'

/** `/home/ana/Projects` -> `~/Projects` (Linux and Mac homes); anything else stays as it is. */
export const shortPath = (path) => String(path ?? '').replace(/^\/(?:home|Users)\/[^/]+(?=\/|$)/, '~')

export function renderPath(state, t = (key) => key) {
  const path = state.workdir
  return path
    ? `<span class="path-icon" aria-hidden="true">▸</span><span class="path-text" title="${esc(path)}">${esc(shortPath(path))}</span>`
    : `<span class="path-text muted">${esc(t('chat.path.unknown'))}</span>`
}


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

/**
 * What the text before the caret asks for: `/model <query>` (the model picker), a slash command at the
 * very start, or an `@` file token at the caret.
 */
export function pickerQuery(text, caret) {
  const before = text.slice(0, caret)
  const model = /^\/model\s+(\S*)$/.exec(before)
  if (model) return { kind: 'model', query: model[1], start: 0 }
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

/** While a rename field is open the tab bar is not redrawn (every streamed delta would rebuild the field and lose what was typed). */
export const keepRenameField = (ui, fieldPresent) => !!ui.renaming && fieldPresent

/** "Closed <title> · Undo", shown for a few seconds after a tab is closed. */
export function renderUndo(tabs, t) {
  if (!tabs.closed) return ''
  const title = tabs.closed.entry.title || t('chat.untitled')
  return `<div class="undo"><span>${esc(t('chat.undo.closed', { title }))}</span><button type="button" class="link" data-chat="undo">${esc(t('chat.undo'))}</button></div>`
}


// ---- the controller ---------------------------------------------------------------------------------

const UNDO_MS = 8000
const TOAST_MS = 8000
const modelLabel = (model) => (model ? model.split('/').pop() : '')

const ICON = {
  plus: '<svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>',
  close: '<svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  gear: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>',
}

/** A selector that finds this control again after its container was redrawn. */
const selectorFor = (el) => {
  const d = el.dataset ?? {}
  if (d.chatField) return `[data-chat-field="${d.chatField}"]`
  if (d.op) return `[data-chat="question"][data-op="${d.op}"]${d.q !== undefined ? `[data-q="${d.q}"]` : ''}`
  return null
}

/** Runs `redraw` (which replaces `container`'s HTML) and gives focus and caret back to the control that had them. */
function keepingFocus(container, redraw) {
  const active = document.activeElement
  const selector = active && container.contains(active) ? selectorFor(active) : null
  const caret = selector ? [active.selectionStart, active.selectionEnd] : null
  redraw()
  if (!selector) return
  const again = container.querySelector(selector)
  if (!again) return
  again.focus()
  try {
    again.setSelectionRange(caret[0], caret[1])
  } catch {
    // not a text field
  }
}

/**
 * `root` is the chat's container. `send(message)` posts to the dock; `request(message)` posts and resolves
 * with the reply; `onSettings()` opens the settings. Every chat message goes out wrapped in
 * `{ type: 'session', session, message }`: each tab is its own session and its own Pi.
 */
export function createChatView({ root, t, send, request, onSettings, makeId = () => crypto.randomUUID() }) {
  let tabs = createTabs(null, makeId)
  const chats = new Map() // session id -> its conversation (state.js)
  const drafts = new Map() // session id -> what was typed and not sent
  const expandedOf = new Map() // session id -> { open, rows, more }: which tool groups and rows are open
  const wizards = new Map() // dialog id -> the question card's wizard state
  const commandsBy = new Map() // session id -> the slash commands its Pi offers
  let picker = null // { kind, items, query, active, start }
  let dismissed = null // { kind, text }: Esc closed the picker for this text
  let settings = null
  const data = { agents: [], tasks: null, tasksKind: 'tasks', memory: [] }
  const ui = { connect: null, menuOpen: false, renaming: null, view: 'chat', agent: null, memory: initialMemoryUi(), tasksOpen: true, index: { expanded: false, my: null, off: 0, height: 360, settled: false } }
  let toast = null
  let connectToken = 0 // a connection in progress that was closed or replaced must not act when its reply arrives
  let undoTimer = null
  let toastTimer = null
  let tick = null
  let pendingClose = null // a closed tab whose Pi is only stopped when its undo window is over
  let fileToken = 0
  let fileTimer = null
  let composerKey = ''
  let headerKey = ''
  let indexKey = ''
  let questionKey = ''
  let input = null
  let banner = ''

  const chatOf = (id) => chats.get(id) ?? createChat()
  const current = () => chatOf(tabs.active)
  const expanded = () => {
    if (!expandedOf.has(tabs.active)) expandedOf.set(tabs.active, { open: new Set(), rows: new Set(), more: new Set() })
    return expandedOf.get(tabs.active)
  }
  const post = (message, session = tabs.active) => send({ type: 'session', session, message })
  const ask = (message, session = tabs.active) => request({ type: 'session', session, message })
  const save = () => send({ type: 'sessionsSave', ...toSaved(tabs) })
  const $ = (id) => root.querySelector(`#${id}`)
  const assistantName = () => (settings?.assistant?.name || '').trim() || 'Aria'
  const activeTab = () => tabs.sessions.find((s) => s.id === tabs.active)

  /** Builds the chat's HTML. Called once the texts are loaded (and again when the language changes). */
  function mount() {
    root.innerHTML = `
    <header class="chat-header" id="chat-header"></header>
    <div id="chat-undo"></div>
    <div class="chat-path-row">
      <div class="chat-path" id="chat-path"></div>
      <button type="button" class="icon-btn" data-chat="settings" data-nav aria-label="${esc(t('chat.settings'))}" title="${esc(t('chat.settings'))}">${ICON.gear}</button>
    </div>
    <div id="chat-banner"></div>
    <div id="chat-problem"></div>
    <div class="chat-body" id="chat-body" data-view="chat">
      <section class="chat-view chat-view-chat" id="view-chat">
        <div class="chat-stage">
          <div class="chat-list" id="chat-list"></div>
          <div class="chat-index" id="chat-index"></div>
          <div class="chat-question" id="chat-question"></div>
        </div>
        <div id="chat-tasks"></div>
        <div id="chat-dialogs"></div>
        <div class="chat-overlay" id="chat-overlay"></div>
        <div id="chat-composer"></div>
      </section>
      <section class="chat-view" id="view-agents" hidden></section>
      <section class="chat-view" id="view-session" hidden></section>
      <section class="chat-view" id="view-memory" hidden></section>
    </div>
    <div id="chat-toast"></div>`
    input = null
    composerKey = ''
    headerKey = ''
    indexKey = ''
    questionKey = ''
    clearInterval(tick)
    tick = setInterval(tickElapsed, 200)
    render()
  }

  function tickElapsed() {
    if (!input) return
    for (const el of root.querySelectorAll('.p-working-time[data-since]')) {
      el.textContent = formatElapsed(Date.now() - Number(el.dataset.since))
    }
  }

  // ---- drawing --------------------------------------------------------------------------------

  function drawHeader() {
    const key = JSON.stringify([
      tabs.sessions.map((s) => [s.id, s.title, s.status, s.unread, chatOf(s.id).model, chatOf(s.id).context, chatOf(s.id).quota]),
      tabs.active, ui.menuOpen, ui.renaming, assistantName(), settings?.compaction?.percent, current().quota,
    ])
    if (key === headerKey && $('chat-header').firstChild) return
    headerKey = key
    const multi = tabs.sessions.length > 1
    const full = tabs.sessions.length >= MAX_SESSIONS
    // Each tab's hover card is the quota card, fed with that session's own model, context and quota windows.
    const sessions = tabs.sessions.map((s) => {
      const chat = chatOf(s.id)
      const model = chat.model ?? ''
      return {
        ...s,
        model: modelLabel(model),
        context: chat.context ?? 0,
        provider: providerLabel(model.split('/')[0]),
        quotaWindows: chat.quota,
        free: /free$/.test(model),
      }
    })
    const left = multi
      ? renderSessionTabs({ ...tabs, sessions }, { maxVisible: 3, hideNew: true, hideUndo: true, menuOpen: ui.menuOpen, renaming: ui.renaming }, t)
      : `<div class="chat-identity"><span class="chat-dot" aria-hidden="true"></span><span class="chat-name">${esc(assistantName())}</span>${renderQuotaBars({ windows: current().quota, provider: providerLabel((current().model ?? '').split('/')[0]), facts: { model: modelLabel(current().model), context: current().context } }, {}, t)}</div>`
    $('chat-header').innerHTML = `${left}
      <button type="button" class="icon-btn chat-new" data-chat="add" ${full ? 'disabled' : ''} aria-label="${esc(t('chat.new'))}">${ICON.plus}</button>
      <button type="button" class="icon-btn" data-chat="close-dock" aria-label="${esc(t('chat.close'))}" title="${esc(t('chat.close'))}">${ICON.close}</button>`
  }

  function drawThread() {
    const state = current()
    const list = $('chat-list')
    const nearBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 80
    const ex = expanded()
    const connectHtml = ui.connect ? `<div class="thread-item" id="chat-connect">${renderConnect(ui.connect, {}, t)}</div>` : ''
    keepingFocus(list, () => {
      list.innerHTML = renderThread(state, { expanded: ex.open, openRows: ex.rows, moreOpen: ex.more, now: Date.now() }, t) + connectHtml
    })
    if (nearBottom || (ui.connect && ui.connect.justOpened)) list.scrollTop = list.scrollHeight
    if (ui.connect) ui.connect.justOpened = false
  }

  function drawIndex() {
    const items = indexItems(current().messages)
    const el = $('chat-index')
    ui.index.height = $('chat-list').clientHeight || ui.index.height
    const key = JSON.stringify([tabs.active, items.map((i) => i.id), ui.activeId, ui.index])
    if (key === indexKey) return
    indexKey = key
    el.innerHTML = items.length > 1 ? renderIndex({ items, activeId: ui.activeId }, ui.index, t) : ''
  }

  /** `force`: the wizard changed. Otherwise the card is only drawn when another one shows up. */
  function drawQuestion(force = false) {
    const state = current()
    const dialog = state.dialogs.find((d) => d.card?.kind === 'question')
    const el = $('chat-question')
    const key = `${tabs.active}:${dialog?.id ?? ''}`
    if (!force && key === questionKey) return
    questionKey = key
    if (!dialog) {
      el.innerHTML = ''
      return
    }
    if (!wizards.has(dialog.id)) wizards.set(dialog.id, questionInit(dialog.card.questions))
    keepingFocus(el, () => {
      el.innerHTML = `<div class="question-fill">${renderQuestion({ id: dialog.id, questions: dialog.card.questions }, wizards.get(dialog.id), t)}</div>`
    })
  }

  function drawDialogs() {
    const state = current()
    const plain = state.dialogs.filter((d) => !d.card)
    const key = `${tabs.active}:${plain.map((d) => d.id).join('|')}`
    const el = $('chat-dialogs')
    if (el.dataset.key === key) return
    el.dataset.key = key
    el.innerHTML = plain.map((d) => renderDialog(d, t)).join('')
    // Never take the keyboard from someone who is typing: the next Space or Enter would answer the dialog.
    if (!(input?.value) && plain.length) el.querySelector('button, input, textarea')?.focus()
  }

  function drawOverlay() {
    const el = $('chat-overlay')
    if (!picker) {
      el.innerHTML = ''
      return
    }
    const props = { items: picker.items, query: picker.query, active: picker.active, open: true }
    keepingFocus(el, () => {
      el.innerHTML =
        picker.kind === 'command' ? renderCommandPicker(props, {}, t)
        : picker.kind === 'file' ? renderFilePicker(props, {}, t)
        : renderModelPicker({ models: picker.items, current: current().model ?? '', query: picker.query, active: picker.active, open: true }, {}, t)
    })
    el.querySelector('.p-cmd-row.on, .p-files-row.on, .p-models-row.on')?.scrollIntoView?.({ block: 'nearest' })
  }

  function drawTasks() {
    const el = $('chat-tasks')
    el.innerHTML = data.tasks?.length ? renderTasks({ tasks: data.tasks, kind: data.tasksKind, show: true }, { open: ui.tasksOpen }, t) : ''
  }

  /** The composer is drawn once; later changes (busy, text, model) swap its buttons, rows and hints in place so the
   * textarea keeps its value, caret, undo history and any composition in progress. */
  function drawComposer() {
    const el = $('chat-composer')
    const value = input ? input.value : (drafts.get(tabs.active) ?? '')
    const model = current().model
    const props = {
      busy: current().running,
      hasText: value.trim().length > 0,
      rows: Math.min(6, Math.max(1, value.split('\n').length)),
      showHints: true,
      model: modelLabel(model) || undefined,
      modelTitle: model ?? undefined,
    }
    const key = JSON.stringify(props)
    if (input && key === composerKey) return
    composerKey = key
    if (!input) {
      el.innerHTML = renderComposer(props, {}, t)
      input = $('chat-input')
      input.value = value
      return
    }
    const next = document.createElement('div')
    next.innerHTML = renderComposer(props, {}, t)
    for (const selector of ['.p-composer-actions', '.p-composer-hints']) {
      const old = el.querySelector(selector)
      const fresh = next.querySelector(selector)
      if (old && fresh) old.replaceWith(fresh)
      else if (old) old.remove()
      else if (fresh) el.querySelector('.p-composer')?.appendChild(fresh)
    }
    input.rows = props.rows
    input.placeholder = next.querySelector('textarea')?.placeholder ?? ''
  }

  function drawViews() {
    const view = ui.view
    $('chat-body').dataset.view = view
    for (const name of ['chat', 'agents', 'session', 'memory']) $(`view-${name}`).hidden = name !== view
    if (view === 'chat') return
    if (view === 'agents') {
      $('view-agents').innerHTML = `
        <div class="chat-panel-scroll">
          ${data.agents.length ? renderSubagents({ agents: data.agents, footnote: t('chat.agents.footnote') }, {}, t) : `<p class="chat-panel-empty">${esc(t('chat.agents.empty'))}</p>`}
        </div>
        <button type="button" class="chat-back" data-chat="panel-back"><kbd>Alt</kbd><kbd>A</kbd><span>${esc(t('chat.agents.back'))}</span></button>`
    } else if (view === 'session') {
      const agent = data.agents.find((a) => (a.id ?? a.role) === ui.agent) ?? data.agents[0]
      $('view-session').innerHTML = `<div class="chat-panel-scroll">${agent ? renderAgentSession({ agent }, { now: Date.now() }, t) : ''}</div>`
    } else if (view === 'memory') {
      keepingFocus($('view-memory'), () => {
        $('view-memory').innerHTML = renderMemory({ memories: data.memory, now: Date.now() }, ui.memory, t)
      })
    }
  }

  function drawToast() {
    $('chat-toast').innerHTML = toast
      ? `<div class="chat-toast" role="status"><span class="chat-dot" aria-hidden="true"></span><div><b>“${esc(toast.title)}”</b><span>${esc(toast.text)}</span></div><button type="button" class="chat-toast-open" data-chat="toast-open" data-session="${esc(toast.session)}">${esc(t('chat.toast.open'))}</button></div>`
      : ''
  }

  function render() {
    if (!$('chat-body')) return
    if (!keepRenameField(ui, !!root.querySelector('[data-chat-rename]'))) drawHeader()
    $('chat-undo').innerHTML = renderUndo(tabs, t)
    $('chat-path').innerHTML = renderPath(current(), t)
    $('chat-problem').innerHTML = renderProblem(current(), t)
    $('chat-banner').innerHTML = banner
    drawThread()
    drawIndex()
    drawQuestion()
    drawDialogs()
    drawOverlay()
    drawTasks()
    drawComposer()
    drawViews()
    drawToast()
    if (ui.renaming) root.querySelector('[data-chat-rename]')?.focus()
  }

  // ---- views -----------------------------------------------------------------------------------

  function openView(view) {
    ui.view = view
    if (view === 'memory') ui.memory = initialMemoryUi()
    render()
    if (view === 'chat') input?.focus()
    else $(`view-${view}`).querySelector('button, input')?.focus()
  }

  // ---- tabs ------------------------------------------------------------------------------------

  /** Shows another session: its draft, its conversation, and a `chatLoad` so its Pi (and any dialog) wakes up. */
  function switchTo(next) {
    if (next === tabs.active) return
    if (input) drafts.set(tabs.active, input.value)
    closePicker(false)
    tabs = selectSession(tabs, next)
    ui.menuOpen = false
    ui.view = 'chat'
    if (input) input.value = drafts.get(next) ?? ''
    save()
    post({ type: 'chatLoad' })
    render()
    input?.focus()
  }

  function addTab() {
    if (tabs.sessions.length >= MAX_SESSIONS) return
    commitClose() // adding a session ends the undo window
    const id = makeId()
    if (input) drafts.set(tabs.active, input.value)
    tabs = addSession(tabs, id)
    if (tabs.active !== id) return
    closePicker(false)
    if (input) input.value = ''
    ui.menuOpen = false
    ui.view = 'chat'
    save()
    post({ type: 'chatLoad' })
    render()
    input?.focus()
  }

  /** The undo window is over (or something else happened): the closed tab's Pi stops and its state goes. */
  function commitClose() {
    clearTimeout(undoTimer)
    if (!pendingClose) return
    const id = pendingClose
    pendingClose = null
    send({ type: 'sessionClose', session: id })
    chats.delete(id)
    drafts.delete(id)
    expandedOf.delete(id)
  }

  function closeTab(id) {
    const wasActive = id === tabs.active
    const before = tabs
    commitClose() // an earlier close can no longer be undone
    tabs = closeSession(tabs, id)
    if (tabs === before) return
    ui.menuOpen = false
    pendingClose = id // its Pi keeps running (and can be brought back) until the window ends
    undoTimer = setTimeout(() => {
      commitClose()
      tabs = { ...tabs, closed: null }
      render()
    }, UNDO_MS)
    if (wasActive) {
      closePicker(false)
      if (input) input.value = drafts.get(tabs.active) ?? ''
      post({ type: 'chatLoad' })
    }
    save()
    render()
  }

  function undo() {
    clearTimeout(undoTimer)
    pendingClose = null // nothing was stopped: the tab simply comes back
    tabs = undoClose(tabs)
    save()
    render()
  }

  function commitRename(id, title) {
    ui.renaming = null
    tabs = renameSession(tabs, id, title)
    save()
    render()
    input?.focus()
  }

  // ---- composer --------------------------------------------------------------------------------

  function clearDraft() {
    input.value = ''
    drafts.delete(tabs.active)
    closePicker(false)
    composerKey = ''
  }

  function submit() {
    const text = input.value.trim()
    if (!text) return
    if (/^\/memory(\s|$)/.test(text)) {
      clearDraft()
      return openView('memory')
    }
    const connect = /^\/connect(?:\s+(\S+))?\s*$/.exec(text)
    if (connect) {
      clearDraft()
      drawComposer()
      return openConnect(connect[1])
    }
    if (/^\/model(\s|$)/.test(text)) return // the model picker answers this one
    const session = tabs.active
    const chat = chatOf(session)
    // Extension commands (and Aria's own) run at once, even during a run; a skill or a prompt template is an ordinary
    // message for Pi and has to wait in the queue like one.
    const name = text.startsWith('/') ? text.slice(1).split(/\s/)[0] : null
    const command = name !== null ? commandsBy.get(session)?.find((c) => c.name === name) : null
    const immediate = name !== null && (!command || command.source === 'extension' || command.source === 'aria')
    if (chat.running && !immediate) {
      post({ type: 'chatSend', text, behavior: 'followUp' }, session)
      chats.set(session, addQueued(chat, text))
    } else {
      post({ type: 'chatSend', text, behavior: null }, session)
      chats.set(session, addUser(chat, text))
    }
    const before = tabs
    tabs = setTitleIfEmpty(tabs, session, text)
    if (tabs !== before) save()
    clearDraft()
    render()
    const list = $('chat-list')
    list.scrollTop = list.scrollHeight
  }

  function stop() {
    post({ type: 'chatAbort' })
  }

  function closePicker(draw = true) {
    const hadModelPicker = picker?.kind === 'model'
    picker = null
    fileToken++
    if (draw) drawOverlay()
    if (hadModelPicker) input?.focus() // the search field is gone: the keyboard goes back to the composer
  }

  /** The model picker opens at once (so Esc works even while Pi is still answering) and fills in when the list arrives. */
  async function openModelPicker(query = '') {
    const session = tabs.active
    picker = { kind: 'model', items: [], query, active: 0, start: 0 }
    drawOverlay()
    $('chat-overlay').querySelector('[data-chat-field="model-search"]')?.focus()
    const reply = await ask({ type: 'listModels' }, session)
    if (picker?.kind !== 'model' || session !== tabs.active) return // closed or moved on meanwhile
    picker = { ...picker, items: reply.models ?? [] }
    drawOverlay()
  }

  const pickerSize = () => {
    if (!picker) return 0
    const { kind, items, query } = picker
    return kind === 'command' ? filterCommands(items, query).length : kind === 'file' ? filterFiles(items, query).length : orderedModels(items, query).length
  }

  async function updatePicker() {
    const asked = pickerQuery(input.value, input.selectionStart ?? input.value.length)
    if (!asked) {
      dismissed = null
      return closePicker()
    }
    if (dismissed && dismissed.kind === asked.kind && input.value.startsWith(dismissed.text)) return closePicker()
    const session = tabs.active
    // A reply that arrives after the owner changed tab or edited the text no longer applies.
    const stale = () => {
      if (session !== tabs.active) return true
      const now = pickerQuery(input.value, input.selectionStart ?? input.value.length)
      return !now || now.kind !== asked.kind || (now.kind !== 'file' && now.query !== asked.query)
    }
    const local = () => [
      { name: 'memory', description: t('chat.cmd.memory'), source: 'aria' },
      { name: 'model', description: t('chat.cmd.model'), source: 'aria' },
      { name: 'connect', description: t('chat.cmd.connect'), source: 'aria' },
    ]
    if (asked.kind === 'command') {
      let items = commandsBy.get(session)
      if (!items) {
        const reply = await ask({ type: 'listCommands' }, session)
        items = [...local(), ...(reply.commands ?? [])]
        // An empty list means Pi could not answer (not started yet, or busy): do not remember that.
        if ((reply.commands ?? []).length) commandsBy.set(session, items)
      }
      if (stale()) return
      picker = { kind: 'command', items, query: asked.query, active: 0, start: asked.start }
      return drawOverlay()
    }
    if (asked.kind === 'model') {
      // "/model " opens the model picker, which has its own search field: the text leaves the composer.
      clearDraft()
      drawComposer()
      return openModelPicker(asked.query)
    }
    clearTimeout(fileTimer)
    const token = ++fileToken
    fileTimer = setTimeout(async () => {
      const reply = await ask({ type: 'listFiles', query: asked.query }, session)
      if (token !== fileToken || session !== tabs.active) return // a newer keystroke, a closed picker or another tab
      picker = { kind: 'file', items: reply.files ?? [], query: asked.query, active: 0, start: asked.start }
      drawOverlay()
    }, 120)
  }

  function choose(index) {
    if (!picker) return
    const { kind, items, query } = picker
    if (kind === 'model') {
      const hit = orderedModels(items, query)[index]?.model
      if (!hit) return
      closePicker()
      ask({ type: 'setModel', model: `${hit.provider}/${hit.id}` })
      return
    }
    const caret = input.selectionStart ?? input.value.length
    if (kind === 'command') {
      const hit = filterCommands(items, query)[index]?.item
      if (!hit) return
      input.value = `/${hit.name} ${input.value.slice(caret)}`
      input.selectionStart = input.selectionEnd = hit.name.length + 2
    } else {
      const hit = filterFiles(items, query)[index]
      if (!hit) return
      const next = insertFile(input.value, caret, picker.start, hit.path)
      input.value = next.text
      input.selectionStart = input.selectionEnd = next.caret
    }
    closePicker(false)
    composerKey = ''
    drawOverlay()
    drawComposer()
    input.focus()
  }

  function onComposerKey(event) {
    if (event.isComposing) return // Escape belongs to `escape()` (called by the page), which also ignores a composition
    if (picker) {
      const size = pickerSize()
      if (size && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
        event.preventDefault()
        event.stopPropagation()
        picker = { ...picker, active: stepActive(picker.active, size, event.key === 'ArrowDown' ? 1 : -1) }
        return drawOverlay()
      }
      if (size && (event.key === 'Tab' || (event.key === 'Enter' && !event.shiftKey))) {
        event.preventDefault()
        return choose(clampActive(picker.active, size))
      }
    }
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.stopPropagation() // the arrows move the caret here, not focus between controls
    }
  }

  // ---- connecting a provider (`/connect`) --------------------------------------------------------------

  const KEY_PROVIDER = { anthropic: 'anthropic-api', 'opencode-go': 'opencode-go' }

  function redrawConnect() {
    drawThread()
  }

  /** `/connect [provider]`: the card opens at the end of the thread. */
  function openConnect(arg) {
    const known = PROVIDERS.some((p) => p.id === arg)
    connectToken++
    ui.connect = { provider: known ? arg : null, step: !known ? 'pick' : arg === 'anthropic' ? 'anthropic' : 'key', error: null, keyDraft: '', revealed: false, justOpened: true }
    render()
    $('chat-list').querySelector('[data-chat="connect"], input')?.focus()
  }

  function closeConnect() {
    connectToken++
    ui.connect = null
    drawThread()
    input?.focus()
  }

  const connectFailed = (code) => {
    // Back to the step the owner can retry from, with the reason.
    const back = ui.connect.provider === 'anthropic' && ui.connect.step !== 'key' ? 'anthropic' : 'key'
    ui.connect = { ...ui.connect, step: back, error: code ?? 'generic' }
    redrawConnect()
  }

  /** The provider is connected: every Pi starts again (so it sees the credential and the package) and the card says so. */
  function connected() {
    const label = PROVIDERS.find((p) => p.id === ui.connect.provider)?.label ?? ''
    send({ type: 'enginesRestart' })
    chats.set(tabs.active, addNote(current(), 'aria', t('chat.connect.done', { provider: label })))
    ui.connect = { ...ui.connect, step: 'done', error: null }
    render()
    post({ type: 'chatLoad' })
  }

  async function saveConnectKey() {
    const c = ui.connect
    const token = connectToken
    const key = (c.keyDraft ?? '').trim()
    if (!key) return connectFailed('key-rejected')
    ui.connect = { ...c, error: null }
    const reply = await request({ type: 'saveKey', provider: KEY_PROVIDER[c.provider], key })
    if (token !== connectToken) return
    if (reply.type === 'keyResult' && reply.ok) return connected()
    connectFailed(reply.error ?? reply.code)
  }

  async function signInWithClaude() {
    const token = connectToken
    ui.connect = { ...ui.connect, step: 'waiting', error: null, url: null }
    redrawConnect()
    const reply = await request({ type: 'startOauth' }, (progress) => {
      if (token === connectToken && progress.url) {
        ui.connect = { ...ui.connect, url: progress.url }
        redrawConnect()
      }
    })
    if (token !== connectToken) return
    if (!(reply.type === 'oauthResult' && reply.ok)) return connectFailed(reply.error ?? reply.code)
    ui.connect = { ...ui.connect, step: 'installing' }
    redrawConnect()
    const installed = await request({ type: 'installClaudeBridge' })
    if (token !== connectToken) return
    if (installed.type !== 'saved') return connectFailed(installed.code)
    connected()
  }

  function connectOp(el) {
    const c = ui.connect
    if (!c) return
    const { op, value, url } = el.dataset
    if (op === 'close') return closeConnect()
    if (op === 'provider') {
      ui.connect = { ...c, provider: value, step: value === 'anthropic' ? 'anthropic' : 'key', error: null }
      return redrawConnect()
    }
    if (op === 'method') {
      if (value === 'oauth') return signInWithClaude()
      ui.connect = { ...c, step: 'key', error: null }
      return redrawConnect()
    }
    if (op === 'back') {
      ui.connect = { ...c, step: c.step === 'key' && c.provider === 'anthropic' ? 'anthropic' : 'pick', error: null }
      return redrawConnect()
    }
    if (op === 'reveal') {
      ui.connect = { ...c, revealed: !c.revealed }
      return redrawConnect()
    }
    if (op === 'save') return saveConnectKey()
    if (op === 'open-url') return send({ type: 'openUrl', url })
    if (op === 'models') {
      const query = c.provider === 'anthropic' ? 'claude' : ''
      closeConnect()
      return openModelPicker(query)
    }
  }

  // ---- answers ---------------------------------------------------------------------------------

  function answer(id, fields, forNote) {
    post({ type: 'uiAnswer', id, ...fields })
    chats.set(tabs.active, answerDialog(current(), id, forNote))
    wizards.delete(id)
    tabs = applyStatus(tabs, tabs.active, current(), true)
    render()
    input?.focus()
  }

  let autoAdvance = null

  /** One step of a question card's wizard (`op` is the part's `data-op`). */
  function questionStep(id, op, q, value) {
    const state = wizards.get(id)
    const dialog = current().dialogs.find((d) => d.id === id)
    if (!state || !dialog) return
    const list = normalizeQuestions(dialog.card.questions)
    clearTimeout(autoAdvance)
    const next = questionReduce(state, op, { q, value })
    wizards.set(id, next)
    if (next.outcome === 'submit') {
      const given = questionAnswers(list, next)
      return answer(id, { value: JSON.stringify(given) }, answersForCard(list, given))
    }
    if (next.outcome === 'dismiss') return answer(id, { cancelled: true }, undefined)
    drawQuestion(true)
    // A single-choice pick moves on by itself after a beat (and sends, on the last question), as the design does.
    if (op === 'pick' && !list[q]?.multi) {
      autoAdvance = setTimeout(() => {
        const cur = wizards.get(id)
        if (cur && cur.step === q) questionStep(id, q < list.length - 1 ? 'next' : 'submit')
      }, 480)
    }
  }

  function onQuestion(el) {
    const { id, op, q, value } = el.dataset
    if (op === 'other') return // typing in the free-text field is handled by the input event, not by clicks
    questionStep(id, op, q === undefined ? undefined : Number(q), value)
  }

  // ---- the thread's scroll and index -------------------------------------------------------------

  // As the design's message index, but it opens only when the pointer is over the thread's scrollbar (its last 8 px),
  // follows the pointer while it stays over the chips' area (300 px), and closes shortly after it leaves. The thread's
  // own scrollbar stays usable: nothing of the index takes the pointer except the chips themselves.
  const INDEX_OPEN_ZONE = 8 // just the thread's scrollbar (6 px wide) and a hair around it
  const INDEX_KEEP_ZONE = 300
  const INDEX_HIDE_MS = 160
  let indexHide = null
  let indexSettle = null
  let indexSuppressed = false

  function updateActive() {
    const items = indexItems(current().messages)
    if (!items.length) return
    const top = $('chat-list').getBoundingClientRect().top
    let at = -1
    for (let i = 0; i < items.length && at === -1; i++) {
      const el = document.getElementById(items[i].id)
      if (el && el.getBoundingClientRect().bottom > top + 8) at = i // the first message still reaching below the top edge
    }
    const id = items[at === -1 ? items.length - 1 : at]?.id
    if (id !== ui.activeId) {
      ui.activeId = id
      drawIndex()
    }
  }

  root.addEventListener('scroll', (event) => {
    if (event.target?.id === 'chat-list') updateActive()
  }, true)

  const indexGeometry = () => indexLayout(indexItems(current().messages).length, { ...ui.index, expanded: true })

  function showIndex(my) {
    clearTimeout(indexHide)
    const height = $('chat-list').clientHeight || ui.index.height
    const items = indexItems(current().messages)
    if (!ui.index.expanded) {
      // Opening: when the chips do not all fit, start with the active one in the middle.
      const g = indexLayout(items.length, { height, expanded: true })
      const at = Math.max(0, items.findIndex((i) => i.id === ui.activeId))
      const off = g.overflow ? Math.min(g.maxOff, Math.max(0, 8 + at * 34 + 15 - height / 2)) : 0
      ui.index = { ...ui.index, expanded: true, settled: false, my, off, height }
      clearTimeout(indexSettle)
      indexSettle = setTimeout(() => {
        ui.index.settled = true
      }, 450)
    } else {
      ui.index = { ...ui.index, my, height }
    }
    drawIndex()
  }

  function scheduleHideIndex() {
    if (!ui.index.expanded) return
    clearTimeout(indexHide)
    indexHide = setTimeout(() => {
      ui.index = { ...ui.index, expanded: false, my: null, settled: false }
      drawIndex()
    }, INDEX_HIDE_MS)
  }

  function hideIndexNow() {
    clearTimeout(indexHide)
    clearTimeout(indexSettle)
    ui.index = { ...ui.index, expanded: false, my: null, settled: false }
    drawIndex()
  }

  root.addEventListener('mousemove', (event) => {
    const list = $('chat-list')
    if (!list || ui.view !== 'chat' || indexItems(current().messages).length < 2) return
    const rect = list.getBoundingClientRect()
    const inside = event.clientY >= rect.top && event.clientY <= rect.bottom && event.clientX >= rect.left && event.clientX <= rect.right
    const fromRight = rect.right - event.clientX
    if (inside && ((ui.index.expanded && fromRight <= INDEX_KEEP_ZONE) || (!indexSuppressed && fromRight <= INDEX_OPEN_ZONE))) {
      showIndex(event.clientY - rect.top)
    } else {
      if (!inside || fromRight > INDEX_OPEN_ZONE) indexSuppressed = false
      scheduleHideIndex()
    }
  })

  // With the index open and more chips than fit, the wheel scrolls the chips; otherwise it scrolls the thread.
  root.addEventListener(
    'wheel',
    (event) => {
      if (!ui.index.expanded) return
      const rect = $('chat-list').getBoundingClientRect()
      if (rect.right - event.clientX > INDEX_KEEP_ZONE) return
      const g = indexGeometry()
      if (!g.overflow) return
      event.preventDefault()
      const dy = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY
      ui.index = { ...ui.index, off: Math.min(g.maxOff, Math.max(0, (ui.index.off ?? 0) + dy)) }
      drawIndex()
    },
    { passive: false },
  )

  /** Jump to a message you sent: the thread scrolls to it, and it gives a soft halo once it is there. */
  function jumpTo(id) {
    const el = document.getElementById(id)
    if (!el) return
    indexSuppressed = true // stay closed until the pointer leaves the edge and comes back
    hideIndexNow()
    const list = $('chat-list')
    list.scrollBy({ top: el.getBoundingClientRect().top - list.getBoundingClientRect().top - 12, behavior: 'smooth' })
    let quiet = null
    const finish = () => {
      list.removeEventListener('scroll', onScroll)
      clearTimeout(quiet)
      clearTimeout(cap)
      flash(el)
    }
    const onScroll = () => {
      clearTimeout(quiet)
      quiet = setTimeout(finish, 110)
    }
    list.addEventListener('scroll', onScroll, { passive: true })
    quiet = setTimeout(finish, 160)
    const cap = setTimeout(finish, 1400)
  }

  function flash(el) {
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return
    const bubble = el.querySelector('.p-user-bubble, p') ?? el
    try {
      bubble.animate(
        [
          { transform: 'scale(1)', transformOrigin: 'right center', boxShadow: '0 0 0 0 rgba(185, 168, 255, 0.5)', offset: 0 },
          { transform: 'scale(1.035)', transformOrigin: 'right center', boxShadow: '0 0 0 4px rgba(185, 168, 255, 0.28)', offset: 0.3 },
          { transform: 'scale(1)', transformOrigin: 'right center', boxShadow: '0 0 0 12px rgba(185, 168, 255, 0)', offset: 1 },
        ],
        { duration: 850, easing: 'ease-in-out' },
      )
    } catch {
      // animations unavailable: the jump alone is enough
    }
  }

  // ---- events ----------------------------------------------------------------------------------

  root.addEventListener('input', (event) => {
    const el = event.target
    if (el === input) {
      drawComposer()
      updatePicker()
    } else if (el?.dataset?.chatField === 'connect-key' && ui.connect) {
      ui.connect.keyDraft = el.value // no redraw: the field keeps its focus and caret
    } else if (el?.dataset?.chatField === 'model-search' && picker?.kind === 'model') {
      picker = { ...picker, query: el.value, active: 0 }
      drawOverlay()
    } else if (el?.dataset?.op === 'other') {
      const id = el.dataset.id
      const state = wizards.get(id)
      if (state) {
        wizards.set(id, questionReduce(state, 'other', { q: Number(el.dataset.q), value: el.value }))
        drawQuestion(true) // the Continue button depends on whether something is picked or typed
      }
    } else if (el?.dataset?.chatField === 'memory-search') {
      ui.memory = reduceMemoryUi(ui.memory, 'search', el.value)
      drawViews()
    }
  })

  root.addEventListener('keydown', (event) => {
    const field = event.target?.dataset?.chatRename !== undefined ? event.target : null
    if (field) {
      if (event.key === 'Enter' && !event.isComposing) commitRename(field.dataset.session, field.value)
      else if (event.key === 'Escape') {
        event.stopPropagation()
        ui.renaming = null
        render()
      }
      return
    }
    if (event.target === input) return onComposerKey(event)
    if (event.target?.dataset?.chatField === 'model-search' && picker?.kind === 'model') {
      const size = pickerSize()
      if (size && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
        event.preventDefault()
        picker = { ...picker, active: stepActive(picker.active, size, event.key === 'ArrowDown' ? 1 : -1) }
        return drawOverlay()
      }
      if (size && (event.key === 'Enter' || event.key === 'Tab') && !event.isComposing) {
        event.preventDefault()
        return choose(clampActive(picker.active, size))
      }
      return // Escape is not handled here: the page calls escape(), which closes just this picker
    }
    if (event.target?.dataset?.chatField === 'connect-key' && event.key === 'Enter' && !event.isComposing) {
      event.preventDefault()
      return saveConnectKey()
    }
        if (event.target?.dataset?.op === 'other' && event.key === 'Enter' && !event.isComposing) {
      event.preventDefault()
      const { id, q } = event.target.dataset
      const dialog = current().dialogs.find((d) => d.id === id)
      const last = Number(q) >= normalizeQuestions(dialog?.card?.questions).length - 1
      return questionStep(id, last ? 'submit' : 'next', Number(q))
    }
  })

  root.addEventListener('focusout', (event) => {
    const field = event.target?.dataset?.chatRename !== undefined ? event.target : null
    if (field && ui.renaming === field.dataset.session) commitRename(field.dataset.session, field.value)
  })

  root.addEventListener('dblclick', (event) => {
    const main = event.target.closest?.('.p-tabs-main')
    if (!main) return
    ui.renaming = main.dataset.session
    drawHeader()
    render()
  })

  root.addEventListener('click', (event) => {
    const el = event.target.closest('[data-chat]')
    if (!el) {
      if (ui.menuOpen) {
        ui.menuOpen = false
        drawHeader()
      }
      return
    }
    const { chat: action, id, session, index, choice } = el.dataset
    switch (action) {
      case 'send': return submit()
      case 'stop': return stop()
      case 'settings': return onSettings?.()
      case 'close-dock': return send({ type: 'closeRequest' })
      case 'open-models':
        if (ui.view !== 'chat') openView('chat')
        return picker?.kind === 'model' ? closePicker() : openModelPicker()
      case 'pick': return choose(Number(index))
      case 'picker-hover':
        if (picker) picker = { ...picker, active: Number(index) }
        return
      case 'select': return switchTo(session)
      case 'add': return addTab()
      case 'close-tab': return closeTab(session)
      case 'undo': return undo()
      case 'menu':
        ui.menuOpen = !ui.menuOpen
        return drawHeader()
      case 'cancel-queued': {
        // Pi can only empty its queue: the other queued messages go back in, in order.
        const at = Number(id)
        const keep = current().queue.filter((_, i) => i !== at).map((q) => q.text)
        post({ type: 'chatClearQueue', keep })
        chats.set(tabs.active, clearQueue(current(), at))
        return render()
      }
      case 'toggle-tools': {
        const set = expanded().open
        if (!set.delete(Number(index))) set.add(Number(index))
        return drawThread()
      }
      case 'toggle-tool-row': {
        const set = expanded().rows
        if (!set.delete(id)) set.add(id)
        return drawThread()
      }
      case 'toggle-tool-diffs': {
        const set = expanded().more
        if (!set.delete(Number(index))) set.add(Number(index))
        return drawThread()
      }
      case 'index-jump': return jumpTo(id)
      case 'permission': return answer(id, { value: choice }, choice)
      // The note says what was decided in the chat's words (free / keep), whatever string the extension used.
      case 'quota': return answer(id, { value: choice }, el.dataset.role || choice)
      case 'question': return onQuestion(el)
      case 'connect': return connectOp(el)
      case 'tasks-toggle':
        ui.tasksOpen = !ui.tasksOpen
        return drawTasks()
      case 'agent-select':
        ui.agent = id
        return openView('session')
      case 'agent-back': return openView('agents')
      case 'panel-back': return openView('chat')
      case 'memory-back': return openView('chat')
      case 'toast-open':
        toast = null
        return switchTo(session)
      case 'answer': return answer(id, { value: el.dataset.value }, el.dataset.value)
      case 'confirm': return answer(id, { confirmed: el.dataset.value === 'true' }, undefined)
      case 'cancel': return answer(id, { cancelled: true }, undefined)
      case 'submit': {
        const field = [...root.querySelectorAll('[data-chat-field]')].find((f) => f.dataset.id === id)
        return answer(id, { value: field?.value ?? '' }, undefined)
      }
      default:
        if (action?.startsWith('memory-')) {
          ui.memory = reduceMemoryUi(ui.memory, action.slice('memory-'.length), el.dataset.value ?? id)
          return drawViews()
        }
    }
  })

  // A row is selected by the pointer MOVING onto it, never by it resting there: after the keyboard moves the selection the
  // list is redrawn, the browser re-sends hover for the row under a pointer that did not move, and that must not win.
  let pointerAt = { x: -1, y: -1 }
  root.addEventListener('mousemove', (event) => {
    const moved = event.clientX !== pointerAt.x || event.clientY !== pointerAt.y
    pointerAt = { x: event.clientX, y: event.clientY }
    if (!moved || !picker) return
    const li = event.target.closest?.('[data-chat="picker-hover"]')
    if (li && picker.active !== Number(li.dataset.index)) {
      picker = { ...picker, active: Number(li.dataset.index) }
      drawOverlay()
    }
  })

  return {
    /** The tabs saved last time (`sessionsLoad`'s reply); creates and saves a first session when there are none. */
    setSessions(saved) {
      tabs = createTabs(saved, makeId)
      if (!saved?.sessions?.length) save()
      render()
    },
    /** What the page knows beyond the chat: the assistant's name and the compaction threshold. */
    setSettings(next) {
      settings = next
      render()
    },
    /** Backend feeds that do not exist yet (subagents, tasks, memory): the UI is ready for them. */
    setAgents(list) {
      data.agents = list ?? []
      render()
    },
    setTasks(tasks, kind = 'tasks') {
      data.tasks = tasks
      data.tasksKind = kind
      render()
    },
    setMemory(memories) {
      data.memory = memories ?? []
      render()
    },
    /** What a session said: its chat messages (`chatState`, `chatEvent`, `chatDialog`, …). */
    receive(session, message) {
      // A late reply for a tab that is gone (closed, its undo window over) must not bring its state back.
      if (!tabs.sessions.some((s) => s.id === session) && tabs.closed?.entry?.id !== session) return
      // A new Pi (a restart, another model) can offer other commands: ask again.
      if ((message.type === 'chatState' && message.model !== chatOf(session).model) || (message.type === 'chatEngine' && message.state !== 'ready')) {
        commandsBy.delete(session)
      }
      const before = chatOf(session)
      chats.set(session, reduce(before, message))
      // A refused message goes back to the composer (when it is empty), so nothing looks sent that was not.
      const giveBack = chatOf(session).giveBack
      if (giveBack !== null && giveBack !== undefined) {
        if (session === tabs.active && input && !input.value) {
          input.value = giveBack
          composerKey = ''
        } else if (session !== tabs.active && !drafts.get(session)) drafts.set(session, giveBack)
        chats.set(session, tookBack(chatOf(session)))
      }
      tabs = applyStatus(tabs, session, chatOf(session), session === tabs.active)
      // A background session that starts waiting for an answer says so.
      if (message.type === 'chatDialog' && session !== tabs.active) {
        const tab = tabs.sessions.find((s) => s.id === session)
        toast = { session, title: tab?.title || t('chat.untitled'), text: t('chat.toast.needs') }
        clearTimeout(toastTimer)
        toastTimer = setTimeout(() => {
          toast = null
          drawToast()
        }, TOAST_MS)
      }
      if (session !== tabs.active) {
        drawHeader()
        drawToast()
        return
      }
      render()
    },
    mount,
    show() {
      post({ type: 'chatLoad' })
      render()
      input?.focus()
    },
    /**
     * Esc, in this order: close a picker, stop a run, leave a panel, close an open tab menu. Returns whether it
     * did something; when it did not, the dock closes.
     */
    escape() {
      if (picker) {
        dismissed = { kind: picker.kind, text: input?.value ?? '' }
        closePicker()
        return true
      }
      if (ui.connect) {
        closeConnect()
        return true
      }
      if (current().running && ui.view === 'chat') {
        stop()
        return true
      }
      if (ui.view !== 'chat') {
        openView('chat')
        return true
      }
      if (ui.menuOpen) {
        ui.menuOpen = false
        drawHeader()
        return true
      }
      return false
    },
    /** Alt+A: the subagents view and back. */
    toggleAgents() {
      openView(ui.view === 'agents' || ui.view === 'session' ? 'chat' : 'agents')
    },
    render,
    setBanner(html) {
      banner = html
      if ($('chat-banner')) $('chat-banner').innerHTML = html
    },
  }
}
