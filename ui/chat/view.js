// The chat's page: the controller that composes the design's parts (header, thread, composer, pickers,
// cards, panels) and wires them to the bridge, plus the few pure helpers it keeps (tested without a
// browser). Conversations live in `state.js`, the tabs in `sessions.js`, the HTML of each element in
// `parts/` and `thread.js`. Everything dynamic is escaped with `esc`.
import { esc } from '../onboarding/components.js'
import { renderAgentSession } from './parts/session.js'
import { renderCommandPicker, filterCommands, stepActive, clampActive, keepVisible } from './parts/commands.js'
import { renderComposer } from './parts/composer.js'
import { renderFilePicker, filterFiles } from './parts/files.js'
import { activeIndexFor, indexItems, indexTargets, renderIndex } from './parts/index.js'
import { renderMemory, initialMemoryUi, reduceMemoryUi } from './parts/memory.js'
import { renderMeter } from './parts/meter.js'
import { renderModelPicker, filterModels } from './parts/models.js'
import { questionAnswers, questionInit, questionReduce, normalizeQuestions, renderQuestion } from './parts/question.js'
import { renderSubagents } from './parts/subagents.js'
import { renderSessionTabs } from './parts/tabs.js'
import { renderTasks } from './parts/tasks.js'
import { formatElapsed } from './parts/working.js'
import {
  MAX_SESSIONS, addSession, applyStatus, closeSession, createTabs, renameSession, selectSession, setTitleIfEmpty, toSaved, undoClose,
} from './sessions.js'
import { addQueued, addUser, answerDialog, clearQueue, createChat, reduce } from './state.js'
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
  const ui = { menuOpen: false, renaming: null, view: 'chat', agent: null, memory: initialMemoryUi(), tasksOpen: true, index: { expanded: false, my: null, off: 0, height: 360, settled: false } }
  let toast = null
  let undoTimer = null
  let toastTimer = null
  let tick = null
  let pendingClose = null // a closed tab whose Pi is only stopped when its undo window is over
  let fileToken = 0
  let fileTimer = null
  let composerKey = ''
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
      <span id="chat-meter-slot"></span>
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

  const meterHtml = () => {
    const state = current()
    return renderMeter(
      { model: modelLabel(state.model) || t('chat.meter.noModel'), used: 0, context: state.context ?? 0, free: false, hasQuota: false, compactAt: settings?.compaction?.percent ?? 40 },
      {},
      t,
    )
  }

  function drawHeader() {
    const multi = tabs.sessions.length > 1
    const full = tabs.sessions.length >= MAX_SESSIONS
    const sessions = tabs.sessions.map((s) => ({ ...s, model: modelLabel(chatOf(s.id).model), context: chatOf(s.id).context ?? 0, quota: 0, free: false }))
    const left = multi
      ? renderSessionTabs({ ...tabs, sessions }, { maxVisible: 3, hideNew: true, hideUndo: true, menuOpen: ui.menuOpen, renaming: ui.renaming, hasQuota: false }, t)
      : `<div class="chat-identity"><span class="chat-dot" aria-hidden="true"></span><span class="chat-name">${esc(assistantName())}</span>${meterHtml()}</div>`
    $('chat-header').innerHTML = `${left}
      <button type="button" class="icon-btn chat-new" data-chat="add" ${full ? 'disabled' : ''} aria-label="${esc(t('chat.new'))}" title="${esc(t('chat.new'))}">${ICON.plus}</button>
      <button type="button" class="icon-btn" data-chat="close-dock" aria-label="${esc(t('chat.close'))}" title="${esc(t('chat.close'))}">${ICON.close}</button>`
    // The meter sits in the header's identity block in the design; with tabs it moves to the path row.
    $('chat-meter-slot').innerHTML = multi ? meterHtml() : ''
  }

  function drawThread() {
    const state = current()
    const list = $('chat-list')
    const nearBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 80
    const ex = expanded()
    list.innerHTML = renderThread(state, { expanded: ex.open, openRows: ex.rows, moreOpen: ex.more, now: Date.now() }, t)
    if (nearBottom) list.scrollTop = list.scrollHeight
  }

  function drawIndex() {
    const items = indexItems(current().messages)
    const el = $('chat-index')
    ui.index.height = $('chat-list').clientHeight || ui.index.height
    el.innerHTML = items.length > 1 ? renderIndex({ items, activeId: ui.activeId }, ui.index, t) : ''
  }

  function drawQuestion() {
    const state = current()
    const dialog = state.dialogs.find((d) => d.card?.kind === 'question')
    const el = $('chat-question')
    if (!dialog) {
      el.innerHTML = ''
      return
    }
    if (!wizards.has(dialog.id)) wizards.set(dialog.id, questionInit(dialog.card.questions))
    el.innerHTML = `<div class="question-fill">${renderQuestion({ id: dialog.id, questions: dialog.card.questions }, wizards.get(dialog.id), t)}</div>`
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
    el.innerHTML =
      picker.kind === 'command' ? renderCommandPicker(props, {}, t)
      : picker.kind === 'file' ? renderFilePicker(props, {}, t)
      : renderModelPicker({ models: picker.items, current: current().model ?? '', query: picker.query, active: picker.active, open: true }, {}, t)
    el.querySelector('.p-cmd-row.on, .p-files-row.on, .p-models-row.on')?.scrollIntoView?.({ block: 'nearest' })
  }

  function drawTasks() {
    const el = $('chat-tasks')
    el.innerHTML = data.tasks?.length ? renderTasks({ tasks: data.tasks, kind: data.tasksKind, show: true }, { open: ui.tasksOpen }, t) : ''
  }

  function drawComposer() {
    const el = $('chat-composer')
    const value = input ? input.value : (drafts.get(tabs.active) ?? '')
    const busy = current().running
    const hasText = value.trim().length > 0
    const rows = Math.min(6, Math.max(1, value.split('\n').length))
    const key = `${busy}|${hasText}|${rows}`
    if (input && key === composerKey) return
    const had = input && document.activeElement === input
    const selection = input ? [input.selectionStart, input.selectionEnd] : null
    composerKey = key
    el.innerHTML = renderComposer({ busy, hasText, rows, showHints: true }, {}, t)
    input = $('chat-input')
    input.value = value
    if (selection) input.setSelectionRange(selection[0], selection[1])
    if (had) input.focus()
  }

  function drawViews() {
    const view = ui.view
    $('chat-body').dataset.view = view
    for (const name of ['chat', 'agents', 'session', 'memory']) $(`view-${name}`).hidden = name !== view
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
      $('view-memory').innerHTML = renderMemory({ memories: data.memory, now: Date.now() }, ui.memory, t)
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
    if (/^\/model(\s|$)/.test(text)) return // the model picker answers this one
    const session = tabs.active
    const chat = chatOf(session)
    const slash = text.startsWith('/') // extension commands run at once, even during a run
    if (chat.running && !slash) {
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
    picker = null
    fileToken++
    if (draw) drawOverlay()
  }

  const pickerSize = () => {
    if (!picker) return 0
    const { kind, items, query } = picker
    return kind === 'command' ? filterCommands(items, query).length : kind === 'file' ? filterFiles(items, query).length : filterModels(items, query).length
  }

  async function updatePicker() {
    const asked = pickerQuery(input.value, input.selectionStart ?? input.value.length)
    if (!asked) {
      dismissed = null
      return closePicker()
    }
    if (dismissed && dismissed.kind === asked.kind && input.value.startsWith(dismissed.text)) return closePicker()
    const session = tabs.active
    if (asked.kind === 'command') {
      if (!commandsBy.has(session)) {
        const reply = await ask({ type: 'listCommands' }, session)
        const local = [
          { name: 'memory', description: t('chat.cmd.memory'), source: 'aria' },
          { name: 'model', description: t('chat.cmd.model'), source: 'aria' },
        ]
        commandsBy.set(session, [...local, ...(reply.commands ?? [])])
      }
      if (session !== tabs.active) return // the owner moved to another tab while Pi was starting
      picker = { kind: 'command', items: commandsBy.get(session), query: asked.query, active: 0, start: asked.start }
      return drawOverlay()
    }
    if (asked.kind === 'model') {
      const reply = await ask({ type: 'listModels' }, session)
      if (session !== tabs.active) return
      picker = { kind: 'model', items: reply.models ?? [], query: asked.query, active: 0, start: 0 }
      return drawOverlay()
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
      const hit = filterModels(items, query)[index]?.model
      if (!hit) return
      closePicker(false)
      clearDraft()
      ask({ type: 'setModel', model: `${hit.provider}/${hit.id}` })
      drawOverlay()
      drawComposer()
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
    if (event.isComposing) return
    if (picker) {
      const size = pickerSize()
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        dismissed = { kind: picker.kind, text: input.value }
        return closePicker()
      }
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
    if (event.key === 'Escape' && current().running) {
      event.preventDefault()
      event.stopPropagation()
      return stop()
    }
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.stopPropagation() // the arrows move the caret here, not focus between controls
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
  function onQuestion(el) {
    const { id, op, q, value } = el.dataset
    const state = wizards.get(id)
    const dialog = current().dialogs.find((d) => d.id === id)
    if (!state || !dialog) return
    const list = normalizeQuestions(dialog.card.questions)
    const next = questionReduce(state, op, { q: q === undefined ? undefined : Number(q), value })
    wizards.set(id, next)
    if (next.outcome === 'submit') return answer(id, { value: JSON.stringify(questionAnswers(list, next)) }, questionAnswers(list, next))
    if (next.outcome === 'dismiss') return answer(id, { cancelled: true }, undefined)
    drawQuestion()
    // A single-choice pick moves on by itself after a beat, as the design does.
    clearTimeout(autoAdvance)
    if (op === 'pick' && !list[Number(q)]?.multi && Number(q) < list.length - 1) {
      autoAdvance = setTimeout(() => {
        const cur = wizards.get(id)
        if (cur && cur.step === Number(q)) {
          wizards.set(id, questionReduce(cur, 'next', {}))
          drawQuestion()
        }
      }, 480)
    }
  }

  // ---- the thread's scroll and index -------------------------------------------------------------

  function updateActive() {
    const items = indexItems(current().messages)
    if (!items.length) return
    const list = $('chat-list')
    const base = list.getBoundingClientRect().top
    const tops = indexTargets(items).map((id) => {
      const el = document.getElementById(id)
      return el ? el.getBoundingClientRect().top - base + list.scrollTop : Infinity
    })
    const at = activeIndexFor(list.scrollTop, tops)
    const id = items[Math.max(0, Math.min(items.length - 1, at))]?.id
    if (id !== ui.activeId) {
      ui.activeId = id
      drawIndex()
    }
  }

  root.addEventListener('scroll', (event) => {
    if (event.target?.id === 'chat-list') updateActive()
  }, true)

  const indexEl = () => $('chat-index')
  root.addEventListener('mouseover', (event) => {
    if (event.target.closest?.('#chat-index') && !ui.index.expanded) {
      ui.index = { ...ui.index, expanded: true, settled: false }
      drawIndex()
      setTimeout(() => { ui.index.settled = true }, 400)
    }
  })
  root.addEventListener('mousemove', (event) => {
    if (!ui.index.expanded) return
    const rect = indexEl()?.getBoundingClientRect()
    if (!rect) return
    ui.index = { ...ui.index, my: event.clientY - rect.top }
    drawIndex()
  })
  root.addEventListener('mouseout', (event) => {
    if (event.target.closest?.('#chat-index') && !event.relatedTarget?.closest?.('#chat-index')) {
      ui.index = { ...ui.index, expanded: false, my: null }
      drawIndex()
    }
  })

  // ---- events ----------------------------------------------------------------------------------

  root.addEventListener('input', (event) => {
    const el = event.target
    if (el === input) {
      drawComposer()
      updatePicker()
    } else if (el?.dataset?.op === 'other') {
      const id = el.dataset.id
      const state = wizards.get(id)
      if (state) wizards.set(id, questionReduce(state, 'other', { q: Number(el.dataset.q), value: el.value }))
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
    if (event.key === 'Escape' && ui.view !== 'chat') {
      event.stopPropagation()
      openView('chat')
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
      case 'open-models': {
        if (!input) return
        input.value = '/model '
        composerKey = ''
        drawComposer()
        input.focus()
        input.setSelectionRange(7, 7)
        return updatePicker()
      }
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
      case 'cancel-queued':
        post({ type: 'chatClearQueue' })
        chats.set(tabs.active, clearQueue(current()))
        return render()
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
      case 'index-jump': {
        document.getElementById(id)?.scrollIntoView({ block: 'start', behavior: 'smooth' })
        return
      }
      case 'permission': return answer(id, { value: choice }, choice)
      case 'quota': return answer(id, { value: choice }, choice)
      case 'question': return onQuestion(el)
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

  root.addEventListener('mouseover', (event) => {
    const li = event.target.closest?.('[data-chat="picker-hover"]')
    if (li && picker && picker.active !== Number(li.dataset.index)) {
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
      if (message.type === 'chatState' && message.model !== chatOf(session).model) commandsBy.delete(session)
      const before = chatOf(session)
      chats.set(session, reduce(before, message))
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
      render()
    },
    mount,
    show() {
      post({ type: 'chatLoad' })
      render()
      input?.focus()
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
