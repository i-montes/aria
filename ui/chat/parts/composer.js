// The composer (design/components/composer): the message field, the stop / queue / send buttons and
// the hints line under it.
//
// Props: { busy, hasText, rows, showHints, agents, background, placeholder, model, modelTitle, attachments }
//   busy         a run is going: the stop button shows, send is replaced by queue once there is text
//   hasText      the field holds text (trimmed, not empty) or files: the stop button turns outlined and queue shows
//   rows         visible rows of the textarea, clamped to 1..6 (the design grows it with its content)
//   showHints    false hides the hints line (default true)
//   agents       the session has `parallel` subagents: the idle hints say "alt+a subagents"
//   background   the session has background jobs: the idle hints say "alt+a background" instead (wins over agents)
//   placeholder  idle placeholder (default `chat.composer.placeholder`); busy always says "A message for the next step…"
//   attachments  [{ id, name, mime, size }]: the files going with the next message, as removable chips above the input
// ui: unused. t: translate(key, vars).
// The controller owns the text and the files: the textarea is `#chat-input` and carries no value here, and the hidden
// file input is `#chat-file` (the paperclip clicks it).
//
// Emits:
//   data-chat="send"  (#chat-send when idle, #chat-queue when busy; the queue button also has
//                      data-queue="true"): send the text, or queue it while a run is going
//   data-chat="stop"  (#chat-stop, only while busy): stop the run
//   data-chat="attach"  (#chat-attach): choose files (the view clicks `#chat-file`)
//   data-chat="attach-remove" data-id: drop that file (attachments.js)
import { esc } from '../../onboarding/components.js'
import { renderAttachments } from './attachments.js'

export const TEXTS = {
  en: {
    'chat.composer.label': 'Message for Aria',
    'chat.composer.placeholder': 'Message Aria…',
    'chat.composer.queuePlaceholder': 'A message for the next step…',
    'chat.composer.send': 'Send',
    'chat.composer.stop': 'Stop',
    'chat.composer.queue': 'Send at the next step',
    'chat.composer.queueTitle': 'Aria reads it once the step going on ends (Alt+Enter keeps it for when Aria finishes)',
    'chat.composer.hint.commands': 'for commands',
    'chat.composer.hint.agents': 'subagents',
    'chat.composer.hint.background': 'background',
    'chat.composer.hint.queue': 'when Aria finishes',
    'chat.composer.hint.steer': 'at the next step',
    'chat.composer.attach': 'Attach files',
  },
  es: {
    'chat.composer.label': 'Mensaje para Aria',
    'chat.composer.placeholder': 'Escríbele a Aria…',
    'chat.composer.queuePlaceholder': 'Un mensaje para el próximo paso…',
    'chat.composer.send': 'Enviar',
    'chat.composer.stop': 'Detener',
    'chat.composer.queue': 'Enviar en el próximo paso',
    'chat.composer.queueTitle': 'Aria lo lee cuando termine el paso en curso (Alt+Enter lo deja para cuando Aria termine)',
    'chat.composer.hint.commands': 'para comandos',
    'chat.composer.hint.agents': 'subagentes',
    'chat.composer.hint.background': 'segundo plano',
    'chat.composer.hint.queue': 'cuando Aria termine',
    'chat.composer.hint.steer': 'en el próximo paso',
    'chat.composer.attach': 'Adjuntar archivos',
  },
}

const ICON = {
  stop: '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="3"/></svg>',
  queue:
    '<svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="13" r="7"/><path d="M12 10v3l2 1.5M9.5 3h5"/></svg>',
  attach:
    '<svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5l-8.6 8.6a5.5 5.5 0 0 1-7.8-7.8l8.6-8.6a3.7 3.7 0 0 1 5.2 5.2l-8.6 8.6a1.8 1.8 0 0 1-2.6-2.6l7.9-7.9"/></svg>',
  send: '<svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>',
}

const key = (name) => `<span class="p-composer-key">${name}</span>`

export function renderComposer({ busy = false, hasText = false, rows = 1, showHints = true, agents = false, background = false, placeholder, model, modelTitle, attachments = [] } = {}, ui = {}, t = (k) => k) {
  const lines = Math.min(6, Math.max(1, Math.round(Number(rows)) || 1))
  const hint = busy ? t('chat.composer.queuePlaceholder') : (placeholder ?? t('chat.composer.placeholder'))
  const stop = busy
    ? `<button type="button" id="chat-stop" class="p-composer-btn p-composer-stop${hasText ? ' has-text' : ''}" data-chat="stop" aria-label="${esc(t('chat.composer.stop'))}" title="${esc(t('chat.composer.stop'))}">${ICON.stop}</button>`
    : ''
  const queue =
    busy && hasText
      ? `<button type="button" id="chat-queue" class="p-composer-btn p-composer-queue" data-chat="send" data-queue="true" aria-label="${esc(t('chat.composer.queue'))}" title="${esc(t('chat.composer.queueTitle'))}">${ICON.queue}</button>`
      : ''
  const send = busy
    ? ''
    : `<button type="button" id="chat-send" class="p-composer-btn p-composer-send" data-chat="send" aria-label="${esc(t('chat.composer.send'))}">${ICON.send}</button>`
  // Under the input: the model (opens the picker), then the shortcuts (alt+a only when the session has subagents or background jobs); while Aria works, that Enter puts the
  // message in at the run's next step and Alt+Enter keeps it for when the run ends.
  const chip = model
    ? `<button type="button" class="p-composer-model" data-chat="open-models"${modelTitle ? ` title="${esc(modelTitle)}"` : ''}>${esc(model)}</button>`
    : ''
  const hints = !showHints
    ? ''
    : `<div class="p-composer-hints">${chip}
${
        busy
          ? `
      <span>${key('enter')} ${esc(t('chat.composer.hint.steer'))}</span>
      <span>${key('alt+enter')} ${esc(t('chat.composer.hint.queue'))}</span>`
          : `
      <span><kbd class="p-composer-kbd">/</kbd> ${esc(t('chat.composer.hint.commands'))}</span>${background || agents ? `
      <span>${key('alt+a')} ${esc(t(background ? 'chat.composer.hint.background' : 'chat.composer.hint.agents'))}</span>` : ''}`
      }
    </div>`
  const files = attachments?.length ? `<div class="p-composer-files">${renderAttachments({ files: attachments, removable: true }, {}, t)}</div>` : ''
  const attach = `<button type="button" id="chat-attach" class="p-composer-btn p-composer-attach" data-chat="attach" aria-label="${esc(t('chat.composer.attach'))}" title="${esc(t('chat.composer.attach'))}">${ICON.attach}</button>`
  return `<div class="p-composer">${files}
    <input type="file" id="chat-file" multiple hidden tabindex="-1">
    <div class="p-composer-box">${attach}
      <div class="p-composer-field">
        <label for="chat-input" class="p-composer-label">${esc(t('chat.composer.label'))}</label>
        <textarea id="chat-input" class="p-composer-input" rows="${lines}" placeholder="${esc(hint)}" spellcheck="false"></textarea>
      </div>
      <div class="p-composer-actions">${stop}${queue}${send}</div>
    </div>${hints}
  </div>`
}
