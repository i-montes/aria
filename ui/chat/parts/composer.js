// The composer (design/components/composer): the message field, the stop / queue / send buttons and
// the hints line under it.
//
// Props: { busy, hasText, rows, showHints, placeholder }
//   busy         a run is going: the stop button shows, send is replaced by queue once there is text
//   hasText      the field holds text (trimmed, not empty): the stop button turns outlined and queue shows
//   rows         visible rows of the textarea, clamped to 1..6 (the design grows it with its content)
//   showHints    false hides the hints line (default true)
//   placeholder  idle placeholder (default `chat.composer.placeholder`); busy always says "Queue a message…"
// ui: unused. t: translate(key, vars).
// The controller owns the text: the textarea is `#chat-input` and carries no value here.
//
// Emits:
//   data-chat="send"  (#chat-send when idle, #chat-queue when busy; the queue button also has
//                      data-queue="true"): send the text, or queue it while a run is going
//   data-chat="stop"  (#chat-stop, only while busy): stop the run
import { esc } from '../../onboarding/components.js'

export const TEXTS = {
  en: {
    'chat.composer.label': 'Message for Aria',
    'chat.composer.placeholder': 'Message Aria…',
    'chat.composer.queuePlaceholder': 'Queue a message…',
    'chat.composer.send': 'Send',
    'chat.composer.stop': 'Stop',
    'chat.composer.queue': 'Queue message',
    'chat.composer.queueTitle': 'Queue it for when Aria finishes',
    'chat.composer.hint.commands': 'for commands',
    'chat.composer.hint.agents': 'subagents',
    'chat.composer.hint.queue': 'queues your message',
  },
  es: {
    'chat.composer.label': 'Mensaje para Aria',
    'chat.composer.placeholder': 'Escríbele a Aria…',
    'chat.composer.queuePlaceholder': 'Deja un mensaje en cola…',
    'chat.composer.send': 'Enviar',
    'chat.composer.stop': 'Detener',
    'chat.composer.queue': 'Poner el mensaje en cola',
    'chat.composer.queueTitle': 'Déjalo en cola para cuando Aria termine',
    'chat.composer.hint.commands': 'para comandos',
    'chat.composer.hint.agents': 'subagentes',
    'chat.composer.hint.queue': 'encola tu mensaje',
  },
}

const ICON = {
  stop: '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="3"/></svg>',
  queue:
    '<svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="13" r="7"/><path d="M12 10v3l2 1.5M9.5 3h5"/></svg>',
  send: '<svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>',
}

const key = (name) => `<span class="p-composer-key">${name}</span>`

export function renderComposer({ busy = false, hasText = false, rows = 1, showHints = true, placeholder, model, modelTitle } = {}, ui = {}, t = (k) => k) {
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
  // Under the input: the model (opens the picker), then the shortcuts; while Aria works, only that Enter queues the message
  // (the line has room for one hint next to the model).
  const chip = model
    ? `<button type="button" class="p-composer-model" data-chat="open-models"${modelTitle ? ` title="${esc(modelTitle)}"` : ''}>${esc(model)}</button>`
    : ''
  const hints = !showHints
    ? ''
    : `<div class="p-composer-hints">${chip}
${
        busy
          ? `
      <span>${key('enter')} ${esc(t('chat.composer.hint.queue'))}</span>`
          : `
      <span><kbd class="p-composer-kbd">/</kbd> ${esc(t('chat.composer.hint.commands'))}</span>
      <span>${key('alt+a')} ${esc(t('chat.composer.hint.agents'))}</span>`
      }
    </div>`
  return `<div class="p-composer">
    <div class="p-composer-box">
      <div class="p-composer-field">
        <label for="chat-input" class="p-composer-label">${esc(t('chat.composer.label'))}</label>
        <textarea id="chat-input" class="p-composer-input" rows="${lines}" placeholder="${esc(hint)}" spellcheck="false"></textarea>
      </div>
      <div class="p-composer-actions">${stop}${queue}${send}</div>
    </div>${hints}
  </div>`
}
