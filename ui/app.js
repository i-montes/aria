// The dock's page: the bridge to the Rust side, the corner notice, the problem banner and the
// onboarding. Everything the dock sends arrives through `window.aria.receive(message)` (see
// crates/aria-core/src/bridge.rs).
import { createI18n, normalizeLanguage } from './i18n.js'
import { createOnboarding } from './onboarding/view.js'

let i18n = createI18n({})
const t = (key, vars) => i18n.t(key, vars)
const state = { problem: null }

// ---- bridge ----------------------------------------------------------------------------------

window.addEventListener('error', (e) => console.error('page error:', e.message, e.filename, e.lineno))
window.addEventListener('unhandledrejection', (e) => console.error('page rejection:', String(e.reason)))

const post = (message) => window.webkit.messageHandlers.aria.postMessage(message)
let nextId = 1
const pending = new Map()

/** Sends a request and resolves with the dock's reply (matched by `id`). `onProgress` gets
 * intermediate messages, such as the sign-in URL. */
function request(message, onProgress) {
  const id = nextId++
  return new Promise((resolve) => {
    pending.set(id, { resolve, onProgress })
    post({ ...message, id })
  })
}

const handlers = {
  open: () => {
    onboarding.onOpen() // after a finished setup, show the settings again
    document.body.classList.remove('show-notice')
    document.body.classList.remove('open')
    void document.body.offsetWidth
    // Two frames, so the closed state is on screen before the transition starts.
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        document.body.classList.add('open')
        document.querySelector('#flow [data-nav]')?.focus()
      }),
    )
  },
  close: () => document.body.classList.remove('open'),
  // These need the catalogs, so they wait for the boot to finish.
  notice: () => booted.then(showNotice),
  problem: (message) =>
    booted.then(() => {
      state.problem = message.code || null // an empty code clears the banner
      renderProblem()
    }),
}

window.aria = {
  send: post,
  request,
  receive(message) {
    const waiting = 'id' in message ? pending.get(message.id) : undefined
    if (waiting) {
      if (message.type === 'oauthUrl') return waiting.onProgress?.(message)
      pending.delete(message.id)
      return waiting.resolve(message)
    }
    handlers[message.type]?.(message)
  },
}

// ---- problem banner and notice ---------------------------------------------------------------

const problem = document.getElementById('problem')

function renderProblem() {
  // Settings problems are not "Aria can't start": it started, and it is protecting your file.
  const title = state.problem?.startsWith('settings-') ? t('problem.settings.title') : t('problem.title')
  problem.innerHTML = state.problem ? `<div class="problem"><b>${title}</b>${t('problem.' + state.problem)}</div>` : ''
}

const NOTICE_MS = 8000

function showNotice() {
  document.getElementById('notice').innerHTML = `
    <span class="dot"></span>
    <div><b>${t('notice.title')}</b><span class="line">${t('notice.press')} <kbd>Super</kbd> <kbd>A</kbd> ${t('notice.toSetUp')}</span></div>`
  document.body.classList.add('show-notice')
  setTimeout(() => document.body.classList.remove('show-notice'), NOTICE_MS)
}

// ---- onboarding ------------------------------------------------------------------------------

const onboarding = createOnboarding({
  root: document.getElementById('flow'),
  t,
  request,
  post,
  browserLanguage: () => normalizeLanguage(navigator.language),
  // The interface follows the conversation language as soon as it is picked.
  setLanguage: (code) => {
    i18n.setLanguage(code)
    renderProblem()
    onboarding.render()
  },
})

// ---- keyboard --------------------------------------------------------------------------------

// Disabled controls cannot take focus, so the arrows have to skip them.
const navigable = () => [...document.querySelectorAll('#flow [data-nav]:not(:disabled)')]

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') return post({ type: 'closeRequest' })
  if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
  if (event.target instanceof HTMLInputElement && event.target.type === 'range') return
  const items = navigable()
  if (!items.length) return
  event.preventDefault()
  const at = items.indexOf(document.activeElement)
  const step = event.key === 'ArrowDown' ? 1 : -1
  items[(at + step + items.length) % items.length].focus()
})

// Tell the dock when the open transition is over; it does not unmap or re-open on its own.
document.querySelector('aside').addEventListener('transitionend', (event) => {
  if (event.propertyName === 'transform' && document.body.classList.contains('open')) post({ type: 'opened' })
})

// ---- boot ------------------------------------------------------------------------------------

async function boot() {
  const [en, es] = await Promise.all(['en', 'es'].map((code) => fetch(`aria://app/i18n/${code}.json`).then((r) => r.json())))
  i18n = createI18n({ en, es })
  i18n.setLanguage(navigator.language)
  const reply = await request({ type: 'loadSettings' })
  if (reply.problem) state.problem = reply.problem
  if (reply.settings?.assistant?.language) i18n.setLanguage(reply.settings.assistant.language)
  renderProblem()
  onboarding.start(reply.settings)
  if (document.body.classList.contains('open')) document.querySelector('#flow [data-nav]')?.focus()
}

// `ready` means the bridge is up (the dock holds its messages until then), so it goes out before
// anything waits on a reply.
post({ type: 'ready' })
const booted = boot()
