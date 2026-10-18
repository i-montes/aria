// The dock's page: the bridge to the Rust side, the corner notice, and the screens. Everything the
// dock sends arrives through `window.aria.receive(message)` (see crates/aria-core/src/bridge.rs).
import { createI18n } from './i18n.js'

let i18n = createI18n({})
const t = (key, vars) => i18n.t(key, vars)
const state = { settings: null, view: 'welcome', problem: null }

// ---- bridge ----------------------------------------------------------------------------------

window.addEventListener('error', (e) => console.error('page error:', e.message, e.filename, e.lineno))
window.addEventListener('unhandledrejection', (e) => console.error('page rejection:', String(e.reason)))

const post = (message) => window.webkit.messageHandlers.aria.postMessage(message)
let nextId = 1
const pending = new Map()

/** Sends a request and resolves with the dock's reply (matched by `id`). `onProgress` gets
 * intermediate messages, such as the sign-in URL. */
export function request(message, onProgress) {
  const id = nextId++
  return new Promise((resolve) => {
    pending.set(id, { resolve, onProgress })
    post({ ...message, id })
  })
}

const handlers = {
  open: () => {
    document.body.classList.remove('show-notice')
    document.body.classList.remove('open')
    void document.body.offsetWidth
    // Two frames, so the closed state is on screen before the transition starts.
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        document.body.classList.add('open')
        focusFirst()
      }),
    )
  },
  close: () => document.body.classList.remove('open'),
  // These need the catalogs, so they wait for the boot to finish.
  notice: () => booted.then(showNotice),
  problem: (message) =>
    booted.then(() => {
      state.problem = message.code
      render()
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

// ---- screens ---------------------------------------------------------------------------------

const content = document.getElementById('content')

function render() {
  const problem = state.problem
    ? `<div class="problem"><b>${t('problem.title')}</b>${t('problem.' + state.problem)}</div>`
    : ''
  content.innerHTML = problem + welcomeView()
  content.querySelectorAll('[data-choice]').forEach((el) =>
    el.addEventListener('click', () => document.dispatchEvent(new CustomEvent('aria:choose', { detail: el.dataset.choice }))),
  )
}

function welcomeView() {
  return `
    <span class="dot"></span>
    <h1>${t('welcome.title')}</h1>
    <p class="intro">${t('welcome.intro')}</p>
    <button class="card" data-nav data-choice="quick">
      <div class="row"><b>${t('welcome.quick.title')}</b><span class="meta">${t('welcome.quick.time')}</span></div>
      <div class="description">${t('welcome.quick.description')}</div>
    </button>
    <button class="card" data-nav data-choice="customize">
      <div class="row"><b>${t('welcome.customize.title')}</b><span class="meta">${t('welcome.customize.steps')}</span></div>
      <div class="description">${t('welcome.customize.description')}</div>
    </button>
    <div class="footer">
      <span>${t('welcome.later')}</span>
      <button class="link" data-nav data-choice="skip">${t('welcome.skip')}</button>
    </div>`
}

// ---- notice ----------------------------------------------------------------------------------

const NOTICE_MS = 8000

function showNotice() {
  document.getElementById('notice').innerHTML = `
    <span class="dot"></span>
    <div><b>${t('notice.title')}</b><span class="line">${t('notice.press')} <kbd>Super</kbd> <kbd>A</kbd> ${t('notice.toSetUp')}</span></div>`
  document.body.classList.add('show-notice')
  setTimeout(() => document.body.classList.remove('show-notice'), NOTICE_MS)
}

// ---- keyboard --------------------------------------------------------------------------------

const navigable = () => [...document.querySelectorAll('[data-nav]')]

function focusFirst() {
  navigable()[0]?.focus()
}

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') return post({ type: 'closeRequest' })
  if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
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
  state.settings = reply.settings
  if (reply.problem) state.problem = reply.problem
  if (reply.settings?.assistant?.language) i18n.setLanguage(reply.settings.assistant.language)
  render()
  if (document.body.classList.contains('open')) focusFirst()
}

// `ready` means the bridge is up (the dock holds its messages until then), so it goes out before
// anything waits on a reply.
post({ type: 'ready' })
const booted = boot()
