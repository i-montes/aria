// The onboarding screens: welcome, the steps of a flow, and the "set up" screen. The working state is
// the settings object (see steps.js); this file turns it into HTML, handles every interaction in one
// delegated listener, and talks to the dock through `env.request`. Widgets live in components.js.
import * as c from './components.js'
import { anyModifierHeld, defaultShortcut, display, fromKeyEvent, keyName, label as shortcutLabel, parseShortcut } from './shortcut.js'
import { renderPanelHead } from '../chat/parts/panelhead.js'
import { orderedModels, renderModelPicker } from '../chat/parts/models.js'
import {
  applyProvider, canContinue, credentialFor, defaultSettings, finish, isSignedIn, languageChoice, nextStep,
  previousStep, PROVIDERS, providerModule, setLanguageChoice, setRouter, skip, stepIds,
} from './steps.js'

const SERVICES = ['notion', 'tavily']
/** What a connector may do without asking (`integrations.access[server]`); a server not listed is at Ask. */
const ACCESS_LEVELS = ['read', 'ask', 'allow']
/** Never gated (the memory), so it has no level to set. */
const UNGATED = new Set(['engram'])
/** A server's name in words, as the connectors extension writes it in its card ("my-docs" -> "My Docs"). The same as
 * `displayName` in extensions/connectors/index.ts (TypeScript the page cannot import): change both together. */
const displayName = (server) =>
  server.split(/[-_]+/).filter(Boolean).map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(' ') || server
const SKILLS = ['brainstorming', 'writing-plans', 'test-driven-development', 'systematic-debugging', 'requesting-code-review', 'verification-before-completion']
const STYLES = ['very-short', 'brief', 'detailed', 'technical', 'teaching']
const LEVELS = ['low', 'medium', 'high']
const DEFAULT_TURN_LIMIT = 30
/** Minutes a run may get nothing from Pi before the dock stops it (`models.turnTimeoutMinutes`; 0 turns it off). */
const DEFAULT_TURN_TIMEOUT = 10
/** The settings screen: the setup's steps grouped by topic, in the order of its overview. A step the provider does not
 * have (the quota without OpenCode Go) is left out. */
const SETTINGS_SECTIONS = [
  { id: 'assistant', steps: ['assistant'] },
  { id: 'provider', steps: ['provider', 'auth', 'quota'] },
  { id: 'models', steps: ['models', 'effort'] },
  { id: 'router', steps: ['router'] },
  { id: 'context', steps: ['compaction', 'memory'] },
  { id: 'integrations', steps: ['integrations'] },
  { id: 'safety', steps: ['sandbox', 'git'] },
  { id: 'agents', steps: ['agents'] },
]
const SAVED_FLASH_MS = 2500
/** The provider Pi files each setup provider's models under. */
const PI_PROVIDER = { 'opencode-go': 'opencode-go', 'anthropic-api': 'anthropic', 'anthropic-oauth': 'claude-bridge' }

/** `env`: { root, t, request, post, setLanguage, browserLanguage, hotkey?, platform?, onDone?, listModels? }. `platform`
 * returns the dock's last `platform` message ({ hotkey, taken, os }): the Shortcut row's names and its "In use". `listModels` returns
 * the models Pi can use; the settings' model fields list them (without it, the provider's catalog in steps.js). `onDone` runs when the user
 * presses the button of the "set up" screen or leaves the settings (the dock goes to the chat); without it the dock
 * closes. */
export function createOnboarding(env) {
  const { root, t, request, post } = env
  const hotkey = () => env.hotkey?.() ?? 'Super + A'
  const platform = () => ({ hotkey: hotkey(), taken: [], os: 'linux', ...env.platform?.() })
  const state = {
    view: 'welcome', // welcome | step | done | settings
    section: null, // settings: the open section, or null for the overview
    saved: null, // settings: the settings as on disk (JSON), to tell what changed
    leaving: false, // settings: Back was pressed with changes not saved
    flash: false, // settings: "Saved" shows for a moment after a save
    models: null, // settings: what Pi can use (env.listModels), or null before it answers
    picker: null, // the open model field: { field: 'daily' | 'planning', query, active }
    mode: 'quick', // quick | full
    step: null,
    editing: false,
    saving: false,
    error: null,
    settings: defaultSettings('en'),
    connected: new Set(), // what the dock found on disk, plus what this session set up
    servers: [], // every server in Aria's mcp.json but engram (the dock's `servers`; an older dock sends none)
    svc: freshServices(), // Notion and Tavily do not depend on the provider, so they live apart from `ui`
    ui: freshUi(),
    found: freshFound(), // the MCP servers other apps have (the dock's `mcpFound`), and what is picked to import
  }

  /** `status`: idle | scanning | ready | error. Replaced on every scan, so a reply to an older one is ignored. */
  function freshFound(result = null) {
    return { status: 'idle', servers: [], picked: [], importing: false, error: null, result }
  }

  function freshServices() {
    return { tavily: { status: 'idle', error: null, draft: '' }, notion: { status: 'idle', error: null, url: null, token: 0 } }
  }

  function freshUi() {
    return {
      signedIn: false,
      typesafeSaved: false,
      reveal: {},
      // The Shortcut row: listening for keys, why the last press (or the typed one) did not count, and the typed text.
      shortcut: { recording: false, note: null, draft: '' },
      drafts: { key: '', typesafeKey: '', extraWritable: '' },
      key: { status: 'idle', error: null, requested: false },
      typesafe: { status: 'idle', error: null },
      oauth: { status: 'idle', error: null, url: null, token: 0, requested: false },
    }
  }

  /** Where the reader is; a reply that arrives after they moved must not move them again. */
  const here = () => `${state.view}:${state.step}:${state.section}`

  const uiFlags = () => ({
    signedIn: state.ui.signedIn,
    keyChecking: state.ui.key.status === 'checking' || state.ui.oauth.status === 'checking',
    typesafeSaved: state.ui.typesafeSaved,
    servicesConnected: SERVICES.filter((id) => state.connected.has(`mcp:${id}`)),
  })

  const update = (mutate) => {
    const next = structuredClone(state.settings)
    mutate(next)
    state.settings = next
  }
  const ids = () => stepIds(state.editing ? 'full' : state.mode, state.settings.provider.id)
  const module = () => providerModule(state.settings.provider.id)
  const assistantName = () => state.settings.assistant.name.trim() || 'Aria'

  // ---- entry points ---------------------------------------------------------------------------

  /** `settings` is what the dock loaded (null on a first run); `connected` is what it found on disk; `servers` the
   * servers configured in Aria's mcp.json. */
  function start(settings, connected = [], servers = []) {
    state.error = null
    state.connected = new Set(connected)
    state.servers = Array.isArray(servers) ? servers : []
    state.svc = freshServices()
    state.ui = freshUi()
    state.found = freshFound()
    state.leaving = false
    if (settings?.setup?.completed) {
      // A finished setup opens as the settings screen, on its overview.
      state.settings = settings
      state.saved = JSON.stringify(settings)
      state.editing = true
      state.mode = 'full'
      state.view = 'settings'
      state.section = null
      state.step = null
    } else {
      state.settings = settings ?? defaultSettings(env.browserLanguage())
      state.editing = false
      state.view = 'welcome'
    }
    state.ui.signedIn = isSignedIn(state.settings, connected)
    state.ui.typesafeSaved = state.connected.has('typesafe:api_key')
    lastView = '' // a (re)opened setup starts at the top, not where the previous one was scrolled
    render()
  }

  // ---- rendering ------------------------------------------------------------------------------

  let lastView = ''
  let focusAfterRender = ''

  function render() {
    const view = `${state.view}:${state.step}:${state.section}`
    const sameStep = view === lastView
    // WebKit does not focus a button when it is clicked, so the active element is not the checkbox that
    // was just pressed. The clicked widget's key is what brings the reader back to it after the redraw.
    const focused = sameStep ? focusAfterRender || document.activeElement?.closest?.('[data-key]')?.dataset.key : ''
    focusAfterRender = ''
    // The step lives in a scrollable panel. Redrawing it and focusing the first control sent the form back
    // to the top on every checkbox; keep the reader where they were instead.
    const panel = globalThis.document?.getElementById?.('content')
    const scrollTop = sameStep ? panel?.scrollTop ?? 0 : 0
    root.innerHTML = state.view === 'welcome' ? welcomeHtml() : state.view === 'done' ? doneHtml() : state.view === 'settings' ? settingsHtml() : stepHtml()
    const target = focused ? root.querySelector(`[data-key="${CSS.escape(focused)}"]`) : null
    lastView = view
    ;(target ?? root.querySelector('[data-nav]'))?.focus({ preventScroll: true })
    if (panel) panel.scrollTop = scrollTop
    if (state.picker) focusModelSearch()
  }

  function welcomeHtml() {
    return `
      <span class="dot"></span>
      <h1>${c.esc(t('welcome.title'))}</h1>
      <p class="intro">${c.esc(t('welcome.intro'))}</p>
      <button class="card" data-nav data-action="choose" data-value="quick">
        <div class="row"><b>${c.esc(t('welcome.quick.title'))}</b><span class="meta">${c.esc(t('welcome.quick.time'))}</span></div>
        <div class="description">${c.esc(t('welcome.quick.description'))}</div>
      </button>
      <button class="card" data-nav data-action="choose" data-value="full">
        <div class="row"><b>${c.esc(t('welcome.customize.title'))}</b><span class="meta">${c.esc(t('welcome.customize.steps'))}</span></div>
        <div class="description">${c.esc(t('welcome.customize.description'))}</div>
      </button>
      <div class="footer">
        <span>${c.esc(t('welcome.later'))}</span>
        <button class="link" data-nav data-action="choose" data-value="skip">${c.esc(t('welcome.skip'))}</button>
      </div>`
  }

  function doneHtml() {
    const name = assistantName()
    return `
      <span class="dot"></span>
      <h1>${c.esc(t('ob.done.title', { name }))}</h1>
      <p class="intro">${c.esc(t('ob.done.desc', { name, hotkey: hotkey() ? keysText(hotkey()) : t('ob.shortcut.noneShort') }))}</p>
      <button class="primary" data-nav data-action="close">${c.esc(t('ob.done.close'))}</button>`
  }

  function stepHtml() {
    const all = ids()
    const at = all.indexOf(state.step)
    const isReview = state.step === 'review'
    const descKey = isReview ? `ob.review.desc.${state.mode}` : `ob.${state.step}.desc`
    const nextLabel = state.saving ? t('ob.saving') : isReview ? t('ob.start', { name: assistantName() }) : t('ob.continue')
    const showDefaults = state.mode === 'full' && !isReview && at >= 2
    return `
      <div class="top"><span>${c.esc(t('ob.step', { n: at + 1, total: all.length }))}</span><span>${c.esc(t(`ob.name.${state.step}`))}</span></div>
      <div class="bars">${all.map((_, i) => `<i class="${i < at ? 'done' : i === at ? 'now' : ''}"></i>`).join('')}</div>
      <h1>${c.esc(t(`ob.${state.step}.title`))}</h1>
      <p class="intro">${c.esc(t(descKey))}</p>
      <div class="fields">${fields()}</div>
      <div class="nav">
        <button class="secondary" data-nav data-action="back">${c.esc(t('ob.back'))}</button>
        <button class="primary" data-nav data-action="next" data-key="next" ${canContinue(state.step, state.settings, uiFlags()) && !state.saving ? '' : 'disabled'}>${c.esc(nextLabel)}</button>
      </div>
      ${showDefaults ? `<button class="link center" data-nav data-action="defaults">${c.esc(t('ob.defaults'))}</button>` : ''}`
  }

  const sectionsByStep = {
    provider: () => providerStep(),
    quota: () => quotaStep(),
    auth: () => authStep(),
    router: () => routerStep(),
    assistant: () => assistantStep(),
    models: () => modelsStep(),
    effort: () => effortStep(),
    compaction: () => compactionStep(),
    memory: () => memoryStep(),
    integrations: () => integrationsStep(),
    sandbox: () => sandboxStep(),
    git: () => gitStep(),
    agents: () => agentsStep(),
    review: () => reviewStep(),
  }
  const fields = () => sectionsByStep[state.step]()

  // ---- steps ----------------------------------------------------------------------------------

  function providerStep() {
    const options = PROVIDERS.map((p) => ({
      value: p.id,
      label: p.label,
      description: t(`ob.provider.${p.id}.desc`),
      badges: p.quota ? [t('ob.badge.quota'), t('ob.badge.jev')] : [],
    }))
    return c.choice('provider', '', options, state.settings.provider.id) + c.info('lavender', t('ob.provider.modules.title'), t('ob.provider.modules.desc'))
  }

  function quotaStep() {
    const quota = state.settings.provider.quota
    const free = module().freeModels.map((m) => ({ value: m.id, label: m.label }))
    let html =
      c.section(t('ob.quota.section.switching')) +
      c.slider('switchAt', t('ob.quota.switchAt'), t('ob.quota.switchAt.desc'), quota.switchAt, 80, 100, 1, '%') +
      c.slider('planSwitchAt', t('ob.quota.planSwitchAt'), t('ob.quota.planSwitchAt.desc'), quota.planSwitchAt, 70, 100, 1, '%') +
      c.choice('freeModel', t('ob.quota.freeModel'), free, quota.freeModel) +
      c.section(t('ob.quota.section.before')) +
      c.toggle('askFirst', t('ob.quota.askFirst'), t('ob.quota.askFirst.desc'), quota.askFirst)
    if (quota.askFirst) {
      const minutes = [1, 3, 5, 10].map((n) => ({ value: String(n), label: t('ob.quota.minutes', { n }) }))
      html +=
        c.segments('countdown', t('ob.quota.countdown'), minutes, String(quota.countdownMinutes), '', true) +
        c.segments('onNoAnswer', t('ob.quota.onNoAnswer'), [
          { value: 'switch', label: t('ob.quota.onNoAnswer.switch') },
          { value: 'stay', label: t('ob.quota.onNoAnswer.stay') },
        ], quota.onNoAnswer, '', true)
    } else {
      html += c.info('neutral', '', t('ob.quota.noAsk'))
    }
    return html
  }

  /** A secret field, its "check and save" button and the state next to it. */
  function keyBlock({ field, label, status, draft, description }) {
    const saving = status.status === 'checking'
    const saved = status.status === 'saved'
    const stateText = saved ? t('ob.signin.saved') : saving ? t('ob.signin.checking') : status.error ? t(`ob.error.${status.error}`) : ''
    return (
      c.textField(field, label, draft, {
        secret: true,
        revealed: !!state.ui.reveal[field],
        placeholder: t('ob.signin.keyPlaceholder'),
        showText: t('ob.signin.show'),
        hideText: t('ob.signin.hide'),
        description,
      }) +
      c.actionButton(`save-${field}`, saving ? t('ob.signin.checking') : t('ob.signin.check'), stateText, {
        disabled: saving || !draft.trim(),
        tone: saved ? 'ok' : status.error ? 'error' : '',
      })
    )
  }

  function authStep() {
    const m = module()
    let html = ''
    if (m.auth === 'key') {
      html += keyBlock({ field: 'key', label: t('ob.signin.keyLabel', { provider: m.label }), status: state.ui.key, draft: state.ui.drafts.key })
      if (state.ui.signedIn && state.ui.key.status === 'saved') html += c.info('green', '', t('ob.signin.existing'))
    } else {
      const o = state.ui.oauth
      const stateText =
        o.status === 'done' ? t('ob.signin.oauth.done')
        : o.status === 'checking' ? t('ob.signin.checking')
        : o.status === 'waiting' ? t('ob.signin.oauth.waiting')
        : o.error ? t(`ob.error.${o.error}`)
        : t('ob.signin.oauth.hint')
      html += c.actionButton('oauth-start', t('ob.signin.oauth.button'), stateText, {
        disabled: o.status === 'waiting' || o.status === 'done' || o.status === 'checking',
        tone: o.status === 'done' ? 'ok' : o.error ? 'error' : '',
      })
      if (o.status === 'waiting') {
        html += `<div class="action-row">${o.url ? `<button class="link" data-nav data-action="oauth-reopen" data-key="oauth-reopen">${c.esc(t('ob.signin.oauth.reopen'))}</button>` : ''}<button class="link" data-nav data-action="oauth-cancel" data-key="oauth-cancel">${c.esc(t('ob.signin.oauth.cancel'))}</button></div>`
      }
    }
    return html + c.info('lavender', t('ob.signin.stored.title'), t('ob.signin.stored.desc'))
  }

  function routerStep() {
    const s = state.settings
    const included = module().jevIncluded
    // Jev through TypeSafe, for providers that do not include it, is not built yet: without OpenCode Go there is no
    // router to turn on (the provider's defaults leave it off).
    if (!included) return c.bullets([t('ob.router.b1'), t('ob.router.b2'), t('ob.router.b3'), t('ob.router.b4')]) + c.info('amber', t('ob.router.unavailable.title'), t('ob.router.unavailable.desc'))
    let html =
      c.bullets([t('ob.router.b1'), t('ob.router.b2'), t('ob.router.b3'), t('ob.router.b4')]) +
      c.choice('router', t('ob.router.question'), [
        { value: 'yes', label: t('ob.router.yes'), description: included ? t('ob.router.yes.included') : t('ob.router.yes.needsKey') },
        { value: 'no', label: t('ob.router.no'), description: t('ob.router.no.desc') },
      ], s.router.enabled ? 'yes' : 'no')
    if (s.router.enabled && included) html += c.info('green', t('ob.router.included.title'), t('ob.router.included.desc'))
    if (s.router.enabled && !included) {
      html += keyBlock({ field: 'typesafeKey', label: t('ob.router.keyLabel'), status: state.ui.typesafe, draft: state.ui.drafts.typesafeKey, description: t('ob.router.keyDesc') })
    }
    if (!s.router.enabled) html += c.info('amber', t('ob.router.off.title'), t('ob.router.off.desc'))
    return html
  }

  function assistantStep() {
    const a = state.settings.assistant
    let html = c.textField('userName', t('ob.assistant.userName'), a.userName, { placeholder: t('ob.assistant.userName.placeholder'), description: t('ob.assistant.userName.desc') })
    if (state.mode === 'quick' && !state.editing) return html // quick setup asks only your name
    const choice = languageChoice(a.language)
    html += c.segments('language', t('ob.assistant.language'), ['es', 'en', 'pt', 'other'].map((v) => ({ value: v, label: t(`ob.assistant.language.${v}`) })), choice)
    if (choice === 'other') html += c.textField('languageOther', t('ob.assistant.languageOther'), a.language, { placeholder: t('ob.assistant.languageOther.placeholder'), indent: true })
    html += c.choice('style', t('ob.assistant.style'), STYLES.map((v) => ({ value: v, label: t(`ob.style.${v}`), description: i18nHas(`ob.style.${v}.desc`) ? t(`ob.style.${v}.desc`) : '' })), a.style)
    // The shortcut is a setting, not a setup question: the dock works with its default from the first run.
    return state.view === 'settings' ? html + shortcutRow() : html
  }

  /** The Shortcut row: the chosen one (or the default, while the setting is empty or unreadable), recorded by
   * pressing it; under it why a press did not count, or which shortcut the dock really uses when the chosen one is
   * taken. */
  function shortcutRow() {
    const { os } = platform()
    const chosen = parseShortcut(state.settings.shortcut, os)
    const shown = chosen ?? defaultShortcut(os)
    const row = state.ui.shortcut
    const { note, tone } = shortcutNote()
    // A shortcut the desktop takes before the page sees it (one already bound, Aria's own) can only be typed.
    const example = os === 'mac' ? 'Command + Shift + A' : 'SUPER + SHIFT + A'
    return (
      c.keyCapture('shortcut', t('ob.shortcut.label'), t('ob.shortcut.desc'), display(shown).split(' + ').map((key) => keyName(key, t)), {
        recording: row.recording,
        recordingText: t('ob.shortcut.recording'),
        meta: chosen ? '' : t('ob.shortcut.default'),
        note,
        tone,
      }) +
      c.textField('shortcutText', '', row.draft, {
        placeholder: t('ob.shortcut.type', { example }),
        mono: true,
        indent: true,
        button: { action: 'shortcut-apply', text: t('ob.shortcut.apply') },
      })
    )
  }

  /** The line under the Shortcut row: why a press or a typed one did not count, else which shortcut the dock really
   * uses when the chosen one is taken (or that none could be). */
  function shortcutNote() {
    const { os, hotkey: inUse, taken } = platform()
    const row = state.ui.shortcut
    if (row.note) {
      const key = row.note === 'needsModifier' ? `ob.shortcut.needsModifier.${os === 'mac' ? 'mac' : 'linux'}` : `ob.shortcut.${row.note}`
      return { note: t(key), tone: 'error' }
    }
    if (!inUse && taken?.length) return { note: t('ob.shortcut.none', { taken: taken.map(keysText).join(', ') }), tone: 'error' }
    // None in use and none tried: the Mac's key handler could not be installed (the dock tries again).
    if (!inUse) return { note: t('ob.shortcut.failed'), tone: 'error' }
    if (taken?.length) {
      const key = taken.length === 1 ? 'ob.shortcut.inUse.one' : 'ob.shortcut.inUse.many'
      return { note: t(key, { hotkey: keysText(inUse), taken: taken.map(keysText).join(', ') }), tone: '' }
    }
    return { note: '', tone: '' }
  }

  const keysText = (text) => String(text).split(' + ').map((key) => keyName(key, t)).join(' + ')

  /** Recording ends. The press that ended it (if one did) is still down: its release is the row's too (see keyup). */
  function stopRecording(event) {
    state.ui.shortcut = { ...state.ui.shortcut, recording: false, note: null }
    heldKeys.clear()
    if (event) heldKeys.add(event.code)
  }

  /** The typed shortcut, applied (Enter or the button): canonical when it parses, a note when it does not. */
  function applyTypedShortcut() {
    const parsed = parseShortcut(state.ui.shortcut.draft, platform().os)
    if (!parsed) {
      state.ui.shortcut = { ...state.ui.shortcut, recording: false, note: 'typedInvalid' }
      return render()
    }
    update((s) => (s.shortcut = shortcutLabel(parsed)))
    state.ui.shortcut = { recording: false, note: null, draft: '' }
    render()
  }

  /** A key pressed while the Shortcut row records. Every key is the row's (the dock's own Esc included); Tab stops
   * recording and moves on. */
  function recordKey(event) {
    if (event.key === 'Tab') {
      stopRecording()
      return render()
    }
    event.preventDefault()
    event.stopPropagation()
    event.stopImmediatePropagation()
    heldKeys.add(event.code)
    const plain = !anyModifierHeld(event)
    if (event.key === 'Escape' || (plain && event.key === 'Backspace')) {
      if (event.key === 'Backspace') update((s) => (s.shortcut = ''))
      stopRecording(event)
      return render()
    }
    const pressed = fromKeyEvent(event, platform().os)
    if (pressed.pending) return
    if (pressed.error) state.ui.shortcut = { ...state.ui.shortcut, recording: true, note: pressed.error }
    else {
      update((s) => (s.shortcut = pressed.label))
      stopRecording(event)
    }
    render()
  }

  /** The dock sent another `platform` message (the shortcut in use changed): the Shortcut row says so at once. */
  function platformChanged() {
    if (state.view === 'settings' && state.section === 'assistant') render()
  }
  const i18nHas = (key) => t(key) !== key

  function modelsStep() {
    let html =
      c.section(t('ob.models.section.daily')) + modelField('daily') +
      c.section(t('ob.models.section.planning')) + modelField('planning') +
      c.segments('planningEffort', t('ob.models.planningEffort'), LEVELS.map((v) => ({ value: v, label: t(`ob.level.${v}`) })), state.settings.models.planningEffort)
    if (module().quota) html += c.toggle('payPerUse', t('ob.models.payPerUse'), t('ob.models.payPerUse.desc'), state.settings.models.allowPayPerUse)
    const timeout = state.settings.models.turnTimeoutMinutes ?? DEFAULT_TURN_TIMEOUT
    html += c.stepper('turnTimeoutMinutes', t('ob.models.turnTimeout'), t('ob.models.turnTimeout.desc'), timeout, 0, 60, 5, ` ${t('ob.models.minutes')}`)
    return html
  }

  // ---- model fields ---------------------------------------------------------------------------

  const refOf = (m) => `${m.provider}/${m.id}`

  /** The models a field offers: Pi's list for the chosen provider (and OpenCode Zen when pay-per-use is allowed), or
   * the provider's catalog while Pi has not answered (or during the first setup, before Pi runs). */
  function providerModels() {
    const s = state.settings
    const own = PI_PROVIDER[s.provider.id]
    const fromPi = (state.models ?? []).filter((m) => m.provider === own || (own === 'opencode-go' && s.models.allowPayPerUse && m.provider === 'opencode'))
    if (fromPi.length) return fromPi
    return (module()?.models ?? []).map((m) => {
      const [provider, ...id] = m.id.split('/')
      return { provider, id: id.join('/'), name: m.label }
    })
  }

  const findModel = (ref) => providerModels().find((m) => refOf(m) === ref) ?? (state.models ?? []).find((m) => refOf(m) === ref)

  /** The field shows the model in use; pressing it opens the chat's model picker under it, with its search. */
  function modelField(field) {
    const ref = state.settings.models[field] ?? ''
    const model = findModel(ref)
    const open = state.picker?.field === field
    let html = `
      <button type="button" class="opt model-field${open ? ' selected' : ''}" data-nav data-action="model-open" data-field="${c.esc(field)}" data-key="model-open:${c.esc(field)}" aria-expanded="${open}">
        <span class="opt-text"><span class="opt-label">${c.esc(model?.name || ref.split('/').pop() || '—')}</span><span class="opt-desc mono">${c.esc(ref)}</span></span>
        <span class="model-field-caret" aria-hidden="true">${open ? '▴' : '▾'}</span>
      </button>`
    if (open) html += `<div class="model-field-picker" data-model-picker>${modelPickerHtml()}</div>`
    return html
  }

  const modelPickerHtml = () =>
    renderModelPicker({ models: providerModels(), current: state.settings.models[state.picker.field] ?? '', query: state.picker.query, active: state.picker.active, open: true }, {}, t)

  /** Redraws only the open picker (typing in its search must not lose the caret), and keeps its marked row in view. */
  function redrawPicker() {
    const box = root.querySelector('[data-model-picker]')
    if (!box) return render()
    box.innerHTML = modelPickerHtml()
    focusModelSearch()
    box.querySelector('.p-models-row.on')?.scrollIntoView?.({ block: 'nearest' })
  }

  function focusModelSearch() {
    const input = root.querySelector('[data-chat-field="model-search"]')
    if (!input) return
    input.focus({ preventScroll: true })
    input.setSelectionRange?.(input.value.length, input.value.length)
  }

  function toggleModelPicker(field) {
    if (state.picker?.field === field) return closeModelPicker()
    const at = orderedModels(providerModels(), '').findIndex(({ model }) => refOf(model) === state.settings.models[field])
    state.picker = { field, query: '', active: Math.max(0, at) }
    render()
    root.querySelector('[data-model-picker] .p-models-row.on')?.scrollIntoView?.({ block: 'nearest' })
    if (state.models === null) void loadModels()
  }

  function closeModelPicker() {
    if (!state.picker) return
    focusAfterRender = `model-open:${state.picker.field}`
    state.picker = null
    render()
  }

  function pickModel(index) {
    const entry = orderedModels(providerModels(), state.picker.query)[index]
    if (!entry) return
    const { field } = state.picker
    update((s) => {
      s.models[field] = refOf(entry.model)
    })
    closeModelPicker()
  }

  /** Asks Pi for its models, once per visit to the section (a provider connected meanwhile shows up). */
  async function loadModels() {
    if (!env.listModels || state.view !== 'settings') return
    const models = await env.listModels().catch(() => [])
    state.models = Array.isArray(models) ? models : []
    if (state.view !== 'settings') return
    state.picker ? redrawPicker() : render()
  }

  function effortStep() {
    const e = state.settings.effort
    const routerOn = state.settings.router.enabled
    const value = e.mode === 'auto' ? 'auto' : e.fixedLevel
    const options = [
      { value: 'auto', label: t('ob.effort.auto'), disabled: !routerOn },
      ...['low', 'medium', 'high', 'max'].map((v) => ({ value: v, label: t(`ob.level.${v}`) })),
    ]
    let html = c.segments('effort', t('ob.effort.label'), options, value, t(`ob.effort.note.${value}`))
    if (!routerOn) html += c.info('amber', t('ob.effort.needsRouter.title'), t('ob.effort.needsRouter.desc'))
    if (e.mode === 'auto' && module().freeModels.length) html += c.toggle('maxOnFree', t('ob.effort.maxOnFree'), t('ob.effort.maxOnFree.desc'), e.maxOnFree)
    return html
  }

  function compactionStep() {
    const percent = state.settings.compaction.percent
    return (
      c.slider('compactionPercent', t('ob.compaction.compactAt'), t('ob.compaction.compactAt.desc'), percent, 20, 80, 5, '%') +
      `<div data-meter>${c.meter(percent, t('ob.compaction.caption', { n: percent }))}</div>` +
      c.info('neutral', '', t(state.settings.memory.enabled ? 'ob.compaction.noteMemory' : 'ob.compaction.noteNoMemory'))
    )
  }

  function memoryStep() {
    const m = state.settings.memory
    let html = c.toggle('memory', t('ob.memory.toggle'), t('ob.memory.toggle.desc'), m.enabled)
    if (m.enabled) {
      html += c.textField('memoryProject', t('ob.memory.project'), m.project, { mono: true, indent: true, placeholder: 'aria', description: t('ob.memory.project.desc') })
      html += c.info('neutral', '', t('ob.memory.words'), true)
    } else {
      html += c.info('amber', t('ob.memory.off.title'), t('ob.memory.off.desc'))
    }
    return html
  }

  const serviceOf = (id) => state.settings.integrations.services.find((s) => s.id === id)
  const isConnected = (id) => state.connected.has(`mcp:${id}`)

  const connectedLine = () => c.info('green', '', t('ob.integrations.connected'), true)

  /** Tavily: its key is checked with Tavily and stored in mcp.json. */
  function tavilyBlock() {
    // Connected is the end of it: nothing left to type or press.
    if (isConnected('tavily')) return connectedLine()
    const tavily = state.svc.tavily
    const checking = tavily.status === 'checking'
    const stateText = checking ? t('ob.signin.checking') : tavily.error ? t(`ob.error.${tavily.error}`) : ''
    return (
      c.textField('tavilyKey', t('ob.integrations.tavily.keyLabel'), tavily.draft, {
        secret: true,
        revealed: !!state.ui.reveal.tavilyKey,
        placeholder: t('ob.signin.keyPlaceholder'),
        showText: t('ob.signin.show'),
        hideText: t('ob.signin.hide'),
        description: t('ob.integrations.tavily.keyDesc'),
        indent: true,
      }) +
      c.actionButton('save-tavilyKey', checking ? t('ob.signin.checking') : t('ob.signin.check'), stateText, {
        disabled: checking || !tavily.draft.trim(),
        tone: tavily.error ? 'error' : '',
      })
    )
  }

  /** Notion: sign in through the browser; the tokens go to mcp-auth.json. */
  function notionBlock() {
    if (isConnected('notion')) return connectedLine()
    const notion = state.svc.notion
    const waiting = notion.status === 'waiting'
    const stateText = waiting ? t('ob.integrations.notion.waiting') : notion.error ? t(`ob.error.${notion.error}`) : t('ob.integrations.notion.hint')
    let html = c.actionButton('mcp-oauth-start', t('ob.integrations.notion.signin'), stateText, {
      disabled: waiting,
      tone: notion.error ? 'error' : '',
    })
    if (waiting) {
      html += `<div class="action-row">${notion.url ? `<button class="link" data-nav data-action="mcp-oauth-reopen" data-key="mcp-oauth-reopen">${c.esc(t('ob.signin.oauth.reopen'))}</button>` : ''}<button class="link" data-nav data-action="mcp-oauth-cancel" data-key="mcp-oauth-cancel">${c.esc(t('ob.signin.oauth.cancel'))}</button></div>`
    }
    return html
  }

  function integrationsStep() {
    const s = state.settings
    let html = c.section(t('ob.integrations.section.services'))
    for (const id of SERVICES) {
      html += c.toggle(`service:${id}`, t(`ob.integrations.${id}`), t(`ob.integrations.${id}.desc`), !!serviceOf(id))
      if (serviceOf(id)) html += id === 'notion' ? notionBlock() : tavilyBlock()
    }
    if (s.integrations.services.some((service) => !isConnected(service.id))) html += c.info('amber', '', t('ob.integrations.needConnect'))
    html += accessRows()
    html += foundBlock()
    const p = s.integrations.packages
    html += c.section(t('ob.integrations.section.packages'))
    // Only a local folder is built (npm, git and a gallery are not): the field is it, and it is used when it names one.
    html += c.textField('packagePath', t('ob.integrations.packagePath'), p.source === 'local' || p.source === 'path' ? p.path : '', { mono: true, placeholder: t('ob.integrations.packagePath.placeholder'), description: t('ob.integrations.packagePath.desc') })
    return html + (s.subagents.planExecute ? c.info('green', t('ob.integrations.auto.title'), t('ob.integrations.auto.desc')) : c.info('neutral', '', t('ob.integrations.auto.none')))
  }

  /** One row per configured MCP server (the dock's `servers`, plus any `mcp:<server>` connected meanwhile or listed by
   * an older dock that sends no `servers`): Read only / Ask / Allow, for what it may change without asking. */
  function accessRows() {
    const fromConnected = [...state.connected].filter((id) => id.startsWith('mcp:')).map((id) => id.slice(4))
    const servers = [...new Set([...state.servers, ...fromConnected])].filter((server) => typeof server === 'string' && server && !UNGATED.has(server)).sort()
    if (!servers.length) return ''
    const access = state.settings.integrations.access ?? {}
    const options = ACCESS_LEVELS.map((level) => ({ value: level, label: t(`ob.integrations.access.${level}`) }))
    return (
      c.section(t('ob.integrations.section.access')) +
      c.info('neutral', '', t('ob.integrations.access.desc')) +
      servers.map((server) => c.segments(`access:${server}`, displayName(server), options, ACCESS_LEVELS.includes(access[server]) ? access[server] : 'ask')).join('')
    )
  }

  /** "Your other MCP servers": what Claude Code, Claude Desktop, Cursor and VS Code have, to copy into Aria's mcp.json.
   * It never blocks Continue: importing is a choice, not a step to finish. */
  function foundBlock() {
    const f = state.found
    let html = c.section(t('ob.integrations.found.section'))
    if (f.status === 'idle' || f.status === 'scanning') return html + c.info('neutral', '', t('ob.integrations.found.scanning'))
    if (f.status === 'error') return html + c.info('amber', '', t(`ob.error.${f.error}`))
    const result = f.result ? c.info('green', '', t(f.result.length === 1 ? 'ob.integrations.found.imported.one' : 'ob.integrations.found.imported.many', { n: f.result.length, names: f.result.join(', ') })) : ''
    if (!f.servers.length) return html + (result || c.info('neutral', '', t('ob.integrations.found.none')))
    const importable = f.servers.filter((server) => server.importable)
    html += c.info('neutral', '', t('ob.integrations.found.desc'))
    if (importable.length) {
      html += c.checks('mcpImport', '', importable.map((server) => ({ id: server.id, label: foundName(server), description: foundDetails(server), on: f.picked.includes(server.id) })))
    }
    for (const server of f.servers.filter((server) => !server.importable)) html += c.info('neutral', foundName(server), foundDetails(server))
    if (importable.length) {
      const stateText = f.importing ? t('ob.integrations.found.importing') : f.error ? t(importErrorKey(f.error)) : ''
      html += c.actionButton('mcp-import', t('ob.integrations.found.import'), stateText, { disabled: f.importing || !f.picked.length, tone: f.error ? 'error' : '' })
    }
    return html + result
  }

  /** An import's error, said for the import: a broken `mcp.json` means nothing was imported. */
  const importErrorKey = (code) => (code === 'mcp-file-broken' ? 'ob.integrations.found.fileBroken' : `ob.error.${code}`)

  /** The name, and the one Aria gives it when Pi would refuse it ("my docs → my-docs"). */
  const foundName = (server) => (server.importAs && server.importAs !== server.name ? `${server.name} → ${server.importAs}` : String(server.name))

  /** Where it came from, what it runs, the secrets it uses (names only), and whether it is off in that app. */
  function foundDetails(server) {
    const secrets = Array.isArray(server.secrets) ? server.secrets.map(String) : []
    const names = secrets.filter((secret) => !secret.startsWith('input:'))
    const parts = [t(`ob.integrations.found.source.${server.source}`), server.target]
    if (names.length) parts.push(t('ob.integrations.found.uses', { names: names.join(', ') }))
    for (const secret of secrets.filter((secret) => secret.startsWith('input:'))) parts.push(t('ob.integrations.found.input', { id: secret.slice(6) }))
    parts.push(...foundWarnings(server))
    if (server.disabled) parts.push(t('ob.integrations.found.disabled'))
    if (!server.importable) parts.push(t('ob.integrations.found.sse'))
    return parts.filter(Boolean).join(' · ')
  }

  /** What may not work the same in Aria (the dock's `warnings`), in plain words, one line per kind: `expands:<NAME>`,
   * `default:<NAME>`, `unsupported:<what>`, `tilde-arg`, `envFile`. */
  function foundWarnings(server) {
    const warnings = Array.isArray(server.warnings) ? server.warnings.map(String) : []
    const named = (prefix, shown = (name) => name) => warnings.filter((w) => w.startsWith(`${prefix}:`)).map((w) => shown(w.slice(prefix.length + 1)))
    const lines = []
    const expands = named('expands')
    if (expands.length) lines.push(t('ob.integrations.found.warn.expands', { names: expands.join(', ') }))
    const defaults = named('default')
    if (defaults.length) lines.push(t('ob.integrations.found.warn.default', { names: defaults.join(', ') }))
    const unsupported = named('unsupported', (what) => `\${${what}}`)
    if (unsupported.length) lines.push(t('ob.integrations.found.warn.unsupported', { names: unsupported.join(', ') }))
    if (warnings.includes('tilde-arg')) lines.push(t('ob.integrations.found.warn.tildeArg'))
    if (warnings.includes('envFile')) lines.push(t('ob.integrations.found.warn.envFile'))
    return lines
  }

  function sandboxStep() {
    const sb = state.settings.sandbox
    let html =
      c.info('neutral', '', t('ob.sandbox.intro')) +
      c.textField('projectsDir', t('ob.sandbox.projectsDir'), sb.projectsDir, { mono: true, placeholder: '~/Projects' }) +
      c.chips('extraWritable', t('ob.sandbox.extraWritable'), sb.extraWritable, state.ui.drafts.extraWritable, t('ob.sandbox.extraWritable.placeholder'), t('ob.removeItem'), t('ob.addItem')) +
      c.toggle('askToLeave', t('ob.sandbox.askToLeave'), t('ob.sandbox.askToLeave.desc'), sb.askToLeave)
    if (state.settings.router.enabled) html += c.toggle('jevReview', t('ob.sandbox.jevReview'), t('ob.sandbox.jevReview.desc'), sb.jevReview)
    return html
  }

  function gitStep() {
    const g = state.settings.git
    return (
      c.toggle('commit', t('ob.git.commit'), t('ob.git.commit.desc'), g.commitWithoutAsking) +
      c.toggle('push', t('ob.git.push'), t('ob.git.push.desc'), g.pushOrMergeWithoutAsking) +
      c.toggle('others', t('ob.git.others'), t('ob.git.others.desc'), g.askBeforeOthersChanges) +
      c.toggle('proposals', t('ob.git.proposals'), t('ob.git.proposals.desc'), g.unaskedAsProposals)
    )
  }

  function agentsStep() {
    const a = state.settings.subagents
    const rows = a.experts.map((e) => ({ id: e.name, label: e.name, description: t(`ob.agents.${e.name}`), enabled: e.enabled, readOnly: e.readOnly }))
    let html =
      c.section(t('ob.agents.section.experts')) +
      c.matrix('experts', [t('ob.agents.on'), t('ob.agents.readOnly')], rows) +
      c.section(t('ob.agents.section.sessions')) +
      c.toggle('saveSessions', t('ob.agents.saveSessions'), t('ob.agents.saveSessions.desc'), a.saveSessions) +
      c.toggle('limitTurns', t('ob.agents.limitTurns'), t('ob.agents.limitTurns.desc'), a.turnLimit !== null)
    if (a.turnLimit !== null) html += c.stepper('turnLimit', t('ob.agents.turnLimit'), '', a.turnLimit, 5, 100, 5, ` ${t('ob.agents.turns', { n: '' }).trim()}`)
    html += c.section(t('ob.agents.section.superpowers')) + c.toggle('planExecute', t('ob.agents.planExecute'), t('ob.agents.planExecute.desc'), a.planExecute)
    if (a.planExecute) {
      html += c.checks('skills', t('ob.agents.skills'), SKILLS.map((id) => ({ id, label: t(`ob.skill.${id}`), description: t(`ob.skill.${id}.desc`), on: a.autoSkills.includes(id) })))
    }
    return html
  }

  const modelLabel = (ref) => findModel(ref)?.name || ref?.split('/').pop() || '—'

  /** One line per step saying what it is set to: the review list and the settings overview read it. */
  function summaryValues() {
    const s = state.settings
    const m = module()
    const q = s.provider.quota
    const language = languageChoice(s.assistant.language) === 'other' ? s.assistant.language : t(`ob.assistant.language.${s.assistant.language}`)
    const services = s.integrations.services.map((x) => t(`ob.integrations.${x.id}`))
    const experts = s.subagents.experts.filter((e) => e.enabled).length
    return {
      provider: (m?.label ?? '—') + (m?.quota ? t('ob.val.quotaMeter') : ''),
      quota: q
        ? t('ob.val.quota', { a: q.switchAt, b: q.planSwitchAt, ask: q.askFirst ? t('ob.val.quotaAsk', { n: q.countdownMinutes }) : t('ob.val.quotaNoAsk') })
        : t('ob.val.quotaNone'),
      auth: t(m?.auth === 'oauth' ? 'ob.val.authOauth' : 'ob.val.authKey'),
      assistant: t('ob.val.assistant', { name: assistantName(), language, style: t(`ob.style.${s.assistant.style}`) }) + (s.assistant.userName.trim() ? t('ob.val.callsYou', { user: s.assistant.userName.trim() }) : ''),
      router: s.router.enabled && m?.jevIncluded ? t('ob.val.routerOn') : t('ob.val.off'),
      models: t('ob.val.models', { daily: modelLabel(s.models.daily), planning: modelLabel(s.models.planning), effort: t(`ob.level.${s.models.planningEffort}`) }),
      effort: s.effort.mode === 'auto' ? t('ob.val.effortAuto') : t(`ob.level.${s.effort.fixedLevel}`),
      compaction: t('ob.val.compaction', { n: s.compaction.percent }),
      memory: s.memory.enabled ? t('ob.val.memoryOn', { project: s.memory.project }) : t('ob.val.off'),
      integrations: services.length ? services.join(', ') : t('ob.val.integrationsNone'),
      sandbox: t('ob.val.sandbox', { dir: s.sandbox.projectsDir, more: s.sandbox.extraWritable.length ? t('ob.val.sandboxMore', { n: s.sandbox.extraWritable.length }) : '', ask: t(s.sandbox.askToLeave ? 'ob.val.sandboxAsks' : 'ob.val.sandboxFree') }),
      git: t('ob.val.git', { commit: t(s.git.commitWithoutAsking ? 'ob.val.commitAlone' : 'ob.val.commitAsks'), push: t(s.git.pushOrMergeWithoutAsking ? 'ob.val.pushAlone' : 'ob.val.pushAsks') }),
      agents: t('ob.val.agents', { n: experts, plan: t(s.subagents.planExecute ? 'ob.val.planOn' : 'ob.val.planOff'), sessions: s.subagents.saveSessions ? t('ob.val.sessionsSaved') : '' }),
    }
  }

  function reviewStep() {
    const quickFlow = state.mode === 'quick'
    const values = summaryValues()
    const order = quickFlow
      ? ['provider', 'auth', 'assistant']
      : ['provider', 'quota', 'auth', 'assistant', 'router', 'models', 'effort', 'compaction', 'memory', 'integrations', 'sandbox', 'git', 'agents']
    const wanted = new Set(ids())
    let html = c.summary(order.map((id) => ({ id, label: t(`ob.row.${id}`), value: values[id], editLabel: wanted.has(id) ? t('ob.edit') : '' })))
    if (quickFlow) html += c.info('lavender', t('ob.review.defaultsNote.title'), t('ob.review.defaultsNote.desc'))
    if (state.error) html += c.info('amber', '', t(`ob.error.${state.error}`))
    return html
  }

  // ---- settings -------------------------------------------------------------------------------

  /** The sections with only the steps this provider has. */
  const settingsSections = () => {
    const have = new Set(ids())
    return SETTINGS_SECTIONS.map((section) => ({ ...section, steps: section.steps.filter((step) => have.has(step)) }))
  }
  const isDirty = () => state.view === 'settings' && JSON.stringify(state.settings) !== state.saved
  /** The sections with something to finish before saving (a sign-in, a service turned on but not connected). */
  const blockedSections = () => settingsSections().filter((section) => section.steps.some((step) => !canContinue(step, state.settings, uiFlags())))

  function settingsHtml() {
    return state.section ? settingsSectionHtml() : settingsOverviewHtml()
  }

  /** The overview's corner: where the settings live, and which Aria this is (`Aria 0.2.0`) once the dock said. */
  function settingsMeta() {
    const version = env.version?.()
    return version ? `~/.aria/settings.json · Aria ${version}` : '~/.aria/settings.json'
  }

  function settingsOverviewHtml() {
    const values = summaryValues()
    const blocked = new Set(blockedSections().map((section) => section.id))
    const cards = settingsSections()
      .map((section) => `
        <button class="card settings-card" data-nav data-action="settings-open" data-value="${c.esc(section.id)}" data-key="settings-open:${c.esc(section.id)}">
          <div class="row"><b>${c.esc(t(`settings.section.${section.id}`))}</b>${blocked.has(section.id) ? `<span class="meta warn">${c.esc(t('settings.attention'))}</span>` : ''}</div>
          <div class="description">${c.esc(section.steps.map((step) => values[step]).join(' · '))}</div>
        </button>`)
      .join('')
    return `
      ${renderPanelHead({ back: { attrs: 'data-nav data-action="settings-exit" data-key="settings-exit"', label: t('settings.back') }, title: t('settings.title'), meta: settingsMeta() })}
      <p class="intro settings-intro">${c.esc(t('settings.desc'))}</p>
      <div class="settings-cards">${cards}</div>
      ${settingsBar()}`
  }

  function settingsSectionHtml() {
    const section = settingsSections().find((s) => s.id === state.section)
    const blocks = section.steps
      .map((step) => `
        <section class="settings-block">
          <h3>${c.esc(t(`ob.${step}.title`))}</h3>
          <p class="hint">${c.esc(t(`ob.${step}.desc`))}</p>
          <div class="fields">${sectionsByStep[step]()}</div>
        </section>`)
      .join('')
    return `
      ${renderPanelHead({ back: { attrs: 'data-nav data-action="settings-home" data-key="settings-home"', label: t('settings.home') }, title: t(`settings.section.${section.id}`), meta: t('settings.title') })}
      ${blocks}
      ${settingsBar()}`
  }

  /** Save and Discard, shown while something changed (or right after a save, to say it was saved). */
  function settingsBar() {
    const dirty = isDirty()
    if (!dirty && !state.flash && !state.error) return ''
    const blocked = blockedSections()
    const canSave = dirty && !blocked.length && !state.saving
    const text = state.saving
      ? t('settings.saving')
      : state.error
        ? t(`ob.error.${state.error}`)
        : !dirty
          ? t('settings.saved')
          : blocked.length
            ? t('settings.blocked', { sections: blocked.map((section) => t(`settings.section.${section.id}`)).join(', ') })
            : state.leaving
              ? t('settings.leave')
              : t('settings.unsaved')
    const tone = state.error || (dirty && blocked.length) ? 'error' : !dirty ? 'ok' : ''
    return `
      <div class="settings-bar${state.leaving ? ' leaving' : ''}" role="status">
        <span class="state ${tone}">${c.esc(text)}</span>
        ${dirty ? `<button type="button" class="secondary" data-nav data-action="settings-discard" data-key="settings-discard">${c.esc(t('settings.discard'))}</button>
        <button type="button" class="primary" data-nav data-action="settings-save" data-key="settings-save" ${canSave ? '' : 'disabled'}>${c.esc(t('settings.save'))}</button>` : ''}
      </div>`
  }

  function openSection(id) {
    state.section = id
    state.leaving = false
    state.picker = null
    if (id === 'models') {
      state.models = null
      void loadModels()
    }
    render()
    // Opening the sign-in's section is what checks a credential already on disk.
    maybeCheckStoredCredential()
    if (id === 'integrations') void scanFound()
  }

  function exitSettings() {
    if (isDirty()) {
      state.leaving = true
      return render()
    }
    state.section = null
    env.onDone ? env.onDone() : post({ type: 'closeRequest' })
  }

  function discardSettings() {
    const before = state.settings.assistant.language
    const leave = state.leaving
    state.settings = JSON.parse(state.saved)
    state.error = null
    state.leaving = false
    // The interface follows the language as it is picked; going back to the saved one takes it back too.
    const language = state.settings.assistant.language
    if (language !== before && languageChoice(language) !== 'other') env.setLanguage(language)
    if (leave) return exitSettings()
    render()
  }

  let flashTimer = null

  async function saveSettings() {
    if (!isDirty() || blockedSections().length || state.saving) return
    const settings = finish(state.settings)
    state.saving = true
    state.error = null
    render()
    const reply = await request({ type: 'saveSettings', settings })
    state.saving = false
    if (reply.type !== 'saved') {
      state.error = reply.code ?? 'generic'
      return render()
    }
    state.saved = JSON.stringify(settings)
    // What the user changed meanwhile stays theirs; only the setup flag comes from the save.
    state.settings = { ...state.settings, setup: settings.setup }
    if (state.leaving) {
      state.leaving = false
      return exitSettings()
    }
    state.flash = true
    clearTimeout(flashTimer)
    flashTimer = setTimeout(() => {
      state.flash = false
      if (state.view === 'settings') render()
    }, SAVED_FLASH_MS)
    render()
  }

  /** settings.json changed outside the page (/effort, /autocompact): the screen shows it, unless the user has changes
   * of their own on it, which are not overwritten under their hands. */
  function refreshSettings(settings, connected = [], servers = []) {
    if (state.view !== 'settings' || isDirty() || state.saving || !settings?.setup?.completed) return
    state.settings = settings
    state.saved = JSON.stringify(settings)
    state.connected = new Set(connected)
    state.servers = Array.isArray(servers) ? servers : []
    state.ui.signedIn = isSignedIn(settings, connected)
    render()
  }

  // ---- navigation -----------------------------------------------------------------------------

  const go = (step) => {
    state.step = step
    state.view = 'step'
    state.error = null
    render()
    // Drawing has no side effects; entering the sign-in step is what checks a credential already on disk.
    maybeCheckStoredCredential()
    if (step === 'integrations') void scanFound()
  }

  function startFlow(mode) {
    state.mode = mode
    state.editing = false
    go('provider')
  }

  async function saveAll() {
    state.saving = true
    state.error = null
    render()
    const reply = await request({ type: 'saveSettings', settings: finish(state.settings, state.mode) })
    state.saving = false
    if (reply.type === 'saved') {
      state.view = 'done'
    } else {
      state.error = reply.code ?? 'generic'
    }
    render()
  }

  async function skipSetup() {
    await request({ type: 'saveSettings', settings: skip(state.settings) })
    post({ type: 'closeRequest' })
  }

  function back() {
    const previous = previousStep(ids(), state.step)
    if (previous !== 'welcome') return go(previous)
    // Before the first step: the welcome screen.
    state.view = 'welcome'
    state.error = null
    render()
  }

  const next = () => {
    if (!canContinue(state.step, state.settings, uiFlags()) || state.saving) return
    const following = nextStep(ids(), state.step)
    following ? go(following) : saveAll()
  }

  // ---- changing settings ----------------------------------------------------------------------

  function pick(field, value) {
    const [name] = field.split(':')
    if (name === 'provider') {
      if (value === state.settings.provider.id) return
      state.settings = applyProvider(state.settings, value)
      const keep = state.ui.drafts
      state.ui = {
        ...freshUi(),
        drafts: { ...keep, key: '' },
        // An API key is signed in only after it is checked (checkStoredKey); an OAuth credential cannot
        // be checked from here, so being on disk is enough.
        signedIn: providerModule(value)?.auth === 'oauth' ? isSignedIn(state.settings, [...state.connected]) : false,
        typesafeSaved: state.connected.has('typesafe:api_key'),
      }
    } else if (name === 'language') {
      state.settings = setLanguageChoice(state.settings, value, languageChoice(state.settings.assistant.language) === 'other' ? state.settings.assistant.language : '')
      if (value !== 'other') env.setLanguage(value)
    } else if (name === 'router') {
      state.settings = setRouter(state.settings, value === 'yes')
    } else if (name === 'access') {
      // The server's name is everything after the first colon; only the three levels are ever written.
      if (!ACCESS_LEVELS.includes(value)) return
      const server = field.slice(field.indexOf(':') + 1)
      update((s) => {
        s.integrations.access = { ...(s.integrations.access ?? {}), [server]: value }
      })
    } else if (name === 'effort') {
      update((s) => {
        s.effort = value === 'auto' ? { ...s.effort, mode: 'auto', fixedLevel: null } : { ...s.effort, mode: 'fixed', fixedLevel: value }
      })
    } else {
      update((s) => {
        if (name === 'style') s.assistant.style = value
        else if (name === 'daily') s.models.daily = value
        else if (name === 'planning') s.models.planning = value
        else if (name === 'planningEffort') s.models.planningEffort = value
        else if (name === 'freeModel') s.provider.quota.freeModel = value
        else if (name === 'countdown') s.provider.quota.countdownMinutes = Number(value)
        else if (name === 'onNoAnswer') s.provider.quota.onNoAnswer = value
      })
    }
    render()
  }

  function toggleField(field) {
    const [name, id] = field.split(':')
    update((s) => {
      if (name === 'service') {
        const at = s.integrations.services.findIndex((x) => x.id === id)
        if (at > -1) s.integrations.services.splice(at, 1)
        else s.integrations.services.push({ id, name: id, auth: id === 'tavily' ? 'key' : 'oauth' })
      } else if (name === 'askFirst') s.provider.quota.askFirst = !s.provider.quota.askFirst
      else if (name === 'payPerUse') s.models.allowPayPerUse = !s.models.allowPayPerUse
      else if (name === 'maxOnFree') s.effort.maxOnFree = !s.effort.maxOnFree
      else if (name === 'memory') s.memory.enabled = !s.memory.enabled
      else if (name === 'askToLeave') s.sandbox.askToLeave = !s.sandbox.askToLeave
      else if (name === 'jevReview') s.sandbox.jevReview = !s.sandbox.jevReview
      else if (name === 'commit') s.git.commitWithoutAsking = !s.git.commitWithoutAsking
      else if (name === 'push') s.git.pushOrMergeWithoutAsking = !s.git.pushOrMergeWithoutAsking
      else if (name === 'others') s.git.askBeforeOthersChanges = !s.git.askBeforeOthersChanges
      else if (name === 'proposals') s.git.unaskedAsProposals = !s.git.unaskedAsProposals
      else if (name === 'saveSessions') s.subagents.saveSessions = !s.subagents.saveSessions
      else if (name === 'limitTurns') s.subagents.turnLimit = s.subagents.turnLimit === null ? DEFAULT_TURN_LIMIT : null
      else if (name === 'planExecute') s.subagents.planExecute = !s.subagents.planExecute
    })
    render()
  }

  /** Typing never re-renders (it would drop the cursor); it updates the state and the Continue button. */
  function typed(field, value) {
    if (field === 'tavilyKey') {
      const tavily = state.svc.tavily
      tavily.draft = value
      if (tavily.status !== 'checking') tavily.status = 'idle'
      tavily.error = null
      const button = root.querySelector('[data-action="save-tavilyKey"]')
      if (button) button.disabled = !value.trim()
      return
    }
    if (field === 'shortcutText') {
      state.ui.shortcut.draft = value
      // Typing again: the note about the last one goes (in place, so the field keeps its focus and caret).
      if (state.ui.shortcut.note) {
        state.ui.shortcut.note = null
        const line = root.querySelector('[data-action="key-capture"][data-field="shortcut"]')?.parentElement?.querySelector('.state')
        const { note, tone } = shortcutNote()
        if (line && note) {
          line.className = `state ${tone}`
          line.textContent = note
        } else line?.remove()
      }
      return
    }
    if (field === 'key' || field === 'typesafeKey') {
      state.ui.drafts[field] = value
      const status = field === 'key' ? state.ui.key : state.ui.typesafe
      if (status.status !== 'checking') status.status = 'idle'
      status.error = null
      const button = root.querySelector(`[data-action="save-${field}"]`)
      if (button) button.disabled = !value.trim()
      return
    }
    update((s) => {
      if (field === 'userName') s.assistant.userName = value
      else if (field === 'languageOther') s.assistant.language = value
      else if (field === 'memoryProject') s.memory.project = value
      else if (field === 'projectsDir') s.sandbox.projectsDir = value
      else if (field === 'packagePath') s.integrations.packages = { ...s.integrations.packages, path: value, source: value.trim() ? 'local' : 'gallery' }
    })
  }

  function slid(field, input) {
    const value = Number(input.value)
    update((s) => {
      if (field === 'switchAt') s.provider.quota.switchAt = value
      else if (field === 'planSwitchAt') s.provider.quota.planSwitchAt = value
      else if (field === 'compactionPercent') s.compaction.percent = value
    })
    const readout = root.querySelector(`[data-readout="${field}"]`)
    if (readout) readout.textContent = `${value}${input.dataset.unit ?? ''}`
    if (field === 'compactionPercent') root.querySelector('[data-meter]').innerHTML = c.meter(value, t('ob.compaction.caption', { n: value }))
  }

  // ---- keys and sign-in -----------------------------------------------------------------------

  // A reply can arrive after the user moved on (changed provider, cancelled): `state.ui` is replaced
  // when the provider changes, so a continuation that finds another `ui` has been overtaken and
  // must not touch the state.
  async function saveKey(field) {
    const ui = state.ui
    const status = field === 'key' ? ui.key : ui.typesafe
    const draft = ui.drafts[field]
    if (status.status === 'checking' || !draft.trim()) return
    // A key the user typed replaces the stored one: the automatic check of an existing key must not
    // fire after this attempt, whichever way it ends.
    if (field === 'key') ui.key.requested = true
    status.status = 'checking'
    status.error = null
    render()
    const provider = field === 'typesafeKey' ? 'typesafe' : module().keyProvider
    const credential = field === 'typesafeKey' ? 'typesafe:api_key' : credentialFor(state.settings.provider.id)
    const at = here()
    const reply = await request({ type: 'saveKey', provider, key: draft })
    if (reply.type === 'keyResult' && reply.ok) state.connected.add(credential) // it IS stored, whatever the page shows now
    if (state.ui !== ui) return
    if (reply.type === 'keyResult' && reply.ok) {
      status.status = 'saved'
      if (field === 'key') {
        ui.signedIn = true
        ui.drafts[field] = ''
        render()
        // A key that just worked does not need a second confirmation: the flow goes on (quick and full).
        if (!state.editing && here() === at) next()
        return
      }
      ui.typesafeSaved = true
      ui.drafts[field] = ''
      render()
      return
    }
    status.status = 'idle'
    status.error = reply.error ?? reply.code ?? 'generic'
    render()
  }

  /** A credential that is already on disk is validated once, when the sign-in step opens. If it still
   * works the flow goes on by itself (quick or full) instead of asking for it again. */
  function maybeCheckStoredCredential() {
    const onSignIn = (state.view === 'step' && state.step === 'auth') || (state.view === 'settings' && state.section === 'provider')
    if (!onSignIn) return
    const m = module()
    if (!m || !state.connected.has(credentialFor(state.settings.provider.id))) return
    const slot = m.auth === 'key' ? state.ui.key : state.ui.oauth
    if (slot.requested) return
    if (m.auth === 'key' && slot.status === 'saved') return
    if (m.auth === 'oauth' && slot.status === 'done') return
    slot.requested = true
    void checkStoredCredential(m)
  }

  async function checkStoredCredential(m) {
    const ui = state.ui
    const slot = m.auth === 'key' ? ui.key : ui.oauth
    slot.status = 'checking'
    slot.error = null
    render()
    const at = here()
    const reply = await request({ type: 'checkKey', provider: m.keyProvider ?? state.settings.provider.id })
    if (state.ui !== ui) return
    // A check that cannot reach the provider does not disprove a credential that is on disk: the dock and
    // Pi will say so when it is really used. Only a rejected credential stops the flow.
    const valid = (reply.type === 'keyResult' && reply.ok) || reply.error === 'network-unavailable'
    if (valid) {
      slot.status = m.auth === 'oauth' ? 'done' : 'saved'
      ui.signedIn = true
      state.connected.add(credentialFor(state.settings.provider.id))
      render()
      // Only a fresh setup moves on its own; while editing, the user decides (they came here to change it).
      if (!state.editing && here() === at && (m.auth === 'oauth' || !ui.drafts.key.trim())) next()
      return
    }
    slot.status = 'idle'
    slot.error = reply.error ?? reply.code ?? 'generic'
    ui.signedIn = false
    render()
  }

  async function startOauth() {
    const ui = state.ui
    const o = ui.oauth
    if (o.status === 'waiting') return
    const token = ++o.token
    const credential = credentialFor(state.settings.provider.id)
    const current = () => state.ui === ui && o.token === token
    o.status = 'waiting'
    o.error = null
    o.url = null
    render()
    const at = here()
    const reply = await request({ type: 'startOauth' }, (progress) => {
      if (!current()) return
      o.url = progress.url
      render()
    })
    if (o.token !== token) return // cancelled or replaced by a newer attempt: ignored
    if (reply.type === 'oauthResult' && reply.ok) state.connected.add(credential)
    if (state.ui !== ui) return
    if (reply.type === 'oauthResult' && reply.ok) {
      o.status = 'done'
      ui.signedIn = true
      render()
      // A sign-in that just worked does not need a second confirmation either.
      if (!state.editing && here() === at) next()
      return
    }
    o.status = 'idle'
    o.error = reply.error ?? reply.code ?? 'oauth-failed'
    render()
  }

  async function saveTavily() {
    const tavily = state.svc.tavily
    if (tavily.status === 'checking' || !tavily.draft.trim()) return
    tavily.status = 'checking'
    tavily.error = null
    render()
    const reply = await request({ type: 'saveKey', provider: 'tavily', key: tavily.draft })
    if (reply.type === 'keyResult' && reply.ok) {
      tavily.status = 'saved'
      tavily.draft = ''
      state.connected.add('mcp:tavily')
    } else {
      tavily.status = 'idle'
      tavily.error = reply.error ?? reply.code ?? 'generic'
    }
    render()
  }

  // Notion signs in through the browser. Like the provider's sign-in, a cancelled attempt is ignored
  // when it ends; unlike it, a provider change does not matter (the service does not depend on it).
  async function startNotion() {
    const notion = state.svc.notion
    if (notion.status === 'waiting') return
    const token = ++notion.token
    notion.status = 'waiting'
    notion.error = null
    notion.url = null
    render()
    const reply = await request({ type: 'startMcpOauth', service: 'notion' }, (progress) => {
      if (notion.token !== token) return
      notion.url = progress.url
      render()
    })
    if (notion.token !== token) return
    if (reply.type === 'oauthResult' && reply.ok) {
      notion.status = 'done'
      state.connected.add('mcp:notion')
    } else {
      notion.status = 'idle'
      notion.error = reply.error ?? reply.code ?? 'oauth-failed'
    }
    render()
  }

  function cancelNotion() {
    const notion = state.svc.notion
    notion.token++
    notion.status = 'idle'
    notion.error = null
    notion.url = null
    render()
  }

  /** Gives up waiting for the browser; whatever that attempt reports later is ignored. */
  function cancelOauth() {
    const o = state.ui.oauth
    o.token++
    o.status = 'idle'
    o.error = null
    o.url = null
    render()
  }

  /** Asks the dock what MCP servers the other apps have. `result` (the names just imported) stays on the new list. */
  async function scanFound(result = null, error = null) {
    const f = freshFound(result)
    f.status = 'scanning'
    state.found = f
    render()
    const reply = await request({ type: 'mcpScan' })
    if (state.found !== f) return
    if (reply.type === 'mcpFound') {
      f.status = 'ready'
      f.servers = Array.isArray(reply.servers) ? reply.servers.filter((server) => server && typeof server.id === 'string') : []
      f.error = error
    } else {
      f.status = 'error'
      f.error = reply.code ?? 'generic'
    }
    render()
  }

  function pickFound(id) {
    const f = state.found
    if (f.importing || !f.servers.some((server) => server.id === id && server.importable)) return
    f.picked = f.picked.includes(id) ? f.picked.filter((x) => x !== id) : [...f.picked, id]
    render()
  }

  /** Copies the picked servers into Aria's mcp.json (the dock restarts Pi itself), then reads the servers again for the
   * access rows and scans again, so what was imported is not offered twice. */
  async function importFound() {
    const f = state.found
    if (f.importing || !f.picked.length) return
    f.importing = true
    f.error = null
    f.result = null
    render()
    const ids = [...f.picked]
    const reply = await request({ type: 'mcpImport', ids })
    f.importing = false
    if (reply.type !== 'saved') {
      f.error = reply.code ?? 'generic'
      if (state.found !== f) return
      // The list is out of date (a server went away or was imported elsewhere): read it again, keeping the note.
      if (f.error === 'nothing-to-import') return scanFound(null, f.error)
      render()
      return
    }
    const before = new Set(state.servers)
    const loaded = await request({ type: 'loadSettings' })
    if (Array.isArray(loaded.servers)) state.servers = loaded.servers
    for (const id of loaded.connected ?? []) state.connected.add(id)
    // The names they got in Aria (a taken one gets -2…); an older dock that sends no servers: the names asked for.
    const added = state.servers.filter((server) => !before.has(server))
    const asked = ids.map((id) => f.servers.find((server) => server.id === id)?.importAs ?? id.slice(id.indexOf(':') + 1))
    await scanFound(added.length ? added : asked)
  }

  /** The dock was opened again: after a finished setup, show the settings (fresh from disk) so they
   * can be changed; in the middle of a setup, keep the place. */
  async function onOpen() {
    if (state.view !== 'done') return
    const reply = await request({ type: 'loadSettings' })
    if (state.view === 'done') start(reply.settings, reply.connected, reply.servers)
  }

  // ---- events ---------------------------------------------------------------------------------

  root.addEventListener('click', (event) => {
    const row = state.picker && event.target.closest('[data-chat="pick"]')
    if (row) return pickModel(Number(row.dataset.index))
    const el = event.target.closest('[data-action]')
    if (!el || el.disabled) return
    const { action, field, value } = el.dataset
    // Any other control closes an open model picker, and stops a recording shortcut.
    if (state.picker && action !== 'model-open') state.picker = null
    if (action !== 'key-capture' && action !== 'shortcut-apply' && state.ui.shortcut.recording) stopRecording()
    if (action !== 'key-capture' && action !== 'shortcut-apply') state.ui.shortcut.note = null
    // WebKit does not focus a button on click; remember the widget so the redraw can focus it again.
    focusAfterRender = el.dataset.key ?? el.closest?.('[data-key]')?.dataset.key ?? ''
    if (action === 'choose') return value === 'skip' ? skipSetup() : startFlow(value)
    if (action === 'back') return back()
    if (action === 'next') return next()
    if (action === 'defaults') return go('review')
    if (action === 'edit') return go(value)
    if (action === 'close') return env.onDone ? env.onDone() : post({ type: 'closeRequest' })
    if (action === 'settings-open') return openSection(value)
    if (action === 'model-open') return toggleModelPicker(field)
    if (action === 'key-capture') {
      // A click (or Enter, or Space) starts listening; a second one stops.
      if (state.ui.shortcut.recording) stopRecording()
      else {
        heldKeys.clear()
        state.ui.shortcut = { ...state.ui.shortcut, recording: true, note: null }
      }
      return render()
    }
    if (action === 'shortcut-apply') return applyTypedShortcut()
    if (action === 'settings-home') {
      state.section = null
      state.picker = null
      state.leaving = false
      return render()
    }
    if (action === 'settings-exit') return exitSettings()
    if (action === 'settings-save') return saveSettings()
    if (action === 'settings-discard') return discardSettings()
    if (action === 'pick') return pick(field, value)
    if (action === 'toggle') return toggleField(field)
    if (action === 'reveal') {
      state.ui.reveal[field] = !state.ui.reveal[field]
      return render()
    }
    if (action === 'save-key') return saveKey('key')
    if (action === 'save-typesafeKey') return saveKey('typesafeKey')
    if (action === 'oauth-start') return startOauth()
    if (action === 'oauth-cancel') return cancelOauth()
    if (action === 'save-tavilyKey') return saveTavily()
    if (action === 'mcp-oauth-start') return startNotion()
    if (action === 'mcp-oauth-cancel') return cancelNotion()
    if (action === 'mcp-oauth-reopen') return post({ type: 'openUrl', url: state.svc.notion.url })
    if (action === 'oauth-reopen') return post({ type: 'openUrl', url: state.ui.oauth.url })
    if (action === 'step') {
      update((s) => {
        if (field === 'turnTimeoutMinutes') {
          const now = s.models.turnTimeoutMinutes ?? DEFAULT_TURN_TIMEOUT
          s.models.turnTimeoutMinutes = Math.max(0, Math.min(60, now + Number(value)))
        } else s.subagents.turnLimit = Math.max(5, Math.min(100, s.subagents.turnLimit + Number(value)))
      })
      return render()
    }
    if (action === 'add-chip') {
      const text = state.ui.drafts.extraWritable.trim()
      if (!text) return
      update((s) => s.sandbox.extraWritable.push(text))
      state.ui.drafts.extraWritable = ''
      return render()
    }
    if (action === 'remove-chip') {
      update((s) => s.sandbox.extraWritable.splice(Number(value), 1))
      return render()
    }
    if (action === 'mcp-import') return importFound()
    if (action === 'check' && field === 'mcpImport') return pickFound(value)
    if (action === 'check') {
      update((s) => {
        const list = s.subagents.autoSkills
        list.includes(value) ? list.splice(list.indexOf(value), 1) : list.push(value)
      })
      return render()
    }
    if (action === 'expert-on' || action === 'expert-ro') {
      update((s) => {
        const expert = s.subagents.experts.find((e) => e.name === value)
        if (action === 'expert-on') expert.enabled = !expert.enabled
        else expert.readOnly = !expert.readOnly
      })
      return render()
    }
  })

  root.addEventListener('input', (event) => {
    const el = event.target
    if (state.picker && el.dataset?.chatField === 'model-search') {
      state.picker = { ...state.picker, query: el.value, active: 0 }
      return redrawPicker()
    }
    if (!(el instanceof HTMLInputElement)) return
    const { action, field } = el.dataset
    if (action === 'input') typed(field, el.value)
    else if (action === 'slide') slid(field, el)
    else if (action === 'chip-draft') state.ui.drafts.extraWritable = el.value
  })

  // The Shortcut row, while it records, takes every key first. The keys it took are let go of on the row's button, and
  // an engine that activates a button on release (Space) would click it again: those releases are the row's too.
  const heldKeys = new Set()
  root.addEventListener('keydown', (event) => {
    if (state.ui.shortcut.recording) recordKey(event)
  })
  root.addEventListener('keyup', (event) => {
    if (!heldKeys.delete(event.code)) return
    event.preventDefault()
    event.stopPropagation()
  })

  // The model picker's search: the arrows move its mark, Enter picks, Esc closes the picker (not the dock).
  root.addEventListener('keydown', (event) => {
    if (!state.picker || event.target.dataset?.chatField !== 'model-search') return
    const n = orderedModels(providerModels(), state.picker.query).length
    const moves = { ArrowDown: 1, ArrowUp: -1 }
    if (event.key in moves) {
      state.picker = { ...state.picker, active: n ? (state.picker.active + moves[event.key] + n) % n : 0 }
      redrawPicker()
    } else if (event.key === 'Enter') pickModel(state.picker.active)
    else if (event.key === 'Escape') closeModelPicker()
    else return
    event.preventDefault()
    event.stopPropagation()
    event.stopImmediatePropagation()
  })

  // The pointer only moves the mark (as in the chat's picker): a redraw would send the list back to the top.
  root.addEventListener('mousemove', (event) => {
    const li = state.picker && event.target.closest?.('[data-chat="picker-hover"]')
    if (!li || state.picker.active === Number(li.dataset.index)) return
    state.picker = { ...state.picker, active: Number(li.dataset.index) }
    for (const item of root.querySelectorAll('[data-model-picker] li[data-chat="picker-hover"]')) {
      item.querySelector('button')?.classList.toggle('on', Number(item.dataset.index) === state.picker.active)
    }
  })

  // Enter in a key field saves it; in any other text field it moves on.
  root.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' || !(event.target instanceof HTMLInputElement) || event.target.type === 'range') return
    event.preventDefault()
    const { action, field } = event.target.dataset
    if (action === 'input' && (field === 'key' || field === 'typesafeKey')) return saveKey(field)
    if (action === 'input' && field === 'shortcutText') return applyTypedShortcut()
    if (action === 'chip-draft') return root.querySelector('[data-action="add-chip"]')?.click()
    const items = [...root.querySelectorAll('[data-nav]:not(:disabled)')]
    items[items.indexOf(event.target) + 1]?.focus()
  })

  /** What a test (or a bug report) needs to see of the controller, without the DOM. */
  const snapshot = () => ({
    view: state.view,
    step: state.step,
    section: state.section,
    dirty: isDirty(),
    leaving: state.leaving,
    mode: state.mode,
    editing: state.editing,
    signedIn: state.ui.signedIn,
    canContinue: canContinue(state.step, state.settings, uiFlags()),
    connected: [...state.connected].sort(),
    typesafeSaved: state.ui.typesafeSaved,
    oauth: state.ui.oauth.status,
    key: state.ui.key.status,
  })

  return { start, render, snapshot, onOpen, refreshSettings, platformChanged }
}
