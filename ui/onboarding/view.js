// The onboarding screens: welcome, the steps of a flow, and the "set up" screen. The working state is
// the settings object (see steps.js); this file turns it into HTML, handles every interaction in one
// delegated listener, and talks to the dock through `env.request`. Widgets live in components.js.
import * as c from './components.js'
import {
  applyProvider, canContinue, credentialFor, defaultSettings, finish, isSignedIn, languageChoice, nextStep,
  previousStep, PROVIDERS, providerModule, setLanguageChoice, setRouter, skip, stepIds,
} from './steps.js'

const SERVICES = ['notion', 'tavily']
const SKILLS = ['brainstorming', 'writing-plans', 'test-driven-development', 'systematic-debugging', 'requesting-code-review', 'verification-before-completion']
const STYLES = ['very-short', 'brief', 'detailed', 'technical', 'teaching']
const LEVELS = ['low', 'medium', 'high']
const DEFAULT_TURN_LIMIT = 30

/** `env`: { root, t, request, post, setLanguage, browserLanguage, hotkey?, onDone? }. `onDone` runs when the user
 * presses the button of the "set up" screen (the dock goes to the chat); without it the dock closes. */
export function createOnboarding(env) {
  const { root, t, request, post } = env
  const hotkey = () => env.hotkey?.() ?? 'Super + A'
  const state = {
    view: 'welcome', // welcome | step | done
    mode: 'quick', // quick | full
    step: null,
    editing: false,
    saving: false,
    error: null,
    settings: defaultSettings('en'),
    connected: new Set(), // what the dock found on disk, plus what this session set up
    svc: freshServices(), // Notion and Tavily do not depend on the provider, so they live apart from `ui`
    ui: freshUi(),
  }

  function freshServices() {
    return { tavily: { status: 'idle', error: null, draft: '' }, notion: { status: 'idle', error: null, url: null, token: 0 } }
  }

  function freshUi() {
    return {
      signedIn: false,
      typesafeSaved: false,
      reveal: {},
      drafts: { key: '', typesafeKey: '', extraWritable: '' },
      key: { status: 'idle', error: null, requested: false },
      typesafe: { status: 'idle', error: null },
      oauth: { status: 'idle', error: null, url: null, token: 0, requested: false },
    }
  }

  const uiFlags = () => ({
    signedIn: state.ui.signedIn,
    keyChecking: state.ui.key.status === 'checking',
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

  /** `settings` is what the dock loaded (null on a first run); `connected` is what it found on disk. */
  function start(settings, connected = []) {
    state.error = null
    state.connected = new Set(connected)
    state.svc = freshServices()
    state.ui = freshUi()
    if (settings?.setup?.completed) {
      state.settings = settings
      state.editing = true
      state.mode = 'full'
      state.view = 'step'
      state.step = 'review'
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
    const view = `${state.view}:${state.step}`
    const sameStep = view === lastView
    // WebKit does not focus a button when it is clicked, so the active element is not the checkbox that
    // was just pressed. The clicked widget's key is what brings the reader back to it after the redraw.
    const focused = sameStep ? focusAfterRender || document.activeElement?.closest?.('[data-key]')?.dataset.key : ''
    focusAfterRender = ''
    // The step lives in a scrollable panel. Redrawing it and focusing the first control sent the form back
    // to the top on every checkbox; keep the reader where they were instead.
    const panel = globalThis.document?.getElementById?.('content')
    const scrollTop = sameStep ? panel?.scrollTop ?? 0 : 0
    root.innerHTML = state.view === 'welcome' ? welcomeHtml() : state.view === 'done' ? doneHtml() : stepHtml()
    const target = focused ? root.querySelector(`[data-key="${CSS.escape(focused)}"]`) : null
    lastView = view
    ;(target ?? root.querySelector('[data-nav]'))?.focus({ preventScroll: true })
    if (panel) panel.scrollTop = scrollTop
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
      <p class="intro">${c.esc(t('ob.done.desc', { name, hotkey: hotkey() }))}</p>
      <button class="primary" data-nav data-action="close">${c.esc(t('ob.done.close'))}</button>`
  }

  function stepHtml() {
    const all = ids()
    const at = all.indexOf(state.step)
    const isReview = state.step === 'review'
    const descKey = isReview ? `ob.review.desc.${state.editing ? 'edit' : state.mode}` : `ob.${state.step}.desc`
    const nextLabel = state.saving
      ? t('ob.saving')
      : isReview
        ? t(state.editing ? 'ob.save' : 'ob.start', { name: assistantName() })
        : t('ob.continue')
    const showDefaults = !state.editing && state.mode === 'full' && !isReview && at >= 2
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
    html += c.textField('assistantName', t('ob.assistant.name'), a.name, { placeholder: t('ob.assistant.name.placeholder'), description: t('ob.assistant.name.desc') })
    const choice = languageChoice(a.language)
    html += c.segments('language', t('ob.assistant.language'), ['es', 'en', 'pt', 'other'].map((v) => ({ value: v, label: t(`ob.assistant.language.${v}`) })), choice)
    if (choice === 'other') html += c.textField('languageOther', t('ob.assistant.languageOther'), a.language, { placeholder: t('ob.assistant.languageOther.placeholder'), indent: true })
    return html + c.choice('style', t('ob.assistant.style'), STYLES.map((v) => ({ value: v, label: t(`ob.style.${v}`), description: i18nHas(`ob.style.${v}.desc`) ? t(`ob.style.${v}.desc`) : '' })), a.style)
  }
  const i18nHas = (key) => t(key) !== key

  function modelsStep() {
    const list = module().models.map((m) => ({ value: m.id, label: m.label }))
    let html =
      c.section(t('ob.models.section.daily')) + c.choice('daily', '', list, state.settings.models.daily) +
      c.section(t('ob.models.section.planning')) + c.choice('planning', '', list, state.settings.models.planning) +
      c.segments('planningEffort', t('ob.models.planningEffort'), LEVELS.map((v) => ({ value: v, label: t(`ob.level.${v}`) })), state.settings.models.planningEffort)
    if (module().quota) html += c.toggle('payPerUse', t('ob.models.payPerUse'), t('ob.models.payPerUse.desc'), state.settings.models.allowPayPerUse)
    return html
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
    const p = s.integrations.packages
    html += c.section(t('ob.integrations.section.packages'))
    html += c.segments('packageSource', t('ob.integrations.packageSource'), ['gallery', 'npm', 'git', 'local'].map((v) => ({ value: v, label: t(`ob.integrations.source.${v}`) })), p.source, t('ob.integrations.packageSource.desc'))
    if (p.source !== 'gallery') html += c.textField('packagePath', t('ob.integrations.packagePath'), p.path, { mono: true, indent: true, placeholder: t('ob.integrations.packagePath.placeholder') })
    html += s.subagents.planExecute ? c.info('green', t('ob.integrations.auto.title'), t('ob.integrations.auto.desc')) : c.info('neutral', '', t('ob.integrations.auto.none'))
    return html + c.toggle('trust', t('ob.integrations.trust'), t('ob.integrations.trust.desc'), p.thirdPartyTrusted)
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

  function modelLabel(id) {
    return module()?.models.find((m) => m.id === id)?.label ?? '—'
  }

  function reviewStep() {
    const s = state.settings
    const m = module()
    const quickFlow = !state.editing && state.mode === 'quick'
    const language = languageChoice(s.assistant.language) === 'other' ? s.assistant.language : t(`ob.assistant.language.${s.assistant.language}`)
    const rows = [
      ['provider', (m?.label ?? '—') + (m?.quota ? t('ob.val.quotaMeter') : '')],
      ['auth', t(m?.auth === 'oauth' ? 'ob.val.authOauth' : 'ob.val.authKey')],
      ['assistant', t('ob.val.assistant', { name: assistantName(), language, style: t(`ob.style.${s.assistant.style}`) }) + (s.assistant.userName.trim() ? t('ob.val.callsYou', { user: s.assistant.userName.trim() }) : '')],
    ]
    if (!quickFlow) {
      const q = s.provider.quota
      const quota = q
        ? t('ob.val.quota', { a: q.switchAt, b: q.planSwitchAt, ask: q.askFirst ? t('ob.val.quotaAsk', { n: q.countdownMinutes }) : t('ob.val.quotaNoAsk') })
        : t('ob.val.quotaNone')
      const effort = s.effort.mode === 'auto' ? t('ob.val.effortAuto') : t(`ob.level.${s.effort.fixedLevel}`)
      const services = s.integrations.services.map((x) => t(`ob.integrations.${x.id}`))
      const experts = s.subagents.experts.filter((e) => e.enabled).length
      rows.splice(1, 0, ['quota', quota])
      rows.push(
        ['router', s.router.enabled ? t('ob.val.routerOn') : t('ob.val.off')],
        ['models', t('ob.val.models', { daily: modelLabel(s.models.daily), planning: modelLabel(s.models.planning), effort: t(`ob.level.${s.models.planningEffort}`) })],
        ['effort', effort],
        ['compaction', t('ob.val.compaction', { n: s.compaction.percent })],
        ['memory', s.memory.enabled ? t('ob.val.memoryOn', { project: s.memory.project }) : t('ob.val.off')],
        ['integrations', services.length ? services.join(', ') : t('ob.val.integrationsNone')],
        ['sandbox', t('ob.val.sandbox', { dir: s.sandbox.projectsDir, more: s.sandbox.extraWritable.length ? t('ob.val.sandboxMore', { n: s.sandbox.extraWritable.length }) : '', ask: t(s.sandbox.askToLeave ? 'ob.val.sandboxAsks' : 'ob.val.sandboxFree') })],
        ['git', t('ob.val.git', { commit: t(s.git.commitWithoutAsking ? 'ob.val.commitAlone' : 'ob.val.commitAsks'), push: t(s.git.pushOrMergeWithoutAsking ? 'ob.val.pushAlone' : 'ob.val.pushAsks') })],
        ['agents', t('ob.val.agents', { n: experts, plan: t(s.subagents.planExecute ? 'ob.val.planOn' : 'ob.val.planOff'), sessions: s.subagents.saveSessions ? t('ob.val.sessionsSaved') : '' })],
      )
    }
    const wanted = new Set(ids())
    let html = c.summary(rows.map(([id, value]) => ({ id, label: t(`ob.row.${id}`), value, editLabel: wanted.has(id) ? t('ob.edit') : '' })).map((r) => (r.editLabel ? r : { ...r, editLabel: '' })))
    if (quickFlow) html += c.info('lavender', t('ob.review.defaultsNote.title'), t('ob.review.defaultsNote.desc'))
    if (state.error) html += c.info('amber', '', t(`ob.error.${state.error}`))
    return html
  }

  // ---- navigation -----------------------------------------------------------------------------

  const go = (step) => {
    state.step = step
    state.view = 'step'
    state.error = null
    render()
    // Drawing has no side effects; entering the sign-in step is what checks a credential already on disk.
    maybeCheckStoredCredential()
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
    const reply = await request({ type: 'saveSettings', settings: finish(state.settings, state.editing ? undefined : state.mode) })
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
    // Before the first step: the welcome screen, or the summary when editing an existing setup.
    if (state.editing) return go('review')
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
        else if (name === 'packageSource') s.integrations.packages.source = value
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
      else if (name === 'trust') s.integrations.packages.thirdPartyTrusted = !s.integrations.packages.thirdPartyTrusted
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
      else if (field === 'assistantName') s.assistant.name = value
      else if (field === 'languageOther') s.assistant.language = value
      else if (field === 'memoryProject') s.memory.project = value
      else if (field === 'projectsDir') s.sandbox.projectsDir = value
      else if (field === 'packagePath') s.integrations.packages.path = value
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
        if (!state.editing) next()
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
    if (state.view !== 'step' || state.step !== 'auth') return
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
      if (!state.editing && (m.auth === 'oauth' || !ui.drafts.key.trim())) next()
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
      if (!state.editing) next()
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

  /** The dock was opened again: after a finished setup, show the settings (fresh from disk) so they
   * can be changed; in the middle of a setup, keep the place. */
  async function onOpen() {
    if (state.view !== 'done') return
    const reply = await request({ type: 'loadSettings' })
    if (state.view === 'done') start(reply.settings, reply.connected)
  }

  // ---- events ---------------------------------------------------------------------------------

  root.addEventListener('click', (event) => {
    const el = event.target.closest('[data-action]')
    if (!el || el.disabled) return
    const { action, field, value } = el.dataset
    // WebKit does not focus a button on click; remember the widget so the redraw can focus it again.
    focusAfterRender = el.dataset.key ?? el.closest?.('[data-key]')?.dataset.key ?? ''
    if (action === 'choose') return value === 'skip' ? skipSetup() : startFlow(value)
    if (action === 'back') return back()
    if (action === 'next') return next()
    if (action === 'defaults') return go('review')
    if (action === 'edit') return go(value)
    if (action === 'close') return env.onDone ? env.onDone() : post({ type: 'closeRequest' })
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
        s.subagents.turnLimit = Math.max(5, Math.min(100, s.subagents.turnLimit + Number(value)))
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
    if (!(el instanceof HTMLInputElement)) return
    const { action, field } = el.dataset
    if (action === 'input') typed(field, el.value)
    else if (action === 'slide') slid(field, el)
    else if (action === 'chip-draft') state.ui.drafts.extraWritable = el.value
  })

  // Enter in a key field saves it; in any other text field it moves on.
  root.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' || !(event.target instanceof HTMLInputElement) || event.target.type === 'range') return
    event.preventDefault()
    const { action, field } = event.target.dataset
    if (action === 'input' && (field === 'key' || field === 'typesafeKey')) return saveKey(field)
    if (action === 'chip-draft') return root.querySelector('[data-action="add-chip"]')?.click()
    const items = [...root.querySelectorAll('[data-nav]:not(:disabled)')]
    items[items.indexOf(event.target) + 1]?.focus()
  })

  /** What a test (or a bug report) needs to see of the controller, without the DOM. */
  const snapshot = () => ({
    view: state.view,
    step: state.step,
    mode: state.mode,
    editing: state.editing,
    signedIn: state.ui.signedIn,
    canContinue: canContinue(state.step, state.settings, uiFlags()),
    connected: [...state.connected].sort(),
    typesafeSaved: state.ui.typesafeSaved,
    oauth: state.ui.oauth.status,
    key: state.ui.key.status,
  })

  return { start, render, snapshot, onOpen }
}
