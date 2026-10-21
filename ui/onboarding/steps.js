// The onboarding as plain data and pure functions: which steps a flow has, what each provider
// brings, and how the answers become `settings.json`. The working state IS the settings object, so
// what the user sees on the review step is exactly what is saved. Nothing here touches the DOM.
//
// The defaults and the provider catalog mirror crates/aria-core (settings.rs, providers.rs); both
// sides are tested against the golden files in crates/aria-core/tests/golden so they cannot drift.

export const PROVIDERS = [
  {
    id: 'opencode-go',
    label: 'OpenCode Go',
    auth: 'key',
    keyProvider: 'opencode-go',
    jevIncluded: true,
    models: [
      { id: 'opencode-go/mimo-v2.6-flash', label: 'MiMo V2.6 Flash' },
      { id: 'opencode-go/mimo-v2.6-pro', label: 'MiMo V2.6 Pro' },
      { id: 'opencode-go/deepseek-v4.1-flash', label: 'DeepSeek V4.1 Flash' },
      { id: 'opencode-go/glm-5.3', label: 'GLM 5.3' },
      { id: 'opencode-go/kimi-k3', label: 'Kimi K3' },
      { id: 'opencode-go/qwen3.8-max', label: 'Qwen 3.8 Max' },
    ],
    freeModels: [
      { id: 'opencode-go/cat-jam-free', label: 'Cat Jam (free)' },
      { id: 'opencode-go/longcat-2.5-preview-free', label: 'Longcat 2.5 preview (free)' },
    ],
    quota: { switchAt: 98, planSwitchAt: 90, freeModel: 'opencode-go/cat-jam-free', askFirst: true, countdownMinutes: 5, onNoAnswer: 'switch' },
    daily: 'opencode-go/mimo-v2.6-flash',
    planning: 'opencode-go/mimo-v2.6-pro',
  },
  {
    id: 'anthropic-api',
    label: 'Anthropic · API key',
    auth: 'key',
    keyProvider: 'anthropic-api',
    jevIncluded: false,
    models: [
      { id: 'anthropic/claude-sonnet-5-5', label: 'Claude Sonnet 5.5' },
      { id: 'anthropic/claude-opus-5-5', label: 'Claude Opus 5.5' },
      { id: 'anthropic/claude-haiku-5-5', label: 'Claude Haiku 5.5' },
      { id: 'anthropic/claude-fable-5-1', label: 'Claude Fable 5.1' },
    ],
    freeModels: [],
    quota: null,
    daily: 'anthropic/claude-sonnet-5-5',
    planning: 'anthropic/claude-opus-5-5',
  },
  {
    id: 'anthropic-oauth',
    label: 'Anthropic · Sign in with Claude',
    auth: 'oauth',
    keyProvider: null,
    jevIncluded: false,
    models: [
      { id: 'claude-bridge/claude-sonnet-5-5', label: 'Claude Sonnet 5.5' },
      { id: 'claude-bridge/claude-opus-5-5', label: 'Claude Opus 5.5' },
      { id: 'claude-bridge/claude-haiku-5-5', label: 'Claude Haiku 5.5' },
      { id: 'claude-bridge/claude-fable-5-1', label: 'Claude Fable 5.1' },
    ],
    freeModels: [],
    quota: null,
    daily: 'claude-bridge/claude-sonnet-5-5',
    planning: 'claude-bridge/claude-opus-5-5',
  },
]

export const providerModule = (id) => PROVIDERS.find((p) => p.id === id) ?? null

const clone = (value) => structuredClone(value)

/** The owner's own setup (the parity rule), with the given default conversation language. */
export function defaultSettings(language = 'en') {
  const expert = (name, readOnly) => ({ name, enabled: true, readOnly })
  return {
    version: 1,
    setup: { completed: false, mode: null },
    provider: { id: null, quota: null },
    router: { enabled: true, backend: 'opencode' },
    assistant: { userName: '', name: 'Aria', language, style: 'very-short' },
    models: { daily: null, planning: null, planningEffort: 'high', allowPayPerUse: false, turnTimeoutMinutes: 10 },
    effort: { mode: 'auto', fixedLevel: null, maxOnFree: true },
    compaction: { percent: 40 },
    memory: { enabled: true, project: 'aria' },
    integrations: { services: [], packages: { source: 'gallery', path: '', thirdPartyTrusted: false }, access: {} },
    sandbox: { projectsDir: '~/Projects', extraWritable: [], askToLeave: true, jevReview: true },
    git: { commitWithoutAsking: false, pushOrMergeWithoutAsking: false, askBeforeOthersChanges: true, unaskedAsProposals: true },
    subagents: {
      experts: [expert('frontend', false), expert('backend', false), expert('security', true), expert('reviewer', true), expert('researcher', true)],
      saveSessions: false,
      turnLimit: null,
      planExecute: true,
      autoSkills: ['systematic-debugging', 'verification-before-completion'],
    },
    shortcut: '',
  }
}

/** Points the settings at a provider: its models and quota, and the router, effort and Jev-review
 * defaults that make sense with it. Returns a new object. */
export function applyProvider(settings, id) {
  const module = providerModule(id)
  if (!module) return settings
  const next = clone(settings)
  next.provider = { id: module.id, quota: module.quota ? clone(module.quota) : null }
  next.models.daily = module.daily
  next.models.planning = module.planning
  if (module.jevIncluded) {
    next.router = { enabled: true, backend: 'opencode' }
    next.effort = { ...next.effort, mode: 'auto', fixedLevel: null }
    next.sandbox.jevReview = true
  } else {
    next.router = { enabled: false, backend: 'typesafe' }
    next.effort = { ...next.effort, mode: 'fixed', fixedLevel: 'medium' }
    next.sandbox.jevReview = false
  }
  return next
}

/** Turns the router on or off. Without it there is no route to read, so auto effort goes away. */
export function setRouter(settings, on) {
  const next = clone(settings)
  const module = providerModule(next.provider.id)
  next.router = { enabled: on, backend: module?.jevIncluded ? 'opencode' : 'typesafe' }
  next.sandbox.jevReview = on
  if (!on && next.effort.mode === 'auto') next.effort = { ...next.effort, mode: 'fixed', fixedLevel: 'medium' }
  if (on && next.effort.mode === 'fixed' && next.effort.fixedLevel === 'medium') next.effort = { ...next.effort, mode: 'auto', fixedLevel: null }
  return next
}

const LANGUAGE_CODES = ['es', 'en', 'pt']

/** `es` | `en` | `pt` | `other` for the language segment. */
export const languageChoice = (value) => (LANGUAGE_CODES.includes(value) ? value : 'other')

export function setLanguageChoice(settings, choice, otherText = '') {
  const next = clone(settings)
  next.assistant.language = choice === 'other' ? otherText : choice
  return next
}

// ---- flow -------------------------------------------------------------------------------------

/** The steps of a flow after the welcome screen. Editing an existing setup uses the full flow. */
export function stepIds(mode, providerId) {
  if (mode === 'quick') return ['provider', 'auth', 'assistant', 'review']
  const hasQuota = !!providerModule(providerId)?.quota
  return [
    'provider',
    ...(hasQuota ? ['quota'] : []),
    'auth', 'router', 'assistant', 'models', 'effort', 'compaction', 'memory', 'integrations', 'sandbox', 'git', 'agents', 'review',
  ]
}

/** The step after `current`, or `null` when `current` was the last one (time to save). */
export function nextStep(ids, current) {
  const at = ids.indexOf(current)
  return at > -1 && at < ids.length - 1 ? ids[at + 1] : null
}

export function previousStep(ids, current) {
  const at = ids.indexOf(current)
  return at > 0 ? ids[at - 1] : 'welcome'
}

/** The credential `auth.json` must hold for each provider (named `provider:type`, as the dock reports it). */
const CREDENTIAL = { 'opencode-go': 'opencode-go:api_key', 'anthropic-api': 'anthropic:api_key', 'anthropic-oauth': 'anthropic:oauth' }

/** Is the chosen provider's credential really there? `connected` is what the dock found on disk. */
export const credentialFor = (providerId) => CREDENTIAL[providerId]
export const isSignedIn = (settings, connected) => connected.includes(credentialFor(settings.provider.id))

/** `ui` is what the page knows beyond the settings: `signedIn`, `keyChecking`, `typesafeSaved`, `servicesConnected`. */
export function canContinue(step, settings, ui) {
  if (step === 'provider') return settings.provider.id !== null
  // A key that is being checked cannot be considered signed in yet.
  if (step === 'auth') return ui.signedIn === true && ui.keyChecking !== true
  // A service you turned on has to be connected; to skip one, turn it off.
  if (step === 'integrations') return settings.integrations.services.every((service) => ui.servicesConnected?.includes(service.id))
  return true
}

/** The settings to save when the setup is done. Without a mode, the saved one stays (editing). Without Jev in the
 * provider the router is off (its TypeSafe backend is not built), whatever an earlier version saved. */
export function finish(settings, mode) {
  const next = clone(settings)
  if (!providerModule(next.provider.id)?.jevIncluded && next.router.enabled) next.router = { ...next.router, enabled: false }
  next.setup = { completed: true, mode: mode ?? settings.setup.mode ?? 'full' }
  return next
}

/** "Skip for now": nothing else changes, and the welcome screen comes back next time. */
export function skip(settings) {
  const next = clone(settings)
  next.setup = { completed: false, mode: 'skipped' }
  return next
}
