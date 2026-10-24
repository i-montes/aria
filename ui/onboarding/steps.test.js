// Run: bun test ui
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import {
  applyProvider, canContinue, credentialFor, defaultSettings, finish, isSignedIn, JEV_PROVIDERS, jevBackend, languageChoice,
  fullStepCount, maxFullSteps, nextStep, previousStep, PROVIDERS, rivals, setLanguageChoice, setRouter, skip, stepIds,
} from './steps.js'

const golden = (name) => JSON.parse(readFileSync(new URL(`../../crates/aria-core/tests/golden/${name}.json`, import.meta.url), 'utf8'))
const PROVIDER_IDS = ['opencode-go', 'anthropic-api', 'anthropic-oauth']

describe('the JS defaults follow the Rust ones (golden files written by aria-core)', () => {
  test('defaults', () => expect(defaultSettings('en')).toEqual(golden('settings-defaults-en')))
  for (const id of PROVIDER_IDS) {
    test(`after choosing ${id}`, () => expect(applyProvider(defaultSettings('en'), id)).toEqual(golden(`settings-provider-${id}`)))
  }
  test('the turn watchdog stops a silent run after ten minutes by default', () => expect(defaultSettings('en').models.turnTimeoutMinutes).toBe(10))
  test("the shortcut starts empty (the platform's default)", () => expect(defaultSettings('en').shortcut).toBe(''))
  test('every connector starts at the default level (no access entries)', () => expect(defaultSettings('en').integrations.access).toEqual({}))
  test('the language is the only thing defaultSettings(language) changes', () => {
    expect(defaultSettings('es').assistant.language).toBe('es')
    expect({ ...defaultSettings('es'), assistant: null }).toEqual({ ...defaultSettings('en'), assistant: null })
  })
  test('applyProvider does not mutate its input', () => {
    const base = defaultSettings('en')
    applyProvider(base, 'opencode-go')
    expect(base).toEqual(golden('settings-defaults-en'))
  })
})

describe('flows', () => {
  test('quick steps', () => expect(stepIds('quick', 'opencode-go')).toEqual(['provider', 'assistant', 'review']))
  test('full steps with a quota', () =>
    expect(stepIds('full', 'opencode-go')).toEqual([
      'provider', 'quota', 'router', 'assistant', 'models', 'effort', 'compaction', 'memory', 'integrations', 'sandbox', 'git', 'agents', 'review',
    ]))
  test('full steps without a quota', () => {
    expect(stepIds('full', 'anthropic-api')).not.toContain('quota')
    expect(stepIds('full', 'anthropic-api')).toHaveLength(12)
  })
  test('the count the welcome screen shows is the length of the real list', () => {
    for (const id of PROVIDER_IDS) expect(fullStepCount(id)).toBe(stepIds('full', id).length)
    expect(fullStepCount('opencode-go')).toBe(fullStepCount('anthropic-api') + 1) // the quota step
    // before a provider is chosen the most a flow can have
    expect(maxFullSteps()).toBe(Math.max(...PROVIDER_IDS.map(fullStepCount)))
    expect(maxFullSteps()).toBe(stepIds('full', 'opencode-go').length)
  })
  test('back from the first step is the welcome screen', () => {
    expect(previousStep(stepIds('full', 'opencode-go'), 'provider')).toBe('welcome')
    expect(previousStep(stepIds('full', 'opencode-go'), 'quota')).toBe('provider')
  })
  test('next after the last step finishes', () => {
    const ids = stepIds('quick', 'opencode-go')
    expect(nextStep(ids, 'provider')).toBe('assistant')
    expect(nextStep(ids, 'review')).toBeNull()
  })
})

describe('what gets saved', () => {
  test('quick setup keeps the owner defaults', () => {
    let s = applyProvider(defaultSettings('es'), 'opencode-go')
    s.assistant.userName = 'Ivan'
    s = finish(s, 'quick')
    expect(s.setup).toEqual({ completed: true, mode: 'quick' })
    expect(s.subagents.turnLimit).toBeNull()
    expect(s.subagents.saveSessions).toBe(false)
    expect(s.git.commitWithoutAsking).toBe(false)
    expect(s.subagents.autoSkills).toEqual(['systematic-debugging'])
    expect(s.compaction.percent).toBe(40)
    expect(s.provider.quota.switchAt).toBe(98)
    expect(s.router).toEqual({ enabled: true, backend: 'opencode' })
    expect(s.assistant).toMatchObject({ userName: 'Ivan', name: 'Aria', language: 'es' })
  })

  test('Anthropic with a Claude sign-in', () => {
    const s = applyProvider(defaultSettings('en'), 'anthropic-oauth')
    expect(s.provider.id).toBe('anthropic-oauth')
    expect(s.provider.quota).toBeNull()
    expect(s.models.daily).toBe('claude-bridge/claude-sonnet-5-5')
  })

  test('turning the router off and on again under OpenCode Go', () => {
    const off = setRouter(applyProvider(defaultSettings('en'), 'opencode-go'), false)
    expect(off.router.enabled).toBe(false)
    expect(off.sandbox.jevReview).toBe(false)
    expect(off.effort).toMatchObject({ mode: 'fixed', fixedLevel: 'medium' })
    const on = setRouter(off, true)
    expect(on.router).toEqual({ enabled: true, backend: 'opencode' })
    expect(on.effort.mode).toBe('auto')
  })

  test('skipping records skipped and changes nothing else', () => {
    const base = defaultSettings('en')
    const s = skip(base)
    expect(s.setup).toEqual({ completed: false, mode: 'skipped' })
    expect({ ...s, setup: null }).toEqual({ ...base, setup: null })
  })

  test('finishing without a mode keeps the one already saved', () => {
    const saved = { ...defaultSettings('en'), setup: { completed: true, mode: 'quick' } }
    expect(finish(saved).setup).toEqual({ completed: true, mode: 'quick' })
  })
})

describe('language', () => {
  test('codes and free text map to the segment choice', () => {
    expect(languageChoice('es')).toBe('es')
    expect(languageChoice('pt')).toBe('pt')
    expect(languageChoice('en')).toBe('en')
    expect(languageChoice('Klingon')).toBe('other')
  })
  test('picking a choice sets the code, "other" keeps the typed text', () => {
    const s = defaultSettings('en')
    expect(setLanguageChoice(s, 'es').assistant.language).toBe('es')
    expect(setLanguageChoice(s, 'other', 'Klingon').assistant.language).toBe('Klingon')
  })
})

describe('when Continue is allowed', () => {
  const opencode = applyProvider(defaultSettings('en'), 'opencode-go')
  test('the providers step needs a main provider that is connected and not being checked', () => {
    expect(canContinue('provider', defaultSettings('en'), { signedIn: true })).toBe(false)
    expect(canContinue('provider', opencode, { signedIn: false })).toBe(false)
    expect(canContinue('provider', opencode, { signedIn: true })).toBe(true)
    expect(canContinue('provider', opencode, { signedIn: true, keyChecking: true })).toBe(false)
  })
  test('the router never blocks, and without Jev in the provider it is saved off', () => {
    const api = setRouter(applyProvider(defaultSettings('en'), 'anthropic-api'), true)
    expect(canContinue('router', api, {})).toBe(true)
    expect(finish(api).router.enabled).toBe(false)
    expect(finish(opencode).router.enabled).toBe(true)
  })
  test('every other step always continues', () => {
    expect(canContinue('models', opencode, {})).toBe(true)
    expect(canContinue('git', opencode, {})).toBe(true)
  })
})

describe('what is already set up on disk', () => {
  const settingsFor = (id) => applyProvider(defaultSettings('en'), id)
  test('a key for the provider counts, a sign-in for another does not', () => {
    expect(isSignedIn(settingsFor('opencode-go'), ['opencode-go:api_key'])).toBe(true)
    expect(isSignedIn(settingsFor('opencode-go'), ['anthropic:oauth'])).toBe(false)
    expect(isSignedIn(settingsFor('opencode-go'), [])).toBe(false)
  })
  test('Anthropic with a key and with a sign-in are different credentials', () => {
    expect(isSignedIn(settingsFor('anthropic-api'), ['anthropic:api_key'])).toBe(true)
    expect(isSignedIn(settingsFor('anthropic-api'), ['anthropic:oauth'])).toBe(false)
    expect(isSignedIn(settingsFor('anthropic-oauth'), ['anthropic:oauth'])).toBe(true)
    expect(isSignedIn(settingsFor('anthropic-oauth'), ['anthropic:api_key'])).toBe(false)
  })
  test('the Anthropic key and the Claude sign-in replace each other; OpenCode Go replaces nothing', () => {
    expect(rivals('anthropic-api')).toEqual(['anthropic-oauth'])
    expect(rivals('anthropic-oauth')).toEqual(['anthropic-api'])
    expect(rivals('opencode-go')).toEqual([])
  })
  test('no provider chosen is never signed in', () => {
    expect(isSignedIn(defaultSettings('en'), ['opencode-go:api_key'])).toBe(false)
  })
})

describe('the integrations step', () => {
  const withServices = (...ids) => {
    const s = defaultSettings('en')
    s.integrations.services = ids.map((id) => ({ id, name: id, auth: id === 'notion' ? 'oauth' : 'key' }))
    return s
  }
  test('no service chosen: always continues', () => expect(canContinue('integrations', withServices(), { servicesConnected: [] })).toBe(true))
  test('a service you turned on must be connected first', () => {
    expect(canContinue('integrations', withServices('notion'), { servicesConnected: [] })).toBe(false)
    expect(canContinue('integrations', withServices('notion'), { servicesConnected: ['notion'] })).toBe(true)
  })
  test('every chosen service counts', () => {
    expect(canContinue('integrations', withServices('notion', 'tavily'), { servicesConnected: ['notion'] })).toBe(false)
    expect(canContinue('integrations', withServices('notion', 'tavily'), { servicesConnected: ['notion', 'tavily'] })).toBe(true)
  })
})

describe('Jev follows the stored keys, not the main provider', () => {
  const GO = 'opencode-go:api_key'
  const TS = 'typesafe:api_key'

  test('Jev comes from OpenCode Go first, then TypeSafe, else none', () => {
    const s = defaultSettings('en')
    expect(jevBackend(applyProvider(s, 'opencode-go'), [])).toBe('opencode')
    expect(jevBackend(applyProvider(s, 'anthropic-oauth'), [GO])).toBe('opencode')
    expect(jevBackend(applyProvider(s, 'anthropic-oauth'), [TS])).toBe('typesafe')
    expect(jevBackend(applyProvider(s, 'anthropic-oauth'), [GO, TS])).toBe('opencode')
    expect(jevBackend(applyProvider(s, 'anthropic-oauth'), [])).toBeNull()
  })

  test('Claude as the main provider keeps the router and the review on with either key', () => {
    for (const [keys, backend] of [[[GO], 'opencode'], [[TS], 'typesafe']]) {
      const s = applyProvider(defaultSettings('en'), 'anthropic-oauth', keys)
      expect(s.router).toEqual({ enabled: true, backend })
      expect(s.sandbox.jevReview).toBe(true)
      expect(s.effort.mode).toBe('fixed') // the main provider's own default
    }
  })

  test('without a Jev key Claude leaves the router off, as before', () => {
    const s = applyProvider(defaultSettings('en'), 'anthropic-oauth')
    expect(s.router.enabled).toBe(false)
    expect(s.sandbox.jevReview).toBe(false)
  })

  test('setRouter and finish follow the stored keys', () => {
    const claude = applyProvider(defaultSettings('en'), 'anthropic-oauth')
    expect(setRouter(claude, true, [TS]).router).toEqual({ enabled: true, backend: 'typesafe' })
    const on = applyProvider(defaultSettings('en'), 'anthropic-oauth', [TS])
    expect(finish(on, 'full', [TS]).router).toEqual({ enabled: true, backend: 'typesafe' })
    expect(finish(on, 'full', [GO]).router).toEqual({ enabled: true, backend: 'opencode' })
    expect(finish(on, 'full', []).router.enabled).toBe(false)
    // Removing the last Jev key turns Jev's review off with the router; any key keeps what was on.
    expect(on.sandbox.jevReview).toBe(true)
    expect(finish(on, 'full', []).sandbox.jevReview).toBe(false)
    expect(finish(on, 'full', [TS]).sandbox.jevReview).toBe(true)
    expect(finish(on, 'full', [GO]).sandbox.jevReview).toBe(true)
  })

  test('TypeSafe is a provider for Jev only: never main, no models', () => {
    expect(JEV_PROVIDERS.map((p) => p.id)).toEqual(['typesafe'])
    expect(PROVIDERS.some((p) => p.id === 'typesafe')).toBe(false)
    expect(credentialFor('typesafe')).toBe(TS)
    expect(stepIds('full', 'typesafe')).not.toContain('quota')
  })
})
