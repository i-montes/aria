// Run: bun test ui
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import {
  applyProvider, canContinue, defaultSettings, finish, isSignedIn, languageChoice, nextStep, previousStep,
  setLanguageChoice, setRouter, skip, stepIds,
} from './steps.js'

const golden = (name) => JSON.parse(readFileSync(new URL(`../../crates/aria-core/tests/golden/${name}.json`, import.meta.url), 'utf8'))
const PROVIDER_IDS = ['opencode-go', 'anthropic-api', 'anthropic-oauth']

describe('the JS defaults follow the Rust ones (golden files written by aria-core)', () => {
  test('defaults', () => expect(defaultSettings('en')).toEqual(golden('settings-defaults-en')))
  for (const id of PROVIDER_IDS) {
    test(`after choosing ${id}`, () => expect(applyProvider(defaultSettings('en'), id)).toEqual(golden(`settings-provider-${id}`)))
  }
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
  test('quick steps', () => expect(stepIds('quick', 'opencode-go')).toEqual(['provider', 'auth', 'assistant', 'review']))
  test('full steps with a quota', () =>
    expect(stepIds('full', 'opencode-go')).toEqual([
      'provider', 'quota', 'auth', 'router', 'assistant', 'models', 'effort', 'compaction', 'memory', 'integrations', 'sandbox', 'git', 'agents', 'review',
    ]))
  test('full steps without a quota', () => {
    expect(stepIds('full', 'anthropic-api')).not.toContain('quota')
    expect(stepIds('full', 'anthropic-api')).toHaveLength(13)
  })
  test('back from the first step is the welcome screen', () => {
    expect(previousStep(stepIds('full', 'opencode-go'), 'provider')).toBe('welcome')
    expect(previousStep(stepIds('full', 'opencode-go'), 'quota')).toBe('provider')
  })
  test('next after the last step finishes', () => {
    const ids = stepIds('quick', 'opencode-go')
    expect(nextStep(ids, 'provider')).toBe('auth')
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
    expect(s.subagents.autoSkills).toEqual(['systematic-debugging', 'verification-before-completion'])
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

  test('a router with a TypeSafe key', () => {
    const base = applyProvider(defaultSettings('en'), 'anthropic-api')
    expect(base.router.enabled).toBe(false)
    const on = setRouter(base, true)
    expect(on.router).toEqual({ enabled: true, backend: 'typesafe' })
    expect(on.sandbox.jevReview).toBe(true)
    expect(on.effort.mode).toBe('auto')
    const off = setRouter(applyProvider(defaultSettings('en'), 'opencode-go'), false)
    expect(off.router.enabled).toBe(false)
    expect(off.sandbox.jevReview).toBe(false)
    expect(off.effort).toMatchObject({ mode: 'fixed', fixedLevel: 'medium' })
    expect(setRouter(off, true).router).toEqual({ enabled: true, backend: 'opencode' })
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
  test('the provider step needs a provider', () => {
    expect(canContinue('provider', defaultSettings('en'), {})).toBe(false)
    expect(canContinue('provider', opencode, {})).toBe(true)
  })
  test('sign-in needs a saved key or finished OAuth', () => {
    expect(canContinue('auth', opencode, { signedIn: false })).toBe(false)
    expect(canContinue('auth', opencode, { signedIn: true })).toBe(true)
  })
  test('the router needs the TypeSafe key unless Jev comes with the provider', () => {
    const api = setRouter(applyProvider(defaultSettings('en'), 'anthropic-api'), true)
    expect(canContinue('router', api, { typesafeSaved: false })).toBe(false)
    expect(canContinue('router', api, { typesafeSaved: true })).toBe(true)
    expect(canContinue('router', setRouter(api, false), { typesafeSaved: false })).toBe(true)
    expect(canContinue('router', opencode, { typesafeSaved: false })).toBe(true)
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
