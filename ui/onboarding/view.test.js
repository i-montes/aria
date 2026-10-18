// Run: bun test ui
// Drives the onboarding controller without a browser: a fake root, delegated events and a bridge whose
// replies the test releases by hand, so slow sign-ins and reopenings can be reproduced exactly.
import { describe, expect, test } from 'bun:test'
import { applyProvider, defaultSettings } from './steps.js'
import { createOnboarding } from './view.js'

globalThis.document = { activeElement: null }
globalThis.CSS = { escape: (value) => value }
globalThis.HTMLInputElement = class {}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0))

function harness() {
  const listeners = {}
  const root = {
    innerHTML: '',
    querySelector: () => null,
    querySelectorAll: () => [],
    addEventListener: (type, fn) => {
      listeners[type] = fn
    },
  }
  const pending = []
  const sent = []
  const env = {
    root,
    t: (key) => key,
    post: (message) => sent.push(message),
    setLanguage: () => {},
    browserLanguage: () => 'en',
    request: (message, onProgress) => new Promise((resolve) => pending.push({ message, onProgress, resolve })),
  }
  const ob = createOnboarding(env)
  const click = (action, field, value) =>
    listeners.click({ target: { closest: () => ({ dataset: { action, field, value }, disabled: false }) } })
  const type = (field, value) => {
    const input = Object.assign(new HTMLInputElement(), { dataset: { action: 'input', field }, value })
    listeners.input({ target: input })
  }
  const lastRequest = () => pending.at(-1)
  return { ob, click, type, pending, sent, lastRequest, root }
}

const oauthOk = { type: 'oauthResult', ok: true, error: null }
const keyOk = { type: 'keyResult', ok: true, error: null }
const completed = { ...applyProvider(defaultSettings('en'), 'opencode-go'), setup: { completed: true, mode: 'quick' } }

describe('a sign-in that finishes late', () => {
  test('a Claude sign-in finishing after the provider changed does not sign the new provider in', async () => {
    const h = harness()
    h.ob.start(null)
    h.click('choose', undefined, 'full')
    h.click('pick', 'provider', 'anthropic-oauth')
    h.click('next')
    h.click('oauth-start')
    const oauth = h.lastRequest()
    expect(oauth.message.type).toBe('startOauth')
    h.click('back')
    h.click('pick', 'provider', 'opencode-go')
    oauth.resolve(oauthOk)
    await flush()
    expect(h.ob.snapshot().signedIn).toBe(false)
  })

  test('a key check finishing after the provider changed does not sign the new provider in', async () => {
    const h = harness()
    h.ob.start(null)
    h.click('choose', undefined, 'full')
    h.click('pick', 'provider', 'opencode-go')
    h.click('next') // quota
    h.click('next') // sign-in
    h.type('key', 'slow-key')
    h.click('save-key')
    const check = h.lastRequest()
    expect(check.message).toMatchObject({ type: 'saveKey', provider: 'opencode-go', key: 'slow-key' })
    h.click('back')
    h.click('back')
    h.click('pick', 'provider', 'anthropic-api')
    check.resolve(keyOk)
    await flush()
    expect(h.ob.snapshot().signedIn).toBe(false)
  })

  test('a sign-in that is still current does sign in', async () => {
    const h = harness()
    h.ob.start(null)
    h.click('choose', undefined, 'full')
    h.click('pick', 'provider', 'anthropic-oauth')
    h.click('next')
    h.click('oauth-start')
    h.lastRequest().resolve(oauthOk)
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ signedIn: true, oauth: 'done' })
  })
})

describe('abandoning a sign-in', () => {
  test('cancelling lets you start again, and the abandoned attempt is ignored when it ends', async () => {
    const h = harness()
    h.ob.start(null)
    h.click('choose', undefined, 'full')
    h.click('pick', 'provider', 'anthropic-oauth')
    h.click('next')
    h.click('oauth-start')
    const first = h.lastRequest()
    expect(h.ob.snapshot().oauth).toBe('waiting')
    h.click('oauth-cancel')
    expect(h.ob.snapshot().oauth).toBe('idle')
    h.click('oauth-start')
    const second = h.lastRequest()
    expect(second).not.toBe(first)
    first.resolve(oauthOk) // the browser tab that was closed finally reports
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ signedIn: false, oauth: 'waiting' })
    second.resolve(oauthOk)
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ signedIn: true, oauth: 'done' })
  })
})

describe('reopening Aria', () => {
  test('after setup is saved, opening the dock again shows the settings, not the finished screen', async () => {
    const h = harness()
    h.ob.start(completed)
    expect(h.ob.snapshot()).toMatchObject({ view: 'step', step: 'review', editing: true })
    h.click('next') // "Save changes"
    const save = h.lastRequest()
    expect(save.message.type).toBe('saveSettings')
    save.resolve({ type: 'saved' })
    await flush()
    expect(h.ob.snapshot().view).toBe('done')
    h.ob.onOpen()
    expect(h.lastRequest().message.type).toBe('loadSettings')
    h.lastRequest().resolve({ type: 'settings', settings: completed })
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ view: 'step', step: 'review', editing: true })
  })

  test('opening the dock in the middle of a setup keeps the place', () => {
    const h = harness()
    h.ob.start(null)
    h.click('choose', undefined, 'quick')
    h.ob.onOpen()
    expect(h.pending).toHaveLength(0)
    expect(h.ob.snapshot()).toMatchObject({ view: 'step', step: 'provider' })
  })
})

describe('what is really connected', () => {
  test('editing a setup knows whether its credential is on disk', () => {
    const h = harness()
    h.ob.start(completed, [])
    expect(h.ob.snapshot().signedIn).toBe(false)
    h.ob.start(completed, ['opencode-go:api_key'])
    expect(h.ob.snapshot().signedIn).toBe(true)
  })
})

describe('the integrations step', () => {
  const connected = ['opencode-go:api_key']
  const open = () => {
    const h = harness()
    h.ob.start(completed, connected)
    h.click('edit', undefined, 'integrations')
    expect(h.ob.snapshot().step).toBe('integrations')
    return h
  }
  const tavilyOk = { type: 'keyResult', ok: true, error: null }

  test('with no service turned on, Continue is open', () => {
    expect(open().ob.snapshot().canContinue).toBe(true)
  })

  test('Tavily: turned on it blocks Continue until its key is checked and saved', async () => {
    const h = open()
    h.click('toggle', 'service:tavily')
    expect(h.ob.snapshot().canContinue).toBe(false)
    h.type('tavilyKey', 'tvly-1')
    h.click('save-tavilyKey')
    expect(h.lastRequest().message).toMatchObject({ type: 'saveKey', provider: 'tavily', key: 'tvly-1' })
    h.lastRequest().resolve(tavilyOk)
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ canContinue: true })
    expect(h.ob.snapshot().connected).toContain('mcp:tavily')
  })

  test('Tavily: a rejected key keeps Continue closed', async () => {
    const h = open()
    h.click('toggle', 'service:tavily')
    h.type('tavilyKey', 'bad')
    h.click('save-tavilyKey')
    h.lastRequest().resolve({ type: 'keyResult', ok: false, error: 'key-rejected' })
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ canContinue: false })
    expect(h.ob.snapshot().connected).not.toContain('mcp:tavily')
  })

  test('Notion: the sign-in opens the browser flow and, once done, Continue opens', async () => {
    const h = open()
    h.click('toggle', 'service:notion')
    expect(h.ob.snapshot().canContinue).toBe(false)
    h.click('mcp-oauth-start', undefined, 'notion')
    expect(h.lastRequest().message).toMatchObject({ type: 'startMcpOauth', service: 'notion' })
    h.lastRequest().resolve(oauthOk)
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ canContinue: true })
    expect(h.ob.snapshot().connected).toContain('mcp:notion')
  })

  test('Notion: cancelling lets you start again and the abandoned attempt is ignored', async () => {
    const h = open()
    h.click('toggle', 'service:notion')
    h.click('mcp-oauth-start', undefined, 'notion')
    const first = h.lastRequest()
    h.click('mcp-oauth-cancel', undefined, 'notion')
    h.click('mcp-oauth-start', undefined, 'notion')
    const second = h.lastRequest()
    expect(second).not.toBe(first)
    first.resolve(oauthOk)
    await flush()
    expect(h.ob.snapshot().connected).not.toContain('mcp:notion')
    second.resolve(oauthOk)
    await flush()
    expect(h.ob.snapshot().connected).toContain('mcp:notion')
  })

  test('a service stays connected when the provider changes meanwhile', async () => {
    const h = open()
    h.click('toggle', 'service:notion')
    h.click('mcp-oauth-start', undefined, 'notion')
    const signIn = h.lastRequest()
    h.click('edit', undefined, 'provider')
    h.click('pick', 'provider', 'anthropic-api')
    signIn.resolve(oauthOk)
    await flush()
    expect(h.ob.snapshot().connected).toContain('mcp:notion')
  })

  test('a service that is already connected on disk lets you continue', () => {
    const h = harness()
    const withNotion = structuredClone(completed)
    withNotion.integrations.services = [{ id: 'notion', name: 'notion', auth: 'oauth' }]
    h.ob.start(withNotion, [...connected, 'mcp:notion'])
    h.click('edit', undefined, 'integrations')
    expect(h.ob.snapshot().canContinue).toBe(true)
  })

  test('turning a service off lets you continue without connecting it', () => {
    const h = open()
    h.click('toggle', 'service:tavily')
    expect(h.ob.snapshot().canContinue).toBe(false)
    h.click('toggle', 'service:tavily')
    expect(h.ob.snapshot().canContinue).toBe(true)
  })
})
