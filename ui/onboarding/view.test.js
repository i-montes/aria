// Run: bun test ui
// Drives the onboarding controller without a browser: a fake root, delegated events and a bridge whose
// replies the test releases by hand, so slow sign-ins and reopenings can be reproduced exactly.
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { applyProvider, defaultSettings } from './steps.js'
import { createOnboarding } from './view.js'

globalThis.document = { activeElement: null }
globalThis.CSS = { escape: (value) => value }
globalThis.HTMLInputElement = class {}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0))

function harness(extra = {}) {
  const listeners = {}
  const root = {
    innerHTML: '',
    querySelector: () => null,
    querySelectorAll: () => [],
    addEventListener: (type, fn) => {
      ;(listeners[type] ??= []).push(fn)
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
    ...extra,
  }
  const ob = createOnboarding(env)
  const click = (action, field, value) => {
    const el = { dataset: { action, field, value }, disabled: false }
    el.closest = () => null
    for (const fn of listeners.click) fn({ target: { closest: (selector) => (selector === '[data-action]' ? el : null) } })
  }
  const type = (field, value) => {
    const input = Object.assign(new HTMLInputElement(), { dataset: { action: 'input', field }, value })
    for (const fn of listeners.input) fn({ target: input })
  }
  const lastRequest = () => pending.at(-1)
  /** A click on an element that carries `data-chat` (the chat's model picker rows) instead of `data-action`. */
  const clickChat = (chat, index) => {
    const el = { dataset: { chat, index: String(index) } }
    for (const fn of listeners.click) fn({ target: { closest: (selector) => (selector === `[data-chat="${chat}"]` ? el : null) } })
  }
  const search = (value) => {
    for (const fn of listeners.input) fn({ target: { value, dataset: { chatField: 'model-search' } } })
  }
  /** A key pressed on the focused control (`init` as a KeyboardEvent's: key, code, the modifier flags). The result
   * says whether the page took it (its default prevented, kept from the dock's own Esc). */
  const press = (init) => {
    const event = { target: { dataset: {} }, defaultPrevented: false, stopped: false, ...init }
    event.preventDefault = () => (event.defaultPrevented = true)
    event.stopPropagation = event.stopImmediatePropagation = () => (event.stopped = true)
    for (const fn of listeners.keydown) {
      if (event.stopped) break
      fn(event)
    }
    return event
  }
  /** A key let go of; whether the page took the release. */
  const release = (init) => {
    const event = { target: { dataset: {} }, defaultPrevented: false, ...init }
    event.preventDefault = () => (event.defaultPrevented = true)
    event.stopPropagation = () => {}
    for (const fn of listeners.keyup ?? []) fn(event)
    return event.defaultPrevented
  }
  return { ob, click, clickChat, search, type, press, release, pending, sent, lastRequest, root }
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
  test('after the setup is saved, opening the dock again shows the settings, not the finished screen', async () => {
    const h = harness()
    h.ob.start(null, ['opencode-go:api_key'])
    h.click('choose', undefined, 'quick')
    h.click('pick', 'provider', 'opencode-go')
    h.click('next') // sign-in: the stored key is checked
    h.lastRequest().resolve(keyOk)
    await flush()
    h.click('next') // assistant -> review
    h.click('next') // "Start using Aria"
    h.lastRequest().resolve({ type: 'saved' })
    await flush()
    expect(h.ob.snapshot().view).toBe('done')
    h.ob.onOpen()
    expect(h.lastRequest().message.type).toBe('loadSettings')
    h.lastRequest().resolve({ type: 'settings', settings: completed })
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ view: 'settings', section: null, editing: true })
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

describe('a key that is already there', () => {
  const connected = ['opencode-go:api_key']
  const openFlow = (mode) => {
    const h = harness()
    h.ob.start(null, connected)
    h.click('choose', undefined, mode)
    h.click('pick', 'provider', 'opencode-go')
    h.click('next')
    if (mode === 'full') h.click('next') // quota
    return h
  }

  test('quick: the stored key is checked and the flow goes on to the assistant', async () => {
    const h = openFlow('quick')
    expect(h.ob.snapshot()).toMatchObject({ step: 'auth', signedIn: false, canContinue: false })
    expect(h.lastRequest().message).toMatchObject({ type: 'checkKey', provider: 'opencode-go' })
    h.lastRequest().resolve(keyOk)
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ step: 'assistant', signedIn: true, canContinue: true })
  })

  test('full: the stored key is checked and the flow goes on to the router', async () => {
    const h = openFlow('full')
    expect(h.ob.snapshot().step).toBe('auth')
    expect(h.lastRequest().message).toMatchObject({ type: 'checkKey', provider: 'opencode-go' })
    h.lastRequest().resolve(keyOk)
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ step: 'router', signedIn: true, canContinue: true })
  })

  test('a stored key that no longer works keeps the step and asks for a new one', async () => {
    const h = openFlow('quick')
    h.lastRequest().resolve({ type: 'keyResult', ok: false, error: 'key-rejected' })
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ step: 'auth', signedIn: false, canContinue: false })
  })

  test('editing: the check does not move the flow by itself', async () => {
    const h = harness()
    h.ob.start(completed, connected)
    h.click('edit', undefined, 'auth')
    expect(h.lastRequest().message).toMatchObject({ type: 'checkKey', provider: 'opencode-go' })
    h.lastRequest().resolve(keyOk)
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ step: 'auth', signedIn: true, canContinue: true })
  })

  test('a key pasted and checked goes on by itself', async () => {
    const h = harness()
    h.ob.start(null)
    h.click('choose', undefined, 'quick')
    h.click('pick', 'provider', 'opencode-go')
    h.click('next')
    expect(h.pending).toHaveLength(0) // nothing on disk: no check
    h.type('key', 'fresh-key')
    h.click('save-key')
    h.lastRequest().resolve(keyOk)
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ step: 'assistant', signedIn: true })
  })
})

describe('the same for Anthropic', () => {
  const openWith = (provider, connected) => {
    const h = harness()
    h.ob.start(null, connected)
    h.click('choose', undefined, 'quick')
    h.click('pick', 'provider', provider)
    h.click('next')
    return h
  }

  test('anthropic-api: a stored key is checked and the flow goes on', async () => {
    const h = openWith('anthropic-api', ['anthropic:api_key'])
    expect(h.ob.snapshot().step).toBe('auth')
    expect(h.lastRequest().message).toMatchObject({ type: 'checkKey', provider: 'anthropic-api' })
    h.lastRequest().resolve(keyOk)
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ step: 'assistant', signedIn: true })
  })

  test('anthropic-oauth: a stored sign-in is checked and the flow goes on', async () => {
    const h = openWith('anthropic-oauth', ['anthropic:oauth'])
    expect(h.lastRequest().message).toMatchObject({ type: 'checkKey', provider: 'anthropic-oauth' })
    h.lastRequest().resolve(keyOk)
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ step: 'assistant', signedIn: true, oauth: 'done' })
  })

  test('anthropic-oauth: a sign-in that no longer works asks for it again', async () => {
    const h = openWith('anthropic-oauth', ['anthropic:oauth'])
    h.lastRequest().resolve({ type: 'keyResult', ok: false, error: 'key-rejected' })
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ step: 'auth', signedIn: false, oauth: 'idle' })
  })

  test('a check that cannot reach the provider trusts the stored credential and goes on', async () => {
    const h = openWith('opencode-go', ['opencode-go:api_key'])
    h.lastRequest().resolve({ type: 'keyResult', ok: false, error: 'network-unavailable' })
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ step: 'assistant', signedIn: true })
  })

  test('signing in from scratch goes on by itself too', async () => {
    const h = harness()
    h.ob.start(null)
    h.click('choose', undefined, 'quick')
    h.click('pick', 'provider', 'anthropic-oauth')
    h.click('next')
    expect(h.pending).toHaveLength(0) // nothing on disk: no check
    h.click('oauth-start')
    h.lastRequest().resolve(oauthOk)
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ step: 'assistant', signedIn: true, oauth: 'done' })
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

describe('a service that is connected shows no way to connect it again', () => {
  const withServices = (...ids) => {
    const settings = structuredClone(completed)
    settings.integrations.services = ids.map((id) => ({ id, name: id, auth: id === 'tavily' ? 'key' : 'oauth' }))
    return settings
  }
  const open = (connected, ...services) => {
    const h = harness()
    h.ob.start(withServices(...services), ['opencode-go:api_key', ...connected])
    h.click('settings-open', undefined, 'integrations')
    return h
  }
  const tavilyControls = ['data-field="tavilyKey"', 'save-tavilyKey']
  const notionControls = ['mcp-oauth-start', 'mcp-oauth-cancel']

  test('Tavily connected: no key field and no button, only that it is connected', () => {
    const { root } = open(['mcp:tavily'], 'tavily')
    for (const control of tavilyControls) expect(root.innerHTML).not.toContain(control)
    expect(root.innerHTML).toContain('ob.integrations.connected')
  })

  test('Notion connected: no sign-in button, not even "sign in again"', () => {
    const { root } = open(['mcp:notion'], 'notion')
    for (const control of notionControls) expect(root.innerHTML).not.toContain(control)
    expect(root.innerHTML).not.toContain('ob.integrations.notion.again')
    expect(root.innerHTML).toContain('ob.integrations.connected')
  })

  test('not connected yet: the key field and the sign-in button are there', () => {
    const { root } = open([], 'tavily', 'notion')
    for (const control of [...tavilyControls, 'mcp-oauth-start']) expect(root.innerHTML).toContain(control)
  })

  test('saving the Tavily key makes its controls disappear', async () => {
    const h = open([], 'tavily')
    h.type('tavilyKey', 'tvly-1')
    h.click('save-tavilyKey')
    h.lastRequest().resolve({ type: 'keyResult', ok: true, error: null })
    await flush()
    for (const control of tavilyControls) expect(h.root.innerHTML).not.toContain(control)
    expect(h.root.innerHTML).toContain('ob.integrations.connected')
  })

  test('finishing the Notion sign-in makes its controls disappear', async () => {
    const h = open([], 'notion')
    h.click('mcp-oauth-start', undefined, 'notion')
    expect(h.root.innerHTML).toContain('mcp-oauth-cancel') // while it waits, you can still cancel
    h.lastRequest().resolve(oauthOk)
    await flush()
    for (const control of notionControls) expect(h.root.innerHTML).not.toContain(control)
    expect(h.root.innerHTML).toContain('ob.integrations.connected')
  })
})

describe('every control redraws the step from the settings (the components draw it)', () => {
  const agentsStep = () => {
    const h = harness()
    h.ob.start(completed, ['opencode-go:api_key'])
    h.click('settings-open', undefined, 'agents')
    return h
  }

  for (const [name, act] of [
    ['a skill checkbox', (h) => h.click('check', 'skills', 'systematic-debugging')],
    ['an expert switch', (h) => h.click('expert-on', undefined, completed.subagents.experts[0].name)],
    ['the turn stepper', (h) => h.click('step', 'turnLimit', '5')],
  ]) {
    test(`${name}`, () => {
      const h = agentsStep()
      const before = h.root.innerHTML
      act(h)
      expect(h.root.innerHTML).not.toBe(before)
    })
  }

  test('a redraw keeps the reader where they were in the step', () => {
    const panel = { scrollTop: 0 }
    globalThis.document.getElementById = (id) => (id === 'content' ? panel : null)
    try {
      const h = agentsStep()
      panel.scrollTop = 420
      h.click('expert-on', undefined, completed.subagents.experts[0].name)
      expect(panel.scrollTop).toBe(420)
    } finally {
      delete globalThis.document.getElementById
    }
  })
})

describe('a reply that arrives after the user moved on', () => {
  test('a stored-key check that finishes after the user went back does not pull them forward', async () => {
    const h = harness()
    h.ob.start(null, ['opencode-go:api_key'])
    h.click('choose', undefined, 'quick')
    h.click('pick', 'provider', 'opencode-go')
    h.click('next')
    const check = h.lastRequest()
    h.click('back')
    expect(h.ob.snapshot().step).toBe('provider')
    check.resolve(keyOk)
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ view: 'step', step: 'provider' })
  })

  test('a browser sign-in that finishes after the user went back does not pull them forward', async () => {
    const h = harness()
    h.ob.start(null)
    h.click('choose', undefined, 'quick')
    h.click('pick', 'provider', 'anthropic-oauth')
    h.click('next')
    h.click('oauth-start')
    const signIn = h.lastRequest()
    h.click('back')
    signIn.resolve(oauthOk)
    await flush()
    expect(h.ob.snapshot().step).toBe('provider')
  })

  test('while a stored Claude sign-in is being checked, Continue waits', () => {
    const h = harness()
    h.ob.start(null, ['anthropic:oauth'])
    h.click('choose', undefined, 'quick')
    h.click('pick', 'provider', 'anthropic-oauth')
    h.click('next')
    expect(h.ob.snapshot()).toMatchObject({ step: 'auth', oauth: 'checking', canContinue: false })
  })
})

describe('the settings screen', () => {
  const settingsHarness = (connected = ['opencode-go:api_key'], settings = completed) => {
    const h = harness()
    h.ob.start(settings, connected)
    return h
  }

  test('a finished setup opens on the overview, one card per section', () => {
    const h = settingsHarness()
    expect(h.ob.snapshot()).toMatchObject({ view: 'settings', section: null, dirty: false })
    for (const id of ['assistant', 'provider', 'models', 'router', 'context', 'integrations', 'safety', 'agents']) {
      expect(h.root.innerHTML).toContain(`data-action="settings-open" data-value="${id}"`)
    }
    expect(h.root.innerHTML).not.toContain('settings-save') // nothing changed, nothing to save
  })

  test('the overview says which Aria this is once the dock told the version', () => {
    expect(settingsHarness().root.innerHTML).not.toContain('Aria 0.')
    const h = harness({ version: () => '0.2.0' })
    h.ob.start(completed, ['opencode-go:api_key'])
    expect(h.root.innerHTML).toContain('~/.aria/settings.json · Aria 0.2.0')
  })

  test('a section shows every step it groups, with the quota only for a provider that has one', () => {
    const h = settingsHarness()
    h.click('settings-open', undefined, 'provider')
    expect(h.root.innerHTML).toContain('ob.provider.title')
    expect(h.root.innerHTML).toContain('ob.auth.title')
    expect(h.root.innerHTML).toContain('ob.quota.title')
    const claude = { ...applyProvider(defaultSettings('en'), 'anthropic-api'), setup: { completed: true, mode: 'full' } }
    const other = settingsHarness(['anthropic:api_key'], claude)
    other.click('settings-open', undefined, 'provider')
    expect(other.root.innerHTML).not.toContain('ob.quota.title')
  })

  test('a change shows the save bar; saving writes it and keeps the reader on the section', async () => {
    const h = settingsHarness()
    h.click('settings-open', undefined, 'safety')
    h.click('toggle', 'commit')
    expect(h.ob.snapshot().dirty).toBe(true)
    expect(h.root.innerHTML).toContain('settings-save')
    h.click('settings-save')
    const save = h.lastRequest()
    expect(save.message.type).toBe('saveSettings')
    expect(save.message.settings.git.commitWithoutAsking).toBe(true)
    expect(save.message.settings.setup).toEqual(completed.setup)
    save.resolve({ type: 'saved' })
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ view: 'settings', section: 'safety', dirty: false })
    expect(h.root.innerHTML).toContain('settings.saved')
  })

  test('discarding brings back what is on disk', () => {
    const h = settingsHarness()
    h.click('settings-open', undefined, 'safety')
    h.click('toggle', 'commit')
    h.click('settings-discard')
    expect(h.ob.snapshot().dirty).toBe(false)
    expect(h.pending).toHaveLength(0)
  })

  test('leaving with changes asks first; leaving without them goes back to the chat', async () => {
    const h = settingsHarness()
    h.click('settings-exit')
    expect(h.sent).toContainEqual({ type: 'closeRequest' }) // no onDone in the harness: the dock closes
    const g = settingsHarness()
    g.click('settings-open', undefined, 'context')
    g.click('toggle', 'memory')
    g.click('settings-exit')
    expect(g.ob.snapshot()).toMatchObject({ view: 'settings', leaving: true })
    expect(g.sent).not.toContainEqual({ type: 'closeRequest' })
    g.click('settings-save')
    g.lastRequest().resolve({ type: 'saved' })
    await flush()
    expect(g.sent).toContainEqual({ type: 'closeRequest' })
  })

  test('a section with something to finish is marked and blocks the save', () => {
    const settings = structuredClone(completed)
    settings.integrations.services = [{ id: 'notion', name: 'notion', auth: 'oauth' }]
    const h = settingsHarness(['opencode-go:api_key'], settings)
    expect(h.root.innerHTML).toContain('settings.attention')
    h.click('settings-open', undefined, 'safety')
    h.click('toggle', 'commit')
    expect(h.root.innerHTML).toContain('settings.blocked')
    expect(h.root.innerHTML).toMatch(/data-action="settings-save"[^>]*disabled/)
    h.click('settings-save')
    expect(h.pending).toHaveLength(0)
  })

  test('opening the provider section checks the stored key, without moving anywhere', async () => {
    const h = settingsHarness()
    h.click('settings-open', undefined, 'provider')
    expect(h.lastRequest().message).toMatchObject({ type: 'checkKey', provider: 'opencode-go' })
    h.lastRequest().resolve(keyOk)
    await flush()
    expect(h.ob.snapshot()).toMatchObject({ view: 'settings', section: 'provider', key: 'saved' })
  })

  test('an outside change refreshes the screen, but never over the reader\'s own changes', () => {
    const outside = structuredClone(completed)
    outside.compaction.percent = 90
    const h = settingsHarness()
    h.ob.refreshSettings(outside, ['opencode-go:api_key'])
    expect(h.root.innerHTML).toContain('ob.val.compaction')
    expect(h.ob.snapshot().dirty).toBe(false)
    h.click('toggle', 'memory') // a change of the reader's own
    h.ob.refreshSettings(completed, ['opencode-go:api_key'])
    expect(h.ob.snapshot().dirty).toBe(true)
  })
})

describe('the model fields list what Pi offers for the connected provider', () => {
  const piModels = [
    { provider: 'opencode-go', id: 'deepseek-v4.1-flash', name: 'DeepSeek V4.1 Flash' },
    { provider: 'opencode-go', id: 'minimax-m3', name: 'MiniMax M3' },
    { provider: 'opencode', id: 'claude-opus-5-5', name: 'Claude Opus 5.5 (Zen)' },
    { provider: 'anthropic', id: 'claude-sonnet-5-5', name: 'Claude Sonnet 5.5' },
  ]
  const openModels = (settings = completed) => {
    let answer
    const h = harness({ listModels: () => new Promise((resolve) => (answer = resolve)) })
    h.ob.start(settings, ['opencode-go:api_key'])
    h.click('settings-open', undefined, 'models')
    return { ...h, answer: async (list = piModels) => { answer(list); await flush() } }
  }

  test('the picker shows the provider\'s models from Pi, not the catalog, and picking one sets the field', async () => {
    const h = openModels()
    await h.answer()
    h.click('model-open', 'daily')
    expect(h.root.innerHTML).toContain('MiniMax M3')
    expect(h.root.innerHTML).not.toContain('Claude Sonnet 5.5') // another provider
    expect(h.root.innerHTML).not.toContain('(Zen)') // pay-per-use is not allowed
    h.search('minimax')
    h.clickChat('pick', 0)
    expect(h.ob.snapshot().dirty).toBe(true)
    h.click('settings-save')
    expect(h.lastRequest().message.settings.models.daily).toBe('opencode-go/minimax-m3')
  })

  test('with pay-per-use allowed, OpenCode Zen\'s models are offered too', async () => {
    const settings = structuredClone(completed)
    settings.models.allowPayPerUse = true
    const h = openModels(settings)
    await h.answer()
    h.click('model-open', 'planning')
    expect(h.root.innerHTML).toContain('(Zen)')
  })

  test('while Pi has not answered (or has nothing for the provider), the provider\'s catalog is offered', async () => {
    const h = openModels()
    h.click('model-open', 'daily')
    expect(h.root.innerHTML).toContain('GLM 5.3')
    await h.answer([])
    expect(h.root.innerHTML).toContain('GLM 5.3')
  })

  test('the turn timeout stepper writes turnTimeoutMinutes, from 0 (off) to 60', () => {
    const h = openModels()
    expect(h.root.innerHTML).toContain('ob.models.turnTimeout')
    expect(h.root.innerHTML).toContain('10 ob.models.minutes')
    h.click('step', 'turnTimeoutMinutes', '-5')
    expect(h.root.innerHTML).toContain('5 ob.models.minutes')
    h.click('step', 'turnTimeoutMinutes', '-5')
    h.click('step', 'turnTimeoutMinutes', '-5') // never below 0
    h.click('settings-save')
    expect(h.lastRequest().message.settings.models.turnTimeoutMinutes).toBe(0)
  })

  test('another control closes the picker', async () => {
    const h = openModels()
    await h.answer()
    h.click('model-open', 'daily')
    expect(h.root.innerHTML).toContain('data-model-picker')
    h.click('pick', 'planningEffort', 'low')
    expect(h.root.innerHTML).not.toContain('data-model-picker')
  })
})

describe('the access level of each connector', () => {
  const connected = ['opencode-go:api_key', 'mcp:notion']
  const servers = ['my-docs', 'notion', 'engram']
  const open = (settings = completed) => {
    const h = harness()
    h.ob.start(settings, connected, servers)
    h.click('settings-open', undefined, 'integrations')
    return h
  }
  const selected = (html, server) => [...html.matchAll(/class="seg selected"[^>]*data-field="([^"]*)" data-value="([^"]*)"/g)].filter((m) => m[1] === `access:${server}`).map((m) => m[2])

  test('integrations list each connected server with its access', () => {
    const settings = structuredClone(completed)
    settings.integrations.access = { notion: 'read' }
    const html = open(settings).root.innerHTML
    expect(html).toContain('ob.integrations.section.access')
    expect(html).toContain('My Docs')
    expect(html).toContain('Notion')
    expect(html).not.toContain('access:engram')
    for (const level of ['read', 'ask', 'allow']) {
      expect(html).toContain(`data-field="access:notion" data-value="${level}"`)
      expect(html).toContain('ob.integrations.access.' + level)
    }
    expect(selected(html, 'notion')).toEqual(['read'])
    expect(selected(html, 'my-docs')).toEqual(['ask']) // nothing set: Ask
  })

  test('no configured server, no access rows', () => {
    const h = harness()
    h.ob.start(completed, ['opencode-go:api_key', 'mcp:engram'], ['engram'])
    h.click('settings-open', undefined, 'integrations')
    expect(h.root.innerHTML).not.toContain('ob.integrations.section.access')
  })

  test('a server configured without a credential has its row too', () => {
    const h = harness()
    h.ob.start(completed, ['opencode-go:api_key'], ['rea'])
    h.click('settings-open', undefined, 'integrations')
    expect(h.root.innerHTML).toContain('data-field="access:rea" data-value="ask"')
    expect(h.root.innerHTML).toContain('Rea')
  })

  test('an older dock without servers: the connected services still get their rows', () => {
    const h = harness()
    h.ob.start(completed, ['opencode-go:api_key', 'mcp:engram', 'mcp:notion'])
    h.click('settings-open', undefined, 'integrations')
    expect(h.root.innerHTML).toContain('data-field="access:notion"')
    expect(h.root.innerHTML).not.toContain('access:engram')
  })

  test('choosing a level saves it', async () => {
    const h = open()
    h.click('pick', 'access:my-docs', 'allow')
    expect(selected(h.root.innerHTML, 'my-docs')).toEqual(['allow'])
    expect(h.ob.snapshot().dirty).toBe(true)
    h.click('pick', 'access:notion', 'nonsense') // only the three levels are ever written
    h.click('settings-save')
    const save = h.lastRequest()
    expect(save.message.type).toBe('saveSettings')
    expect(save.message.settings.integrations.access).toEqual({ 'my-docs': 'allow' })
    save.resolve({ type: 'saved' })
    await flush()
    expect(h.ob.snapshot().dirty).toBe(false)
  })
})

describe('the MCP servers the person already has', () => {
  const found = [
    { id: 'claude-code:files', name: 'files', importAs: 'files', source: 'claude-code', kind: 'stdio', target: 'npx', secrets: [], importable: true, disabled: false },
    { id: 'cursor:docs', name: 'my docs', importAs: 'my-docs', source: 'cursor', kind: 'http', target: 'docs.example.com', secrets: ['X-Api-Key'], importable: true, disabled: true },
    { id: 'vscode:tracker', name: 'tracker', importAs: 'tracker', source: 'vscode', kind: 'stdio', target: 'node', secrets: ['TRACKER_TOKEN', 'input:tracker-token'], importable: true, disabled: false },
    { id: 'cursor:feed', name: 'feed', importAs: 'feed', source: 'cursor', kind: 'sse', target: 'feed.example.com', secrets: [], importable: false, disabled: false },
  ]
  // The texts with their placeholders, so the names that fill them can be seen.
  const t = (key, vars) => (vars ? `${key}${JSON.stringify(vars)}` : key)
  const scanOf = (h) => h.pending.find((p) => p.message.type === 'mcpScan' && !p.answered)
  const answer = async (p, reply) => {
    p.answered = true
    p.resolve(reply)
    await flush()
  }
  const inSettings = async (servers = found) => {
    const h = harness({ t })
    h.ob.start(completed, ['opencode-go:api_key'], ['notion'])
    h.click('settings-open', undefined, 'integrations')
    await answer(scanOf(h), { type: 'mcpFound', id: 1, servers })
    return h
  }

  test('found servers are listed with source and target', async () => {
    const h = await inSettings()
    const html = h.root.innerHTML
    expect(html).toContain('ob.integrations.found.section')
    expect(html).toContain('data-field="mcpImport" data-value="claude-code:files"')
    expect(html).toContain('ob.integrations.found.source.claude-code')
    expect(html).toContain('npx')
    expect(html).toContain('my docs → my-docs') // the name Aria gives it, only when it differs
    expect(html).not.toContain('files → files')
    expect(html).toContain('ob.integrations.found.disabled')
    // SSE is listed, but has no checkbox
    expect(html).toContain('feed')
    expect(html).toContain('ob.integrations.found.sse')
    expect(html).not.toContain('data-value="cursor:feed"')
    // the onboarding step scans too, and the scan never blocks Continue
    const s = harness({ t })
    s.ob.start(null)
    s.click('choose', undefined, 'full')
    s.click('edit', undefined, 'integrations')
    expect(scanOf(s)).toBeTruthy()
    expect(s.ob.snapshot().canContinue).toBe(true)
  })

  test('secrets show only their names', async () => {
    const h = await inSettings()
    const html = h.root.innerHTML
    expect(html).toContain('ob.integrations.found.uses{&quot;names&quot;:&quot;X-Api-Key&quot;}')
    expect(html).toContain('ob.integrations.found.uses{&quot;names&quot;:&quot;TRACKER_TOKEN&quot;}')
    expect(html).toContain('ob.integrations.found.input{&quot;id&quot;:&quot;tracker-token&quot;}')
    expect(html).not.toContain('input:tracker-token&quot;')
  })

  test('import sends the checked ids', async () => {
    const h = await inSettings()
    expect(h.root.innerHTML).toMatch(/data-action="mcp-import"[^>]*disabled/) // nothing checked yet
    h.click('check', 'mcpImport', 'claude-code:files')
    h.click('check', 'mcpImport', 'cursor:docs')
    h.click('check', 'mcpImport', 'cursor:docs') // unchecked again
    h.click('check', 'mcpImport', 'vscode:tracker')
    h.click('check', 'mcpImport', 'cursor:feed') // not importable: ignored
    expect(h.root.innerHTML).not.toMatch(/data-action="mcp-import"[^>]*disabled/)
    expect(h.ob.snapshot().dirty).toBe(false) // picking is not a change to the settings
    h.click('mcp-import')
    const sent = h.lastRequest()
    expect(sent.message).toEqual({ type: 'mcpImport', ids: ['claude-code:files', 'vscode:tracker'] })
    await answer(sent, { type: 'error', code: 'mcp-write-failed' })
    expect(h.root.innerHTML).toContain('ob.error.mcp-write-failed')
  })

  test('what may not work the same in Aria is said in plain words', async () => {
    const odd = { id: 'vscode:db', name: 'db', importAs: 'db', source: 'vscode', kind: 'stdio', target: 'db', secrets: [], importable: true, disabled: false,
      warnings: ['expands:TOOLS', 'expands:KEY', 'default:PORT', 'unsupported:workspaceFolder', 'tilde-arg', 'envFile'] }
    const h = await inSettings([odd, ...found])
    const html = h.root.innerHTML
    expect(html).toContain('ob.integrations.found.warn.expands{&quot;names&quot;:&quot;TOOLS, KEY&quot;}')
    expect(html).toContain('ob.integrations.found.warn.default{&quot;names&quot;:&quot;PORT&quot;}')
    expect(html).toContain('ob.integrations.found.warn.unsupported{&quot;names&quot;:&quot;${workspaceFolder}&quot;}')
    expect(html).toContain('ob.integrations.found.warn.tildeArg')
    expect(html).toContain('ob.integrations.found.warn.envFile')
    // a server without warnings (or from an older dock, with no field) has none
    expect(html.match(/ob\.integrations\.found\.warn\.envFile/g)).toHaveLength(1)
  })

  test('the warnings read well in both languages', () => {
    const catalog = (code) => JSON.parse(readFileSync(new URL(`../i18n/${code}.json`, import.meta.url), 'utf8'))
    for (const code of ['en', 'es']) {
      const texts = catalog(code)
      for (const key of ['expands', 'default', 'unsupported', 'tildeArg', 'envFile']) expect(texts[`ob.integrations.found.warn.${key}`]).toBeTruthy()
    }
    expect(catalog('en')['ob.integrations.found.imported.one'].replace('{names}', 'files')).toBe('Imported 1: files')
    expect(catalog('es')['ob.integrations.found.imported.one'].replace('{names}', 'files')).toBe('Importado 1: files')
    expect(catalog('es')['ob.integrations.found.imported.many'].replace('{n}', '2').replace('{names}', 'a, b')).toBe('Importados 2: a, b')
  })

  test('a server name is shown as text, never as markup', async () => {
    const evil = { id: 'cursor:<img src=x onerror=alert(1)>', name: '<img src=x onerror=alert(1)>', importAs: 'img-src-x-onerror-alert-1', source: 'cursor', kind: 'stdio', target: '<b>run</b>', secrets: ['<i>KEY</i>'], importable: true, disabled: false }
    const h = await inSettings([evil])
    const html = h.root.innerHTML
    expect(html).not.toContain('<img')
    expect(html).not.toContain('<b>run</b>')
    expect(html).not.toContain('<i>KEY</i>')
    expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;')
    expect(html).toContain('&lt;b&gt;run&lt;/b&gt;')
  })

  test('one server imported is said in the singular, more in the plural', async () => {
    const h = await inSettings()
    h.click('check', 'mcpImport', 'claude-code:files')
    h.click('mcp-import')
    await answer(h.lastRequest(), { type: 'saved', id: 2 })
    await answer(h.lastRequest(), { type: 'settings', settings: completed, connected: [], servers: ['notion', 'files'] })
    await answer(scanOf(h), { type: 'mcpFound', id: 3, servers: found.slice(1) })
    expect(h.root.innerHTML).toContain('ob.integrations.found.imported.one{&quot;n&quot;:1,&quot;names&quot;:&quot;files&quot;}')
    h.click('check', 'mcpImport', 'cursor:docs')
    h.click('check', 'mcpImport', 'vscode:tracker')
    h.click('mcp-import')
    await answer(h.lastRequest(), { type: 'saved', id: 4 })
    await answer(h.lastRequest(), { type: 'settings', settings: completed, connected: [], servers: ['notion', 'files', 'my-docs', 'tracker'] })
    await answer(scanOf(h), { type: 'mcpFound', id: 5, servers: found.slice(3) })
    expect(h.root.innerHTML).toContain('ob.integrations.found.imported.many{&quot;n&quot;:2,&quot;names&quot;:&quot;my-docs, tracker&quot;}')
  })

  test('nothing left to import reads the list again and says why', async () => {
    const h = await inSettings()
    h.click('check', 'mcpImport', 'claude-code:files')
    h.click('mcp-import')
    await answer(h.lastRequest(), { type: 'error', code: 'nothing-to-import' })
    const again = scanOf(h)
    expect(again).toBeTruthy() // a new scan
    await answer(again, { type: 'mcpFound', id: 3, servers: found.slice(1) })
    const html = h.root.innerHTML
    expect(html).not.toContain('data-value="claude-code:files"') // gone from the other app
    expect(html).toContain('ob.error.nothing-to-import')
  })

  test('a broken mcp.json says nothing was imported', async () => {
    const h = await inSettings()
    h.click('check', 'mcpImport', 'claude-code:files')
    h.click('mcp-import')
    await answer(h.lastRequest(), { type: 'error', code: 'mcp-file-broken' })
    const html = h.root.innerHTML
    expect(html).toContain('ob.integrations.found.fileBroken')
    expect(html).not.toContain('ob.error.mcp-file-broken')
    expect(scanOf(h)).toBeFalsy() // no new scan for this one
  })

  test('nothing found says so', async () => {
    const h = await inSettings([])
    expect(h.root.innerHTML).toContain('ob.integrations.found.none')
    expect(h.root.innerHTML).not.toContain('data-action="mcp-import"')
  })

  test('already imported servers are not offered again after a refresh', async () => {
    const h = await inSettings()
    h.click('check', 'mcpImport', 'claude-code:files')
    h.click('mcp-import')
    await answer(h.lastRequest(), { type: 'saved', id: 2 })
    const load = h.lastRequest()
    expect(load.message.type).toBe('loadSettings')
    await answer(load, { type: 'settings', settings: completed, connected: ['opencode-go:api_key'], servers: ['notion', 'files'] })
    // the access rows show the new server, at Ask
    expect(h.root.innerHTML).toContain('data-field="access:files" data-value="ask"')
    await answer(scanOf(h), { type: 'mcpFound', id: 3, servers: found.slice(1) })
    const html = h.root.innerHTML
    expect(html).not.toContain('data-value="claude-code:files"')
    expect(html).toContain('ob.integrations.found.imported.one{&quot;n&quot;:1,&quot;names&quot;:&quot;files&quot;}')
    expect(h.ob.snapshot().dirty).toBe(false)
  })
})

describe('the shortcut row', () => {
  // The texts with their placeholders, so the values that fill them can be seen.
  const t = (key, vars) => (vars ? `${key}${JSON.stringify(vars)}` : key)
  const open = (platform = { hotkey: 'Super + A', taken: [], os: 'linux' }, settings = completed) => {
    const h = harness({ t, platform: () => platform })
    h.ob.start(settings, ['opencode-go:api_key'])
    h.click('settings-open', undefined, 'assistant')
    return h
  }
  const kbds = (html) => [...html.matchAll(/<kbd>([^<]*)<\/kbd>/g)].map((m) => m[1])
  const saved = (h) => {
    h.click('settings-save')
    return h.lastRequest().message.settings.shortcut
  }

  test('the shortcut row records keys', () => {
    const h = open()
    expect(h.root.innerHTML).toContain('ob.shortcut.label')
    expect(kbds(h.root.innerHTML)).toEqual(['Super', 'A']) // the default, while the setting is empty
    expect(h.root.innerHTML).toContain('ob.shortcut.default')
    h.click('key-capture', 'shortcut')
    expect(h.root.innerHTML).toContain('ob.shortcut.recording')
    // a modifier alone waits for the key
    expect(h.press({ key: 'Shift', code: 'ShiftLeft', shiftKey: true }).defaultPrevented).toBe(true)
    expect(h.root.innerHTML).toContain('ob.shortcut.recording')
    // a key without a modifier, or one the dock cannot bind, says so and keeps recording
    h.press({ key: 'k', code: 'KeyK' })
    expect(h.root.innerHTML).toContain('ob.shortcut.needsModifier.linux')
    h.press({ key: 'Enter', code: 'Enter', ctrlKey: true })
    expect(h.root.innerHTML).toContain('ob.shortcut.unsupported')
    expect(h.ob.snapshot().dirty).toBe(false)
    // the key by its position (event.code), the modifiers in their canonical order
    const taken = h.press({ key: 'K', code: 'KeyK', shiftKey: true, metaKey: true })
    expect(taken.defaultPrevented && taken.stopped).toBe(true)
    expect(h.root.innerHTML).not.toContain('ob.shortcut.recording')
    expect(kbds(h.root.innerHTML)).toEqual(['Super', 'Shift', 'K'])
    expect(h.ob.snapshot().dirty).toBe(true)
    expect(saved(h)).toBe('SUPER + SHIFT + K')
    // keys outside a recording are not taken
    expect(h.press({ key: 'k', code: 'KeyK', ctrlKey: true }).defaultPrevented).toBe(false)
  })

  test('on a Mac the modifiers have their Mac names, and Option does not change the letter', () => {
    const h = open({ hotkey: 'Option + Space', taken: [], os: 'mac' })
    expect(kbds(h.root.innerHTML)).toEqual(['Option', 'key.space']) // Space in the reader's language
    h.click('key-capture', 'shortcut')
    h.press({ key: 'å', code: 'KeyA', altKey: true, metaKey: true })
    expect(kbds(h.root.innerHTML)).toEqual(['Command', 'Option', 'A'])
    h.click('key-capture', 'shortcut')
    h.press({ key: 'F5', code: 'F5', ctrlKey: true })
    expect(saved(h)).toBe('Control + F5')
  })

  test('escape cancels', () => {
    const h = open()
    h.click('key-capture', 'shortcut')
    const esc = h.press({ key: 'Escape', code: 'Escape' })
    expect(esc.stopped).toBe(true) // the dock does not close
    expect(h.root.innerHTML).not.toContain('ob.shortcut.recording')
    expect(kbds(h.root.innerHTML)).toEqual(['Super', 'A'])
    expect(h.ob.snapshot().dirty).toBe(false)
    // a click elsewhere cancels too
    h.click('key-capture', 'shortcut')
    h.click('pick', 'style', 'brief')
    expect(h.root.innerHTML).not.toContain('ob.shortcut.recording')
  })

  test('backspace resets to the default', () => {
    const h = open({ hotkey: 'Ctrl + Alt + K', taken: [], os: 'linux' }, { ...completed, shortcut: 'CTRL + ALT + K' })
    expect(kbds(h.root.innerHTML)).toEqual(['Ctrl', 'Alt', 'K'])
    expect(h.root.innerHTML).not.toContain('ob.shortcut.default')
    h.click('key-capture', 'shortcut')
    h.press({ key: 'Backspace', code: 'Backspace' })
    expect(kbds(h.root.innerHTML)).toEqual(['Super', 'A'])
    expect(h.root.innerHTML).toContain('ob.shortcut.default')
    expect(saved(h)).toBe('')
  })

  test('backspace with Super held (however the engine says it) is a key, not a reset', () => {
    for (const name of ['Super', 'OS', 'Meta']) {
      const h = open({ hotkey: 'Ctrl + Alt + K', taken: [], os: 'linux' }, { ...completed, shortcut: 'CTRL + ALT + K' })
      h.click('key-capture', 'shortcut')
      h.press({ key: 'Backspace', code: 'Backspace', getModifierState: (n) => n === name })
      expect(h.root.innerHTML).toContain('ob.shortcut.recording') // still recording, with a note
      expect(h.root.innerHTML).toContain('ob.shortcut.unsupported')
      expect(h.ob.snapshot().dirty).toBe(false) // not reset to the default
    }
  })

  test('typing in the field again takes the "not a shortcut" note away', () => {
    const h = open()
    h.type('shortcutText', 'super + return')
    h.click('shortcut-apply', 'shortcutText')
    expect(h.root.innerHTML).toContain('ob.shortcut.typedInvalid')
    h.type('shortcutText', 'super + r')
    h.ob.platformChanged() // any redraw
    expect(h.root.innerHTML).not.toContain('ob.shortcut.typedInvalid')
    // the fallback note comes back once the person's own note is gone
    const p = { hotkey: 'Super + Alt + A', taken: ['Super + A'], os: 'linux' }
    const g = open(p)
    g.type('shortcutText', 'nope')
    g.click('shortcut-apply', 'shortcutText')
    g.type('shortcutText', 'nope2')
    g.ob.platformChanged()
    expect(g.root.innerHTML).toContain('ob.shortcut.inUse.one')
  })

  test('in use shows a fallback', () => {
    const platform = { hotkey: 'Super + A', taken: [], os: 'linux' }
    const h = open(platform, { ...completed, shortcut: 'super + shift + a' })
    expect(kbds(h.root.innerHTML)).toEqual(['Super', 'Shift', 'A']) // a hand-written setting, shown canonical
    expect(h.root.innerHTML).not.toContain('ob.shortcut.inUse')
    // the dock says the chosen one is taken: the row says which one works
    Object.assign(platform, { hotkey: 'Super + Alt + A', taken: ['Super + Shift + A', 'Super + A'] })
    h.ob.platformChanged()
    expect(h.root.innerHTML.replaceAll('&quot;', '"')).toContain('ob.shortcut.inUse.many{"hotkey":"Super + Alt + A","taken":"Super + Shift + A, Super + A"}')
    Object.assign(platform, { hotkey: 'Super + A', taken: ['Super + Shift + A'] })
    h.ob.platformChanged()
    expect(h.root.innerHTML.replaceAll('&quot;', '"')).toContain('ob.shortcut.inUse.one{"hotkey":"Super + A","taken":"Super + Shift + A"}')
  })

  test('Super is recognized however the engine reports it', () => {
    const shapes = [
      { metaKey: true },
      { getModifierState: (name) => name === 'Super' },
      { getModifierState: (name) => name === 'OS' },
      { getModifierState: (name) => name === 'Meta' },
    ]
    for (const shape of shapes) {
      const h = open()
      h.click('key-capture', 'shortcut')
      // the modifier alone, under each name WebKitGTK and others give it, waits for the key
      for (const key of ['Super_L', 'Super_R', 'Meta', 'OS', 'Super']) {
        h.press({ key, code: 'MetaLeft', ...shape })
        expect(h.root.innerHTML).toContain('ob.shortcut.recording')
      }
      h.press({ key: 'j', code: 'KeyJ', ...shape })
      expect(saved(h)).toBe('SUPER + J')
    }
  })

  test('a shortcut can be typed instead', () => {
    const h = open()
    expect(h.root.innerHTML).toContain('data-field="shortcutText"')
    h.type('shortcutText', 'super + shift + 1') // one the desktop would take before the page sees it
    h.click('shortcut-apply', 'shortcutText')
    expect(kbds(h.root.innerHTML)).toEqual(['Super', 'Shift', '1'])
    expect(saved(h)).toBe('SUPER + SHIFT + 1')
    h.type('shortcutText', 'super + return')
    h.click('shortcut-apply', 'shortcutText')
    expect(h.root.innerHTML).toContain('ob.shortcut.typedInvalid')
    expect(kbds(h.root.innerHTML)).toEqual(['Super', 'Shift', '1']) // nothing changed
    // Enter in the field applies it too
    h.type('shortcutText', 'ctrl+alt+space')
    const enter = { key: 'Enter', code: 'Enter', target: Object.assign(new HTMLInputElement(), { dataset: { action: 'input', field: 'shortcutText' }, type: 'text' }) }
    h.press(enter)
    expect(kbds(h.root.innerHTML)).toEqual(['Ctrl', 'Alt', 'key.space'])
    expect(h.root.innerHTML).not.toContain('ob.shortcut.typedInvalid')
    // on a Mac the names are the Mac ones
    const m = open({ hotkey: 'Option + Space', taken: [], os: 'mac' })
    m.type('shortcutText', 'cmd + shift + k')
    m.click('shortcut-apply', 'shortcutText')
    expect(saved(m)).toBe('Command + Shift + K')
  })

  test('none available says so', () => {
    const h = open({ hotkey: '', taken: ['Option + Space', 'Control + Option + Space', 'Command + Option + Space'], os: 'mac' })
    const html = h.root.innerHTML.replaceAll('&quot;', '"')
    expect(html).toContain('ob.shortcut.none{"taken":"Option + key.space, Control + Option + key.space, Command + Option + key.space"}')
    expect(html).not.toContain('ob.shortcut.inUse')
  })

  test('no shortcut and none tried (the Mac\'s key handler failed) says the dock keeps trying', () => {
    const h = open({ hotkey: '', taken: [], os: 'mac' })
    expect(h.root.innerHTML).toContain('ob.shortcut.failed')
    expect(h.root.innerHTML).not.toContain('ob.shortcut.none')
  })

  test('only the release of the key that ended a recording is taken', () => {
    const h = open()
    h.click('key-capture', 'shortcut')
    h.press({ key: 'Shift', code: 'ShiftLeft', shiftKey: true })
    h.press({ key: ' ', code: 'Space', shiftKey: true, ctrlKey: true })
    expect(h.release({ key: ' ', code: 'Space' })).toBe(true) // it would click the row again
    expect(h.release({ key: 'Shift', code: 'ShiftLeft' })).toBe(false) // cleared when recording stopped
    expect(h.release({ key: ' ', code: 'Space' })).toBe(false) // once
    // a recording stopped by a click leaves nothing behind
    h.click('key-capture', 'shortcut')
    h.press({ key: 'Control', code: 'ControlLeft', ctrlKey: true })
    h.click('key-capture', 'shortcut')
    expect(h.release({ key: 'Control', code: 'ControlLeft' })).toBe(false)
  })

  test('the row is in the settings only, not in the setup', () => {
    const h = harness({ t })
    h.ob.start(null)
    h.click('choose', undefined, 'full')
    h.click('edit', undefined, 'assistant')
    expect(h.root.innerHTML).not.toContain('ob.shortcut.label')
  })
})
