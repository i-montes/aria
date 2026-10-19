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
  const click = (action, field, value) => {
    const el = { dataset: { action, field, value }, disabled: false }
    el.closest = () => null
    listeners.click({ target: { closest: () => el } })
  }
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
    h.click('edit', undefined, 'integrations')
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
    h.click('edit', undefined, 'agents')
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
