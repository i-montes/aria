// Run: bun test ui/chat/parts/connect.test.js
import { describe, expect, test } from 'bun:test'
import { PROVIDERS, TEXTS, errorKey, renderConnect } from './connect.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const hostile = '<script>alert("x")</script> & \'q\''

describe('the provider list', () => {
  test('Anthropic and OpenCode Go can be connected', () => {
    expect(PROVIDERS.map((p) => p.id)).toEqual(['anthropic', 'opencode-go'])
  })

  test('the first step offers each provider', () => {
    const html = renderConnect({ step: 'pick' }, {}, t)
    expect(html).toContain('data-chat="connect" data-op="provider" data-value="anthropic"')
    expect(html).toContain('data-chat="connect" data-op="provider" data-value="opencode-go"')
    expect(html).toContain('chat.connect.title')
  })
})

describe('Anthropic', () => {
  test('it offers signing in with Claude or an API key, and a way back', () => {
    const html = renderConnect({ provider: 'anthropic', step: 'anthropic' }, {}, t)
    expect(html).toContain('data-op="method" data-value="oauth"')
    expect(html).toContain('data-op="method" data-value="key"')
    expect(html).toContain('chat.connect.method.oauth')
    expect(html).toContain('chat.connect.method.key')
    expect(html).toContain('data-op="back"')
  })

  test('the key step has a password field, a save button and show/hide', () => {
    const hidden = renderConnect({ provider: 'anthropic', step: 'key' }, {}, t)
    expect(hidden).toContain('type="password"')
    expect(hidden).toContain('data-chat-field="connect-key"')
    expect(hidden).toContain('data-op="save"')
    expect(hidden).toContain('data-op="reveal"')
    expect(renderConnect({ provider: 'anthropic', step: 'key', revealed: true }, {}, t)).toContain('type="text"')
  })

  test('a typed key is kept in the field, escaped', () => {
    const html = renderConnect({ provider: 'anthropic', step: 'key', keyDraft: hostile }, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).toContain('&lt;script&gt;')
  })

  test('waiting for the browser can reopen the link and be cancelled', () => {
    const html = renderConnect({ provider: 'anthropic', step: 'waiting', url: 'https://claude.ai/oauth?a=1&b="2"' }, {}, t)
    expect(html).toContain('chat.connect.waiting')
    expect(html).toContain('data-op="open-url"')
    expect(html).toContain('data-op="close"')
    expect(html).not.toContain('b="2"')
  })

  test('installing the Claude provider says so and offers nothing to click but closing', () => {
    const html = renderConnect({ provider: 'anthropic', step: 'installing' }, {}, t)
    expect(html).toContain('chat.connect.installing')
    expect(html).not.toContain('data-op="save"')
  })

  test('done offers to choose a model', () => {
    const html = renderConnect({ provider: 'anthropic', step: 'done' }, {}, t)
    expect(html).toContain('chat.connect.done')
    expect(html).toContain('data-op="models"')
  })
})

describe('OpenCode Go and errors', () => {
  test('OpenCode Go goes straight to the key', () => {
    const html = renderConnect({ provider: 'opencode-go', step: 'key' }, {}, t)
    expect(html).toContain('OpenCode Go')
    expect(html).toContain('data-chat-field="connect-key"')
  })

  test('an error shows its text, and an unknown code a generic one', () => {
    expect(renderConnect({ provider: 'anthropic', step: 'key', error: 'key-rejected' }, {}, t)).toContain('chat.connect.error.key-rejected')
    expect(renderConnect({ provider: 'anthropic', step: 'key', error: 'something-new' }, {}, t)).toContain('chat.connect.error.generic')
    expect(errorKey('claude-code-missing')).toBe('chat.connect.error.claude-code-missing')
    expect(errorKey(undefined)).toBeNull()
  })

  test('every error code the Rust side can answer has a text', () => {
    for (const code of ['key-rejected', 'network-unavailable', 'auth-file-broken', 'oauth-denied', 'oauth-timeout', 'oauth-state', 'oauth-failed', 'pi-missing', 'claude-code-missing', 'claude-bridge-install-failed', 'unknown-provider', 'generic']) {
      expect(TEXTS.en[`chat.connect.error.${code}`], code).toBeDefined()
      expect(TEXTS.es[`chat.connect.error.${code}`], code).toBeDefined()
    }
  })
})

describe('escaping and texts', () => {
  test('an unknown provider id is shown escaped, never as markup', () => {
    const html = renderConnect({ provider: hostile, step: 'key' }, {}, t)
    expect(html).not.toContain('<script>')
  })

  test('both languages have the same keys, and every key the renderer uses exists', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    const html = ['pick', 'anthropic', 'key', 'waiting', 'installing', 'done'].map((step) => renderConnect({ provider: 'anthropic', step, error: 'generic' }, {}, (k) => `«${k}»`)).join('')
    for (const key of new Set([...html.matchAll(/«(chat\.connect\.[a-zA-Z.-]+)»/g)].map((m) => m[1]))) expect(TEXTS.en[key], key).toBeDefined()
  })
})
