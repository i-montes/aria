// Run: bun test ui/chat/parts/secret.test.js
import { describe, expect, test } from 'bun:test'
import { TEXTS, renderSecret } from './secret.js'
import { parseCardTitle } from './cards.js'
import { answerDialog, createChat } from '../state.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const evil = '<script>alert("x")</script>'

describe('the secret card the model asks for', () => {
  const props = { mode: 'asked', id: 'd1', name: 'github-token', purpose: 'Push to the CI', env: 'ARIA_SECRET_GITHUB_TOKEN' }

  test('a password field, the purpose, the variable, Save and Cancel', () => {
    const html = renderSecret(props, {}, t)
    expect(html).toContain('type="password"')
    expect(html).toContain('data-secret-value data-id="d1"')
    expect(html).toContain('Push to the CI')
    expect(html).toContain('$ARIA_SECRET_GITHUB_TOKEN')
    expect(html).toContain('chat.secret.asked name=github-token')
    expect(html).toContain('data-chat="secret" data-op="save" data-id="d1"')
    expect(html).toContain('data-chat="secret" data-op="cancel" data-id="d1"')
  })

  test('busy disables it; an error is said', () => {
    expect(renderSecret({ ...props, busy: true }, {}, t)).toContain(' disabled')
    expect(renderSecret({ ...props, error: 'keyring-off' }, {}, t)).toContain('chat.secret.error.keyring-off')
  })

  test('every dynamic value is escaped', () => {
    const html = renderSecret({ mode: 'asked', id: evil, name: evil, purpose: evil, env: evil }, {}, t)
    expect(html).not.toContain('<script>')
  })

  test('its dialog title parses, and answering it leaves the name in the thread', () => {
    const card = parseCardTitle(JSON.stringify({ aria: 1, kind: 'secret', name: 'db', purpose: 'staging', env: 'ARIA_SECRET_DB' }))
    expect(card).toEqual({ kind: 'secret', name: 'db', purpose: 'staging', env: 'ARIA_SECRET_DB' })
    const state = { ...createChat(), dialogs: [{ id: 'd1', card }] }
    const after = answerDialog(state, 'd1', 'db')
    expect(after.dialogs).toEqual([])
    expect(after.messages.at(-1)).toMatchObject({ kind: 'note', label: 'secret', key: 'chat.secret.saved', vars: { name: 'db' } })
    expect(answerDialog(state, "d1", undefined).messages.length).toBe(state.messages.length)
  })
})

describe('the secrets the composer held back', () => {
  test('one name field per secret, with Keep, Send as is and Edit; never the value', () => {
    const html = renderSecret({ mode: 'pasted', found: [{ kind: 'OpenAI key', name: 'openai-key', value: 'sk-proj-0123456789abcdef' }] }, {}, t)
    expect(html).toContain('chat.secret.pasted.one')
    expect(html).toContain('data-secret-name data-index="0" value="openai-key"')
    for (const op of ['keep', 'send', 'edit']) expect(html).toContain(`data-chat="secret-paste" data-op="${op}"`)
    expect(html).not.toContain('sk-proj')
  })
})

test('both languages say the same things', () => {
  expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
})
