// Run: bun test ui/chat/secrets.test.js
import { describe, expect, test } from 'bun:test'
import { envName, findSecrets, suggestName, validName, withSecretsSaved } from './secrets.js'

describe('the composer catches a secret before it goes out', () => {
  test('key shapes, with a name to offer', () => {
    const found = findSecrets('use this: sk-proj-0123456789abcdefXYZ and ghp_0123456789abcdefghijABCD')
    expect(found.map((f) => [f.kind, f.name])).toEqual([
      ['OpenAI key', 'openai-key'],
      ['GitHub token', 'github-token'],
    ])
    expect(found[0].value).toBe('sk-proj-0123456789abcdefXYZ')
  })

  test('a secret-named assignment, not an ordinary one', () => {
    const found = findSecrets('PORT=3000\nDB_PASSWORD=hunter2long')
    expect(found).toHaveLength(1)
    expect(found[0]).toMatchObject({ value: 'hunter2long', name: 'db-password' })
  })

  test('a name that only mentions a secret, or a colon with no space, is not an assignment', () => {
    expect(findSecrets('secrets_list: something-long')).toEqual([])
    expect(findSecrets('key:anthropic-account')).toEqual([])
    expect(findSecrets('"api_key": "abcdef123"')[0]).toMatchObject({ value: 'abcdef123', name: 'api-key' })
  })

  test('ordinary text, variables and short values are left alone', () => {
    expect(findSecrets('the skeleton key is in the drawer')).toEqual([])
    expect(findSecrets('TOKEN=$ARIA_SECRET_GITHUB_TOKEN')).toEqual([])
    expect(findSecrets('sk-short')).toEqual([])
    expect(findSecrets(undefined)).toEqual([])
  })

  test('the same value twice is one secret; two of a kind get a number', () => {
    expect(findSecrets('sk-proj-0123456789abcdefXYZ sk-proj-0123456789abcdefXYZ')).toHaveLength(1)
    expect(findSecrets('sk-proj-0123456789abcdefXYZ sk-proj-9999999999abcdefXYZ').map((f) => f.name)).toEqual(['openai-key', 'openai-key-2'])
  })

  test('the message that goes out names the secret, never its value', () => {
    const text = 'deploy with sk-proj-0123456789abcdefXYZ please'
    const found = findSecrets(text)
    const out = withSecretsSaved(text, found, ['openai-key'])
    expect(out).toBe('deploy with [secret saved as openai-key: use $ARIA_SECRET_OPENAI_KEY] please')
  })

  test('names', () => {
    expect(suggestName('DB_PASSWORD')).toBe('db-password')
    expect(validName('github-token')).toBe(true)
    expect(validName('-x')).toBe(false)
    expect(envName('db-password')).toBe('ARIA_SECRET_DB_PASSWORD')
  })
})
