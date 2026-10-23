// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { envName, foldName, listSecrets, maskNote, maskText, secretEnv } from './secrets.ts'

describe('names', () => {
  test('fold like the dock does', () => {
    expect(foldName('GitHub token')).toBe('github-token')
    expect(foldName('openai_key')).toBe('openai-key')
    expect(foldName('-x')).toBeNull()
    expect(foldName('a/b')).toBeNull()
    expect(envName('github-token')).toBe('ARIA_SECRET_GITHUB_TOKEN')
  })

  test('the index lists names and purposes, never values', () => {
    const dir = mkdtempSync(join(tmpdir(), 'aria-secrets-'))
    const path = join(dir, 'secrets.json')
    writeFileSync(path, JSON.stringify({ secrets: [{ name: 'db', purpose: 'staging', saved: 1 }] }))
    expect(listSecrets(path)).toEqual([{ name: 'db', purpose: 'staging' }])
    expect(listSecrets(join(dir, 'none.json'))).toEqual([])
  })
})

describe('a command gets the secrets it names, and only those', () => {
  const entries = [{ name: 'github-token', purpose: '' }, { name: 'db', purpose: '' }]
  const read = (name: string) => (name === 'github-token' ? 'ghp_value' : 'db_value')

  test('$X and ${X}', () => {
    expect(secretEnv('curl -H "Authorization: Bearer $ARIA_SECRET_GITHUB_TOKEN" api', entries, read)).toEqual({ ARIA_SECRET_GITHUB_TOKEN: 'ghp_value' })
    expect(secretEnv('psql "${ARIA_SECRET_DB}"', entries, read)).toEqual({ ARIA_SECRET_DB: 'db_value' })
  })

  test('a command naming none gets none, and a longer name is not a match', () => {
    expect(secretEnv('ls -la', entries, read)).toEqual({})
    expect(secretEnv('echo $ARIA_SECRET_DB_OTHER', entries, read)).toEqual({})
  })
})

describe('masking', () => {
  test("a stored secret is replaced by its name, wherever it shows", () => {
    const { text } = maskText('token is hunter2-long and again hunter2-long', [{ name: 'db', value: 'hunter2-long' }])
    expect(text).toBe('token is [secret db] and again [secret db]')
  })

  test('key shapes are masked, with their line', () => {
    const { text, found } = maskText('a\nOPENAI=sk-proj-0123456789abcdefXYZ\ngh ghp_0123456789abcdefghijABCD')
    expect(text).not.toContain('sk-proj')
    expect(text).not.toContain('ghp_0123')
    expect(found.map((f) => f.line)).toEqual([2, 3])
  })

  test('secret-named assignments in env, JSON and YAML are masked; others are left', () => {
    const env = 'PORT=3000\nDB_PASSWORD=hunter2\nexport API_TOKEN="abcd1234"\n'
    const { text, found } = maskText(env)
    expect(text).toContain('PORT=3000')
    expect(text).toContain('DB_PASSWORD=[masked]')
    expect(text).toContain('export API_TOKEN="[masked]"')
    expect(found).toHaveLength(2)
    expect(maskText('{\n  "client_secret": "s3cr3tvalue",\n  "name": "x"\n}').text).toContain('"client_secret": "[masked]"')
    expect(maskText('auth_token: abcdef12\n').text).toBe('auth_token: [masked]\n')
  })

  test('names that only mention a secret, and account names, are not assignments', () => {
    expect(maskText('secrets_list: TypeError: tools.secrets_list does not exist').found).toEqual([])
    expect(maskText('keyring: on\n  key:anthropic\n  key:openai\n').found).toEqual([])
    expect(maskText('SECRET_KEY: abcdef12').text).toBe('SECRET_KEY: [masked]')
  })

  test('a reference to a variable is not a value', () => {
    expect(maskText('TOKEN=$ARIA_SECRET_GITHUB_TOKEN').found).toEqual([])
  })

  test('the note says where and how to keep it, never the value', () => {
    const { found } = maskText('DB_PASSWORD=hunter2')
    const note = maskNote(found, '.env')
    expect(note).toContain('`.env`')
    expect(note).toContain('secret_import')
    expect(note).not.toContain('hunter2')
  })
})
