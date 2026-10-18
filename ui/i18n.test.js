// Run: bun test ui
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { createI18n } from './i18n.js'

const load = (code) => JSON.parse(readFileSync(new URL(`./i18n/${code}.json`, import.meta.url), 'utf8'))
const en = load('en')
const es = load('es')
const i18n = () => createI18n({ en, es })

test('English strings after setLanguage("en")', () => {
  const { t, setLanguage } = i18n()
  setLanguage('en')
  expect(t('welcome.title')).toBe(en['welcome.title'])
})

test('Spanish strings after setLanguage("es")', () => {
  const { t, setLanguage } = i18n()
  setLanguage('es')
  expect(t('welcome.title')).toBe(es['welcome.title'])
  expect(es['welcome.title']).not.toBe(en['welcome.title'])
})

test('locale-style and free-text values are understood', () => {
  const { t, setLanguage } = i18n()
  setLanguage('es-CO')
  expect(t('welcome.title')).toBe(es['welcome.title'])
  setLanguage('es_CO.UTF-8')
  expect(t('welcome.title')).toBe(es['welcome.title'])
})

test('unsupported languages fall back to English', () => {
  const { t, setLanguage } = i18n()
  setLanguage('pt')
  expect(t('welcome.title')).toBe(en['welcome.title'])
  setLanguage('de')
  expect(t('welcome.title')).toBe(en['welcome.title'])
  setLanguage('Klingon')
  expect(t('welcome.title')).toBe(en['welcome.title'])
})

test('a missing key returns the key itself', () => {
  const { t } = i18n()
  expect(t('missing.key')).toBe('missing.key')
})

test('variables are filled in', () => {
  const { t } = createI18n({ en: { greet: 'Hello {name}, {name}!' } })
  expect(t('greet', { name: 'Ivan' })).toBe('Hello Ivan, Ivan!')
})

test('English and Spanish have exactly the same keys', () => {
  expect(Object.keys(es).sort()).toEqual(Object.keys(en).sort())
})

// Every code the Rust side can send to the page must have a text, or the page shows the raw key.
describe('every code the dock can emit has a string in both languages', () => {
  const rust = (file) => readFileSync(new URL(`../crates/${file}`, import.meta.url), 'utf8').split('#[cfg(test)]')[0]
  const handlers = rust('aria/src/handlers.rs')
  const main = rust('aria/src/main.rs')
  const problemCodes = [...main.matchAll(/set_problem\("([a-z-]+)"\)/g)].map((m) => m[1])
  // load_settings reports its problem as `problem: Some("...")` / a match arm that yields the code
  problemCodes.push(...[...handlers.slice(handlers.indexOf('fn load_settings'), handlers.indexOf('fn save_settings')).matchAll(/=> "([a-z-]+)"/g)].map((m) => m[1]))
  const errorCodes = [...handlers.matchAll(/(?:Some\(|=> |code: )"([a-z]+-[a-z-]+)"/g)].map((m) => m[1]).filter((code) => !problemCodes.includes(code) || code.startsWith('settings-'))

  test('the scan finds the codes it should', () => {
    expect(problemCodes).toContain('hyprland-classic-config')
    expect(problemCodes).toContain('settings-invalid')
    expect(errorCodes).toContain('key-rejected')
    expect(errorCodes).toContain('oauth-timeout')
  })
  for (const [name, catalog] of [['en', en], ['es', es]]) {
    test(`problems in ${name}`, () => {
      for (const code of new Set(problemCodes)) expect(catalog[`problem.${code}`], `problem.${code}`).toBeDefined()
    })
    test(`errors in ${name}`, () => {
      for (const code of new Set(errorCodes)) expect(catalog[`ob.error.${code}`], `ob.error.${code}`).toBeDefined()
    })
  }
})
