// Run: bun test ui
import { expect, test } from 'bun:test'
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
