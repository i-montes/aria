// Every visible string of the dock comes from a catalog (`i18n/<code>.json`). English and Spanish
// are complete; any other language falls back to English until it is translated.

export const SUPPORTED = ['en', 'es']

/** `es`, `es-CO`, `es_CO.UTF-8` -> `es`; anything unsupported -> `en`. */
export function normalizeLanguage(value) {
  const code = String(value ?? '').slice(0, 2).toLowerCase()
  return SUPPORTED.includes(code) ? code : 'en'
}

export function createI18n(catalogs) {
  let language = 'en'
  return {
    setLanguage(value) {
      language = normalizeLanguage(value)
    },
    language: () => language,
    /** The string for `key` in the current language, `{name}` placeholders filled from `vars`. */
    t(key, vars = {}) {
      const text = catalogs[language]?.[key] ?? catalogs.en?.[key] ?? key
      return text.replace(/\{(\w+)\}/g, (match, name) => (name in vars ? String(vars[name]) : match))
    },
  }
}
