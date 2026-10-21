// The dock's shortcut (the `shortcut` setting) as the page reads and records it. The same rules as
// crates/aria-core/src/shortcut.rs (Rust the page cannot import): change both together. A shortcut is
// { mods, key }: canonical modifiers in canonical order (at least one), and a letter, a digit, `Space` or `F1`–`F12`.

/** The modifiers in their canonical spelling and order, and the names each one is also typed as. */
const MODIFIERS = {
  linux: [['SUPER', 'super', 'win', 'meta', 'logo'], ['CTRL', 'ctrl', 'control'], ['ALT', 'alt'], ['SHIFT', 'shift']],
  mac: [['Command', 'command', 'cmd'], ['Control', 'control', 'ctrl'], ['Option', 'option', 'opt', 'alt'], ['Shift', 'shift']],
}
const DEFAULTS = { linux: 'SUPER + A', mac: 'Option + Space' }
/** A KeyboardEvent's flag for each modifier, in the canonical order of both platforms (Super is Meta on Linux). */
const FLAGS = ['metaKey', 'ctrlKey', 'altKey', 'shiftKey']
const MODIFIER_KEYS = new Set(['Shift', 'Control', 'Alt', 'AltGraph', 'Meta', 'OS', 'Super', 'Hyper'])

const platform = (os) => (os === 'mac' ? 'mac' : 'linux')

/** The canonical key: `A`–`Z`, `0`–`9`, `Space`, `F1`–`F12`; null for anything else. */
function canonicalKey(name) {
  const upper = name.toUpperCase()
  if (/^[A-Z0-9]$/.test(upper)) return upper
  if (upper === 'SPACE') return 'Space'
  const f = /^F([1-9]|1[0-2])$/.exec(upper)
  return f ? `F${f[1]}` : null
}

/** The shortcut `text` names on `os` (`linux`, `mac`), or null when it is not one. */
export function parseShortcut(text, os) {
  const parts = String(text ?? '').split('+').map((part) => part.trim())
  const key = canonicalKey(parts.pop() ?? '')
  if (!key) return null
  const named = []
  for (const part of parts) {
    const found = MODIFIERS[platform(os)].find((names) => names.slice(1).includes(part.toLowerCase()))
    if (!found) return null
    named.push(found[0])
  }
  const mods = MODIFIERS[platform(os)].map((names) => names[0]).filter((name) => named.includes(name))
  return mods.length ? { mods, key } : null
}

export const defaultShortcut = (os) => parseShortcut(DEFAULTS[platform(os)], os)

/** The setting's form: `SUPER + SHIFT + A`, `Option + Space`. */
export const label = (shortcut) => [...shortcut.mods, shortcut.key].join(' + ')

/** The page's form (one `<kbd>` per part): `Super + Shift + A`, `Option + Space`. */
export const display = (shortcut) =>
  [...shortcut.mods.map((m) => m.charAt(0) + m.slice(1).toLowerCase()), shortcut.key].join(' + ')

/** A key press while the row records: `{ label }` for a shortcut, `{ pending: true }` for a modifier alone (the key
 * is still to come), `{ error: 'needsModifier' | 'unsupported' }` otherwise. The key is read by its position
 * (`event.code`), so Option or Shift changing the character (`å`, `!`) does not change the shortcut. */
export function fromKeyEvent(event, os) {
  if (MODIFIER_KEYS.has(event.key)) return { pending: true }
  const fromCode = /^Key([A-Z])$|^Digit([0-9])$|^(Space)$|^(F[0-9]{1,2})$/.exec(event.code ?? '')
  const name = fromCode ? fromCode.slice(1).find(Boolean) : event.code ? '' : event.key === ' ' ? 'Space' : (event.key ?? '')
  const key = name ? canonicalKey(name) : null
  const names = MODIFIERS[platform(os)].map((m) => m[0])
  const mods = FLAGS.map((flag, i) => (event[flag] ? names[i] : null)).filter(Boolean)
  if (!key) return { error: mods.length ? 'unsupported' : 'needsModifier' }
  if (!mods.length) return { error: 'needsModifier' }
  return { label: label({ mods, key }) }
}
