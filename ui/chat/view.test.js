// Run: bun test ui
// The chat's HTML is built by pure functions, so it is tested without a browser.
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { createChat } from './state.js'
import { chatTexts } from './texts.js'
import { insertFile, keepRenameField, pickerQuery, renderDialog, renderPath, renderProblem, renderUndo, preloadOrder, shortPath } from './view.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const ev = (event) => ({ type: 'chatEvent', event })
const run = (state, ...messages) => messages.reduce(reduce, state)

describe('the working folder', () => {
  test('the path is always shown, shortened under the home folder, with the full path as its title', () => {
    const html = renderPath({ ...createChat(), workdir: '/home/ana/Projects' })
    expect(html).toContain('~/Projects')
    expect(html).toContain('title="/home/ana/Projects"')
  })

  test('shortPath handles Linux and Mac homes and leaves other paths alone', () => {
    expect(shortPath('/home/ana/Projects/x')).toBe('~/Projects/x')
    expect(shortPath('/Users/ana/Code')).toBe('~/Code')
    expect(shortPath('/srv/code')).toBe('/srv/code')
    expect(shortPath('/home/ana')).toBe('~')
    expect(shortPath('')).toBe('')
  })

  test('an unknown folder still shows something', () => {
    expect(renderPath(createChat())).toContain('chat.path.unknown')
  })
})

describe('dialogs', () => {
  const dialog = (method, request) => ({ id: 'u<1', method, request })

  test('select renders one button per option plus cancel, escaped', () => {
    const html = renderDialog(dialog('select', { title: 'Allow <rm>?', options: ['Allow', 'Block & log'] }), t)
    expect(html).toContain('Allow &lt;rm&gt;?')
    expect(html.match(/data-chat="answer"/g)).toHaveLength(2)
    expect(html).toContain('data-value="Block &amp; log"')
    expect(html).toContain('data-chat="cancel"')
    expect(html).toContain('data-id="u&lt;1"')
  })

  test('confirm renders yes and no', () => {
    const html = renderDialog(dialog('confirm', { title: 'Switch model?', message: 'Free model' }), t)
    expect(html).toContain('Free model')
    expect(html).toContain('data-chat="confirm" data-id="u&lt;1" data-value="true"')
    expect(html).toContain('data-value="false"')
    expect(html).toContain('chat.yes')
    expect(html).toContain('chat.no')
  })

  test('input and editor render a field and OK, and keep the prefill', () => {
    const input = renderDialog(dialog('input', { title: 'Name?', placeholder: 'type…' }), t)
    expect(input).toContain('<input')
    expect(input).toContain('data-chat="submit"')
    const editor = renderDialog(dialog('editor', { title: 'Edit', prefill: 'line <1>' }), t)
    expect(editor).toContain('<textarea')
    expect(editor).toContain('line &lt;1&gt;')
  })
})

describe('problems', () => {
  test('a problem names what to do', () => {
    expect(renderProblem({ problem: 'pi-missing' }, t)).toContain('chat.problem.pi-missing')
    expect(renderProblem({ problem: null }, t)).toBe('')
  })

  test('every problem the chat can report has a string in both languages', () => {
    const rust = readFileSync(new URL('../../crates/aria/src/chat.rs', import.meta.url), 'utf8').split('#[cfg(test)]')[0]
    const core = readFileSync(new URL('../../crates/aria-core/src/pi.rs', import.meta.url), 'utf8').split('#[cfg(test)]')[0]
    const codes = new Set([
      ...[...rust.matchAll(/(?:Err\(|map_err\(\|_\| |=> )"([a-z]+-[a-z-]+)"/g)].map((m) => m[1]),
      ...[...core.matchAll(/=> "([a-z]+-[a-z]+)"/g)].map((m) => m[1]),
    ])
    expect(codes.has('pi-missing')).toBe(true)
    expect(codes.has('node-missing')).toBe(true)
    for (const lang of ['en', 'es']) {
      const catalog = JSON.parse(readFileSync(new URL(`../i18n/${lang}.json`, import.meta.url), 'utf8'))
      for (const code of codes) expect(catalog[`chat.problem.${code}`], `${lang}: chat.problem.${code}`).toBeDefined()
    }
  })
})

describe('pickers', () => {
  test('pickerQuery finds the model picker after "/model ", a command at the start and an @ file at the caret', () => {
    expect(pickerQuery('/model ', 7)).toEqual({ kind: 'model', query: '', start: 0 })
    expect(pickerQuery('/model gl', 9)).toEqual({ kind: 'model', query: 'gl', start: 0 })
    expect(pickerQuery('/model', 6)).toEqual({ kind: 'command', query: 'model', start: 0 })
    expect(pickerQuery('/ef', 3)).toEqual({ kind: 'command', query: 'ef', start: 0 })
    expect(pickerQuery('/effort high', 12)).toBeNull() // after a space it is an argument
    expect(pickerQuery('look at @src/ma', 15)).toEqual({ kind: 'file', query: 'src/ma', start: 8 })
    expect(pickerQuery('mail me@x.com', 13)).toBeNull() // an @ inside a word is not a file
    expect(pickerQuery('hello', 5)).toBeNull()
  })

  test('insertFile replaces the @ token with the path and one space', () => {
    expect(insertFile('look at @src/ma now', 15, 8, 'src/main.rs')).toEqual({ text: 'look at @src/main.rs now', caret: 21 })
    expect(insertFile('see @sr', 7, 4, 'src/a.rs')).toEqual({ text: 'see @src/a.rs ', caret: 14 })
  })
})

describe('the tab bar helpers', () => {
  const tab = (id, extra = {}) => ({ id, title: id, status: 'idle', unread: false, ...extra })
  const bar = (sessions, active = sessions[0].id, closed = null) => ({ sessions, active, closed })

  test('the tab bar is left alone while a rename field is open', () => {
    expect(keepRenameField({ renaming: 'a' }, true)).toBe(true)
    expect(keepRenameField({ renaming: 'a' }, false)).toBe(false) // the field has to be drawn first
    expect(keepRenameField({ renaming: null }, true)).toBe(false)
  })

  test('the undo bar names the closed session', () => {
    expect(renderUndo(bar([tab('a')]), t)).toBe('')
    const html = renderUndo(bar([tab('a')], 'a', { entry: tab('x', { title: 'Old <one>' }), index: 1 }), t)
    expect(html).toContain('Old &lt;one&gt;')
    expect(html).toContain('data-chat="undo"')
  })
})

describe('texts', () => {
  test('every text key the chat view, thread and state use exists in both languages (parts included)', () => {
    const read = (file) => readFileSync(new URL(file, import.meta.url), 'utf8')
    const source = read('./view.js') + read('./state.js') + read('./thread.js')
    const keys = new Set([...source.matchAll(/\bt\('(chat\.[a-zA-Z.-]+)'/g)].map((m) => m[1]))
    for (const key of [...source.matchAll(/key: '(chat\.[a-zA-Z.-]+)'/g)].map((m) => m[1])) keys.add(key)
    for (const code of ['allow', 'always', 'deny', 'free', 'keep']) keys.add(`chat.result.${code}`)
    expect(keys.size).toBeGreaterThan(15)
    const texts = chatTexts()
    for (const lang of ['en', 'es']) {
      const catalog = { ...JSON.parse(read(`../i18n/${lang}.json`)), ...texts[lang] }
      for (const key of keys) expect(catalog[key], `${lang}: ${key}`).toBeDefined()
    }
  })

  test('the parts bring the same keys in both languages', () => {
    const texts = chatTexts()
    expect(Object.keys(texts.es).sort()).toEqual(Object.keys(texts.en).sort())
  })
})

describe('preloading the saved tabs', () => {
  test('the active tab first, then the others in their order', () => {
    const tabs = { sessions: [{ id: 'a' }, { id: 'b' }, { id: 'c' }], active: 'b' }
    expect(preloadOrder(tabs)).toEqual(['b', 'a', 'c'])
  })
  test('no tabs, nothing to load', () => expect(preloadOrder({ sessions: [], active: null })).toEqual([]))
})
