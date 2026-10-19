// Run: bun test ui/chat/parts/models.test.js
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { TEXTS, filterModels, renderModelPicker } from './models.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const hostile = '<script>alert("x")</script> & \'q\''
const models = [
  { provider: 'opencode-go', id: 'glm-5', name: 'GLM 5' },
  { provider: 'opencode-go', id: 'kimi-k2', name: 'Kimi K2' },
  { provider: 'opencode', id: 'big-pickle', name: 'Big Pickle', tag: 'no quota', tier: 'free' },
]
const ids = (query, list = models) => filterModels(list, query).map((x) => x.model.id)

describe('filterModels', () => {
  test('no query keeps the order', () => {
    expect(ids('')).toEqual(['glm-5', 'kimi-k2', 'big-pickle'])
  })

  test('the name first (prefix, then inside), then the id, provider, description, tag and tier', () => {
    const r = filterModels(models, 'k')
    expect(r.map((x) => x.model.id)).toEqual(['kimi-k2', 'big-pickle']) // "Kimi" prefix, then "pic-k-le"
    expect(r[0].score).toBe(0)
    expect(r[0].ranges).toEqual([[0, 1]])
    expect(r[1].score).toBe(1 + 'big pickle'.indexOf('k'))
    expect(ids('opencode-go')).toEqual(['glm-5', 'kimi-k2']) // provider only
    expect(filterModels(models, 'opencode-go')[0].score).toBe(50)
    expect(ids('free')).toEqual(['big-pickle']) // tier
    expect(ids('no quota')).toEqual(['big-pickle']) // tag
  })

  test('matching ignores case; nothing matches gives an empty list', () => {
    expect(ids('GLM')).toEqual(['glm-5'])
    expect(ids('zzz')).toEqual([])
  })
})

describe('renderModelPicker', () => {
  const picker = (extra = {}) => ({ models, current: 'opencode-go/glm-5', query: '', active: 0, open: true, ...extra })

  test('one option per match with pick and hover emits, a count and the title', () => {
    const html = renderModelPicker(picker(), {}, t)
    expect(html.match(/data-chat="pick"/g)).toHaveLength(3)
    expect(html.match(/data-chat="picker-hover"/g)).toHaveLength(3)
    expect(html).toContain('chat.models.count.other n=3')
    expect(html).toContain('chat.models.title')
    expect(html).toContain('role="listbox"')
  })

  test('the current model is selected (by provider/id or by id) and the active row is marked apart', () => {
    const html = renderModelPicker(picker({ active: 2 }), {}, t)
    expect(html.match(/aria-selected="true"/g)).toHaveLength(1)
    expect(html.match(/p-models-ring is-current/g)).toHaveLength(1)
    expect(html).toMatch(/class="p-models-row on" data-chat="pick" data-index="2"/)
    const byId = renderModelPicker(picker({ current: 'kimi-k2' }), {}, t)
    expect(byId.match(/p-models-ring is-current/g)).toHaveLength(1)
  })

  test('a free model has its green tag and a tag only shows when there is one', () => {
    const html = renderModelPicker(picker(), {}, t)
    expect(html.match(/class="p-models-tag is-free"/g)).toHaveLength(1)
    expect(html.match(/class="p-models-tag"/g) ?? []).toHaveLength(0)
    expect(html).toContain('no quota')
  })

  test('the second line is the description, or the provider and id', () => {
    const html = renderModelPicker(picker({ models: [{ provider: 'p', id: 'i', name: 'N' }, { provider: 'p', id: 'j', name: 'M', description: 'Fast one' }] }), {}, t)
    expect(html).toContain('p/i')
    expect(html).toContain('Fast one')
  })

  test('a model without a name shows its id', () => {
    expect(renderModelPicker(picker({ models: [{ provider: 'p', id: 'only-id' }] }), {}, t)).toContain('only-id')
  })

  test('the matched letters are highlighted', () => {
    expect(renderModelPicker(picker({ query: 'ki' }), {}, t)).toContain('<span class="p-models-hl">Ki</span>')
  })

  test('open and closed share one element', () => {
    expect(renderModelPicker(picker(), {}, t)).toContain('class="p-models is-open"')
    expect(renderModelPicker(picker({ open: false }), {}, t)).toContain('class="p-models"')
  })

  test('an empty result names the query, escaped', () => {
    const html = renderModelPicker(picker({ query: hostile }), {}, t)
    expect(html).not.toContain('<ul')
    expect(html).not.toContain('<script>')
    expect(html).toContain('chat.models.empty')
    expect(html).toContain('&lt;script&gt;')
  })

  test('names, ids, providers and tags are escaped', () => {
    const bad = [{ provider: hostile, id: hostile, name: hostile, description: hostile, tag: hostile }]
    for (const query of ['', 'script']) {
      const html = renderModelPicker(picker({ models: bad, current: '', query }), {}, t)
      expect(html).not.toContain('<script>')
      expect(html).toContain('&lt;')
      expect(html).toContain('&amp; &#39;q&#39;')
    }
    expect(renderModelPicker(picker({ models: bad, current: '' }), {}, t)).toContain('&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &#39;q&#39;')
  })
})

describe('texts', () => {
  test('every key used exists in both languages with the same keys', () => {
    const source = readFileSync(new URL('./models.js', import.meta.url), 'utf8')
    const used = [...source.matchAll(/\bt\('(chat\.[a-zA-Z.]+)'/g)].map((m) => m[1])
    expect(used.length).toBeGreaterThan(5)
    for (const lang of ['en', 'es']) for (const key of used) expect(TEXTS[lang][key], `${lang}: ${key}`).toBeTruthy()
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.models.')).toBe(true)
  })
})
