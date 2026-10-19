// Run: bun test ui/chat/parts/models.test.js
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { TEXTS, filterModels, orderedModels, providerLabel, renderModelPicker } from './models.js'

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

describe('grouping by provider', () => {
  const mixed = [
    { provider: 'opencode-go', id: 'glm-5', name: 'GLM 5' },
    { provider: 'anthropic', id: 'claude-sonnet-5-5', name: 'Claude Sonnet 5.5' },
    { provider: 'opencode-go', id: 'kimi-k2', name: 'Kimi K2' },
    { provider: 'anthropic', id: 'claude-opus-5-5', name: 'Claude Opus 5.5' },
    { provider: 'weird<provider>', id: 'x', name: 'X' },
  ]

  test('models of one provider sit together, providers in order of first appearance, models in their own order', () => {
    expect(orderedModels(mixed, '').map((x) => x.model.id)).toEqual(['glm-5', 'kimi-k2', 'claude-sonnet-5-5', 'claude-opus-5-5', 'x'])
  })

  test('with a query the provider whose best match ranks first comes first', () => {
    const found = orderedModels(mixed, 'claude').map((x) => x.model.id)
    expect(found).toEqual(['claude-sonnet-5-5', 'claude-opus-5-5'])
    const byName = orderedModels(mixed, 'kimi').map((x) => x.model.id)
    expect(byName).toEqual(['kimi-k2'])
  })

  test('each entry knows its flat index and where its provider group starts', () => {
    const list = orderedModels(mixed, '')
    expect(list.map((x) => x.index)).toEqual([0, 1, 2, 3, 4])
    expect(list.filter((x) => x.firstOfProvider).map((x) => x.model.provider)).toEqual(['opencode-go', 'anthropic', 'weird<provider>'])
  })

  test('provider names are readable', () => {
    expect(providerLabel('opencode-go')).toBe('OpenCode Go')
    expect(providerLabel('opencode')).toBe('OpenCode Zen')
    expect(providerLabel('anthropic')).toBe('Anthropic')
    expect(providerLabel('claude-bridge')).toBe('Claude (sign-in)')
    expect(providerLabel('some-new-one')).toBe('some-new-one')
    expect(providerLabel('')).toBe('')
  })

  test('the picker shows one header per provider with its models counted, and row indexes run across groups', () => {
    const html = renderModelPicker({ models: mixed, current: '', query: '', active: 0, open: true }, {}, t)
    expect(html.match(/class="p-models-group"/g)).toHaveLength(3)
    expect(html).toContain('OpenCode Go')
    expect(html).toContain('Anthropic')
    expect(html).toContain('weird&lt;provider&gt;')
    expect(html).not.toContain('weird<provider>')
    expect(html.match(/data-chat="pick" data-index="4"/g)).toHaveLength(1)
    expect(html.indexOf('OpenCode Go')).toBeLessThan(html.indexOf('GLM 5'))
    expect(html.indexOf('GLM 5')).toBeLessThan(html.indexOf('Kimi K2'))
    expect(html.indexOf('Kimi K2')).toBeLessThan(html.indexOf('Anthropic'))
  })

  test('the picker has its own search field, filled with the query and escaped', () => {
    const html = renderModelPicker({ models: mixed, current: '', query: 'cla"<b>', active: 0, open: true }, {}, t)
    expect(html).toContain('data-chat-field="model-search"')
    expect(html).toContain('value="cla&quot;&lt;b&gt;"')
    expect(html).toContain('placeholder="chat.models.search"')
    expect(html).not.toContain('<b>')
  })

  test('the search field shows even when nothing matches, so the query can be fixed', () => {
    const html = renderModelPicker({ models: mixed, current: '', query: 'zzz', active: 0, open: true }, {}, t)
    expect(html).toContain('data-chat-field="model-search"')
    expect(html).toContain('chat.models.empty')
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
