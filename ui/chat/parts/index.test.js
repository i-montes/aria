// Run: bun test ui/chat/parts
import { describe, expect, test } from 'bun:test'
import { TEXTS, activeIndexFor, indexItems, indexLayout, indexTargets, messageDomId, renderIndex } from './index.js'

const t = (key, vars = {}) => (TEXTS.en[key] ?? key).replace(/\{(\w+)\}/g, (m, name) => (name in vars ? String(vars[name]) : m))
const items = (n) => Array.from({ length: n }, (_, i) => ({ id: `m${i}`, label: `Message ${i}` }))
const NASTY = '<script>alert(1)</script> "q" \'s\' & co'
const ESCAPED = '&lt;script&gt;alert(1)&lt;/script&gt; &quot;q&quot; &#39;s&#39; &amp; co'

describe('indexItems', () => {
  test('only user messages, keyed by their position in the thread', () => {
    const messages = [
      { kind: 'user', text: 'first question' },
      { kind: 'assistant', text: 'answer' },
      { kind: 'tools', calls: [] },
      { kind: 'user', text: '  second\n  question   here ' },
      { kind: 'notice', tone: 'info', text: 'x' },
    ]
    expect(indexItems(messages)).toEqual([
      { id: messageDomId(0), label: 'first question' },
      { id: messageDomId(3), label: 'second question here' },
    ])
  })

  test('skips empty messages and tolerates a missing list', () => {
    expect(indexItems([{ kind: 'user', text: '   ' }, { kind: 'user' }])).toEqual([])
    expect(indexItems(undefined)).toEqual([])
  })

  test('long messages are cut', () => {
    const [item] = indexItems([{ kind: 'user', text: 'a'.repeat(500) }])
    expect(item.label.length).toBeLessThanOrEqual(121)
    expect(item.label.endsWith('…')).toBe(true)
  })

  test('indexTargets lists the element ids to jump to', () => {
    expect(indexTargets(items(3))).toEqual(['m0', 'm1', 'm2'])
  })
})

describe('activeIndexFor', () => {
  test('tops only: the last message that has reached the top edge', () => {
    const tops = [0, 300, 700, 1200]
    expect(activeIndexFor(0, tops)).toBe(0)
    expect(activeIndexFor(250, tops)).toBe(0)
    expect(activeIndexFor(290, tops)).toBe(1) // jump lands 12px below the edge
    expect(activeIndexFor(700, tops)).toBe(2)
    expect(activeIndexFor(5000, tops)).toBe(3)
  })

  test('with bottoms: the first message still visible under the edge, like the design', () => {
    const spans = [{ top: 0, bottom: 100 }, { top: 120, bottom: 400 }, { top: 500, bottom: 560 }]
    expect(activeIndexFor(0, spans)).toBe(0)
    expect(activeIndexFor(95, spans)).toBe(1)
    expect(activeIndexFor(300, spans)).toBe(1)
    expect(activeIndexFor(9999, spans)).toBe(2)
  })

  test('nothing to pick', () => {
    expect(activeIndexFor(0, [])).toBe(-1)
    expect(activeIndexFor(0, undefined)).toBe(-1)
  })
})

describe('indexLayout', () => {
  test('a short list is centred and nothing scrolls', () => {
    const l = indexLayout(3, { height: 360 })
    expect(l.overflow).toBe(false)
    expect(l.rows).toHaveLength(3)
    const total = 3 * 30 + 2 * 4
    expect(l.rows[0].top).toBeCloseTo((360 - total) / 2)
    expect(l.rows[1].top - l.rows[0].top).toBeCloseTo(34)
    expect(l.rows.every((r) => r.scale === 1)).toBe(true)
  })

  test('a long list overflows and offsets by the scroll', () => {
    const l = indexLayout(30, { height: 360, off: 100 })
    expect(l.overflow).toBe(true)
    expect(l.contentH).toBe(30 * 34 - 4 + 16)
    expect(l.maxOff).toBe(l.contentH - 360)
    expect(l.rows[0].top).toBeCloseTo(8 - 100)
    expect(l.thumbH).toBeGreaterThanOrEqual(24)
    const past = indexLayout(30, { height: 360, off: 99999 })
    expect(past.rows[0].top).toBeCloseTo(8 - past.maxOff)
  })

  test('the pointer magnifies the rows near it, up to 1.4', () => {
    const l = indexLayout(7, { height: 360, my: 180, expanded: true })
    const peak = Math.max(...l.rows.map((r) => r.scale))
    expect(peak).toBeGreaterThan(1.3)
    expect(peak).toBeLessThanOrEqual(1.4)
    expect(l.rows[3].scale).toBeGreaterThan(l.rows[0].scale)
    expect(l.rows[3].near).toBe(true)
    expect(l.rows[0].near).toBe(false)
    for (let i = 1; i < 7; i++) expect(l.rows[i].top).toBeGreaterThan(l.rows[i - 1].top)
  })

  test('entry delay fans out from the middle until settled', () => {
    const fresh = indexLayout(5, { height: 360, expanded: true, settled: false })
    expect(fresh.rows[2].delay).toBe(0)
    expect(fresh.rows[0].delay).toBe(60)
    const settled = indexLayout(5, { height: 360, expanded: true, settled: true })
    expect(settled.rows.every((r) => r.delay === 0)).toBe(true)
    expect(indexLayout(5, { height: 360, expanded: false }).rows.every((r) => r.delay === 0)).toBe(true)
  })
})

describe('renderIndex', () => {
  test('one button per item that jumps to its message', () => {
    const html = renderIndex({ items: items(3), activeId: 'm1' }, { expanded: true }, t)
    expect(html.match(/data-chat="index-jump"/g)).toHaveLength(3)
    expect(html).toContain('data-id="m0"')
    expect(html).toContain('data-id="m2"')
    expect(html).toContain('role="navigation"')
    expect(html).toContain('aria-label="Jump to a message you sent"')
    expect(html).toContain('Message 1')
  })

  test('only the active item is current', () => {
    const html = renderIndex({ items: items(3), activeId: 'm1' }, { expanded: true }, t)
    expect(html.match(/aria-current="true"/g)).toHaveLength(1)
    const active = html.split('<li').find((li) => li.includes('aria-current="true"'))
    expect(active).toContain('data-id="m1"')
    expect(active).toContain('is-active')
  })

  test('collapsed until the pointer arrives; open shows the panel', () => {
    const closed = renderIndex({ items: items(3), activeId: 'm0' }, {}, t)
    expect(closed).not.toContain('is-open')
    expect(renderIndex({ items: items(3), activeId: 'm0' }, { expanded: true }, t)).toContain('is-open')
  })

  test('a long list gets a scroll thumb and a fade; a short one does not', () => {
    const long = renderIndex({ items: items(30), activeId: 'm0' }, { expanded: true }, t)
    expect(long).toContain('p-index-thumb')
    expect(long).toContain('is-overflow')
    const short = renderIndex({ items: items(3), activeId: 'm0' }, { expanded: true }, t)
    expect(short).not.toContain('p-index-thumb')
    expect(short).not.toContain('is-overflow')
  })

  test('nothing to index renders nothing', () => {
    expect(renderIndex({ items: [] }, {}, t)).toBe('')
    expect(renderIndex({}, {}, t)).toBe('')
  })

  test('labels and ids are escaped everywhere they appear', () => {
    const html = renderIndex({ items: [{ id: NASTY, label: NASTY }, ...items(2)], activeId: NASTY }, { expanded: true }, t)
    expect(html).not.toContain('<script>')
    expect(html).not.toContain('alert(1)</')
    expect(html).toContain(ESCAPED)
    expect(html).toContain(`data-id="${ESCAPED}"`)
    expect(html).toContain(`title="${ESCAPED}"`)
  })

  test('a translator that returns markup is escaped', () => {
    expect(renderIndex({ items: items(2) }, {}, () => '<b>x</b>')).not.toContain('<b>x</b>')
  })

  test('numeric ui values cannot inject markup', () => {
    const html = renderIndex({ items: items(2) }, { expanded: true, height: '"><script>', my: '"><x>', off: 'x' }, t)
    expect(html).not.toContain('<script>')
    expect(html).not.toContain('<x>')
    expect(html).not.toContain('NaN')
  })
})

describe('texts', () => {
  test('English and Spanish define the same keys, all prefixed', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.index.')).toBe(true)
    expect(TEXTS.es['chat.index.label']).toBe('Ir a un mensaje que enviaste')
  })
})
