// Run: bun test ui/chat/parts/document.test.js
import { describe, expect, test } from 'bun:test'
import { TEXTS, renderDocument } from './document.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const EVIL = `<script>alert("x")</script> & 'q'`
const spec = {
  doc: 'spec',
  status: 'draft',
  title: 'Push through a queue',
  path: 'docs/specs/push-queue.md',
  sections: [
    { heading: 'GOAL', text: 'Send push from a queue.' },
    { heading: 'REQUIREMENTS', items: ['One', 'Two'] },
  ],
}

describe('the document card', () => {
  test('shows kind, status, title, path, headings, text and items', () => {
    const html = renderDocument(spec, {}, t)
    expect(html).toContain('chat.doc.kind.spec')
    expect(html).toContain('chat.doc.status.draft')
    expect(html).toContain('Push through a queue')
    expect(html).toContain('docs/specs/push-queue.md')
    expect(html).toContain('GOAL')
    expect(html).toContain('Send push from a queue.')
    expect(html.match(/<li class="p-doc-item"/g)).toHaveLength(2)
    expect(html).toContain('is-draft')
  })
  test('an approved plan pops its chip', () => {
    const html = renderDocument({ ...spec, doc: 'plan', status: 'approved' }, {}, t)
    expect(html).toContain('chat.doc.kind.plan')
    expect(html).toContain('is-approved')
    expect(html).not.toContain('is-draft')
  })
  test('unknown doc and status fall back to spec and draft', () => {
    const html = renderDocument({ ...spec, doc: 'novel', status: 'weird' }, {}, t)
    expect(html).toContain('chat.doc.kind.spec')
    expect(html).toContain('is-draft')
  })
  test('empty or missing data does not throw', () => {
    expect(renderDocument({}, {}, t)).toContain('class="p-doc')
    expect(renderDocument(undefined, undefined, t)).toContain('class="p-doc')
    const html = renderDocument({ sections: [null, {}, { heading: 'H', items: [] }] }, {}, t)
    expect(html).not.toContain('<ul class="p-doc-items"')
  })
  test('static mode turns animations off', () => {
    expect(renderDocument(spec, { animate: false }, t)).toContain('is-static')
  })
  test('every text field is escaped', () => {
    const html = renderDocument({ doc: 'spec', status: 'draft', title: EVIL, path: EVIL, sections: [{ heading: EVIL, text: EVIL, items: [EVIL] }] }, {}, t)
    expect(html).not.toContain('<script')
    expect(html).not.toContain('alert("x")')
    expect(html).toContain('&lt;script&gt;')
    expect(html).toContain('&amp;')
    expect(html).toContain('&#39;q&#39;')
  })
})

test('texts: both languages have the same keys, prefixed', () => {
  expect(Object.keys(TEXTS.en).sort()).toEqual(Object.keys(TEXTS.es).sort())
  for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.doc.')).toBe(true)
})
