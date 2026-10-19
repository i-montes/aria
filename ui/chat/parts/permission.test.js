// Run: bun test ui/chat/parts/permission.test.js
import { describe, expect, test } from 'bun:test'
import { TEXTS, renderPermission } from './permission.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const evil = '<script>alert("x")</script> & \'q\''
const escaped = '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &#39;q&#39;'

describe('permission card', () => {
  const props = { id: 'd1', kind: 'sandbox', summary: 'Wants to leave the sandbox', detail: 'pnpm install', jev: 'Only the registry.' }

  test('shows the title, the command, the jev review and three choices', () => {
    const html = renderPermission(props, {}, t)
    expect(html).toContain('class="p-perm"')
    expect(html).toContain('Wants to leave the sandbox')
    expect(html).toContain('<code class="p-perm-code">pnpm install</code>')
    expect(html).toContain('Only the registry.')
    for (const choice of ['allow', 'always', 'deny']) {
      expect(html).toContain(`data-chat="permission" data-id="d1" data-choice="${choice}"`)
    }
    expect(html.match(/<button/g)).toHaveLength(3)
  })

  test('falls back to the kind title without a summary and omits empty parts', () => {
    const html = renderPermission({ id: 'a', kind: 'write', detail: 'src/a.rs' }, {}, t)
    expect(html).toContain('chat.perm.write')
    expect(html).not.toContain('p-perm-review')
    expect(renderPermission({ id: 'a', kind: 'sandbox' }, {}, t)).not.toContain('p-perm-cmd')
    expect(renderPermission({ id: 'a' }, {}, t)).toContain('chat.perm.sandbox')
  })

  test('jev can also be an object with a review or text field', () => {
    expect(renderPermission({ ...props, jev: { review: 'From review' } }, {}, t)).toContain('From review')
    expect(renderPermission({ ...props, jev: { text: 'From text' } }, {}, t)).toContain('From text')
    expect(renderPermission({ ...props, jev: { other: 1 } }, {}, t)).not.toContain('p-perm-review')
  })

  test('every dynamic value is escaped', () => {
    const html = renderPermission({ id: evil, kind: evil, summary: evil, detail: evil, jev: evil }, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).toContain(escaped)
    expect(html).toContain(`data-id="${escaped}"`)
    const object = renderPermission({ ...props, jev: { review: evil } }, {}, t)
    expect(object).not.toContain('<script>')
  })

  test('buttons carry no inline colors that could change on hover', () => {
    expect(renderPermission(props, {}, t)).not.toContain('style=')
  })

  test('texts exist in both languages with the same keys', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.perm.')).toBe(true)
  })
})
