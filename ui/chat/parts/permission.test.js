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

  test('a connector card has its icon and title', () => {
    const options = [{ value: 'Yes, just this once', label: 'Yes, just this once', primary: true }, { value: 'No', label: 'No' }]
    const html = renderPermission({ id: 'c1', kind: 'connector', detail: 'title: Weekly notes', options }, {}, t)
    expect(html).toContain('chat.perm.connector')
    expect(html).toContain('data-icon="plug"')
    expect(html).toContain('p-perm-path') // the arguments, not a shell command: no "$" prompt
    expect(html).not.toContain('p-perm-prompt')
    expect(html).toContain('<code class="p-perm-code">title: Weekly notes</code>')
    expect(renderPermission({ id: 'c1', kind: 'connector', summary: 'Notion wants to create pages' }, {}, t)).toContain('Notion wants to create pages')
    expect(renderPermission({ id: 's', kind: 'sandbox' }, {}, t)).not.toContain('data-icon="plug"')
    expect(TEXTS.en['chat.perm.connector']).toBe('A connector wants to act')
    expect(TEXTS.es['chat.perm.connector']).toBe('Un conector quiere actuar')
  })

  test('texts exist in both languages with the same keys', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.perm.')).toBe(true)
  })
})

describe('the options an extension gives', () => {
  const options = [
    { value: 'Yes, just this once', label: 'Yes, just this once', primary: true },
    { value: "Yes, and don't ask again for git push this session", label: "Yes, and don't ask again for git push this session" },
    { value: 'Yes, no sandbox for the rest of the session', label: 'Yes, no sandbox for the rest of the session' },
    { value: 'No', label: 'No' },
  ]

  test('one button per option, in order, each sending its own value', () => {
    const html = renderPermission({ id: 'p1', kind: 'sandbox', summary: 'Restart the database', options }, {}, t)
    const choices = [...html.matchAll(/data-choice="([^"]*)"/g)].map((m) => m[1])
    expect(choices).toEqual(['Yes, just this once', 'Yes, and don&#39;t ask again for git push this session', 'Yes, no sandbox for the rest of the session', 'No'])
    expect(html).not.toContain('data-choice="allow"')
  })

  test('the primary option is the filled button', () => {
    const html = renderPermission({ id: 'p1', options }, {}, t)
    expect(html).toMatch(/class="p-perm-btn p-perm-primary"[^>]*data-choice="Yes, just this once"/)
  })

  test('values and labels are escaped', () => {
    const html = renderPermission({ id: 'p1', options: [{ value: '"><script>', label: '<b>x</b>' }] }, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).not.toContain('<b>x')
  })
})
