// Run: bun test ui/chat/parts/quota.test.js
import { describe, expect, test } from 'bun:test'
import { TEXTS, renderQuota } from './quota.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const evil = '<script>alert("x")</script> & \'q\''

describe('quota card', () => {
  test('shows the percentage, a progress bar and the two choices', () => {
    const html = renderQuota({ id: 'q1', percent: 92, freeModel: 'big-pickle', secondsLeft: 25 }, {}, t)
    expect(html).toContain('class="p-quota"')
    expect(html).toContain('role="progressbar"')
    expect(html).toContain('aria-valuenow="92"')
    expect(html).toContain('width:92%')
    expect(html).toContain('chat.quota.used n=92')
    expect(html).toContain('data-chat="quota" data-id="q1" data-choice="free"')
    expect(html).toContain('data-chat="quota" data-id="q1" data-choice="keep"')
    expect(html).toContain('chat.quota.free_model model=big-pickle')
    expect(html).toContain('chat.quota.countdown n=25')
  })

  test('the percentage is clamped and rounded; bad values become 0', () => {
    expect(renderQuota({ id: 'a', percent: 250 }, {}, t)).toContain('aria-valuenow="100"')
    expect(renderQuota({ id: 'a', percent: -4 }, {}, t)).toContain('aria-valuenow="0"')
    expect(renderQuota({ id: 'a', percent: 91.6 }, {}, t)).toContain('aria-valuenow="92"')
    expect(renderQuota({ id: 'a', percent: 'x' }, {}, t)).toContain('aria-valuenow="0"')
    expect(renderQuota({ id: 'a' }, {}, t)).toContain('width:0%')
  })

  test('the free model and the countdown are optional', () => {
    const html = renderQuota({ id: 'a', percent: 90 }, {}, t)
    expect(html).not.toContain('chat.quota.free_model')
    expect(html).not.toContain('chat.quota.countdown')
    expect(renderQuota({ id: 'a', percent: 90, secondsLeft: 0 }, {}, t)).toContain('chat.quota.countdown n=0')
  })

  test('every dynamic value is escaped', () => {
    const html = renderQuota({ id: evil, percent: evil, freeModel: evil, secondsLeft: evil }, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).toContain('data-id="&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &#39;q&#39;"')
    expect(html).toContain('model=&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &#39;q&#39;')
  })

  test('texts exist in both languages with the same keys', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.quota.')).toBe(true)
  })
})
