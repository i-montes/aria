// Run: bun test ui/chat/parts/meter.test.js
import { describe, expect, test } from 'bun:test'
import { TEXTS, renderMeter } from './meter.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const evil = '<script>alert("x")</script> & \'q\''

describe('header meter', () => {
  const props = { model: 'OpenCode Go', used: 62, context: 18, free: false, hasQuota: true, compactAt: 40 }

  test('the button opens the model picker and shows the model, quota bar and percentage', () => {
    const html = renderMeter(props, {}, t)
    expect(html).toContain('data-chat="open-models"')
    expect(html).toContain('OpenCode Go')
    expect(html).toContain('width:62%')
    expect(html).toContain('p-meter-pct')
    expect(html).toContain('62%')
    expect(html).not.toContain('p-meter-free')
  })

  test('the popover has the quota bar, the context bar and the compaction mark', () => {
    const html = renderMeter(props, {}, t)
    expect(html).toContain('role="tooltip"')
    expect(html).toContain('chat.meter.subscription')
    expect(html).toContain('width:18%')
    expect(html).toContain('left:40%')
    expect(html).toContain('chat.meter.compacts n=40')
    expect(html).toContain('chat.meter.model_hint')
  })

  test('quota colors follow the thresholds 80 and 95', () => {
    expect(renderMeter({ ...props, used: 79 }, {}, t)).toContain('p-meter-ok')
    expect(renderMeter({ ...props, used: 80 }, {}, t)).toContain('p-meter-warn')
    expect(renderMeter({ ...props, used: 95 }, {}, t)).toContain('p-meter-crit')
    expect(renderMeter({ ...props, used: 79 }, {}, t)).not.toContain('p-meter-warn')
  })

  test('a free model shows the badge and no quota bar', () => {
    const html = renderMeter({ ...props, free: true }, {}, t)
    expect(html).toContain('class="p-meter-free"')
    expect(html).not.toContain('p-meter-pct')
    expect(html).toContain('chat.meter.free_note')
    expect(html).toContain('chat.meter.tier_free')
  })

  test('a provider without quota says it is pay per use', () => {
    const html = renderMeter({ ...props, hasQuota: false }, {}, t)
    expect(html).not.toContain('p-meter-pct')
    expect(html).toContain('chat.meter.no_quota')
    expect(html).toContain('chat.meter.tier_payg')
  })

  test('numbers are clamped and rounded', () => {
    const html = renderMeter({ ...props, used: 140, context: -3, compactAt: 33.4 }, {}, t)
    expect(html).toContain('width:100%')
    expect(html).toContain('width:0%')
    expect(html).toContain('left:33%')
    expect(renderMeter({ model: 'm' }, {}, t)).toContain('left:40%')
  })

  test('the reset time is optional', () => {
    expect(renderMeter({ ...props, used: 62 }, {}, t)).not.toContain('chat.meter.note_ok')
    expect(renderMeter({ ...props, used: 62, resets: '12 days' }, {}, t)).toContain('chat.meter.note_ok resets=12 days')
    expect(renderMeter({ ...props, used: 85, resets: '12 days' }, {}, t)).toContain('chat.meter.note_low resets=12 days')
    expect(renderMeter({ ...props, used: 99 }, {}, t)).toContain('chat.meter.note_out_plain')
  })

  test('every dynamic value is escaped', () => {
    const html = renderMeter({ model: evil, used: evil, context: evil, free: false, hasQuota: true, compactAt: evil, resets: evil }, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).toContain('&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &#39;q&#39;')
    expect(renderMeter({ ...props, model: evil }, {}, t)).not.toContain('<script>')
  })

  test('texts exist in both languages with the same keys', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.meter.')).toBe(true)
  })
})
