import { describe, expect, test } from 'bun:test'
import { TEXTS, renderAnswers } from './answers.js'

const t = (key) => key
const hostile = '<script>alert("x")</script> & \'q\''
const enc = '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &#39;q&#39;'

describe('renderAnswers', () => {
  test('escapes headers, selected values and the title', () => {
    const html = renderAnswers({ title: hostile, answers: [{ header: hostile, selected: [hostile, hostile] }] }, {}, t)
    expect(html).not.toContain('<script>')
    // the title appears twice (aria-label and caption), the header once, the values twice
    expect(html.split(enc).length - 1).toBe(5)
    const talk = renderAnswers({ answers: [{ header: 'h', talk: true }], talkText: hostile }, {}, t)
    expect(talk).not.toContain('<script>')
    expect(talk).toContain(enc)
  })
  test('one row per answer, one line per selected value', () => {
    const html = renderAnswers({ answers: [{ header: 'Fallback', selected: ['Commit hash'] }, { header: 'Docs', selected: ['a', 'b'] }] }, {}, t)
    expect(html.match(/p-answers-row"/g)).toHaveLength(2)
    expect(html.match(/p-answers-value"/g)).toHaveLength(3)
  })
  test('rows fade in staggered', () => {
    const html = renderAnswers({ answers: [{ header: 'a', selected: [] }, { header: 'b', selected: [] }] }, {}, t)
    expect(html).toContain('animation-delay: 70ms')
    expect(html).toContain('animation-delay: 140ms')
  })
  test('a "talk" answer shows the talk line instead of values', () => {
    const html = renderAnswers({ answers: [{ header: 'Tests', talk: true, selected: ['ignored'] }] }, {}, t)
    expect(html).toContain('p-answers-talk')
    expect(html).toContain('chat.answers.talk')
    expect(html).not.toContain('ignored')
  })
  test('default title and header come from the texts', () => {
    const html = renderAnswers({ answers: [{ selected: ['x'] }] }, {}, t)
    expect(html).toContain('chat.answers.title')
    expect(html).toContain('chat.answers.question')
  })
  test('no answers still renders the card', () => {
    expect(renderAnswers({ answers: [] }, {}, t)).toContain('role="group"')
    expect(renderAnswers({}, {}, t)).toContain('role="group"')
  })
})

describe('texts', () => {
  test('both languages have the same keys, all prefixed', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.answers.')).toBe(true)
  })
})
