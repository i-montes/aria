// Run: bun test ui/chat/parts/question.test.js
import { describe, expect, test } from 'bun:test'
import { TEXTS, normalizeQuestions, questionAnswers, questionInit, questionReduce, renderQuestion } from './question.js'

const t = (key, vars) => (vars ? `${key} ${Object.entries(vars).map(([k, v]) => `${k}=${v}`).join(',')}` : key)
const evil = '<script>alert("x")</script> & \'q\''
const escaped = '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &#39;q&#39;'

const questions = [
  { question: 'Fallback?', header: 'Fallback', options: [{ label: 'dev' }, { label: 'Commit hash', description: 'From git', recommended: true }, { label: 'Required' }] },
  { question: 'Tests?', header: 'Tests', options: [{ label: 'One' }, { label: 'Two' }] },
  { question: 'Docs?', header: 'Docs', multi: true, options: [{ label: '.env.example', recommended: true }, { label: 'README' }, { label: 'CHANGELOG' }] },
]
const run = (state, ...steps) => steps.reduce((s, [op, payload]) => questionReduce(s, op, payload), state)

describe('normalizeQuestions', () => {
  test('puts the recommended option first (stable), caps at 4 questions and 4 options', () => {
    const [first] = normalizeQuestions(questions)
    expect(first.options.map((o) => o.label)).toEqual(['Commit hash', 'dev', 'Required'])
    expect(first.options[0].recommended).toBe(true)
    const many = Array.from({ length: 6 }, (_, i) => ({ question: `q${i}`, options: Array.from({ length: 6 }, (_, k) => ({ label: `o${k}`, recommended: k === 5 })) }))
    const n = normalizeQuestions(many)
    expect(n).toHaveLength(4)
    expect(n[0].options).toHaveLength(4)
    expect(n[0].options[0].label).toBe('o5')
  })

  test('is defensive about shapes', () => {
    expect(normalizeQuestions(undefined)).toEqual([])
    expect(normalizeQuestions([null, 5, 'x'])).toEqual([])
    const [q] = normalizeQuestions([{ question: 7, options: ['A', null, { label: 3 }], type: 'check' }])
    expect(q.question).toBe('7')
    expect(q.multi).toBe(true)
    expect(q.options.map((o) => o.label)).toEqual(['A', '3'])
  })
})

describe('wizard state', () => {
  test('init', () => {
    const s = questionInit(questions)
    expect(s.step).toBe(0)
    expect(s.picked).toEqual([[], [], []])
    expect(s.other).toEqual(['', '', ''])
    expect(s.outcome).toBe(null)
  })

  test('single choice replaces the pick and does not advance', () => {
    let s = run(questionInit(questions), ['pick', { q: 0, value: '1' }], ['pick', { q: 0, value: 2 }])
    expect(s.picked[0]).toEqual([2])
    expect(s.step).toBe(0)
  })

  test('multi choice toggles in option order', () => {
    let s = run(questionInit(questions), ['pick', { q: 2, value: 2 }], ['pick', { q: 2, value: 0 }])
    expect(s.picked[2]).toEqual([0, 2])
    s = questionReduce(s, 'pick', { q: 2, value: 2 })
    expect(s.picked[2]).toEqual([0])
  })

  test('picks out of range are ignored', () => {
    const s0 = questionInit(questions)
    expect(questionReduce(s0, 'pick', { q: 0, value: 9 })).toBe(s0)
    expect(questionReduce(s0, 'pick', { q: 7, value: 0 })).toBe(s0)
    expect(questionReduce(s0, 'pick', { q: 0, value: 'x' })).toBe(s0)
    expect(questionReduce(s0, 'pick', { q: 0, value: -1 })).toBe(s0)
  })

  test('other text replaces a single pick and a pick clears it; multi keeps both', () => {
    let s = run(questionInit(questions), ['pick', { q: 0, value: 0 }], ['other', { q: 0, value: 'my own' }])
    expect(s.picked[0]).toEqual([])
    expect(s.other[0]).toBe('my own')
    s = questionReduce(s, 'pick', { q: 0, value: 1 })
    expect(s.other[0]).toBe('')
    expect(s.picked[0]).toEqual([1])
    s = run(questionInit(questions), ['pick', { q: 2, value: 1 }], ['other', { q: 2, value: 'extra' }])
    expect(s.picked[2]).toEqual([1])
    expect(s.other[2]).toBe('extra')
  })

  test('next needs an answer, back goes back, goto jumps', () => {
    const s0 = questionInit(questions)
    expect(questionReduce(s0, 'next').step).toBe(0)
    let s = run(s0, ['pick', { q: 0, value: 0 }], ['next'])
    expect(s.step).toBe(1)
    expect(questionReduce(s, 'back').step).toBe(0)
    expect(questionReduce(s0, 'back').step).toBe(0)
    expect(questionReduce(s0, 'goto', { q: 2 }).step).toBe(2)
    expect(questionReduce(s0, 'goto', { q: 9 })).toBe(s0)
  })

  test('next on the last question does nothing; submit needs an answer on the current one', () => {
    let s = run(questionInit(questions), ['goto', { q: 2 }])
    expect(questionReduce(s, 'next')).toBe(s)
    expect(questionReduce(s, 'submit')).toBe(s)
    s = run(s, ['pick', { q: 2, value: 0 }], ['submit'])
    expect(s.outcome).toBe('submit')
  })

  test('skip advances, and on the last question submits if anything was answered, else dismisses', () => {
    let s = run(questionInit(questions), ['skip'], ['skip'])
    expect(s.step).toBe(2)
    expect(questionReduce(s, 'skip').outcome).toBe('dismiss')
    s = run(questionInit(questions), ['pick', { q: 0, value: 0 }], ['goto', { q: 2 }])
    expect(questionReduce(s, 'skip').outcome).toBe('submit')
  })

  test('dismiss and talk set the outcome; talk answers the current question', () => {
    expect(questionReduce(questionInit(questions), 'dismiss').outcome).toBe('dismiss')
    const s = run(questionInit(questions), ['pick', { q: 0, value: 1 }], ['talk'])
    expect(s.outcome).toBe('submit')
    expect(s.picked[0]).toEqual([])
    expect(s.talk[0]).toBe(true)
  })

  test('unknown ops, and a state without questions, are harmless', () => {
    const s0 = questionInit(questions)
    expect(questionReduce(s0, 'explode')).toBe(s0)
    expect(questionReduce(s0, 'pick')).toBe(s0)
    const empty = questionInit([])
    expect(questionReduce(empty, 'next')).toBe(empty)
    expect(questionReduce(empty, 'skip').outcome).toBe('dismiss')
    expect(() => questionReduce(null, 'pick', {})).not.toThrow()
  })

  test('the reducer never mutates its input', () => {
    const s0 = questionInit(questions)
    const snapshot = JSON.stringify(s0)
    questionReduce(s0, 'pick', { q: 0, value: 0 })
    questionReduce(s0, 'other', { q: 0, value: 'x' })
    expect(JSON.stringify(s0)).toBe(snapshot)
  })
})

describe('questionAnswers', () => {
  test('returns [{question, answers:[label...]}] in option order, plus the other text', () => {
    const s = run(questionInit(questions), ['pick', { q: 0, value: 0 }], ['other', { q: 1, value: ' mine ' }], ['pick', { q: 2, value: 2 }], ['pick', { q: 2, value: 1 }], ['other', { q: 2, value: 'x' }])
    expect(questionAnswers(questions, s)).toEqual([
      { question: 'Fallback?', answers: ['Commit hash'] },
      { question: 'Tests?', answers: ['mine'] },
      { question: 'Docs?', answers: ['README', 'CHANGELOG', 'x'] },
    ])
  })

  test('unanswered questions are left out; talk is flagged with no answers', () => {
    expect(questionAnswers(questions, questionInit(questions))).toEqual([])
    const s = run(questionInit(questions), ['pick', { q: 0, value: 0 }], ['goto', { q: 1 }], ['talk'])
    expect(questionAnswers(questions, s)).toEqual([
      { question: 'Fallback?', answers: ['Commit hash'] },
      { question: 'Tests?', answers: [], talk: true },
    ])
  })

  test('indexes that no longer match are ignored', () => {
    const s = { ...questionInit(questions), picked: [[0, 99], [], []] }
    expect(questionAnswers(questions, s)).toEqual([{ question: 'Fallback?', answers: ['Commit hash'] }])
    expect(questionAnswers(undefined, undefined)).toEqual([])
  })
})

describe('renderQuestion', () => {
  const props = { id: 'd1', questions }
  const ui = (patch = {}) => ({ ...questionInit(questions), ...patch })

  test('shows tabs, the current question, options with the recommended first and marked', () => {
    const html = renderQuestion(props, ui(), t)
    expect(html).toContain('role="tablist"')
    expect(html.match(/role="tab"/g)).toHaveLength(3)
    expect(html).toContain('aria-selected="true"')
    expect(html).toContain('Fallback?')
    expect(html).not.toContain('Tests?')
    expect(html.indexOf('Commit hash')).toBeLessThan(html.indexOf('>dev<'))
    expect(html).toContain('chat.question.recommended')
    expect(html).toContain('From git')
    expect(html).toContain('role="tabpanel"')
    expect(html).toContain('aria-live="polite"')
  })

  test('options emit pick with the question and value', () => {
    const html = renderQuestion(props, ui(), t)
    expect(html).toContain('data-chat="question" data-id="d1" data-op="pick" data-q="0" data-value="0"')
    expect(html).toContain('data-op="pick" data-q="0" data-value="2"')
  })

  test('pressed state follows picked; radio and check indicators differ', () => {
    const picked = renderQuestion(props, ui({ picked: [[1], [], []] }), t)
    expect(picked.match(/aria-pressed="true"/g)).toHaveLength(1)
    expect(picked).toContain('p-q-radio')
    const multi = renderQuestion(props, ui({ step: 2, picked: [[], [], [0, 1]] }), t)
    expect(multi.match(/aria-pressed="true"/g)).toHaveLength(2)
    expect(multi).toContain('p-q-check')
    expect(multi).not.toContain('p-q-radio')
  })

  test('the other row holds a text input that emits other', () => {
    const html = renderQuestion(props, ui({ other: ['my <b>own</b>', '', ''] }), t)
    expect(html).toContain('<input')
    expect(html).toContain('data-op="other" data-q="0"')
    expect(html).toContain('value="my &lt;b&gt;own&lt;/b&gt;"')
    expect(html).toContain('p-q-other selected')
  })

  test('footer: primary is Continue then Send, disabled until answered; back appears after the first step', () => {
    const first = renderQuestion(props, ui(), t)
    expect(first).toContain('data-op="next"')
    expect(first).toContain('chat.question.continue')
    expect(first).toMatch(/data-op="next"[^>]*disabled/)
    expect(first).not.toContain('data-op="back"')
    expect(first).toContain('data-op="skip"')
    expect(first).toContain('data-op="talk"')
    expect(first).toContain('data-op="dismiss"')
    const answered = renderQuestion(props, ui({ picked: [[0], [], []] }), t)
    expect(answered).not.toMatch(/data-op="next"[^>]*disabled/)
    const last = renderQuestion(props, ui({ step: 2, picked: [[], [], [1]] }), t)
    expect(last).toContain('data-op="submit"')
    expect(last).toContain('chat.question.send')
    expect(last).toContain('data-op="back"')
    expect(last).not.toMatch(/data-op="submit"[^>]*disabled/)
  })

  test('tabs show a check when answered, a chat icon for talk, and jump with goto', () => {
    const html = renderQuestion(props, ui({ picked: [[0], [], []], talk: [false, true, false] }), t)
    expect(html).toContain('data-op="goto" data-q="1"')
    expect(html).toContain('p-q-tab-done')
    expect(html).toContain('p-q-tab-talk')
  })

  test('works with partial ui and without questions', () => {
    expect(() => renderQuestion(props, {}, t)).not.toThrow()
    expect(() => renderQuestion(props, undefined, t)).not.toThrow()
    expect(() => renderQuestion({ id: 'x' }, {}, t)).not.toThrow()
    expect(() => renderQuestion({ id: 'x', questions: [] }, { step: 3 }, t)).not.toThrow()
    expect(renderQuestion(props, { step: 99 }, t)).toContain('Docs?')
  })

  test('every dynamic value is escaped', () => {
    const hostile = [{ question: evil, header: evil, multi: true, options: [{ label: evil, description: evil, recommended: true }] }]
    const html = renderQuestion({ id: evil, questions: hostile }, { step: 0, picked: [[0]], other: [evil], talk: [] }, t)
    expect(html).not.toContain('<script>')
    expect(html).toContain(escaped)
    expect(html).toContain(`data-id="${escaped}"`)
    expect(html).toContain(`value="${escaped}"`)
    expect(html).not.toMatch(/ value="[^"]*"[^>]*"[^>]*<script/)
  })

  test('texts exist in both languages with the same keys', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.question.')).toBe(true)
  })
})
