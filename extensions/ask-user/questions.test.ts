// Run: bun test ~/.pi/agent/extensions/ask-user/questions.test.ts
import { describe, expect, test } from 'bun:test'
import { formatAnswers, noUserText, normalize, parseCardAnswers, questionCard } from './questions.ts'

const q = (options: any[], extra: object = {}) => ({ header: 'Repo', question: '¿Dónde guardo el spec?', options, ...extra })

describe('normalize', () => {
  test('the recommended option goes first, marked', () => {
    const [n] = normalize({ questions: [q([{ label: 'gateway' }, { label: 'widget', recommended: true }])] })
    expect(n.options.map((o) => o.label)).toEqual(['widget (recommended)', 'gateway'])
  })
  test('an "Otra" the model added is dropped, the form adds its own', () => {
    const [n] = normalize({ questions: [q([{ label: 'a' }, { label: 'b' }, { label: 'Otra opción' }])] })
    expect(n.options.map((o) => o.label)).toEqual(['a', 'b'])
  })
  test('duplicates and blanks are dropped', () => {
    const [n] = normalize({ questions: [q([{ label: 'A' }, { label: 'a ' }, { label: ' ' }, { label: 'b' }])] })
    expect(n.options.map((o) => o.label)).toEqual(['A', 'b'])
  })
  test('fewer than two real options is an error the model can fix', () => {
    expect(() => normalize({ questions: [q([{ label: 'a' }, { label: 'Other' }])] })).toThrow('at least 2')
  })
  test('more than four questions is an error', () => {
    expect(() => normalize({ questions: Array(5).fill(q([{ label: 'a' }, { label: 'b' }])) })).toThrow('at most 4')
  })
  test('long headers are cut and missing ones numbered', () => {
    const n = normalize({ questions: [q([{ label: 'a' }, { label: 'b' }], { header: 'Una cabecera larguísima' }), q([{ label: 'a' }, { label: 'b' }], { header: '' })] })
    expect(n.map((x) => x.header)).toEqual(['Una cabecera', 'P2'])
  })
})

test('formatAnswers strips the marker and tells written answers apart', () => {
  const qs = normalize({ questions: [q([{ label: 'gateway' }, { label: 'widget', recommended: true }]), q([{ label: 'sí' }, { label: 'no' }], { question: '¿Commit?' })] })
  const text = formatAnswers(qs, [
    { id: 'q1', label: 'widget (recommended)', wasCustom: false, index: 1 },
    { id: 'q2', label: 'solo después de los tests', wasCustom: true },
  ])
  expect(text).toBe('¿Dónde guardo el spec?\n→ The user chose: widget\n\n¿Commit?\n→ The user wrote: solo después de los tests')
})

test('no user: subagents report NEEDS_CONTEXT, print mode goes on', () => {
  expect(noUserText('json')).toContain('NEEDS_CONTEXT')
  expect(noUserText('print')).toContain('recommended option')
})

describe("asking in Aria's dock", () => {
  const questions = normalize({
    questions: [
      { header: 'Fallback', question: 'Which fallback?', options: [{ label: 'Version' }, { label: 'Git hash', description: 'Always there', recommended: true }] },
      { header: 'Checks', question: 'Which checks?', options: [{ label: 'Unit' }, { label: 'Lint' }] },
    ],
  })

  test('the card carries the questions, the recommended option first and marked, without the text suffix', () => {
    const card = JSON.parse(questionCard(questions))
    expect(card.aria).toBe(1)
    expect(card.kind).toBe('question')
    expect(card.questions[0]).toEqual({ question: 'Which fallback?', header: 'Fallback', options: [{ label: 'Git hash', description: 'Always there', recommended: true }, { label: 'Version' }] })
    expect(card.questions[1].options.map((o: any) => o.label)).toEqual(['Unit', 'Lint'])
  })

  test("the dock's answers become the tool's answers: a choice, or what the user wrote", () => {
    const reply = JSON.stringify([{ question: 'Which fallback?', answers: ['Git hash'] }, { question: 'Which checks?', answers: ['Both, and types'] }])
    const answers = parseCardAnswers(reply, questions)!
    expect(answers).toEqual([
      { id: 'q1', label: 'Git hash', wasCustom: false, index: 1 },
      { id: 'q2', label: 'Both, and types', wasCustom: true },
    ])
    expect(formatAnswers(questions, answers)).toContain('The user chose: Git hash')
    expect(formatAnswers(questions, answers)).toContain('The user wrote: Both, and types')
  })

  test('"talk about this" is an answer the model can act on', () => {
    const answers = parseCardAnswers(JSON.stringify([{ question: 'Which fallback?', answers: [], talk: true }]), questions)!
    expect(answers[0].wasCustom).toBe(true)
    expect(answers[0].label).toContain('talk')
  })

  test('a reply that is not the answers is nothing (the tool then says the user did not answer)', () => {
    expect(parseCardAnswers('nope', questions)).toBeNull()
    expect(parseCardAnswers('{"a":1}', questions)).toBeNull()
    expect(parseCardAnswers(undefined, questions)).toBeNull()
  })
})
