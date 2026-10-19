// Run: bun test ~/.pi/agent/extensions/ask-user/questions.test.ts
import { describe, expect, test } from 'bun:test'
import { formatAnswers, noUserText, normalize } from './questions.ts'

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
