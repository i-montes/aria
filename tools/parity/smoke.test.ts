import { expect, test } from 'bun:test'
import { judge } from './smoke.ts'

test('a short Spanish greeting passes', () => expect(judge('¡Hola! ¿En qué te ayudo?')).toEqual([]))
test('an English answer is not Spanish', () => expect(judge('Hello! How can I help?')).toEqual(['not in Spanish']))
test('a heading fails', () => expect(judge('## Resumen\nHola, ¿qué tal?')).toContain('heading'))
test('a table fails', () => expect(judge('Hola.\n| a | b |\n|---|---|')).toContain('table'))
test('a bullet list fails', () => expect(judge('Hola, puedo:\n- leer\n- escribir')).toContain('bullet list'))
test('more than four sentences fails', () =>
  expect(judge('Hola. Estoy aquí. Puedo ayudarte. Dime qué necesitas. Tengo tiempo. Empecemos.')).toContain('more than 4 sentences'))
test('an empty answer fails', () => expect(judge('  ')).toEqual(['empty']))
