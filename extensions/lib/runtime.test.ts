import { expect, test } from 'bun:test'
import { modelRuntime } from './runtime.ts'

test('modelRuntime rejects without ARIA_PI_DIST', async () => {
  const saved = process.env.ARIA_PI_DIST
  delete process.env.ARIA_PI_DIST
  try {
    await expect(modelRuntime()).rejects.toThrow('scratch-home.sh')
  } finally {
    if (saved !== undefined) process.env.ARIA_PI_DIST = saved
  }
})
