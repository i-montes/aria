// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { PROBE_ARGS, probeBox } from './probe.ts'

describe('the probe of the box', () => {
  test('a bwrap that runs `true` in a user namespace works', () => {
    const seen: string[][] = []
    expect(probeBox((program, args) => void seen.push([program, ...args]))).toBe(true)
    expect(seen).toEqual([['bwrap', '--unshare-user', '--ro-bind', '/', '/', 'true']])
    expect(PROBE_ARGS).toContain('--unshare-user')
  })
  test('a missing bwrap does not', () => {
    expect(probeBox(() => { throw Object.assign(new Error('spawn bwrap ENOENT'), { code: 'ENOENT' }) })).toBe(false)
  })
  test('a bwrap that is installed but cannot make the namespace (AppArmor, sysctl) does not either', () => {
    expect(probeBox(() => { throw Object.assign(new Error('Command failed'), { status: 1 }) })).toBe(false)
  })
})
