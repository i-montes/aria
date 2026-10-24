// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { bwrapArgs } from './policy.ts'
import { PROBE_ARGS, probeBox } from './probe.ts'

describe('the probe of the box', () => {
  test('mounts /dev and /proc like the real call, so a host where --proc fails is caught', () => {
    const real = bwrapArgs('/home/ana/Projects/acme/checkout-widget', { home: '/home/ana', runtimeDir: '/run/user/1000', exists: () => false, env: {} }).join(' ')
    expect(real).toContain('--dev /dev --proc /proc')
    expect(PROBE_ARGS.join(' ')).toContain('--dev /dev --proc /proc')
    // a fake bwrap that refuses to mount /proc, as a container-style server does
    const refuseProc = (_program: string, args: string[]) => {
      if (args.join(' ').includes('--proc /proc')) throw Object.assign(new Error("bwrap: Can't mount proc"), { status: 1 })
    }
    expect(probeBox(refuseProc)).toBe(false)
  })
  test('a bwrap that runs `true` in a user namespace works', () => {
    const seen: string[][] = []
    expect(probeBox((program, args) => void seen.push([program, ...args]))).toBe(true)
    expect(seen).toEqual([['bwrap', '--unshare-user', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', 'true']])
    expect(PROBE_ARGS).toContain('--unshare-user')
  })
  test('a missing bwrap does not', () => {
    expect(probeBox(() => { throw Object.assign(new Error('spawn bwrap ENOENT'), { code: 'ENOENT' }) })).toBe(false)
  })
  test('a bwrap that is installed but cannot make the namespace (AppArmor, sysctl) does not either', () => {
    expect(probeBox(() => { throw Object.assign(new Error('Command failed'), { status: 1 }) })).toBe(false)
  })
})
