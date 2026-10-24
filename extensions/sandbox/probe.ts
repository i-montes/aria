// Does the box work here? `bwrap --version` succeeding says only that the program is installed: on Ubuntu 24.04 AppArmor
// (and some kernels' sysctl) refuse the user namespace bwrap needs, and then every boxed command fails. So the check is
// the real thing: a user namespace with the whole disk read-only, running `true`.
import { execFileSync } from 'node:child_process'

export const PROBE_ARGS = ['--unshare-user', '--ro-bind', '/', '/', 'true']

type Run = (program: string, args: string[]) => void

const spawn: Run = (program, args) => {
  execFileSync(program, args, { stdio: 'ignore', timeout: 10_000 })
}

/** True when bwrap exits 0 running `true` in a user namespace. `run` throws on a non-zero exit or a missing program. */
export function probeBox(run: Run = spawn): boolean {
  try {
    run('bwrap', PROBE_ARGS)
    return true
  } catch {
    return false
  }
}

let cached: boolean | undefined

/** The probe, once for the process. */
export const boxWorks = (): boolean => (cached ??= probeBox())
