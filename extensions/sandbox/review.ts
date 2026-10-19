// Jev's independent look at a command before the user lets it out of the sandbox: the description
// comes from the model running the command, so Jev checks it against the command and flags the
// effects that matter. Jev only answers choice/score/bool questions, so it reviews rather than
// writes its own summary.
export { CLASSIFIER } from '../jev-router/route.ts'

export const REVIEW_TIMEOUT_MS = 4000
const YES = 0.5

const FLAGS = {
  deletes: {
    instructions: 'Does running `command` delete or overwrite existing files, directories, data, containers, branches or commits (rm, git reset --hard, force push, docker rm, DROP, truncating redirects…)?',
    label: 'deletes or overwrites data',
  },
  sends: {
    instructions: 'Does `command` send data out of this machine: upload, push to a remote, post to an API, copy over ssh/scp/rsync, publish a package? Only downloading or reading from the network does not count.',
    label: 'sends data out',
  },
  installs: {
    instructions: 'Does `command` install, update or remove software or packages, globally or in the project?',
    label: 'installs software',
  },
  system: {
    instructions: 'Does `command` change the system outside the project: system or user config files, services, cron, shell startup files, permissions, other users, or use sudo?',
    label: 'changes the system',
  },
} as const
type Flag = keyof typeof FLAGS

export interface Review {
  matches: boolean
  flags: Flag[]
}

export function reviewRequest(command: string, description: string) {
  const questions: Record<string, unknown> = {
    matches: {
      type: 'bool',
      instructions:
        'An AI agent wants to run `command` and told the user only `description`. Is `description` an accurate and complete account of what `command` does? Any important action it leaves out (deleting, uploading, installing, changing system settings, running a downloaded script) makes it inaccurate. An empty description is inaccurate.',
      criteria: { true: 'The description is accurate and leaves out nothing important', false: 'The description is wrong, vague or hides an important action' },
    },
  }
  for (const [key, f] of Object.entries(FLAGS)) questions[key] = { type: 'bool', instructions: f.instructions, criteria: { true: 'Yes', false: 'No' } }
  return { state: { command: command.slice(0, 6000), description: description.slice(0, 500) }, questions }
}

/** A Jev result -> review, or null when it failed or answered something unexpected. */
export function parseReview(result: any): Review | null {
  if (result?.stopReason !== 'stop') return null
  const a = result.answers ?? {}
  const p = (key: string) => (a[key]?.type === 'bool' ? Number(a[key].probability) : NaN)
  if (!Number.isFinite(p('matches'))) return null
  return { matches: p('matches') >= YES, flags: (Object.keys(FLAGS) as Flag[]).filter((k) => p(k) >= YES) }
}

/** The line shown in the dialog. */
export function reviewLine(r: Review | null): string {
  if (!r) return 'Jev: no review (unavailable)'
  const verdict = r.matches ? '✓ matches the description' : '⚠ does NOT match the description'
  return [`Jev: ${verdict}`, ...r.flags.map((f) => FLAGS[f].label)].join(' · ')
}
