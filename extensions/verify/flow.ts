// When a run that edited files ends, the project's checks run in order and stop at the first failure. All passed: the
// status says so. A failure: the status says it, and the model is told the command and the end of its output once per
// failing streak (its fix runs again; a second failure in a row is only shown, never fed back again).
import { lastLines, type Check, type Outcome } from './checks.ts'

export interface Status {
  ok: boolean
  passed?: number
}

export class Verifier {
  private dirty = false
  private on = true
  private toldOnce = false

  constructor(
    private deps: {
      checks: () => Check[]
      run: (command: string) => Promise<Outcome>
      tell: (text: string) => void
      status: (value: Status) => void
    },
  ) {}

  setOn(on: boolean) {
    this.on = on
  }

  /** An edit or write happened in this run. */
  edited() {
    this.dirty = true
  }

  async runEnded() {
    const edited = this.dirty
    this.dirty = false
    if (!edited || !this.on) return
    const checks = this.deps.checks()
    if (!checks.length) return
    let passed: number | undefined
    for (const check of checks) {
      const outcome = await this.deps.run(check.command)
      const count = /(\d+)\s+pass/.exec(outcome.output)
      if (count) passed = Number(count[1])
      if (!outcome.ok) {
        this.deps.status({ ok: false })
        if (!this.toldOnce) {
          this.toldOnce = true
          const why = outcome.timedOut ? ' (stopped after 5 minutes)' : ''
          this.deps.tell(`The project's check \`${check.command}\` failed after your edits${why}. Its last lines:\n\n${lastLines(outcome.output)}\n\nFix it.`)
        }
        return
      }
    }
    this.toldOnce = false
    this.deps.status(passed === undefined ? { ok: true } : { ok: true, passed })
  }
}
