// Scheduled tasks for Aria (docs/superpowers/specs/2026-10-21-schedules-design.md): the tools `schedule_create`,
// `schedule_list`, `schedule_update` and `schedule_delete`. Aria's dock keeps the tasks and runs each one at its time in
// the task's own tab; these tools reach it through the session's jobs socket (`ARIA_JOBS_SOCK`). In a subagent
// (`ctx.mode === 'json'`) or without the dock they refuse.
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { DOCK_REFUSAL, jobsClient, refusal, textResult } from '../background/client.ts'
import { isChild, object, str } from '../background/run.ts'

const SUBAGENT_REFUSAL = 'Scheduled tasks are not available inside a subagent.'
const EVERY_DAY = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

const CREATE_DESCRIPTION = [
  'Schedule a task Aria runs by itself at a time of day, in its own tab, with a desktop notification: "every day at 7:30 give me my Notion tasks", "on weekdays at 18:00 summarise my day", "on 2026-11-02 at 9:00 remind me to…".',
  'Before calling this, tell the person in one line when it will run and what it may do (each connector read-only unless they said it may write, and no folder outside the projects folder unless they named one), and wait for their yes.',
  '`prompt` is the whole instruction as it will be sent at that time (the run does not see this conversation). `at` is local 24-hour HH:MM. `days`: some of mon tue wed thu fri sat sun (default every day), or `date` YYYY-MM-DD for a single run.',
  '`connectors`: a level per connector the task uses, "read" or "allow" (allow = it may write there without asking). Nothing is ever asked during a scheduled run: what is not granted is refused.',
  'A task that may write (any "allow", or writable folders) is saved paused until the person presses Approve in /schedules; a read-only task starts at once.',
].join('\n')

const DAYS_SCHEMA = { type: 'array', items: { type: 'string', enum: EVERY_DAY }, description: 'Days it runs on (default every day).' }
const CONNECTORS_SCHEMA = {
  type: 'object',
  additionalProperties: { type: 'string', enum: ['read', 'allow'] },
  description: 'Level per connector (MCP server name): "read" or "allow". Unlisted connectors are read-only.',
}
const WRITABLE_SCHEMA = { type: 'array', items: { type: 'string' }, description: 'Absolute folders outside the projects folder it may write (default none).' }

const text = (t: string, details: any = undefined) => textResult(t, false, details)

/** "every day at 07:30", "mon, wed at 18:00", "on 2026-11-02 at 09:00". */
export function whenText(when: { at: string; days?: string[]; date?: string | null }): string {
  if (when.date) return `on ${when.date} at ${when.at}`
  const days = when.days ?? []
  const every = EVERY_DAY.every((d) => days.includes(d))
  return `${every ? 'every day' : days.join(', ')} at ${when.at}`
}

const at = (ms: number | null | undefined) => (typeof ms === 'number' ? new Date(ms).toString().slice(0, 21) : 'not again')

/** One line per task, for the model: what, when, next run, what it may do, how it last went. */
export function taskLine(task: any): string {
  const grants = Object.entries(task.grants?.connectors ?? {}).map(([name, level]) => `${name}: ${level}`)
  const writable = task.grants?.writable?.length ? [`writes ${task.grants.writable.join(', ')}`] : []
  const last = task.lastResult ? `last: ${task.lastResult.ok ? '' : 'failed — '}${task.lastResult.summary}` : 'never ran'
  return [
    `${task.title} (${task.id})${task.enabled ? '' : ' [paused]'}: ${whenText(task.when)}`,
    `next ${at(task.next)}`,
    grants.length || writable.length ? [...grants, ...writable].join(', ') : 'read-only',
    last,
  ].join(' · ')
}

function clientFor(ctx: any) {
  if (isChild(ctx)) return { refused: refusal(SUBAGENT_REFUSAL) }
  const client = jobsClient()
  return client ? { client } : { refused: refusal(DOCK_REFUSAL.replace('Background work', 'Scheduled tasks')) }
}

async function ask(ctx: any, op: string, fields: object, done: (answer: any) => string) {
  const found = clientFor(ctx)
  if ('refused' in found) return found.refused
  try {
    const answer = await found.client.request(op, fields)
    // The dock answers a refusal as `ok: false` with its reason (a bad time, an unknown connector…).
    if (!answer?.ok) return refusal(`${answer?.error ?? "Aria's dock did not answer"}`)
    return text(done(answer))
  } catch (e: any) {
    return refusal(`${e?.message ?? e}`)
  }
}

export default function schedule(pi: ExtensionAPI) {
  pi.registerTool({
    name: 'schedule_create',
    label: 'Schedule a task',
    description: CREATE_DESCRIPTION,
    parameters: object(
      {
        title: str('At most 5 words, in the person\'s language.'),
        prompt: str('The whole instruction, as it will be sent at that time.'),
        at: str('Local time, 24-hour HH:MM.'),
        days: DAYS_SCHEMA,
        date: str('YYYY-MM-DD for a single run (then days are ignored).'),
        connectors: CONNECTORS_SCHEMA,
        writable: WRITABLE_SCHEMA,
      },
      ['title', 'prompt', 'at'],
    ),
    async execute(_id: string, params: any, _signal: any, _onUpdate: any, ctx: any) {
      return ask(ctx, 'schedule_create', params, (answer) =>
        answer.task?.awaiting
          ? `Saved, waiting for the person: a task that writes runs only once they approve it with Approve in /schedules (it just opened). Tell them so. ${taskLine(answer.task)}`
          : `Scheduled: ${taskLine({ ...answer.task, next: answer.task?.next })}`,
      )
    },
  } as any)

  pi.registerTool({
    name: 'schedule_list',
    label: 'Scheduled tasks',
    description: 'List the scheduled tasks: when each runs, its next run, what it may do and how its last run went.',
    parameters: object({}, []),
    async execute(_id: string, _params: any, _signal: any, _onUpdate: any, ctx: any) {
      return ask(ctx, 'schedule_list', {}, (answer) => {
        const tasks = answer.tasks ?? []
        return tasks.length ? tasks.map(taskLine).join('\n') : 'No scheduled tasks.'
      })
    },
  } as any)

  pi.registerTool({
    name: 'schedule_update',
    label: 'Change a scheduled task',
    description: 'Change a scheduled task (its id is in schedule_list): any of title, prompt, at, days, date, connectors, writable, or `enabled: false` to pause it. Say the change and wait for a yes when it widens what the task may do.',
    parameters: object(
      {
        task: str('The task id (s-…).'),
        title: str('New title.'),
        prompt: str('New instruction.'),
        at: str('New local time, HH:MM.'),
        days: DAYS_SCHEMA,
        date: str('YYYY-MM-DD for a single run.'),
        connectors: CONNECTORS_SCHEMA,
        writable: WRITABLE_SCHEMA,
        enabled: { type: 'boolean', description: 'false pauses it, true resumes it.' },
      },
      ['task'],
    ),
    async execute(_id: string, params: any, _signal: any, _onUpdate: any, ctx: any) {
      return ask(ctx, 'schedule_update', params, (answer) =>
        answer.task?.awaiting
          ? `Changed, and waiting for the person: it runs again once they approve it with Approve in /schedules. Tell them so. ${taskLine(answer.task)}`
          : `Changed: ${taskLine(answer.task ?? {})}`,
      )
    },
  } as any)

  pi.registerTool({
    name: 'schedule_delete',
    label: 'Delete a scheduled task',
    description: 'Delete a scheduled task (its id is in schedule_list) and its tab. Confirm with the person first.',
    parameters: object({ task: str('The task id (s-…).') }, ['task']),
    async execute(_id: string, params: any, _signal: any, _onUpdate: any, ctx: any) {
      return ask(ctx, 'schedule_delete', params, () => `Deleted ${params.task}.`)
    },
  } as any)
}
