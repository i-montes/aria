// Background work for Aria (only under Aria's dock, which runs the jobs and gives each session's Pi a socket in
// `ARIA_JOBS_SOCK`): the tools `background_agent`, `background_list`, `background_output`, `background_stop`, and
// `background_run` when the sandbox is off (`PI_SANDBOX=0`; otherwise the sandbox extension registers it, so the
// command goes through the sandbox's route). Each user message gets a hidden `[background] running: …` line while
// something runs, and each job that ends becomes an `aria-background` message (a new turn when it ended by itself),
// acked to the dock only after it was sent, so a Pi that restarts in between gets it again. The rules for when to
// use the background live in the tools' descriptions (Pi ignores `promptGuidelines` with Aria's SYSTEM.md).
// In a subagent (`ctx.mode === 'json'`) nothing is opened and every tool refuses.
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { subagentProblem } from '../sandbox/policy.ts'
import { DOCK_REFUSAL, jobsClient, refusal, startProcess, textResult, type JobsClient } from './client.ts'
import { folderOf, isChild, object, prepareRun, RUN_DESCRIPTION, RUN_PROPERTIES, str, SUBAGENT_REFUSAL, WHEN } from './run.ts'
import { endedMessage, listText, nowLine } from './text.ts'

export { SUBAGENT_REFUSAL }
const LIST_TIMEOUT_MS = 150
const MAX_REMEMBERED = 500

const AGENT_DESCRIPTION = [
  'Start a subagent that works in the background while you keep talking with the person. It returns at once with the job id; you are told when it ends (do not wait for it or poll it).',
  WHEN,
  '`task` is the whole brief: the subagent does not see this conversation, so say the goal, the files and facts it needs, the constraints and what "done" means.',
  '`title`: at most 6 words, in the person\'s language. `folder`: the project it works in (absolute or ~/…; default: your folder), inside the projects folder.',
  '`writes: false` for research or a review (it runs without the folder lock and changes no files). `isolation: "worktree"` for a large job (its own git worktree; the changes come back when it ends); "auto" (default) uses the folder, or a worktree when the folder is taken.',
  'While a writing subagent works in a folder (not a worktree), do not edit files in that folder yourself until it ends.',
  '`model: "heavy"` for heavy work (planning-grade model); "session" (default) uses the model of this chat.',
].join('\n')

const text = (t: string, details: any = undefined) => textResult(t, false, details)

/** The client for a tool call, or the refusal to return. */
function clientFor(ctx: any): { client: JobsClient } | { refused: ReturnType<typeof refusal> } {
  if (isChild(ctx)) return { refused: refusal(SUBAGENT_REFUSAL) }
  const client = jobsClient()
  return client ? { client } : { refused: refusal(DOCK_REFUSAL) }
}

/** `j3`, `#3` or `3` → the job id. */
async function jobId(client: JobsClient, ref: unknown): Promise<string> {
  const s = String(ref ?? '').trim()
  const n = s.match(/^#?(\d+)$/)
  if (!n) return s
  try {
    const list = await client.request('list')
    const found = (list?.jobs ?? []).find((j: any) => j.number === Number(n[1]))
    if (found) return found.id
  } catch {
    // fall back to the usual id form
  }
  return `j${n[1]}`
}

const failure = (what: string, e: any) => refusal(`${what}: ${e?.message ?? e}`)

// Per process, on globalThis: the one listener on the jobs client (a reloaded or second instance of this extension
// replaces it, so an end is never told twice) and the ends already told (an event given again after a lost ack).
type Listener = { client: JobsClient; owner: object; unsubscribe: () => void }
const LISTENER = Symbol.for('aria.background.listener')
const TOLD = Symbol.for('aria.background.told')
const shared = globalThis as unknown as { [LISTENER]?: Listener; [TOLD]?: Set<string> }
const told = (shared[TOLD] ??= new Set<string>())

export default function background(pi: ExtensionAPI) {
  const me = {}

  const onEvent = (client: JobsClient, e: any) => {
    try {
      if (!e || typeof e.seq !== 'number') return
      const key =
        e.event === 'cut'
          ? `cut:${(e.jobs ?? []).map((j: any) => `${j.id}@${j.startedAt}`).join(',')}`
          : `${e.event}:${e.job?.id}@${e.job?.startedAt}:${e.job?.state}:${e.job?.endedAt}`
      if (!told.has(key)) {
        const { text: content, triggerTurn } = endedMessage(e)
        pi.sendMessage(
          { customType: 'aria-background', content, display: true, details: { job: e.job ?? null, jobs: e.jobs, by: e.by ?? null, event: e.event } },
          triggerTurn ? { triggerTurn: true, deliverAs: 'followUp' } : undefined,
        )
        told.add(key)
        if (told.size > MAX_REMEMBERED) told.delete(told.values().next().value as string)
      }
      client.ack(e.seq)
    } catch {
      // not acked: the dock gives it again on the next hello
    }
  }

  /** Makes this instance the one listening on the current client (again, if the client changed). */
  const listen = (client: JobsClient | null) => {
    if (!client) return
    const current = shared[LISTENER]
    if (current && current.client === client && current.owner === me) return
    try {
      current?.unsubscribe()
    } catch {}
    shared[LISTENER] = { client, owner: me, unsubscribe: client.onEvent((e) => onEvent(client, e)) }
  }

  /** The client for a tool call (listening on it), or the refusal. */
  const use = (ctx: any) => {
    const got = clientFor(ctx)
    if ('client' in got) listen(got.client)
    return got
  }

  pi.on('session_start', (_event: any, ctx: any) => {
    try {
      if (!isChild(ctx)) listen(jobsClient())
    } catch {
      // no background events this session
    }
  })

  pi.on('session_shutdown', () => {
    try {
      const current = shared[LISTENER]
      if (current?.owner === me) {
        current.unsubscribe()
        delete shared[LISTENER]
      }
    } catch {}
  })

  pi.on('before_agent_start', async (event: any, ctx: any) => {
    try {
      if (event?.messageType !== 'user' || isChild(ctx)) return
      const client = jobsClient()
      if (!client) return
      listen(client)
      const answer = await client.request('list', {}, LIST_TIMEOUT_MS)
      const line = answer?.ok ? nowLine(answer.jobs ?? []) : null
      if (line) return { message: { customType: 'aria-background-now', content: line, display: false } }
    } catch {
      // the dock did not answer in time: no line this time
    }
  })

  pi.registerTool({
    name: 'background_agent',
    label: 'Background subagent',
    description: AGENT_DESCRIPTION,
    parameters: object(
      {
        task: str('The full brief. The subagent does not see this conversation.'),
        title: str("At most 6 words, in the person's language."),
        folder: str('The project it works in: absolute or ~/…, inside the projects folder. Default: your folder.'),
        writes: { type: 'boolean', description: 'false for research or a review: it changes no files. Default true.' },
        isolation: { type: 'string', enum: ['auto', 'folder', 'worktree'], description: '"worktree" for a large job; "auto" (default) uses the folder, or a worktree when the folder is taken.' },
        model: { type: 'string', enum: ['session', 'heavy'], description: '"heavy" for heavy work; "session" (default) is this chat\'s model.' },
      },
      ['task', 'title'],
    ),
    async execute(_id: string, params: any, _signal: any, _onUpdate: any, ctx: any) {
      const got = use(ctx)
      if ('refused' in got) return got.refused
      const folder = folderOf(params.folder, ctx)
      const problem = subagentProblem({ cwd: folder }, ctx?.cwd ?? process.cwd())
      if (problem) return refusal(`Refused: ${problem}. A background subagent works inside the projects folder.`)
      let answer: any
      try {
        answer = await got.client.request('start', {
          kind: 'agent',
          task: params.task,
          title: params.title,
          folder,
          writes: params.writes ?? true,
          isolation: params.isolation ?? 'auto',
          model: params.model ?? 'session',
        })
      } catch (e) {
        return failure('Could not start the subagent', e)
      }
      if (!answer?.ok || !answer.job) return refusal(`Could not start the subagent: ${answer?.error ?? 'no answer from the dock'}`)
      const job = answer.job
      const state =
        job.state === 'queued'
          ? 'It is queued (three subagents already run, or another writer holds that folder) and starts by itself.'
          : `It is running${job.isolation === 'worktree' ? ' in its own worktree' : ''}.`
      const hold = job.writes !== false && job.isolation === 'folder' ? ` Do not edit files in ${folder} until it ends.` : ''
      return text(`Started background subagent #${job.number} (${job.title}, job ${job.id}) in ${folder}. ${state}${hold} Aria will be told when it ends; do not wait for it.`, { job })
    },
  } as any)

  pi.registerTool({
    name: 'background_list',
    label: 'Background work',
    description: "List this session's background work: what runs (processes, subagents) and what finished.",
    parameters: object({}),
    async execute(_id: string, _params: any, _signal: any, _onUpdate: any, ctx: any) {
      const got = use(ctx)
      if ('refused' in got) return got.refused
      try {
        const answer = await got.client.request('list')
        if (!answer?.ok) return refusal(`Could not list the background work: ${answer?.error ?? 'no answer'}`)
        return text(listText(answer.jobs ?? []), { jobs: answer.jobs ?? [] })
      } catch (e) {
        return failure('Could not list the background work', e)
      }
    },
  } as any)

  pi.registerTool({
    name: 'background_output',
    label: 'Background output',
    description: "The last output lines of a background job (a process's output, or what a subagent did).",
    parameters: object({ job: str('The job: its id (j3) or number (#3).'), lines: { type: 'number', description: 'How many of the last lines: default 60, at most 400.' } }, ['job']),
    async execute(_id: string, params: any, _signal: any, _onUpdate: any, ctx: any) {
      const got = use(ctx)
      if ('refused' in got) return got.refused
      try {
        const id = await jobId(got.client, params.job)
        const lines = Math.max(1, Math.min(400, Math.floor(Number(params.lines) || 60)))
        const answer = await got.client.request('output', { job: id, lines })
        if (!answer?.ok) return refusal(`Could not read ${params.job}: ${answer?.error ?? 'no answer'}`)
        const out: string[] = answer.lines ?? []
        return text(out.length ? out.join('\n') : '(no output yet)', { job: id, lines: out.length })
      } catch (e) {
        return failure(`Could not read ${params.job}`, e)
      }
    },
  } as any)

  pi.registerTool({
    name: 'background_stop',
    label: 'Stop background job',
    description: 'Stop a background job (a process and everything it started, or a subagent). Only when the person asks, or the job is no longer needed.',
    parameters: object({ job: str('The job: its id (j3) or number (#3).') }, ['job']),
    async execute(_id: string, params: any, _signal: any, _onUpdate: any, ctx: any) {
      const got = use(ctx)
      if ('refused' in got) return got.refused
      try {
        const id = await jobId(got.client, params.job)
        const answer = await got.client.request('stop', { job: id })
        if (!answer?.ok) return refusal(`Could not stop ${params.job}: ${answer?.error ?? 'no answer'}`)
        if (answer.already) return text(`${params.job} had already ended.`, { job: id, already: true })
        return text(`Stopping ${params.job}.`, { job: id })
      } catch (e) {
        return failure(`Could not stop ${params.job}`, e)
      }
    },
  } as any)

  if (process.env.PI_SANDBOX === '0') {
    pi.registerTool({
      name: 'background_run',
      label: 'Run in the background',
      description: RUN_DESCRIPTION,
      parameters: object(RUN_PROPERTIES, ['command', 'title']),
      async execute(_id: string, params: any, _signal: any, _onUpdate: any, ctx: any) {
        const got = use(ctx)
        if ('refused' in got) return got.refused
        const ready = prepareRun(params, ctx)
        if ('refused' in ready) return ready.refused
        return startProcess({ command: params.command, line: params.command, title: params.title, folder: ready.folder })
      },
    } as any)
  }
}
