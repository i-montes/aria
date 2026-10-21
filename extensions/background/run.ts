// What `background_run` shares wherever it is registered (here with `PI_SANDBOX=0`, in the sandbox extension
// otherwise): the rules for deciding, its description and parameters, and the checks that come before anything runs
// or is asked (a subagent, no dock, a folder outside the projects folder).
import { resolvePath, subagentProblem } from '../sandbox/policy.ts'
import { DOCK_REFUSAL, jobsClient, refusal } from './client.ts'

export const SUBAGENT_REFUSAL = 'Background work is not available inside a subagent.'

export const WHEN = [
  'When to use the background (the person saying "in the background" or "do it here" always wins):',
  '- a command that does not end by itself (a dev server, `watch`, `tail -f`, a tunnel, `docker compose up` without -d): always background_run;',
  '- a job of more than a few steps that can be described up front (fix failing tests, refactor one module, research, a review): background_agent;',
  '- anything short or conversational (one or two tool calls, iterative design, debugging step by step with the person): do it here, in the foreground.',
].join('\n')

export const RUN_DESCRIPTION = [
  'Run a command that does not end by itself (a dev server, `watch`, `tail -f`, a tunnel, `docker compose up` without -d) in the background. It waits up to 3 s and returns the job id, the first output lines and the URL if one appeared; you are told if it dies by itself.',
  WHEN,
  '`title`: at most 6 words, in the person\'s language. `folder`: where it runs (default: your folder; ~ is expanded).',
].join('\n')

// Plain JSON Schema (Pi validates it like a TypeBox schema), so the extension loads under `bun test` without typebox.
export const str = (description: string) => ({ type: 'string', description })
export const object = (properties: Record<string, any>, required: string[] = []) => ({ type: 'object', properties, required, additionalProperties: false })

export const RUN_PROPERTIES = {
  command: str('The command, as bash would run it.'),
  title: str("At most 6 words, in the person's language."),
  folder: str('Where it runs (default: your folder; ~ is expanded).'),
}

export const isChild = (ctx: any) => ctx?.mode === 'json'

/** The folder a job works in: `folder` resolved against the session's cwd (`~` and relative paths work). */
export const folderOf = (folder: unknown, ctx: any): string => {
  const cwd = ctx?.cwd ?? process.cwd()
  return typeof folder === 'string' && folder.trim() ? resolvePath(folder, cwd).lexical : cwd
}

/** The checks before a background command runs (or the sandbox asks the person about it), in this order: not in a
 * subagent, under Aria's dock, a folder inside the projects folder. The folder to run in, or the refusal. */
export function prepareRun(params: { folder?: unknown }, ctx: any): { folder: string } | { refused: ReturnType<typeof refusal> } {
  if (isChild(ctx)) return { refused: refusal(SUBAGENT_REFUSAL) }
  if (!jobsClient()) return { refused: refusal(DOCK_REFUSAL) }
  const folder = folderOf(params.folder, ctx)
  const problem = subagentProblem({ cwd: folder }, ctx?.cwd ?? process.cwd())
  return problem ? { refused: refusal(`Refused: ${problem}. A background command runs inside the projects folder.`) } : { folder }
}
