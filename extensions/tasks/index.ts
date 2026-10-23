// The `tasks_update` tool: the model keeps the task list of multi-step work (the whole list each time); Aria's terminal
// draws it above the editor from the tool's result. Subagents (`--mode json`) do not get it.
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { Type } from 'typebox'
import { MAX_TASKS, normalizeTasks, summary, toolsFor } from './tasks.ts'

const Params = Type.Object({
  tasks: Type.Array(
    Type.Object({
      id: Type.String({ description: 'A short stable id for the task, the same in every update.' }),
      title: Type.String({ description: 'What the task does, in a few words, in the user\'s language.' }),
      status: Type.Union([Type.Literal('todo'), Type.Literal('active'), Type.Literal('done')]),
      agent: Type.Optional(Type.String({ description: 'The subagent working on it, when one is.' })),
    }),
    { maxItems: MAX_TASKS },
  ),
})

export default function tasks(pi: ExtensionAPI) {
  pi.on('session_start', (_event: any, ctx: any) => {
    if (ctx.mode === 'json') pi.setActiveTools(toolsFor(pi.getActiveTools(), ctx.mode))
  })

  pi.registerTool({
    name: 'tasks_update',
    label: 'Tasks',
    description:
      'Keep the task list of work with more than one step: call it with the whole list when you start such work, and again each time a task starts or ends (status todo, active or done; at most one or two active at once). The user sees the list while you work. Do not use it for a single-step answer.',
    promptSnippet: 'tasks_update: keep the task list of multi-step work',
    parameters: Params,
    executionMode: 'sequential',
    async execute(_id: string, params: any) {
      const list = normalizeTasks(params)
      return { content: [{ type: 'text', text: summary(list) }], details: { tasks: list } }
    },
  })
}
