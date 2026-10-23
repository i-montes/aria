// The task list of multi-step work, as the model keeps it with `tasks_update`: the whole list each time. Aria's terminal
// reads the last one from the conversation (its result's details) and draws it above the editor.

export type Status = 'todo' | 'active' | 'done'
export interface Task {
  id: string
  title: string
  status: Status
  agent?: string
}

export const MAX_TASKS = 20
const STATUSES: Status[] = ['todo', 'active', 'done']

/** The tasks as given, cleaned: a title is required, ids are filled in, an unknown status is `todo`, at most 20. */
export function normalizeTasks(params: any): Task[] {
  const list = Array.isArray(params?.tasks) ? params.tasks : []
  const tasks: Task[] = []
  list.forEach((raw: any, at: number) => {
    if (!raw || typeof raw !== 'object') return
    const title = String(raw.title ?? '').trim()
    if (!title) return
    const id = String(raw.id ?? '').trim() || `task-${at + 1}`
    const status: Status = STATUSES.includes(raw.status) ? raw.status : 'todo'
    const agent = typeof raw.agent === 'string' && raw.agent.trim() ? raw.agent.trim() : undefined
    tasks.push(agent ? { id, title, status, agent } : { id, title, status })
  })
  return tasks.slice(0, MAX_TASKS)
}

/** One line for the model: how far it is and what runs now. */
export function summary(tasks: Task[]): string {
  if (!tasks.length) return 'no tasks'
  const done = tasks.filter((t) => t.status === 'done').length
  const active = tasks.filter((t) => t.status === 'active').map((t) => t.title)
  return `${done}/${tasks.length} done` + (active.length ? `; now: ${active.join(', ')}` : '')
}

/** The active tools for a Pi in `mode`: a subagent (`json`) has nobody to show a task list to. */
export const toolsFor = (active: string[], mode: string): string[] =>
  mode === 'json' ? active.filter((name) => name !== 'tasks_update') : active
