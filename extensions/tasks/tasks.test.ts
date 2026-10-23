import { describe, expect, test } from 'bun:test'
import { normalizeTasks, MAX_TASKS, summary, toolsFor } from './tasks.ts'

describe('the task list', () => {
  test('the list is validated and capped', () => {
    const many = Array.from({ length: 30 }, (_, n) => ({ id: `t${n}`, title: `step ${n}`, status: 'todo' }))
    expect(normalizeTasks({ tasks: many })).toHaveLength(MAX_TASKS)
    expect(MAX_TASKS).toBe(20)
    const odd = normalizeTasks({ tasks: [
      { id: 'a', title: '  Write the test  ', status: 'active', agent: 'backend' },
      { id: '', title: 'no id', status: 'done' },
      { title: 'x', status: 'weird' },
      { id: 'b', title: '', status: 'todo' },
      'nonsense',
    ] })
    expect(odd).toEqual([
      { id: 'a', title: 'Write the test', status: 'active', agent: 'backend' },
      { id: 'task-2', title: 'no id', status: 'done' },
      { id: 'task-3', title: 'x', status: 'todo' },
    ])
  })

  test('the result says the progress in one line', () => {
    expect(summary([{ id: 'a', title: 'A', status: 'done' }, { id: 'b', title: 'B', status: 'active' }, { id: 'c', title: 'C', status: 'todo' }])).toBe('1/3 done; now: B')
    expect(summary([])).toBe('no tasks')
  })
})

describe('the tool', () => {
  test('subagents do not get the tool', () => {
    expect(toolsFor(['read', 'tasks_update'], 'json')).toEqual(['read'])
    expect(toolsFor(['read', 'tasks_update'], 'rpc')).toEqual(['read', 'tasks_update'])
    expect(toolsFor(['read', 'tasks_update'], 'tui')).toEqual(['read', 'tasks_update'])
  })
})
