// Run: bun test extensions
import { afterEach, describe, expect, test } from 'bun:test'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { agentDir, projectsDir } from './agent-dir.ts'

const saved = { agent: process.env.PI_CODING_AGENT_DIR, projects: process.env.ARIA_PROJECTS_DIR }
afterEach(() => {
  for (const [key, value] of [['PI_CODING_AGENT_DIR', saved.agent], ['ARIA_PROJECTS_DIR', saved.projects]] as const) {
    if (value === undefined) delete process.env[key]
    else process.env[key] = value
  }
})

describe('where the extensions find their files', () => {
  test('agentDir uses PI_CODING_AGENT_DIR (Aria sets it to ~/.aria/agent)', () => {
    process.env.PI_CODING_AGENT_DIR = '/srv/aria/agent'
    expect(agentDir()).toBe('/srv/aria/agent')
  })

  test('agentDir falls back to ~/.pi/agent, so the same files run in the owner\'s own Pi', () => {
    delete process.env.PI_CODING_AGENT_DIR
    expect(agentDir()).toBe(join(homedir(), '.pi', 'agent'))
  })

  test('projectsDir uses ARIA_PROJECTS_DIR, and ~/Projects without it', () => {
    process.env.ARIA_PROJECTS_DIR = '/srv/code'
    expect(projectsDir()).toBe('/srv/code')
    delete process.env.ARIA_PROJECTS_DIR
    expect(projectsDir()).toBe(join(homedir(), 'Projects'))
  })

  test('an empty variable counts as not set', () => {
    process.env.PI_CODING_AGENT_DIR = ''
    expect(agentDir()).toBe(join(homedir(), '.pi', 'agent'))
  })
})
