// Subagent mode: a child `pi` started by a subagent tool (`parallel` runs `pi --mode json`) gets
// a short prompt of its own instead of the main agent's SYSTEM.md (persona, memory cycle, router
// and delegation rules it does not need), may read memory but not write it, and ends its answer
// with the lessons it learned so the main agent can save them. It also gets the same process guard
// as the main agent (../jev-router/process-guard.ts).
//
// PI_SUBAGENT_LEAN=0 turns it off, to compare runs with and without it.
import { ariaConfig } from '../lib/config.ts'
import { turnLimitReached } from './limit.ts'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { checkKill } from '../jev-router/process-guard.ts'

const PROMPT_FILE = join(dirname(fileURLToPath(import.meta.url)), 'SUBAGENT.md')

// Engram tools that change memory; reading stays allowed.
const MEMORY_WRITES = new Set([
  'mem_save',
  'mem_update',
  'mem_judge',
  'mem_compare',
  'mem_session_summary',
  'mem_save_prompt',
  'mem_session_start',
  'mem_session_end',
  'mem_capture_passive',
  'mem_pin',
  'mem_unpin',
  'mem_review',
])

const isChild = (ctx: any) => ctx?.mode === 'json' && process.env.PI_SUBAGENT_LEAN !== '0'

export default function (pi: ExtensionAPI) {
  pi.on('before_agent_start', (event: any, ctx: any) => {
    if (!isChild(ctx)) return
    const options = event.systemPromptOptions
    if (!options) return
    try {
      // Takes the place SYSTEM.md has in the main agent; the expert (appended prompt), skills,
      // cwd and MCP sections are still added after it. Read on every run, so editing
      // SUBAGENT.md needs no reload.
      options.customPrompt = readFileSync(PROMPT_FILE, 'utf8')
    } catch {
      // keep the default prompt
    }
  })

  // A subagent that reached the turn limit gets no more tools: it has to deliver what it has.
  let turns = 0
  pi.on('turn_end', (_event: any, ctx: any) => {
    if (isChild(ctx)) turns += 1
  })

  pi.on('tool_call', (event: any, ctx: any) => {
    if (!isChild(ctx)) return
    const config = ariaConfig()
    if (turnLimitReached(turns, config)) {
      return { block: true, reason: `Stopped at the turn limit (${config.subagents.turnLimit}): no more tools. Write your report now with what you have.` }
    }
    // Same process guard as the main agent: a subagent may only stop its own or the project's processes.
    if (event.toolName === 'bash') {
      const verdict = checkKill(String(event.input?.command ?? ''), process.pid, ctx.cwd)
      if (verdict.block) return { block: true, reason: verdict.reason }
    }
    const name = String(event.toolName ?? '').replace(/^mcp__engram__/, '')
    if (String(event.toolName ?? '').startsWith('mcp__engram__') && MEMORY_WRITES.has(name)) {
      return {
        block: true,
        reason: 'You are a subagent: you do not write to memory. Put what is worth remembering in the "Learnings" section of your report.',
      }
    }
  })
}
