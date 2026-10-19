// Jev router for Pi: before each user message, the free Jev classifier answers a few questions
// about it in one call (route, memory, web, Notion, read-only) and the answers reach the model as
// a hidden `[router]` message right after the user's, not shown in the TUI. The conversation stays
// on one model, so its prompt cache never breaks.
//
// In a read-only turn it also guards writes: `edit`/`write` outside the scratch dirs are blocked,
// suspicious `bash` commands are checked with Jev, and `parallel` subagents are told the turn is
// read-only.
//
// `/plan` and `/execute` switch to Superpowers modes (see modes.ts): routing pauses, a fixed hint
// is sent instead, and plan mode has its own write guard and model. Plan mode picks a free model
// when the Go quota is nearly spent, and ends by itself once the plan is written or when the user
// sends something that is not planning.
//
// Every `bash` call also goes through the process guard (process-guard.ts): the agent may only
// stop processes of its own session or project.
//
// Reasoning effort: in auto it follows the route on paid models and a free model runs at the
// highest level it supports, since thinking costs nothing there; a level the user picks by hand
// stays fixed, across sessions, until `/effort auto`.
//
// Only the main agent is routed (TUI and print modes); `parallel` subagents run in JSON mode and are
// left alone. Any failure (no classifier, no credential, a slow answer) sends the message unlabelled
// and leaves the guard off.
import { existsSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { isAbsolute, join, resolve } from 'node:path'
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import {
  bashWriteRequest,
  CLASSIFIER,
  classifierRequest,
  type Expert,
  expertRequest,
  expertsLine,
  type Hints,
  hintLine,
  isScratchPath,
  lastAssistantText,
  mayWrite,
  parseExpertChoice,
  parseExpert,
  parseHints,
  parsePlanning,
  planContinuationRequest,
  READ_ONLY_NOTE, jevStatus } from './route.ts'
import {
  executePrompt,
  FREE_PLAN_MODEL,
  gitRoot,
  isPlanPath,
  LARGE_CONTEXT_TOKENS,
  type Mode,
  modeHint,
  notAPlan,
  PLAN_BLOCK_REASON,
  PLAN_DIRS_TEXT,
  PLAN_MODEL,
  PLAN_THINKING,
  planMayWrite,
  planPrompt,
  QUOTA_THRESHOLD,
  routedMode,
} from './modes.ts'
import { checkKill } from './process-guard.ts'
import { exhausted, fetchGoUsage } from './usage.ts'
import { confirmFreeSwitch } from '../quota-fallback/ask.ts'
import { isFree } from '../quota-fallback/fallback.ts'
import { agentDir } from '../lib/agent-dir.ts'

const CLASSIFY_TIMEOUT_MS = 3000
// Repeated every turn, right after the user's message: a rule buried in a long system prompt gets
// ignored by smaller models (the user wants short, direct answers).
const STYLE_NOTE =
  'Answer style: very short and direct, the answer first, at most ~80 words. No tables, no headings, no closing summary ("en resumen…") or offer. Longer only if the user asked for detail.'
// The fixed reasoning effort the user picked, shared by every session until `/effort auto`.
const EFFORT_FILE = join(agentDir(), 'jev-router-effort.json')

function readFixedEffort(): string | null {
  try {
    const level = JSON.parse(readFileSync(EFFORT_FILE, 'utf8')).level
    return typeof level === 'string' ? level : null
  } catch {
    return null
  }
}

function writeFixedEffort(level: string | null) {
  try {
    if (level) writeFileSync(EFFORT_FILE, JSON.stringify({ level }) + '\n')
    else rmSync(EFFORT_FILE, { force: true })
  } catch {
    // stays for this session only
  }
}
const TMP_DIRS = [...new Set(['/tmp', tmpdir()])]
const EXPERTS_DIR = join(agentDir(), 'agents')

// Read on every use, so adding or editing an expert file needs no reload.
function loadExperts(): Expert[] {
  try {
    return readdirSync(EXPERTS_DIR)
      .filter((f) => f.endsWith('.md'))
      .flatMap((f) => {
        const expert = parseExpert(readFileSync(join(EXPERTS_DIR, f), 'utf8'))
        return expert ? [expert] : []
      })
  } catch {
    return []
  }
}

const BLOCK_REASON =
  'Blocked by the router: this turn is read-only and this would write outside /tmp. Put temporary files in /tmp; if files really have to change, ask the user with `ask_user`: once they answer, the turn is no longer read-only.'

export default function (pi: ExtensionAPI) {
  // Hints of the current turn; null when the last message went unlabelled.
  let current: Hints | null = null
  let mode: Mode = 'normal'
  // Model and thinking level to go back to when plan mode ends.
  let restore: { model: any; thinking: string } | null = null
  // The model plan mode is using (GLM, or the free one when the quota is nearly spent).
  let planModelId: string = PLAN_MODEL.id
  // Set when plan mode writes a plan file; plan mode ends when that run settles.
  let planWritten = false

  const routed = (ctx: any) => routedMode(ctx?.mode)

  // Reasoning effort has two modes. Auto (the default): a quick turn drops to low and a think turn
  // goes at least to high from `base`, the level the session started with, and a free model is
  // pinned to its top level (`max`, which Pi clamps to what the model supports). Fixed: a level the
  // user picks by hand (/thinking, shift+tab or /effort <level>) stays put, also in later
  // sessions (EFFORT_FILE), until `/effort auto`. Our own changes are told apart by `ours` and
  // `lastSet`; Pi re-clamping the level when the model changes is told apart by the model key.
  let base: string | null = null
  let lastSet: string | null = null
  let ours = false
  let fixed: string | null = null
  let modelKey: string | null = null
  const keyOf = (m: any) => (m ? `${m.provider}/${m.id}` : null)
  const showEffort = (ctx: any) => {
    try {
      ctx?.ui?.setStatus?.('jev-router-effort', fixed ? `effort fixed: ${fixed} · /effort auto` : 'effort auto')
    } catch {
      // no UI
    }
  }
  const fix = (level: string | null, ctx: any) => {
    fixed = level
    writeFixedEffort(level)
    showEffort(ctx)
  }
  pi.on('thinking_level_select', (event: any, ctx: any) => {
    if (!routed(ctx) || ours || event.level === lastSet || mode !== 'normal') return
    if (keyOf(ctx.model) !== modelKey) return // Pi carrying the level over to a new model
    fix(event.level, ctx)
  })
  const setLevel = (level: string) => {
    ours = true
    try {
      pi.setThinkingLevel(level as any)
    } finally {
      ours = false
    }
    lastSet = pi.getThinkingLevel()
  }
  const RANK = ['off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max']
  const levelFor = (route: Hints['route'], b: string) => {
    if (route === 'quick') return 'low'
    if (route === 'think') return RANK.indexOf(b) >= RANK.indexOf('high') ? b : 'high'
    return b
  }
  const applyThinking = (model: any, route?: Hints['route']) => {
    try {
      base ??= pi.getThinkingLevel()
      const level = fixed ?? (isFree(model) ? 'max' : route ? levelFor(route, base) : base)
      if (pi.getThinkingLevel() !== level) setLevel(level)
    } catch {
      // keep whatever level is set
    }
  }

  // A new session or a model switch (by hand or by quota-fallback) applies the mode right away.
  // Plan mode sets its own level.
  pi.on('session_start', (_event: any, ctx: any) => {
    if (!routed(ctx)) return
    modelKey = keyOf(ctx.model)
    base = pi.getThinkingLevel()
    fixed = readFixedEffort()
    showEffort(ctx)
    if (mode === 'normal') applyThinking(ctx.model)
  })
  pi.on('model_select', (event: any, ctx: any) => {
    if (!routed(ctx)) return
    modelKey = keyOf(event.model)
    if (mode === 'normal') applyThinking(event.model, current?.route)
  })

  pi.registerCommand('effort', {
    description: 'Reasoning effort: auto (set by the route; top level on free models) or a fixed level kept across sessions.',
    getArgumentCompletions: (prefix: string) =>
      ['auto', ...RANK].filter((o) => o.startsWith(prefix.trim().toLowerCase())).map((o) => ({ value: o, label: o })),
    handler: async (args: string, ctx: any) => {
      let pick: string | undefined = args.trim().toLowerCase()
      if (!pick) {
        const now = fixed ? `fixed at ${fixed}` : 'auto'
        pick = await ctx.ui.select(`Reasoning effort (now: ${now})`, ['auto', ...RANK])
        if (!pick) return
      }
      if (pick === 'auto') {
        fix(null, ctx)
        if (mode === 'normal') applyThinking(ctx.model, current?.route)
        ctx.ui.notify(`Effort is auto: ${pi.getThinkingLevel()} now, it follows the route.`, 'info')
        return
      }
      if (!RANK.includes(pick)) {
        ctx.ui.notify(`Unknown level "${pick}". Use auto or ${RANK.join(', ')}.`, 'error')
        return
      }
      fix(pick, ctx)
      if (mode === 'normal') setLevel(pick)
      ctx.ui.notify(`Effort fixed at ${pick} (this model uses ${pi.getThinkingLevel()}). /effort auto to go back.`, 'info')
    },
  })

  const classifier = (ctx: any) => ctx.modelRegistry.findOfType('classifier', CLASSIFIER.provider, CLASSIFIER.id)

  // Classify when a run starts from a user prompt and add the hints as a hidden message: it is
  // stored in the session and sent to the model after the prompt, but not shown in the TUI, and the
  // user's own message stays as typed. Steering messages join a running turn and keep its hints.
  pi.on('before_agent_start', async (event: any, ctx: any) => {
    if (!routed(ctx) || event.messageType !== 'user') return
    const text = String(event.prompt ?? '')
    current = null
    if (mode !== 'normal') {
      return { message: { customType: 'jev-router', content: modeHint(mode), display: false, details: { mode } } }
    }
    if (!text.trim() || text.startsWith('/')) return
    try {
      const jev = classifier(ctx)
      if (!jev) return
      let previous = ''
      try {
        previous = lastAssistantText(ctx.sessionManager.getBranch())
      } catch {
        // no session yet
      }
      const result = await ctx.modelRegistry.classify(jev, classifierRequest(text, previous), {
        signal: AbortSignal.timeout(CLASSIFY_TIMEOUT_MS),
      })
      const hints = parseHints(result)
      if (!hints) return
      current = hints
      applyThinking(ctx.model, hints.route)
      const experts =
        hints.route === 'parallel' ? expertsLine(loadExperts().filter((e) => !hints.readOnly || e.readOnly)) : ''
      // Aria's dock shows the route as a line before the answer.
      if (ctx.mode === 'rpc') ctx.ui?.setStatus?.('aria-jev', JSON.stringify(jevStatus(hints)))
      const content = [hintLine(hints), experts, STYLE_NOTE].filter(Boolean).join('\n')
      return { message: { customType: 'jev-router', content, display: false, details: hints } }
    } catch {
      return
    } finally {
      // An unlabelled turn goes back to the base level instead of keeping the last route's.
      if (!current) applyThinking(ctx.model)
    }
  })

  // Give each `parallel` task without an expert the one Jev picks for it (all tasks at once).
  // In a read-only turn only experts that cannot edit files are allowed: a task given a writing
  // expert is reassigned among the read-only ones.
  async function assignExperts(input: any, ctx: any, readOnly: boolean) {
    const all = loadExperts()
    const experts = readOnly ? all.filter((e) => e.readOnly) : all
    const allowed = new Set(experts.map((e) => e.name))
    const items: any[] = [
      ...(typeof input.task === 'string' ? [input] : []),
      ...(Array.isArray(input.tasks) ? input.tasks : []),
      ...(Array.isArray(input.chain) ? input.chain : []),
    ].filter((item) => item && typeof item.task === 'string')
    for (const item of items) {
      if (item.agent && all.some((e) => e.name === item.agent) && !allowed.has(item.agent)) delete item.agent
    }
    const pending = items.filter((item) => !item.agent)
    const jev = experts.length && pending.length ? classifier(ctx) : null
    if (!jev) return
    await Promise.allSettled(
      pending.map(async (item) => {
        const result = await ctx.modelRegistry.classify(jev, expertRequest(item.task, experts), {
          signal: AbortSignal.timeout(CLASSIFY_TIMEOUT_MS),
        })
        const name = parseExpertChoice(result)
        if (name) item.agent = name
      }),
    )
  }

  // Where this turn may write, or null when writes are free: plan mode allows `docs/superpowers/`
  // and `.git` of a repo under ~/Projects plus /tmp; a read-only turn only the scratch dirs.
  type Guard = { allows: (path: string) => boolean; dirsText: string[]; reason: string }
  const writeGuard = (ctx: any): Guard | null => {
    if (mode === 'plan') return { allows: (p) => planMayWrite(p, ctx.cwd), dirsText: PLAN_DIRS_TEXT, reason: PLAN_BLOCK_REASON }
    if (mode === 'normal' && current?.readOnly) return { allows: (p) => isScratchPath(p, TMP_DIRS), dirsText: TMP_DIRS, reason: BLOCK_REASON }
    return null
  }

  pi.on('tool_call', async (event: any, ctx: any) => {
    if (!routed(ctx)) return
    if (event.toolName === 'bash') {
      const verdict = checkKill(String(event.input?.command ?? ''), process.pid, ctx.cwd)
      if (verdict.block) return { block: true, reason: verdict.reason }
    }
    const guard = writeGuard(ctx)
    if (event.toolName === 'parallel') {
      try {
        await assignExperts(event.input ?? {}, ctx, !!guard)
      } catch {
        // tasks keep running without an expert
      }
    }
    if (!guard) return
    const BLOCK = { block: true, reason: guard.reason }

    if (event.toolName === 'edit' || event.toolName === 'write') {
      const raw = String(event.input?.path ?? '')
      const path = isAbsolute(raw) ? raw : resolve(ctx.cwd, raw)
      if (!guard.allows(path)) return BLOCK
      return
    }

    if (event.toolName === 'bash') {
      const command = String(event.input?.command ?? '')
      if (!mayWrite(command)) return
      try {
        const jev = classifier(ctx)
        if (!jev) return BLOCK
        const result = await ctx.modelRegistry.classify(jev, bashWriteRequest(command, ctx.cwd, guard.dirsText), {
          signal: AbortSignal.timeout(CLASSIFY_TIMEOUT_MS),
        })
        const p = result?.stopReason === 'stop' ? result.answers?.writes?.probability : undefined
        if (typeof p !== 'number' || p >= 0.5) return BLOCK
      } catch {
        return BLOCK
      }
      return
    }

    if (event.toolName === 'parallel') {
      const note = (task: unknown) => (typeof task === 'string' && !task.includes(READ_ONLY_NOTE) ? `${READ_ONLY_NOTE}\n\n${task}` : task)
      const input = event.input ?? {}
      if (typeof input.task === 'string') input.task = note(input.task)
      for (const list of [input.tasks, input.chain, input.team?.tasks]) {
        if (Array.isArray(list)) for (const item of list) if (item && typeof item === 'object') item.task = note(item.task)
      }
      if (input.race && typeof input.race === 'object') input.race.task = note(input.race.task)
    }
  })

  // --- Superpowers modes ---------------------------------------------------------------------

  const setMode = (next: Mode, ctx: any) => {
    mode = next
    if (next !== 'plan') planWritten = false
    const status = next === 'plan' ? `◆ plan mode · ${planModelId}` : next === 'execute' ? '▶ execute mode · /execute off' : undefined
    try {
      ctx.ui?.setStatus?.('jev-router-mode', status)
    } catch {
      // no UI
    }
  }

  // Leave plan mode: back to the model and thinking level the user had before /plan.
  const leavePlan = async () => {
    if (!restore) return
    const { model, thinking } = restore
    restore = null
    if (model) await pi.setModel(model)
    setLevel(thinking)
  }

  const endPlan = async (ctx: any, why: string) => {
    await leavePlan()
    setMode('normal', ctx)
    try {
      ctx.ui?.notify?.(why, 'info')
    } catch {
      // no UI
    }
  }

  // The user answered an `ask_user` question: they are taking part in this turn, so it stops being
  // read-only (the agent asks precisely when a read-only turn blocked something it needs).
  pi.on('tool_result', (event: any, ctx: any) => {
    if (!routed(ctx) || event.toolName !== 'ask_user' || event.isError || event.details?.cancelled !== false) return
    if (mode === 'normal' && current?.readOnly) current = { ...current, readOnly: false }
  })

  // A plan file written in plan mode: planning is over once this run settles.
  pi.on('tool_result', (event: any, ctx: any) => {
    if (!routed(ctx) || mode !== 'plan' || event.isError) return
    if (event.toolName !== 'write' && event.toolName !== 'edit') return
    const raw = String(event.input?.path ?? '')
    const path = isAbsolute(raw) ? raw : resolve(ctx.cwd, raw)
    if (isPlanPath(path)) planWritten = true
  })

  pi.on('agent_settled', async (_event: any, ctx: any) => {
    if (routed(ctx) && mode === 'plan' && planWritten) {
      await endPlan(ctx, 'Plan written: plan mode off. Review it and run it with /execute <plan path>.')
    }
  })

  // In plan mode, a message that is not part of planning ends it before the turn starts, so that
  // turn runs on the user's model with normal routing.
  pi.on('input', async (event: any, ctx: any) => {
    if (!routed(ctx) || mode !== 'plan' || event.source === 'extension' || event.streamingBehavior) return { action: 'continue' }
    const text = String(event.text ?? '')
    if (!text.trim() || text.startsWith('/')) return { action: 'continue' }
    try {
      const jev = classifier(ctx)
      if (!jev) return { action: 'continue' }
      let previous = ''
      try {
        previous = lastAssistantText(ctx.sessionManager.getBranch())
      } catch {
        // no session yet
      }
      const result = await ctx.modelRegistry.classify(jev, planContinuationRequest(text, previous), {
        signal: AbortSignal.timeout(CLASSIFY_TIMEOUT_MS),
      })
      const p = parsePlanning(result)
      if (p !== null && p < 0.3) await endPlan(ctx, 'That message is not about planning: plan mode off.')
    } catch {
      // stay in plan mode
    }
    return { action: 'continue' }
  })

  /** The plan model for now: GLM, or the free one when the Go quota is at the threshold. */
  async function choosePlanModel(ctx: any): Promise<{ model: any; note: string | null }> {
    const usage = await fetchGoUsage()
    let spent = usage ? exhausted(usage, QUOTA_THRESHOLD) : null
    // Same account and quota as quota-fallback: one shared answer decides both.
    if (spent && (await confirmFreeSwitch(ctx, { window: spent.name, ...spent.window }, PLAN_MODEL.id)) === 'stay') spent = null
    const pick = spent ? FREE_PLAN_MODEL : PLAN_MODEL
    const note = spent
      ? `Go ${spent.name} quota at ${spent.window.percent}%: planning with ${FREE_PLAN_MODEL.id} (free) instead of ${PLAN_MODEL.id}.`
      : null
    return { model: ctx.modelRegistry.find(pick.provider, pick.id), note }
  }

  pi.registerCommand('plan', {
    description:
      'Design and plan with Superpowers (brainstorming → writing-plans) on a stronger model. Turns itself off once the plan is written; /plan off to leave earlier.',
    handler: async (args: string, ctx: any) => {
      const idea = String(args ?? '').trim()
      if (idea === 'off') {
        await endPlan(ctx, 'Plan mode off.')
        return
      }
      if (!idea) {
        ctx.ui.notify('Usage: /plan <idea>  ·  /plan off', 'info')
        return
      }
      // Planning re-reads the whole session on a pricier model: in a long session that is costly.
      const used = ctx.getContextUsage?.()?.tokens
      if (mode !== 'plan' && typeof used === 'number' && used > LARGE_CONTEXT_TOKENS && ctx.hasUI) {
        const k = Math.round(used / 1000)
        const go = await ctx.ui.confirm(
          `This session is already at ~${k}k tokens`,
          'The planning model re-reads them on every call and spends quota. Better to open a new session (/new) or compact (/compact) first. Plan here anyway?',
        )
        if (!go) return
      }
      const { model, note } = await choosePlanModel(ctx)
      if (mode !== 'plan') restore = { model: ctx.model, thinking: pi.getThinkingLevel() }
      if (!model || !(await pi.setModel(model))) {
        restore = null
        ctx.ui.notify('Could not switch to the planning model.', 'error')
        return
      }
      planModelId = model.id
      setLevel(isFree(model) ? 'max' : PLAN_THINKING)
      setMode('plan', ctx)
      if (note) ctx.ui.notify(note, 'warning')
      pi.sendUserMessage(planPrompt(idea))
    },
  })

  /** The newest plan under docs/superpowers/plans of the repo Pi runs in, or null. */
  function latestPlan(cwd: string): string | null {
    const root = gitRoot(cwd) ?? cwd
    const dir = join(root, 'docs', 'superpowers', 'plans')
    if (!existsSync(dir)) return null
    const files = readdirSync(dir)
      .filter((f) => f.endsWith('.md'))
      .map((f) => join(dir, f))
      .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs)
    return files[0] ?? null
  }

  pi.registerCommand('execute', {
    description:
      'Execute a Superpowers plan with subagents (subagent-driven-development). Takes a plan, not a spec. /execute off to leave.',
    handler: async (args: string, ctx: any) => {
      const arg = String(args ?? '').trim()
      if (arg === 'off') {
        await leavePlan()
        setMode('normal', ctx)
        ctx.ui.notify('Execute mode off.', 'info')
        return
      }
      const plan = arg || latestPlan(ctx.cwd)
      if (!plan) {
        ctx.ui.notify('No plan found in docs/superpowers/plans/. Give me the path: /execute <plan path>.', 'error')
        return
      }
      const problem = notAPlan(plan, ctx.cwd)
      if (problem) {
        ctx.ui.notify(`${problem} /execute needs the implementation plan: write it first (writing-plans) and review it.`, 'error')
        return
      }
      await leavePlan()
      setMode('execute', ctx)
      pi.sendUserMessage(executePrompt(plan))
    },
  })
}
