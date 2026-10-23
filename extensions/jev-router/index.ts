// Jev router for Pi: before each user message, the free Jev classifier answers a few questions
// about it in one call (route, memory, web, Notion, read-only) and the answers reach the model as
// a hidden `[router]` message right after the user's, not shown in the TUI. The conversation stays
// on one model, so its prompt cache never breaks.
//
// In a read-only turn it also guards writes: `edit`/`write` outside the scratch dirs are blocked,
// suspicious `bash` commands are checked with Jev, and `parallel` subagents are told the turn is
// read-only.
//
// `/plan` starts plan mode, phase A of the slice skill (see modes.ts and slice.ts): routing pauses,
// a fixed hint is sent instead, and plan mode has its own write guard and model. Plan mode picks a
// free model when the Go quota is nearly spent, and ends when the owner approves the plan
// (`plan_ready`), with `/plan off`, or when the user sends something that is not planning. The
// approved plan, or `/execute`, builds its next open phase in the background.
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
import { execFileSync } from 'node:child_process'
import { readdirSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import {
  bashWriteRequest,
  classifierFor,
  classifierRequest,
  type Expert,
  expertRequest,
  expertsLine,
  type Hints,
  hintLine,
  methodNote,
  isScratchPath,
  lastAssistantText,
  mayWrite,
  parseExpertChoice,
  parseExpert,
  parseHints,
  parsePlanning,
  planContinuationRequest,
  READ_ONLY_NOTE, jevStatus } from './route.ts'
import { askApproval, pendingBuild } from './approve.ts'
import {
  executePrompt,
  gitRoot,
  LARGE_CONTEXT_TOKENS,
  type Mode,
  modeHint,
  PLAN_BLOCK_REASON,
  PLAN_DIRS_TEXT,
  planModel,
  planThinking,
  planMayWrite,
  planPrompt,
  planQuotaThreshold,
  REVIEW_SKILL,
  routedMode,
  shouldClassify,
  SLICE_SKILL,
} from './modes.ts'
import { buildBrief, firstOpenPhase, latestPlan, nextOpenPhase, nextStepHint, parseMarker, parsePlan, planProblem } from './slice.ts'
import { createTaskFeed } from './tasks-feed.ts'
import { checkKill } from './process-guard.ts'
import { exhausted, fetchGoUsage } from './usage.ts'
import { confirmFreeSwitch } from '../quota-fallback/ask.ts'
import { freeModel, isFree } from '../quota-fallback/fallback.ts'
import { ariaConfig, hasAriaConfig } from '../lib/config.ts'
import { styleNote } from './style.ts'
import { EFFORT_FILE, effortFor, RANK, saveEffort } from './effort.ts'
import { agentDir } from '../lib/agent-dir.ts'

const CLASSIFY_TIMEOUT_MS = 3000

function readFixedEffort(): string | null {
  try {
    const level = JSON.parse(readFileSync(EFFORT_FILE, 'utf8')).level
    return typeof level === 'string' ? level : null
  } catch {
    return null
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

/** The commit the repository of `path` is at now, or '' when git cannot say. */
function headOf(path: string): string {
  try {
    const dir = dirname(path)
    return execFileSync('git', ['-C', gitRoot(dir) ?? dir, 'rev-parse', 'HEAD'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim()
  } catch {
    return ''
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
  let planModelId: string = planModel().id
  // The plan the owner approved with `plan_ready`: plan mode ends and its build starts when that run settles.
  const build = pendingBuild()
  // The plan's marks of the phase that builds, published as the status `aria-tasks` (the dock's task dock, the terminal's strip).
  const taskFeed = createTaskFeed()
  const showTasks = (ctx: any, tasks: any[] | null) => {
    try {
      ctx.ui?.setStatus?.('aria-tasks', tasks ? JSON.stringify({ tasks }) : undefined)
    } catch {
      // no UI
    }
  }

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
    saveEffort(level)
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
  const applyThinking = (model: any, route?: Hints['route']) => {
    try {
      base ??= pi.getThinkingLevel()
      const level = effortFor({ route, free: isFree(model), base, fixed, config: ariaConfig() })
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
    // Under Aria the settings say it (aria.json, effortFor); the owner's own Pi keeps its effort file.
    fixed = hasAriaConfig() ? null : readFixedEffort()
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

  const classifier = (ctx: any) => {
    const c = classifierFor(ariaConfig().router.backend)
    return ctx.modelRegistry.findOfType('classifier', c.provider, c.id)
  }

  // Classify when a run starts from a user prompt and add the hints as a hidden message: it is
  // stored in the session and sent to the model after the prompt, but not shown in the TUI, and the
  // user's own message stays as typed. Steering messages join a running turn and keep its hints.
  pi.on('before_agent_start', async (event: any, ctx: any) => {
    // A slice job ended: say what comes next (ask the owner, review, the next phase). Plain background work is untouched.
    if (event.messageType === 'custom' && event.message?.customType === 'aria-background') {
      const job = event.message.details?.job
      const marker = job ? parseMarker(String(job.task ?? '')) : null
      if (!marker) return
      if (marker.kind === 'build') taskFeed.finish(String(job.id))
      let plan = ''
      try {
        plan = readFileSync(marker.plan, 'utf8')
      } catch {
        // gone: the hint says the phase did not finish
      }
      // A review with no phase left after it ends the plan: its list goes away. Otherwise it stays until the next phase's job replaces it.
      if (marker.kind === 'review' && plan && !nextOpenPhase(parsePlan(plan), marker.phase)) {
        taskFeed.stop()
        showTasks(ctx, null)
      }
      const hint = nextStepHint(job, plan, { skill: SLICE_SKILL, review: REVIEW_SKILL, head: () => headOf(marker.plan) })
      return hint ? { message: { customType: 'jev-router', content: hint, display: false, details: { slice: marker } } } : undefined
    }
    if (!routed(ctx) || event.messageType !== 'user') return
    const text = String(event.prompt ?? '')
    current = null
    if (mode !== 'normal') {
      return { message: { customType: 'jev-router', content: modeHint(), display: false, details: { mode } } }
    }
    if (!text.trim() || text.startsWith('/')) return
    if (!shouldClassify(ctx?.mode, ariaConfig())) return // the router is off in Aria's settings
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
      const content = [hintLine(hints), methodNote(hints), experts, styleNote(ariaConfig().style)].filter(Boolean).join('\n')
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

  // Where this turn may write, or null when writes are free: plan mode allows `docs/plans/`
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

  // --- Plan mode and the build ---------------------------------------------------------------------

  const setMode = (next: Mode, ctx: any) => {
    mode = next
    // Leaving plan mode (/plan off, /execute, a message that is not planning) cancels an approved build that has not started.
    if (next !== 'plan') build.cancel()
    const status = next === 'plan' ? `◆ plan mode · ${planModelId}` : undefined
    try {
      ctx.ui?.setStatus?.('jev-router-mode', status)
    } catch {
      // no UI
    }
  }

  // Leave plan mode: back to the model and thinking level the user had before /plan.
  const tell = (ctx: any, text: string, level: 'info' | 'error') => {
    try {
      ctx.ui?.notify?.(text, level)
    } catch {
      // no UI
    }
  }

  const endPlan = async (ctx: any, why: string) => {
    if (restore) {
      const { model, thinking } = restore
      restore = null
      if (model) await pi.setModel(model)
      setLevel(thinking)
    }
    setMode('normal', ctx)
    tell(ctx, why, 'info')
  }

  // The user answered an `ask_user` question: they are taking part in this turn, so it stops being
  // read-only (the agent asks precisely when a read-only turn blocked something it needs).
  pi.on('tool_result', (event: any, ctx: any) => {
    if (!routed(ctx) || event.toolName !== 'ask_user' || event.isError || event.details?.cancelled !== false) return
    if (mode === 'normal' && current?.readOnly) current = { ...current, readOnly: false }
  })

  // A slice phase starts building in the background: its plan file's marks become the task list while the job runs.
  pi.on('tool_result', (event: any, ctx: any) => {
    if (event.toolName !== 'background_agent' || event.isError) return
    const job = event.details?.job
    const marker = job ? parseMarker(String(job.task ?? '')) : null
    if (!job || marker?.kind !== 'build') return
    const plan = isAbsolute(marker.plan) ? marker.plan : resolve(ctx.cwd ?? process.cwd(), marker.plan)
    taskFeed.start({ id: String(job.id), number: Number(job.number) }, { plan, phase: marker.phase }, (tasks) => showTasks(ctx, tasks))
  })
  pi.on('session_shutdown', () => taskFeed.stop())

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
    let spent = usage ? exhausted(usage, planQuotaThreshold()) : null
    // Same account and quota as quota-fallback: one shared answer decides both.
    const PLAN_MODEL = planModel()
    const FREE_PLAN_MODEL = freeModel()
    if (spent && (await confirmFreeSwitch(ctx, { window: spent.name, ...spent.window }, PLAN_MODEL.id)) === 'stay') spent = null
    const pick = spent ? FREE_PLAN_MODEL : PLAN_MODEL
    const note = spent
      ? `Go ${spent.name} quota at ${spent.window.percent}%: planning with ${FREE_PLAN_MODEL.id} (free) instead of ${PLAN_MODEL.id}.`
      : null
    return { model: ctx.modelRegistry.find(pick.provider, pick.id), note }
  }

  // `/plan` and `/execute` exist unless Aria's settings turn them off (subagents.planExecute).
  const planExecute = ariaConfig().subagents.planExecute
  if (planExecute) pi.registerCommand('plan', {
    description:
      'Plan sizable work with slice: one round of questions, then a plan of phases and tasks to approve. /plan off to leave.',
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
      await startPlan(idea, ctx)
    },
  })

  // The assistant offered to plan the work first (the router's `plan=offer`) and the user accepted: plan mode starts
  // the way /plan does, once this answer has ended.
  let pendingPlan: string | null = null
  if (planExecute) pi.registerTool({
    name: 'start_plan',
    label: 'Plan first',
    description:
      'Start plan mode (slice) for `idea` once your answer ends, as /plan does. Only after the user accepted your offer to plan the work first.',
    parameters: { type: 'object', properties: { idea: { type: 'string', description: 'What to plan, in one or two sentences, in the user\'s words.' } }, required: ['idea'] },
    async execute(_id: string, params: any) {
      const idea = String(params?.idea ?? '').trim()
      if (!idea) return { content: [{ type: 'text', text: 'start_plan needs the idea to plan.' }], isError: true }
      if (mode !== 'normal') return { content: [{ type: 'text', text: `Already in ${mode} mode.` }], isError: true }
      pendingPlan = idea
      return { content: [{ type: 'text', text: 'Plan mode starts as soon as this answer ends: end it now with one short line saying so.' }] }
    },
  } as any)

  // The owner approves a plan the model wrote and committed: plan mode ends and the build starts once the answer has ended.
  if (planExecute) pi.registerTool({
    name: 'plan_ready',
    label: 'Plan ready',
    description:
      'Show the user the slice plan you wrote and committed (docs/plans/…) and ask them to approve it. Only in plan mode. Approved, the build starts once your answer ends.',
    parameters: { type: 'object', properties: { path: { type: 'string', description: 'The plan file.' } }, required: ['path'] },
    async execute(_id: string, params: any, signal: any, _onUpdate: any, ctx: any) {
      const r = await askApproval(ctx, String(params?.path ?? ''), ctx.cwd, { mode, noAsk: !!ariaConfig().noAsk, signal })
      if (r.approved) build.queue(r.approved)
      return { content: [{ type: 'text', text: r.text }], details: r.details, isError: !!r.isError }
    },
  } as any)

  pi.on('agent_settled', async (_event: any, ctx: any) => {
    const plan = routed(ctx) ? build.take() : null
    if (plan) {
      await endPlan(ctx, 'Plan approved: plan mode off.')
      startBuild(plan, ctx)
      return
    }
    if (!routed(ctx) || !pendingPlan) return
    const idea = pendingPlan
    pendingPlan = null
    if (mode === 'normal') await startPlan(idea, ctx)
  })

  /** Plan mode for `idea`: the planning model, then the plan prompt (from /plan, or the accepted offer). */
  async function startPlan(idea: string, ctx: any) {
    // Planning re-reads the whole session on a pricier model: in a long session that is costly.
    const used = ctx.getContextUsage?.()?.tokens
    if (mode !== 'plan' && typeof used === 'number' && used > LARGE_CONTEXT_TOKENS && ctx.hasUI && !ariaConfig().noAsk) {
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
    setLevel(isFree(model) ? 'max' : planThinking())
    setMode('plan', ctx)
    if (note) ctx.ui.notify(note, 'warning')
    pi.sendUserMessage(planPrompt(idea))
  }

  /** Build the first open phase of `plan` in the background: the model starts the job with the brief. */
  function startBuild(plan: string, ctx: any) {
    let parsed
    try {
      parsed = parsePlan(readFileSync(plan, 'utf8'))
    } catch {
      tell(ctx, `Can't read \`${plan}\`.`, 'error')
      return
    }
    const phase = firstOpenPhase(parsed)
    if (!phase) {
      tell(ctx, `Every phase of ${plan} is done.`, 'info')
      return
    }
    const brief = buildBrief({ plan, phase: phase.number, name: phase.name, base: headOf(plan), skill: SLICE_SKILL })
    pi.sendUserMessage(executePrompt(brief, phase.name))
  }

  if (planExecute) pi.registerCommand('execute', {
    description: 'Build the next open phase of a slice plan in the background (docs/plans/…).',
    handler: async (args: string, ctx: any) => {
      const arg = String(args ?? '').trim()
      const path = arg || latestPlan(ctx.cwd)
      const problem = path ? planProblem(path, ctx.cwd) : 'No plan found in docs/plans/.'
      if (!path || problem) {
        ctx.ui.notify(`${problem} Usage: /execute [plan path]`, 'error')
        return
      }
      if (mode === 'plan') await endPlan(ctx, 'Plan mode off.')
      startBuild(isAbsolute(path) ? path : resolve(ctx.cwd, path), ctx)
    },
  })
}
