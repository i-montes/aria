// What the router asks Jev about each new user message, and how the answers become a hint line the
// conversation model follows. The conversation stays on one model; only the hint changes.

// Only the free Jev: the same OpenCode key also reaches the paid one.
export const CLASSIFIER = { provider: 'opencode', id: 'jev-1.13-free' } as const

export type Route = 'quick' | 'focused' | 'parallel' | 'think'

export const ROUTE_CRITERIA: Record<Route, string> = {
  quick:
    'Small talk; telling the assistant a fact or preference; recalling something the user said before (their name, favourites, earlier topics); a question answered from what the assistant knows; one short lookup or action (show a path, list a folder, run one command, open something). Few or no tool calls.',
  focused:
    'One coherent piece of work done step by step: fix a bug, change a config, write or edit a file, build one feature, look something up in one source (one web search, one Notion query).',
  parallel:
    'Broad work that splits into independent parts: analyse or audit a whole project or repository, review several aspects or many files, compare several sources or options by researching each one. The message often lists several goals.',
  think:
    'Asks the assistant to reason more than to do or recall: give an opinion, compare options from knowledge, advise on a design, explain something hard, help make a decision. Not for recalling facts or preferences. No investigation of files needed.',
}

const ROUTE_LABEL: Record<Route, string> = {
  quick: 'quick',
  focused: 'focused',
  parallel: 'parallel',
  think: 'think',
}

const BOOL_QUESTIONS = {
  memory: {
    instructions:
      'Does `message` involve the user\'s personal memory: something said in earlier conversations ("remember", "last time", "my favourite…", "what do you know about me"), or a personal fact, preference, person or ongoing project the user mentions or tells the assistant to keep?',
    criteria: { true: 'Earlier conversations may matter', false: 'Self-contained, nothing from the past needed' },
  },
  web: {
    instructions: 'Does answering `message` need current information from the internet (news, docs, versions, prices, searching the web)?',
    criteria: { true: 'Needs the web', false: 'No web needed' },
  },
  notion: {
    instructions: "Does `message` concern the user's Notion workspace? The user keeps their tasks, to-dos, deadlines and work logs in Notion, so questions about their tasks count even without the word Notion.",
    criteria: { true: 'Involves Notion', false: 'Does not involve Notion' },
  },
  sizable: {
    instructions:
      'Does `message` ask to build or change something sizable in a project: a new feature, a redesign, a migration, a change across several parts of a system, where agreeing on a plan before writing code pays off? A single fix, a small edit, a config change, a question, an explanation or an analysis is NOT sizable.',
    criteria: { true: 'Sizable work worth planning first', false: 'Small or not a build' },
  },
  changes: {
    instructions:
      'Does `message` ask the assistant to create, modify, delete, install or run something that changes files or the system? Analysing, reviewing, auditing, explaining, reading, searching or researching how something could be done is NOT a change. Use `previous_reply` for short follow-ups such as "yes, do it".',
    criteria: { true: 'Asks for changes', false: 'Read-only: look, analyse or answer' },
  },
} as const

export type BoolKey = keyof typeof BOOL_QUESTIONS

export interface Hints {
  route: Route
  routeP: number
  memory: boolean
  web: boolean
  notion: boolean
  readOnly: boolean
  /** Sizable work that changes things: the assistant offers to plan it first (`plan=offer`). */
  planWorth: boolean
}

export function classifierRequest(message: string, previousReply: string) {
  const questions: Record<string, unknown> = {
    route: {
      type: 'choice',
      instructions:
        'What kind of work does `message` ask a personal assistant for? `previous_reply` is the assistant\'s last answer: use it when `message` is a short follow-up such as "yes, do it".',
      criteria: ROUTE_CRITERIA,
    },
  }
  for (const [key, q] of Object.entries(BOOL_QUESTIONS)) questions[key] = { type: 'bool', ...q }
  return {
    state: { message: message.slice(0, 8000), previous_reply: previousReply.slice(-2000) },
    questions,
  }
}

// A bool is "yes" above this probability. Read-only needs `changes` to be clearly unlikely.
const YES = 0.5
const READ_ONLY_BELOW = 0.3
// Offering a plan interrupts: only when Jev is fairly sure the work is sizable.
const PLAN_WORTH = 0.6

// A Jev result -> hints, or null when it failed or answered something unexpected.
export function parseHints(result: any): Hints | null {
  if (result?.stopReason !== 'stop') return null
  const a = result.answers ?? {}
  const route = a.route?.choice
  if (a.route?.type !== 'choice' || !(route in ROUTE_CRITERIA)) return null
  const p = (key: BoolKey) => (a[key]?.type === 'bool' ? Number(a[key].probability) : NaN)
  const changes = p('changes')
  const readOnly = Number.isFinite(changes) && changes < READ_ONLY_BELOW
  return {
    route,
    routeP: Number(a.route.probabilities?.[route] ?? 0),
    memory: p('memory') >= YES,
    web: p('web') >= YES,
    notion: p('notion') >= YES,
    readOnly,
    planWorth: !readOnly && (route === 'focused' || route === 'parallel') && p('sizable') >= PLAN_WORTH,
  }
}

export const HINT_PREFIX = '[router]'

export function hintLine(h: Hints): string {
  const yn = (b: boolean) => (b ? 'yes' : 'no')
  return [
    `${HINT_PREFIX} route=${ROUTE_LABEL[h.route]} (${h.routeP.toFixed(2)})`,
    `memory=${yn(h.memory)}`,
    `web=${yn(h.web)}`,
    `notion=${yn(h.notion)}`,
    `changes=${h.readOnly ? 'no (read-only)' : 'yes'}`,
    ...(h.planWorth ? ['plan=offer'] : []),
  ].join(' · ')
}

/** For focused work that changes things (and is not sizable, which gets the plan offer): start from the outcome and
 * the check that proves it. Never names a method: that is the router's business, not the person's. */
export function methodNote(h: Hints): string {
  if (h.route !== 'focused' || h.readOnly || h.planWorth) return ''
  return 'Before changing anything, state in one line the outcome and the check that will prove it (a test when there is one), then make the check fail, then pass.'
}

// The text of the last assistant message on the branch (Pi session entries), or ''.
export function lastAssistantText(entries: readonly any[]): string {
  for (let i = entries.length - 1; i >= 0; i--) {
    const message = entries[i]?.type === 'message' ? entries[i].message : null
    if (message?.role !== 'assistant') continue
    const content = message.content
    if (typeof content === 'string') return content
    if (Array.isArray(content)) {
      const text = content.filter((b: any) => b?.type === 'text').map((b: any) => b.text).join('\n')
      if (text) return text
    }
  }
  return ''
}

// --- Read-only guard -------------------------------------------------------------------------

// Paths a read-only turn may still write: scratch space only.
export function isScratchPath(path: string, tmpDirs: readonly string[]): boolean {
  return tmpDirs.some((dir) => path === dir || path.startsWith(dir.endsWith('/') ? dir : `${dir}/`))
}

// Cheap pre-filter: commands without any of these can only read. Anything that matches goes to
// Jev, which tells a real write from a redirect to /dev/null or a write under /tmp.
const MAYBE_WRITES =
  /(^|[^0-9&2])>{1,2}(?!\s*\/dev\/null)|\b(rm|rmdir|mv|cp|ln|touch|mkdir|tee|truncate|chmod|chown|dd|patch|rsync|install|unlink)\b|\bsed\b[^|;&]*\s-i|\bgit\s+(add|commit|checkout|switch|reset|restore|stash|apply|merge|rebase|clean|rm|mv|push|pull|tag|branch\s+-[dD])\b|\b(npm|pnpm|yarn|bun)\s+(i|install|add|remove|uninstall|update|upgrade|link|run\s+build|build)\b|\bbun\s+build\b|\b(pip|pipx|cargo|go)\s+(install|uninstall)\b|\bperl\s+-[a-z]*i|\bnode\s+-e|\bpython3?\s+-c|<<\s*['"]?\w+/

export function mayWrite(command: string): boolean {
  return MAYBE_WRITES.test(command)
}

export function bashWriteRequest(command: string, cwd: string, tmpDirs: readonly string[]) {
  return {
    state: { command: command.slice(0, 6000), cwd, scratch_dirs: tmpDirs },
    questions: {
      writes: {
        type: 'bool' as const,
        instructions:
          'Does running `command` in `cwd` create, modify, move or delete any file or directory OUTSIDE `scratch_dirs`? Writing only to /dev/null, stdout or inside `scratch_dirs` does not count. Commands that only read, list, search, print, test or type-check do not count. Starting or running a program — a dev server, a mock server, a test runner, a `package.json` script, in the background or not — does not count either, even though it may touch its own caches: judge only what the command line itself clearly does (redirects into project files, rm, mv, cp, git commit/stash/checkout, package installs, code generators, sed -i).',
        criteria: { true: 'Writes outside the scratch dirs', false: 'Only reads, or writes only inside the scratch dirs' },
      },
    },
  }
}

export const READ_ONLY_NOTE =
  'This turn is read-only: do not create, change or delete files outside /tmp; temporary files go in /tmp.'

// --- Experts ---------------------------------------------------------------------------------

// Experts are pi-parallel-agents agent files (~/.pi/agent/agents/*.md). The main model only sees
// their names and descriptions; their prompts reach the subagents through `parallel`'s `agent`.
export interface Expert {
  name: string
  description: string
  // Its `tools` leave out edit and write: safe for read-only turns.
  readOnly: boolean
}

export const NO_EXPERT = 'ninguno'

export function parseExpert(markdown: string): Expert | null {
  const front = markdown.match(/^---\n([\s\S]*?)\n---/)?.[1]
  if (!front) return null
  const field = (key: string) => front.match(new RegExp(`^${key}:\\s*(.+)$`, 'm'))?.[1]?.trim()
  const name = field('name')
  const description = field('description')
  const tools = field('tools')?.split(',').map((t) => t.trim()) ?? []
  const readOnly = tools.length > 0 && !tools.includes('edit') && !tools.includes('write')
  return name && description ? { name, description, readOnly } : null
}

export function expertRequest(task: string, experts: readonly Expert[]) {
  const criteria: Record<string, string> = {}
  for (const e of experts) criteria[e.name] = e.description
  criteria[NO_EXPERT] = 'None of the above fits: general work such as counting, listing, summarising or running a simple command.'
  return {
    state: { task: task.slice(0, 6000) },
    questions: {
      expert: {
        type: 'choice' as const,
        instructions: 'Which specialist should carry out `task`? Pick the one whose description best matches the work the task asks for.',
        criteria,
      },
    },
  }
}

// A Jev result -> the expert name, or null when none fits clearly enough.
export function parseExpertChoice(result: any, minP = 0.4): string | null {
  if (result?.stopReason !== 'stop') return null
  const a = result.answers?.expert
  if (a?.type !== 'choice' || a.choice === NO_EXPERT) return null
  return Number(a.probabilities?.[a.choice] ?? 0) >= minP ? a.choice : null
}

export function expertsLine(experts: readonly Expert[]): string {
  if (!experts.length) return ''
  return `Experts for \`parallel\` (each task's \`agent\` field): ${experts.map((e) => `${e.name} — ${e.description}`).join(' | ')}`
}

// --- Plan mode: is the user still planning? ---------------------------------------------------

export function planContinuationRequest(message: string, previousReply: string) {
  return {
    state: { message: message.slice(0, 6000), previous_reply: previousReply.slice(-2000) },
    questions: {
      planning: {
        type: 'bool' as const,
        instructions:
          'The assistant is in the middle of designing and planning a change with the user (asking questions one at a time, proposing a design, writing a spec and an implementation plan). `previous_reply` is the assistant\'s last message. Is `message` part of that planning conversation — answering its questions, choosing an option, approving, correcting or adjusting the design, spec, scope or plan, or asking about them? A new, separate request to do, check, fix, run or verify something right now is NOT planning.',
        criteria: { true: 'Continues the planning conversation', false: 'A separate request, not part of planning' },
      },
    },
  }
}

/** Probability that the message continues planning, or null if Jev could not answer. */
export function parsePlanning(result: any): number | null {
  if (result?.stopReason !== 'stop') return null
  const a = result.answers?.planning
  return a?.type === 'bool' && typeof a.probability === 'number' ? a.probability : null
}

/** The route line Aria's dock shows before an answer (`{ mode, action }`, words the chat's Jev part knows). */
export function jevStatus(h: Hints): { mode: string; action: string } {
  const MODE: Record<Route, string> = { quick: 'quick', focused: 'focused task', parallel: 'work to split', think: 'needs thinking' }
  const action = [h.memory && 'searches memory', h.web && 'searches the web', h.notion && 'checks Notion', h.readOnly && 'read-only'].filter(Boolean).join(' · ')
  return { mode: MODE[h.route], action }
}
