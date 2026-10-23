// What fills the context window, part by part, in tokens estimated at 4 characters each: the system prompt (without
// the skills' list, counted apart), the tools' declarations, the skills, the memory hint, and the conversation (what Pi
// reports as used, minus the rest).

export interface Parts {
  system: number
  tools: number
  skills: number
  memory: number
  messages: number
}

export const tokens = (text: string): number => Math.ceil((text ?? '').length / 4)

export function measure(input: {
  systemPrompt: string
  skills: { name?: string; description?: string }[]
  tools: { name?: string; description?: string; parameters?: unknown }[]
  memory: string
  usedTokens: number | null
}): Parts {
  const skills = input.skills.reduce((n, s) => n + tokens(s.description ?? ''), 0)
  const system = Math.max(0, tokens(input.systemPrompt) - skills)
  const tools = input.tools.reduce(
    (n, t) => n + tokens(t.name ?? '') + tokens(t.description ?? '') + tokens(JSON.stringify(t.parameters ?? {})),
    0,
  )
  const memory = tokens(input.memory)
  const messages = input.usedTokens === null ? 0 : Math.max(0, input.usedTokens - system - skills - tools - memory)
  return { system, tools, skills, memory, messages }
}
