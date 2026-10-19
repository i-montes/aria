// The subagents' turn limit (Aria's settings: subagents.turnLimit; none by default, as in the owner's Pi).
import type { AriaConfig } from '../lib/config.ts'

export const turnLimitReached = (turns: number, config: AriaConfig): boolean =>
  config.subagents.turnLimit !== null && turns >= config.subagents.turnLimit
