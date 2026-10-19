// Where the extensions find their files. In Aria, Pi runs with `PI_CODING_AGENT_DIR=~/.aria/agent` and
// `ARIA_PROJECTS_DIR=<the working folder>`; in the owner's own Pi neither is set and the old places are used, so the same
// files run in both. (This folder has no index.ts: Pi does not load it as an extension.)
import { homedir } from 'node:os'
import { join } from 'node:path'

export const agentDir = (): string => process.env.PI_CODING_AGENT_DIR || join(homedir(), '.pi', 'agent')

export const projectsDir = (): string => process.env.ARIA_PROJECTS_DIR || join(homedir(), 'Projects')
