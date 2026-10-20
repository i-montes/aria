// `install`: the agent installs what the user asks for, usually from a link: a skill (into the agent folder's `skills/`),
// an MCP server (into `mcp.json`) or a Pi package (`pi install`). The user confirms every install on a card that shows
// exactly what goes in (secrets masked). The tool does the parts the sandbox keeps from the model on purpose: `mcp.json`
// holds credentials and Aria's own folder is not writable, so the model never edits them itself, and a key the user gave
// goes straight into the file instead of through a shell command. The `install-from-link` skill tells the model how to
// read a link before calling it.
//
// A skill and a Pi package take effect in the next session (Pi lists skills and loads packages when it starts); a new
// MCP server also connects right away, for this session, through `registerMcpServer`.
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { execFile } from 'node:child_process'
import { chmodSync, cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { Type } from 'typebox'
import { agentDir } from '../lib/agent-dir.ts'
import { canAsk, sandboxCard } from '../sandbox/card.ts'
import { describeMcp, findSkills, mcpEntry, mcpHome, namedSkills, parseSkillSource, skillName, validPackage, validSkillName, withServer } from './install.ts'

const Params = Type.Object({
  kind: Type.Union([Type.Literal('skill'), Type.Literal('mcp'), Type.Literal('package')], { description: 'What to install.' }),
  source: Type.Optional(
    Type.String({
      description:
        'skill: the link (a GitHub repository, folder or SKILL.md, a raw SKILL.md URL, or a git URL). package: what `pi install` takes (npm:<name>@<version>, git:github.com/<owner>/<repo>@<ref>, or an https git URL).',
    }),
  ),
  skills: Type.Optional(Type.Array(Type.String(), { description: 'skill: when the source holds several skills, the names to install (all of them when omitted).' })),
  name: Type.Optional(Type.String({ description: 'mcp: the server name (letters, digits, _ and -), short and lowercase, e.g. "github".' })),
  url: Type.Optional(Type.String({ description: 'mcp: a remote server\'s streamable HTTP URL.' })),
  headers: Type.Optional(Type.Record(Type.String(), Type.String(), { description: 'mcp (remote): HTTP headers, e.g. {"Authorization": "Bearer <key>"}.' })),
  command: Type.Optional(Type.String({ description: 'mcp: a local server\'s executable (one word, e.g. "npx").' })),
  args: Type.Optional(Type.Array(Type.String(), { description: 'mcp (local): the command\'s arguments, e.g. ["-y", "@scope/server"].' })),
  env: Type.Optional(Type.Record(Type.String(), Type.String(), { description: 'mcp (local): environment variables, e.g. API keys.' })),
  description: Type.Optional(Type.String({ description: 'mcp: what the server offers, in one sentence.' })),
  signIn: Type.Optional(Type.Boolean({ description: 'mcp (remote): the server signs in with OAuth; open the sign-in in the browser after installing.' })),
})

const INSTALL = 'Install'
const CANCEL = 'Cancel'

const result = (text: string) => ({ content: [{ type: 'text' as const, text }], details: {} })

/** Runs a program (not a shell) and gives its output; a failure carries what it printed. */
function run(program: string, args: string[], signal: AbortSignal | undefined, timeout: number): Promise<string> {
  return new Promise((resolve, reject) =>
    execFile(program, args, { signal, timeout, maxBuffer: 4 * 1024 * 1024 }, (error, stdout, stderr) =>
      error ? reject(new Error(`${program} ${args[0]} failed: ${(stderr || stdout || error.message).toString().trim().slice(-800)}`)) : resolve(stdout.toString()),
    ),
  )
}

/** Aria's `pi` (its pinned engine, with its agent folder), else the one on the PATH. */
const piBin = (): string => {
  const own = join(mcpHome(agentDir()), 'bin', 'pi')
  return existsSync(own) ? own : 'pi'
}

/** Writes JSON through a temp file and a rename, mode 600 (it may hold keys). */
function writeJson(path: string, value: unknown) {
  mkdirSync(dirname(path), { recursive: true })
  const tmp = `${path}.install-${process.pid}.tmp`
  writeFileSync(tmp, `${JSON.stringify(value, null, 2)}\n`, { mode: 0o600 })
  chmodSync(tmp, 0o600)
  renameSync(tmp, path)
}

/** Under Aria the agent folder links to the files in `~/.aria`; a file that did not exist yet gets its link now. */
function linkIntoAgent(name: string) {
  const home = mcpHome(agentDir())
  const at = join(agentDir(), name)
  if (home !== agentDir() && !existsSync(at)) {
    if (!existsSync(join(home, name))) writeJson(join(home, name), {})
    symlinkSync(join(home, name), at)
  }
}

export default function install(pi: ExtensionAPI) {
  // Subagents (`--mode json`) have nobody to confirm with: keep the tool out of their declared tools.
  pi.on('session_start', (_event: any, ctx: any) => {
    if (ctx.mode === 'json') pi.setActiveTools(pi.getActiveTools().filter((n) => n !== 'install'))
  })

  /** The user's yes on the permission card (dock) or the terminal's dialog. */
  const confirm = async (ctx: any, summary: string, detail: string, signal?: AbortSignal): Promise<boolean> => {
    const options = [INSTALL, CANCEL]
    const title = ctx.mode === 'rpc' ? sandboxCard('sandbox', { summary, detail }, options) : `${summary}\n${detail}`
    return (await ctx.ui.select(title, options, { signal })) === INSTALL
  }

  async function installSkill(params: any, signal: AbortSignal | undefined, ctx: any) {
    const source = parseSkillSource(params.source ?? '')
    if (!source) return result(`"${params.source ?? ''}" is not a skill link this tool reads (a GitHub repository, folder or SKILL.md, a raw SKILL.md URL, or a git URL). Find the repository or SKILL.md the page points to and call again.`)
    const skillsDir = join(agentDir(), 'skills')
    const tmp = mkdtempSync(join(tmpdir(), 'aria-skill-'))
    try {
      let found: { name: string; folder: string }[]
      if (source.kind === 'file') {
        const response = await fetch(source.url, { signal })
        if (!response.ok) return result(`Downloading ${source.url} failed: HTTP ${response.status}.`)
        const text = await response.text()
        const folder = join(tmp, dirname(new URL(source.url).pathname).split('/').pop() || 'skill')
        mkdirSync(folder)
        writeFileSync(join(folder, 'SKILL.md'), text)
        found = [{ name: skillName(text, folder), folder }]
      } else {
        const repo = join(tmp, 'repo')
        await run('git', ['clone', '--depth', '1', ...(source.ref ? ['--branch', source.ref] : []), source.repo, repo], signal, 180_000)
        found = namedSkills(findSkills(join(repo, source.path)))
      }
      if (!found.length) return result(`There is no SKILL.md in ${params.source}.`)
      const wanted: string[] | undefined = params.skills?.length ? params.skills : undefined
      const chosen = wanted ? found.filter((s) => wanted.includes(s.name)) : found
      if (!chosen.length) return result(`None of ${wanted!.join(', ')} is in ${params.source}; it holds: ${found.map((s) => s.name).join(', ')}.`)
      const invalid = chosen.filter((s) => !validSkillName(s.name))
      if (invalid.length) return result(`Not a valid skill name (lowercase letters, digits and hyphens): ${invalid.map((s) => s.name).join(', ')}.`)

      const lines = chosen.map((s) => `${s.name}${existsSync(join(skillsDir, s.name)) ? ' (replaces the installed one)' : ''}`)
      const summary = chosen.length === 1 ? `Install the skill ${chosen[0].name}?` : `Install ${chosen.length} skills?`
      if (!(await confirm(ctx, summary, `${params.source}\n${lines.join('\n')}`, signal))) return result('The user did not want it installed; nothing changed.')

      for (const skill of chosen) {
        const target = join(skillsDir, skill.name)
        rmSync(target, { recursive: true, force: true })
        cpSync(skill.folder, target, { recursive: true, filter: (from) => !from.split('/').includes('.git') })
      }
      const paths = chosen.map((s) => join(skillsDir, s.name, 'SKILL.md'))
      return result(`Installed ${chosen.map((s) => s.name).join(', ')} in ${skillsDir}. Pi lists skills when a session starts, so it appears from the next session; to use it now, read ${paths.join(', ')}.`)
    } finally {
      rmSync(tmp, { recursive: true, force: true })
    }
  }

  async function installMcp(params: any, signal: AbortSignal | undefined, ctx: any) {
    const built = mcpEntry(params)
    if ('error' in built) return result(built.error)
    const file = join(mcpHome(agentDir()), 'mcp.json')
    let current: any = {}
    if (existsSync(file)) {
      try {
        current = JSON.parse(readFileSync(file, 'utf8'))
      } catch {
        return result(`${file} does not parse; it was left as it is. Ask the user to fix it first.`)
      }
    }
    const replaces = !!current?.mcpServers?.[params.name]
    const summary = `${replaces ? 'Replace' : 'Install'} the MCP server ${params.name}?`
    if (!(await confirm(ctx, summary, describeMcp(params.name, built.entry), signal))) return result('The user did not want it installed; nothing changed.')

    writeJson(file, withServer(current, params.name, built.entry))
    linkIntoAgent('mcp.json')
    const notes: string[] = []
    if (params.signIn) {
      linkIntoAgent('mcp-auth.json')
      try {
        await run(piBin(), ['mcp', 'login', params.name, '--timeout', '300'], signal, 330_000)
        notes.push('The user signed in.')
      } catch (error) {
        notes.push(`The sign-in did not finish (${(error as Error).message}); the user can sign in from the next session with /mcp login ${params.name}.`)
      }
    }
    if (replaces) {
      notes.push('The old entry stays in use until the next session.')
    } else {
      try {
        pi.registerMcpServer(params.name, built.entry as any)
        notes.push(`It connects now; its tools are mcp__${params.name}__<tool>.`)
      } catch (error) {
        notes.push(`It connects from the next session (${(error as Error).message}).`)
      }
    }
    return result(`Added the MCP server ${params.name} to ${file}. ${notes.join(' ')}`)
  }

  async function installPackage(params: any, signal: AbortSignal | undefined, ctx: any) {
    const source = (params.source ?? '').trim()
    if (!validPackage(source)) return result(`"${source}" is not a package source \`pi install\` takes (npm:<name>@<version>, git:github.com/<owner>/<repo>@<ref>, or an https git URL).`)
    const detail = `pi install ${source}\nA package can bring extensions: code that runs inside Aria with your permissions.`
    if (!(await confirm(ctx, `Install the Pi package ${source}?`, detail, signal))) return result('The user did not want it installed; nothing changed.')
    const output = await run(piBin(), ['install', source], signal, 600_000)
    return result(`Installed ${source}; Pi loads it from the next session.\n${output.trim().slice(-600)}`)
  }

  pi.registerTool({
    name: 'install',
    label: 'Install',
    description:
      'Install a skill, an MCP server or a Pi package for the user, who confirms it on a card first. Read the `install-from-link` skill before the first call. Use it only when the user asked to install something. Never write mcp.json, Pi\'s settings or the skills folder yourself: this tool is the only way in.',
    promptSnippet: 'install: install a skill, MCP server or Pi package the user asked for (they confirm it)',
    parameters: Params,
    exposure: 'model-only',
    executionMode: 'sequential',

    async execute(_id, params: any, signal, _onUpdate, ctx: any) {
      if (!canAsk(ctx)) return result('Nobody can confirm an install here (no terminal or dock); nothing was installed. Tell the user what to install instead.')
      try {
        if (params.kind === 'skill') return await installSkill(params, signal, ctx)
        if (params.kind === 'mcp') return await installMcp(params, signal, ctx)
        return await installPackage(params, signal, ctx)
      } catch (error) {
        return result(`The install failed: ${(error as Error).message}`)
      }
    },
  })
}
