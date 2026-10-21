// Run: bun test ~/.pi/agent/extensions/sandbox
import { afterAll, describe, expect, test } from 'bun:test'
import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { bwrapArgs, containsSecret, isSecret, isWritable, looksBlocked, projectRoot, shellQuote, subagentProblem, wrapCommand, writableRoots, isAgentCode, isAriaOwn } from './policy.ts'

const home = '/home/u'
const all = () => true
const none = () => false

describe('projectRoot', () => {
  const repo = mkdtempSync('/tmp/sbx-repo-')
  mkdirSync(join(repo, '.git'))
  mkdirSync(join(repo, 'src', 'deep'), { recursive: true })
  test('git root from a subdir', () => expect(projectRoot(join(repo, 'src', 'deep'), home)).toBe(repo))
  test('a plain dir is its own project', () => expect(projectRoot('/tmp', home)).toBe('/tmp'))
  test('home is no project', () => expect(projectRoot(home, home)).toBeNull())
  test('/ is no project', () => expect(projectRoot('/', home)).toBeNull())
})

describe('writableRoots', () => {
  test('project, /tmp and existing caches', () => {
    const roots = writableRoots('/srv/app', home, (p) => p === `${home}/.npm`)
    expect(roots).toEqual(['/srv/app', '/tmp', `${home}/.npm`])
  })
  test('from home only /tmp and caches', () => expect(writableRoots(home, home, none)).toEqual(['/tmp']))
})

describe('isWritable', () => {
  test('inside the project', () => expect(isWritable('src/a.ts', '/srv/app', home, none)).toBe(true))
  test('.. leaving the project', () => expect(isWritable('../other/x', '/srv/app', home, none)).toBe(false))
  test('/tmp', () => expect(isWritable('/tmp/x', '/srv/app', home, none)).toBe(true))
  test('a cache', () => expect(isWritable(`${home}/.npm/_cacache/x`, '/srv/app', home, all)).toBe(true))
  test('dotfile in home', () => expect(isWritable(`${home}/.bashrc`, '/srv/app', home, all)).toBe(false))
  test('prefix trick /srv/app2', () => expect(isWritable('/srv/app2/x', '/srv/app', home, none)).toBe(false))
})

describe('isSecret', () => {
  test('ssh key', () => expect(isSecret(`${home}/.ssh/id_ed25519`, '/srv/app', home)).toBe(true))
  test('relative from home', () => expect(isSecret('.ssh/id_ed25519', home, home)).toBe(true))
  test('pi auth', () => expect(isSecret(`${home}/.pi/agent/auth.json`, '/srv/app', home)).toBe(true))
  test('aria\'s credentials too', () => {
    for (const f of ['.aria/auth.json', '.aria/mcp.json', '.aria/mcp-auth.json', '.aria/agent/auth.json', '.aria/agent/mcp.json', '.aria/agent/mcp-auth.json']) {
      expect(isSecret(`${home}/${f}`, '/srv/app', home)).toBe(true)
    }
  })
  test('runtime dir socket', () => expect(isSecret('/run/user/1000/hypr/x/.socket.sock', '/srv/app', home, '/run/user/1000')).toBe(true))
  test('a normal file', () => expect(isSecret(`${home}/.pi/agent/SYSTEM.md`, '/srv/app', home)).toBe(false))
})

describe('bwrapArgs', () => {
  const args = bwrapArgs('/srv/app/src', { home, runtimeDir: '/run/user/1000', exists: all })
  const pairs = (flag: string) => args.flatMap((a, i) => (a === flag ? [args[i + 1]] : []))
  test('read-only root first', () => expect(args.slice(0, 3)).toEqual(['--ro-bind', '/', '/']))
  test('project and /tmp writable', () => expect(pairs('--bind')).toEqual(expect.arrayContaining(['/srv/app/src', '/tmp'])))
  test('secret dirs and the runtime dir hidden', () => expect(pairs('--tmpfs')).toEqual(expect.arrayContaining([`${home}/.ssh`, '/run/user/1000'])))
  test('secret files masked', () => expect(args.join(' ')).toContain(`--ro-bind /dev/null ${home}/.pi/agent/auth.json`))
  test('cwd kept', () => expect(pairs('--chdir')).toEqual(['/srv/app/src']))
  test('no namespace isolation of net or pid', () => {
    expect(args).not.toContain('--unshare-net')
    expect(args).not.toContain('--unshare-pid')
    expect(args).not.toContain('--die-with-parent')
  })
  test('missing secrets are skipped', () => expect(bwrapArgs('/srv/app', { home, runtimeDir: '/run/user/1000', exists: none }).join(' ')).not.toContain('.ssh'))
  test("the dock's sockets are neither named nor seen", () => {
    const env = { ARIA_JOBS_SOCK: '/tmp/aria/jobs-s1.sock', PI_WATCH_SOCK: '/tmp/aria-watch/watch-s1.sock' }
    const boxed = bwrapArgs('/srv/app', { home, runtimeDir: '/run/user/1000', exists: all, env })
    const tmpfs = boxed.flatMap((a, i) => (a === '--tmpfs' ? [boxed[i + 1]] : []))
    expect(tmpfs).toEqual(expect.arrayContaining(['/tmp/aria', '/tmp/aria-watch']))
    expect(boxed.lastIndexOf('/tmp/aria')).toBeGreaterThan(boxed.indexOf('/tmp'))
    const unset = boxed.flatMap((a, i) => (a === '--unsetenv' ? [boxed[i + 1]] : []))
    expect(unset).toEqual(expect.arrayContaining(['ARIA_JOBS_SOCK', 'PI_WATCH_SOCK']))
  })
  test('a socket folder that is not there is not masked', () => {
    const env = { ARIA_JOBS_SOCK: '/tmp/aria/jobs-s1.sock' }
    const boxed = bwrapArgs('/srv/app', { home, runtimeDir: '/run/user/1000', exists: (p) => p !== '/tmp/aria', env })
    expect(boxed).not.toContain('/tmp/aria')
    expect(boxed.join(' ')).toContain('--unsetenv ARIA_JOBS_SOCK')
  })
  test('no sockets set, nothing extra masked', () => {
    const count = (env: Record<string, string>) =>
      bwrapArgs('/srv/app', { home, runtimeDir: '/run/user/1000', exists: all, env }).filter((a) => a === '--tmpfs').length
    expect(count({}) + 1).toBe(count({ ARIA_JOBS_SOCK: '/tmp/aria/jobs-s1.sock' }))
    expect(bwrapArgs('/srv/app', { home, exists: all, env: {} }).join(' ')).toContain('--unsetenv ARIA_JOBS_SOCK --unsetenv PI_WATCH_SOCK')
  })
})

describe('wrapCommand', () => {
  test('shellQuote survives a real shell', () => {
    const tricky = `echo "a b" 'c' $(printf x) \\\n; cat <<EOF\nhi $HOME\nEOF`
    expect(execFileSync('bash', ['-c', `printf %s ${shellQuote(tricky)}`]).toString()).toBe(tricky)
  })
  test('starts with exec bwrap and ends with the quoted command', () => {
    const w = wrapCommand('ls -la', '/srv/app', { home, runtimeDir: '/run/user/1000', exists: none })
    expect(w.startsWith('exec bwrap ')).toBe(true)
    expect(w.endsWith(`-- bash -c 'ls -la'`)).toBe(true)
  })
})

describe('looksBlocked', () => {
  test('read-only fs', () => expect(looksBlocked("touch: cannot touch '/home/u/x': Read-only file system")).toBe(true))
  test('ordinary failure', () => expect(looksBlocked('Error: test failed')).toBe(false))
})

// Final-review fixes (2026-10-14): each case is an escape the reviewer showed.
describe('review fixes', () => {
  // The fake home must live outside /tmp, which is writable on its own.
  const real = realpathSync(mkdtempSync(join(homedir(), '.cache', 'sbx-test-')))
  afterAll(() => rmSync(real, { recursive: true, force: true }))
  mkdirSync(join(real, '.ssh'))
  const repo = join(real, 'proj')
  mkdirSync(join(repo, '.git', 'hooks'), { recursive: true })
  const { symlinkSync, writeFileSync } = require('node:fs')
  symlinkSync(join(real, '.ssh'), join(repo, 'keys'))
  symlinkSync(join(real, '.bashrc'), join(repo, 'rc'))
  const rt = '/run/user/1000'
  test('C1 @ prefix is resolved like Pi', () => expect(isSecret('@~/.ssh/id_ed25519', repo, real, rt)).toBe(true))
  test('C1 file:// URL is resolved like Pi', () => expect(isSecret(`file://${real}/.ssh/id_ed25519`, repo, real, rt)).toBe(true))
  test('C1 /proc paths are refused', () => expect(isSecret('/proc/self/root/etc/passwd', repo, real, rt)).toBe(true))
  test('C1 symlink into a secret dir', () => expect(isSecret('keys/id_ed25519', repo, real, rt)).toBe(true))
  test('C1 symlink out of the project is not writable', () => expect(isWritable('rc', repo, real, none)).toBe(false))
  test('C1 @ write outside', () => expect(isWritable('@~/.bashrc', repo, real, none)).toBe(false))
  const args = bwrapArgs(repo, { home: real, runtimeDir: rt, exists: all })
  const s = args.join(' ')
  test('C2 docker socket masked', () => expect(s).toContain('--ro-bind /dev/null /run/docker.sock'))
  test('C3 herdr dir hidden', () => expect(s).toContain(`--tmpfs ${real}/.config/herdr`))
  test('C4 .git/config and hooks read-only after the project bind', () => {
    expect(s).toContain(`--ro-bind ${repo}/.git/config ${repo}/.git/config`)
    expect(s).toContain(`--ro-bind ${repo}/.git/hooks ${repo}/.git/hooks`)
    expect(args.lastIndexOf(`${repo}/.git/hooks`)).toBeGreaterThan(args.indexOf(repo))
  })
  test('C4/I5 Pi file tools may not write .git/config, hooks or .pi', () => {
    expect(isWritable('.git/config', repo, real, none)).toBe(false)
    expect(isWritable('.git/hooks/pre-commit', repo, real, none)).toBe(false)
    expect(isWritable('.pi/extensions/x.ts', repo, real, none)).toBe(false)
    expect(isWritable('.git/index', repo, real, none)).toBe(true)
  })
  test('C5 jiti cache read-only', () => expect(s).toContain('--ro-bind /tmp/jiti /tmp/jiti'))
  test('C5 Pi file tools may not write the jiti cache', () => expect(isWritable('/tmp/jiti/x.mjs', repo, real, none)).toBe(false))
  test('I1 more credentials hidden', () => {
    expect(isSecret(`${real}/.claude/.credentials.json`, repo, real, rt)).toBe(true)
    expect(isSecret(`${real}/.config/wayvnc/config`, repo, real, rt)).toBe(true)
    expect(isSecret(`${real}/.local/share/opencode/auth.json`, repo, real, rt)).toBe(true)
  })
  test('I2 only download caches are writable, not bin/installs', () => {
    expect(isWritable(`${real}/.bun/bin/git`, repo, real, all)).toBe(false)
    expect(isWritable(`${real}/.local/share/mise/installs/node/x`, repo, real, all)).toBe(false)
    expect(isWritable(`${real}/.cache/blesh/x`, repo, real, all)).toBe(false)
    expect(isWritable(`${real}/.npm/_cacache/x`, repo, real, all)).toBe(true)
    expect(isWritable(`${real}/.bun/install/cache/x`, repo, real, all)).toBe(true)
  })
  test('I3 grep over a dir that contains secrets is refused', () => {
    expect(containsSecret(real, repo, real, rt)).toBe(true)
    expect(containsSecret(repo, repo, real, rt)).toBe(false)
  })
  test('I4 X11 sockets hidden and DISPLAY unset', () => {
    expect(s).toContain('--tmpfs /tmp/.X11-unix')
    expect(s).toContain('--unsetenv DISPLAY')
    expect(s).toContain('--ro-bind /dev/null /run/dbus/system_bus_socket')
  })
  test('I6 a worktree also gets its common git dir', () => {
    const main = join(real, 'main')
    mkdirSync(join(main, '.git', 'worktrees', 'wt'), { recursive: true })
    writeFileSync(join(main, '.git', 'worktrees', 'wt', 'commondir'), '../..\n')
    const wt = join(real, 'wt')
    mkdirSync(wt)
    writeFileSync(join(wt, '.git'), `gitdir: ${main}/.git/worktrees/wt\n`)
    expect(writableRoots(wt, real, none)).toEqual(expect.arrayContaining([wt, `${main}/.git`]))
    expect(isWritable(`${main}/.git/worktrees/wt/index.lock`, wt, real, none)).toBe(true)
    expect(isWritable(`${main}/.git/config`, wt, real, none)).toBe(false)
  })
  test('C6 subagent inputs: cwd must stay in the project, contextFiles not secret', () => {
    expect(subagentProblem({ tasks: [{ task: 'x', cwd: `${real}/.config/hypr` }] }, repo, real, rt)).toMatch(/cwd/)
    expect(subagentProblem({ task: 'x', contextFiles: [`${real}/.ssh/id_ed25519`] }, repo, real, rt)).toMatch(/contextFiles/)
    expect(subagentProblem({ chain: [{ task: 'x', cwd: `${repo}/sub` }] }, repo, real, rt)).toBeNull()
  })
})

describe("Aria's own files", () => {
  test("the extensions, persona, experts and settings of Aria's Pi can never be written by the model", () => {
    const agent = '/home/u/.aria/agent'
    for (const p of [`${agent}/extensions/sandbox/policy.ts`, `${agent}/agents/security.md`, `${agent}/SYSTEM.md`, `${agent}/settings.json`, `${agent}/.aria-files`]) {
      expect(isAgentCode(p, agent)).toBe(true)
    }
    expect(isAgentCode(`${agent}/sessions/aria/x.jsonl`, agent)).toBe(false)
    expect(isAgentCode('/home/u/Projects/app/SYSTEM.md', agent)).toBe(false)
  })

  test("in the owner's own Pi (no Aria) nothing is locked this way", () => {
    expect(isAgentCode('/home/u/.pi/agent/extensions/x.ts', null)).toBe(false)
  })
})

describe('extra writable folders (settings)', () => {
  const { mkdtempSync: mk, writeFileSync: wr } = require('node:fs')
  const { tmpdir } = require('node:os')
  const withExtra = (extra: string[], run: () => void) => {
    const dir = mk(join(tmpdir(), 'aria-extra-'))
    wr(join(dir, 'aria.json'), JSON.stringify({ sandbox: { extraWritable: extra } }))
    const saved = process.env.PI_CODING_AGENT_DIR
    process.env.PI_CODING_AGENT_DIR = dir
    try { run() } finally { if (saved === undefined) delete process.env.PI_CODING_AGENT_DIR; else process.env.PI_CODING_AGENT_DIR = saved }
  }
  const all = () => true
  test('an extra folder is writable', () =>
    withExtra(['/srv/shared'], () => expect(isWritable('/srv/shared/notes.md', '/home/u/Projects/app', '/home/u', all)).toBe(true)))
  test('an extra folder inside a secret dir is not', () =>
    withExtra(['/home/u/.ssh'], () => expect(isWritable('/home/u/.ssh/config', '/home/u/Projects/app', '/home/u', all)).toBe(false)))
  test('an extra folder that does not exist is left out (bwrap cannot bind it)', () =>
    withExtra(['/srv/missing'], () => expect(writableRoots('/home/u/Projects/app', '/home/u', (p) => p !== '/srv/missing')).not.toContain('/srv/missing')))
  test("an extra folder that holds Aria's own files is refused", () => {
    const { mkdtempSync: mk, mkdirSync: md, writeFileSync: wr } = require('node:fs')
    const home = join(mk(join(require('node:os').tmpdir(), 'aria-home-')), '.aria')
    const agent = join(home, 'agent')
    md(agent, { recursive: true })
    wr(join(agent, 'aria.json'), JSON.stringify({ sandbox: { extraWritable: [home, join(home, 'bin')] } }))
    const saved = [process.env.PI_CODING_AGENT_DIR, process.env.ARIA_SETTINGS]
    process.env.PI_CODING_AGENT_DIR = agent
    process.env.ARIA_SETTINGS = join(home, 'settings.json')
    try {
      const roots = writableRoots('/home/u/Projects/app', '/home/u', all)
      expect(roots).not.toContain(home)
      expect(roots).not.toContain(join(home, 'bin'))
    } finally {
      if (saved[0] === undefined) delete process.env.PI_CODING_AGENT_DIR; else process.env.PI_CODING_AGENT_DIR = saved[0]
      if (saved[1] === undefined) delete process.env.ARIA_SETTINGS; else process.env.ARIA_SETTINGS = saved[1]
    }
  })
  test('without it nothing changes', () =>
    withExtra([], () => expect(isWritable('/srv/shared/notes.md', '/home/u/Projects/app', '/home/u', all)).toBe(false)))
})

// Aria links its agent dir's credentials to the files in ~/.aria; bwrap cannot mount over a symlink, so the link's
// target is what gets hidden. Runs the real bwrap when there is one.
describe.skipIf(!existsSync('/usr/bin/bwrap'))('a credential that is a symlink', () => {
  const fake = realpathSync(mkdtempSync('/tmp/sbx-link-home-'))
  mkdirSync(join(fake, '.aria', 'agent'), { recursive: true })
  writeFileSync(join(fake, '.aria', 'auth.json'), 'SECRET-123')
  symlinkSync(join(fake, '.aria', 'auth.json'), join(fake, '.aria', 'agent', 'auth.json'))
  const project = mkdtempSync('/tmp/sbx-link-repo-')
  afterAll(() => rmSync(fake, { recursive: true, force: true }) || rmSync(project, { recursive: true, force: true }))
  const run = (cmd: string) => {
    try {
      return execFileSync('bash', ['-c', wrapCommand(cmd, project, { home: fake, runtimeDir: '/nonexistent' })], { stdio: 'pipe' }).toString()
    } catch (error: any) {
      return `${error.stdout ?? ''}${error.stderr ?? ''}`
    }
  }
  test('commands still run in the box', () => expect(run('echo inside')).toBe('inside\n'))
  test('neither the link nor its target can be read', () => {
    const out = run(`cat ${fake}/.aria/agent/auth.json ${fake}/.aria/auth.json; echo done`)
    expect(out).toContain('done')
    expect(out).not.toContain('SECRET-123')
  })
})

// A credential linked into the project being worked on (dotfiles): the project's protected .pi folder is bound
// read-only after the writable roots, and must not bring the hidden file back.
describe.skipIf(!existsSync('/usr/bin/bwrap'))('a credential linked into the project', () => {
  const fake = realpathSync(mkdtempSync('/tmp/sbx-dot-home-'))
  const dotfiles = join(fake, 'dotfiles')
  mkdirSync(join(dotfiles, '.git'), { recursive: true })
  mkdirSync(join(dotfiles, '.pi', 'agent'), { recursive: true })
  writeFileSync(join(dotfiles, '.pi', 'agent', 'auth.json'), 'DOT-SECRET-9')
  mkdirSync(join(fake, '.pi', 'agent'), { recursive: true })
  symlinkSync(join(dotfiles, '.pi', 'agent', 'auth.json'), join(fake, '.pi', 'agent', 'auth.json'))
  afterAll(() => rmSync(fake, { recursive: true, force: true }))
  test('stays hidden', () => {
    let out = ''
    try {
      out = execFileSync('bash', ['-c', wrapCommand(`cat ${fake}/.pi/agent/auth.json; echo done`, dotfiles, { home: fake, runtimeDir: '/nonexistent' })], { stdio: 'pipe' }).toString()
    } catch (error: any) {
      out = `${error.stdout ?? ''}${error.stderr ?? ''}`
    }
    expect(out).toContain('done')
    expect(out).not.toContain('DOT-SECRET-9')
  })
})

// Aria in a home of its own (ARIA_HOME elsewhere, as in the parity checks): its credentials are found from the
// environment Aria gives Pi, not only under ~/.aria.
describe('an Aria home outside ~/.aria', () => {
  const saved = { agent: process.env.PI_CODING_AGENT_DIR, settings: process.env.ARIA_SETTINGS }
  afterAll(() => {
    for (const [key, value] of [['PI_CODING_AGENT_DIR', saved.agent], ['ARIA_SETTINGS', saved.settings]] as const) {
      if (value === undefined) delete process.env[key]
      else process.env[key] = value
    }
  })
  test('its credentials and its agent dir links are secrets', () => {
    process.env.PI_CODING_AGENT_DIR = '/srv/aria-copy/agent'
    process.env.ARIA_SETTINGS = '/srv/aria-copy/settings.json'
    for (const file of ['auth.json', 'mcp.json', 'mcp-auth.json']) {
      expect(isSecret(`/srv/aria-copy/${file}`, '/srv/app', home)).toBe(true)
      expect(isSecret(`/srv/aria-copy/agent/${file}`, '/srv/app', home)).toBe(true)
    }
    expect(isSecret('/srv/aria-copy/settings.json', '/srv/app', home)).toBe(false)
  })
})

// Final review (2026-10-21): the files that make aria.json, aria.json itself, and the attachments only the dock writes.
describe("Aria's settings, aria.json and attachments", () => {
  const env = { PI_CODING_AGENT_DIR: '/srv/aria-home/agent', ARIA_PROJECTS_DIR: '/srv/work', ARIA_SETTINGS: '/srv/aria-home/settings.json' }
  test('none of them can be changed under Aria', () => {
    for (const p of ['/srv/aria-home/agent/aria.json', '/srv/aria-home/settings.json', '/srv/aria-home/attachments/s1/2026-10-21/1-a.png', '/srv/aria-home/attachments']) {
      expect(isAriaOwn(p, '/srv/work', env, home)).toBe(true)
    }
  })
  test('the rest of the Aria home and the projects are not locked this way', () => {
    expect(isAriaOwn('/srv/aria-home/agent/sessions/x.jsonl', '/srv/work', env, home)).toBe(false)
    expect(isAriaOwn('/srv/work/settings.json', '/srv/work', env, home)).toBe(false)
    expect(isAriaOwn('/srv/work/attachments/a.png', '/srv/work', env, home)).toBe(false)
  })
  test('a relative or ~ path is resolved first', () => {
    expect(isAriaOwn('../aria-home/agent/aria.json', '/srv/work', env, home)).toBe(true)
  })
  test('the owner\'s own Pi (no ARIA_PROJECTS_DIR) locks nothing', () => {
    expect(isAriaOwn('/srv/aria-home/settings.json', '/srv/work', { ...env, ARIA_PROJECTS_DIR: undefined }, home)).toBe(false)
  })
  test('a link to the attachments folder is the attachments folder', () => {
    const root = realpathSync(mkdtempSync('/tmp/sbx-own-'))
    mkdirSync(join(root, 'aria', 'agent'), { recursive: true })
    mkdirSync(join(root, 'aria', 'attachments'))
    mkdirSync(join(root, 'work'))
    symlinkSync(join(root, 'aria', 'attachments'), join(root, 'work', 'files'))
    const linked = { PI_CODING_AGENT_DIR: join(root, 'aria', 'agent'), ARIA_PROJECTS_DIR: join(root, 'work'), ARIA_SETTINGS: join(root, 'aria', 'settings.json') }
    try {
      expect(isAriaOwn('files/x.png', join(root, 'work'), linked, home)).toBe(true)
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })
})
