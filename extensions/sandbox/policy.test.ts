// Run: bun test ~/.pi/agent/extensions/sandbox
import { afterAll, describe, expect, test } from 'bun:test'
import { execFileSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, realpathSync, rmSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { bwrapArgs, containsSecret, isSecret, isWritable, looksBlocked, projectRoot, shellQuote, subagentProblem, wrapCommand, writableRoots, isAgentCode } from './policy.ts'

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
  test('without it nothing changes', () =>
    withExtra([], () => expect(isWritable('/srv/shared/notes.md', '/home/u/Projects/app', '/home/u', all)).toBe(false)))
})
