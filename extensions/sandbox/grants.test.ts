// Run: bun test ~/.pi/agent/extensions/sandbox
import { describe, expect, test } from 'bun:test'
import { commandPreview, covers, coversPath, emptyGrants, grantDir, grantUnits } from './grants.ts'

const units = (c: string) => grantUnits(c)?.units ?? null
const broad = (c: string) => grantUnits(c)?.broad

describe('grantUnits', () => {
  test('program and subcommand', () => expect(units('docker ps -a')).toEqual(['docker ps']))
  test('pipes and lists', () => expect(units('docker ps | grep pg && systemctl --user restart x; echo ok')).toEqual(['docker ps', 'grep', 'systemctl restart', 'echo']))
  test('a file argument is not a subcommand', () => expect(units('rm -f notes.txt')).toEqual(['rm']))
  test('git flags before the subcommand', () => expect(units('git --no-pager log -5')).toEqual(['git log']))
  test('an unclear subcommand allows only the exact command', () => expect(broad('git -c x=1 push origin main')).toBe(true))
  test('env vars and wrappers are skipped', () => expect(units('FOO=1 timeout 10 git status')).toEqual(['git status']))
  test('redirections are not programs', () => expect(units('git log 2>&1 >/dev/null &>log')).toEqual(['git log']))
  test('quoted separators stay arguments', () => expect(units(`grep -E "a|b;c" f && echo 'x && y'`)).toEqual(['grep', 'echo']))
  test('control keywords', () => expect(units('if test -f x; then rm x; fi')).toEqual(['test', 'rm']))
  test('a quoted heredoc body is data', () => expect(units("python3 - <<'EOF'\nimport os; os.system('rm -rf ~')\nEOF\necho done")).toEqual(['python3', 'echo']))
  test('$( ) in an unquoted heredoc body is seen', () => expect(grantUnits('cat <<EOF\n$(rm -rf ~)\nEOF')).toBeNull())
  for (const hidden of ['echo $(rm x)', 'echo `rm x`', 'eval "$X"', 'bash -c "rm x"', 'sh -ec "rm x"', 'ls | xargs rm', 'find . -exec rm {} +', 'source ./env.sh', '. ./env.sh', 'diff <(ls a) <(ls b)'])
    test(`hidden code: ${hidden}`, () => expect(grantUnits(hidden)).toBeNull())
  test('a quoted program name', () => expect(grantUnits('"./run me.sh"')).toBeNull())
  test('empty', () => expect(grantUnits('   ')).toBeNull())
  for (const c of ['python3 x.py', 'node build.js', 'npx vite', 'npm run dev', 'make install', 'sudo systemctl restart x', 'bash deploy.sh', './deploy.sh', '~/bin/tool', 'ssh host ls', 'python3.12 -m http.server'])
    test(`runs arbitrary code: ${c}`, () => expect(broad(c)).toBe(true))
  for (const c of ['git status', 'docker compose up -d', 'systemctl --user restart x', '/usr/bin/git log'])
    test(`ordinary: ${c}`, () => expect(broad(c)).toBe(false))
})

describe('covers', () => {
  const g = emptyGrants()
  g.programs.add('docker ps')
  g.programs.add('grep')
  g.programs.add('npm run')
  g.exact.add('npm run build')
  test('all units granted', () => expect(covers(g, 'docker ps | grep pg')).toBe(true))
  test('another subcommand is not covered', () => expect(covers(g, 'docker rm -f db')).toBe(false))
  test('one program missing', () => expect(covers(g, 'docker ps && rm -rf x')).toBe(false))
  test('hidden code is never covered by programs', () => expect(covers(g, 'docker ps $(rm x)')).toBe(false))
  test('a broad program is never covered as a program', () => expect(covers(g, 'npm run deploy')).toBe(false))
  test('but its exact command is', () => expect(covers(g, '  npm run build ')).toBe(true))
  test('all covers anything', () => expect(covers({ ...emptyGrants(), all: true }, 'echo $(rm x)')).toBe(true))
  test('nothing granted', () => expect(covers(emptyGrants(), 'docker ps')).toBe(false))
})

describe('coversPath', () => {
  const g = emptyGrants()
  g.dirs.add(grantDir('/home/u/.config/app/settings.json'))
  test('same folder', () => expect(coversPath(g, '/home/u/.config/app/other.json')).toBe(true))
  test('subfolder', () => expect(coversPath(g, '/home/u/.config/app/sub/x')).toBe(true))
  test('a sibling with the same prefix', () => expect(coversPath(g, '/home/u/.config/apple/x')).toBe(false))
  test('the parent', () => expect(coversPath(g, '/home/u/.config/x')).toBe(false))
})

describe('commandPreview', () => {
  test('short command as is', () => expect(commandPreview('docker ps')).toBe('docker ps'))
  test('long script shows two lines and a count', () => expect(commandPreview('a\nb\nc\nd')).toBe('a\nb\n(+2 more lines)'))
  test('long line is cut', () => expect(commandPreview('x'.repeat(150))).toBe(`${'x'.repeat(99)}…`))
})
