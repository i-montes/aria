import { describe, expect, test } from 'bun:test'
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describeMcp, findSkills, mask, mcpEntry, mcpHome, namedSkills, parseSkillSource, skillName, validPackage, validSkillName, withServer } from './install.ts'

describe('parseSkillSource', () => {
  test('a GitHub repository, folder and SKILL.md', () => {
    expect(parseSkillSource('https://github.com/acme/skills')).toEqual({ kind: 'git', repo: 'https://github.com/acme/skills', path: '' })
    expect(parseSkillSource('https://github.com/acme/skills.git')).toEqual({ kind: 'git', repo: 'https://github.com/acme/skills', path: '' })
    expect(parseSkillSource('https://github.com/acme/skills/tree/main/skills/pdf')).toEqual({ kind: 'git', repo: 'https://github.com/acme/skills', ref: 'main', path: 'skills/pdf' })
    expect(parseSkillSource('https://github.com/acme/skills/blob/v2/skills/pdf/SKILL.md')).toEqual({ kind: 'git', repo: 'https://github.com/acme/skills', ref: 'v2', path: 'skills/pdf' })
    expect(parseSkillSource('https://github.com/acme/pdf/blob/main/SKILL.md')).toEqual({ kind: 'git', repo: 'https://github.com/acme/pdf', ref: 'main', path: '' })
  })

  test('a raw SKILL.md and other git URLs', () => {
    expect(parseSkillSource('https://raw.githubusercontent.com/acme/skills/main/pdf/SKILL.md')).toEqual({ kind: 'file', url: 'https://raw.githubusercontent.com/acme/skills/main/pdf/SKILL.md' })
    expect(parseSkillSource('https://gitlab.com/acme/skills.git')).toEqual({ kind: 'git', repo: 'https://gitlab.com/acme/skills.git', path: '' })
    expect(parseSkillSource('git@github.com:acme/skills.git')).toEqual({ kind: 'git', repo: 'git@github.com:acme/skills.git', path: '' })
  })

  test('anything else is not a skill link', () => {
    expect(parseSkillSource('https://github.com/acme/skills/issues/3')).toBeNull()
    expect(parseSkillSource('https://example.com/blog/my-skill')).toBeNull()
    expect(parseSkillSource('file:///etc/SKILL.md')).toBeNull()
    expect(parseSkillSource('not a link')).toBeNull()
  })
})

describe('skills in a folder', () => {
  const root = mkdtempSync(join(tmpdir(), 'install-test-'))
  const skill = (dir: string, text: string) => {
    mkdirSync(join(root, dir), { recursive: true })
    writeFileSync(join(root, dir, 'SKILL.md'), text)
  }
  skill('skills/pdf', '---\nname: pdf-tools\ndescription: PDFs\n---\n# PDF')
  skill('skills/pdf/inner', '---\nname: inner\n---\n')
  skill('skills/notes', '# no front matter')
  skill('node_modules/dep', '---\nname: dep\n---\n')

  test('finds each skill once, not inside another skill or node_modules', () => {
    expect(namedSkills(findSkills(root))).toEqual([
      { name: 'notes', folder: join(root, 'skills/notes') },
      { name: 'pdf-tools', folder: join(root, 'skills/pdf') },
    ])
  })

  test('a folder that is a skill is the only one', () => {
    expect(findSkills(join(root, 'skills/pdf'))).toEqual([join(root, 'skills/pdf')])
  })

  test('the name comes from the front matter, quoted or not, else the folder', () => {
    expect(skillName('---\nname: "web-search"\n---\n', '/x/y')).toBe('web-search')
    expect(skillName('# title', '/x/my-skill')).toBe('my-skill')
    expect(validSkillName('pdf-tools')).toBe(true)
    expect(validSkillName('PDF Tools')).toBe(false)
    expect(validSkillName('../evil')).toBe(false)
  })
})

describe('mcpEntry', () => {
  test('a remote server with a key', () => {
    expect(mcpEntry({ name: 'github', url: 'https://api.example.com/mcp', headers: { Authorization: 'Bearer abc' }, description: ' GitHub ' })).toEqual({
      entry: { url: 'https://api.example.com/mcp', headers: { Authorization: 'Bearer abc' }, description: 'GitHub' },
    })
  })

  test('a local server', () => {
    expect(mcpEntry({ name: 'rea', command: 'npx', args: ['-y', 'rea-mcp'], env: { KEY: 'x' } })).toEqual({ entry: { command: 'npx', args: ['-y', 'rea-mcp'], env: { KEY: 'x' } } })
  })

  test('what cannot be an entry', () => {
    for (const bad of [
      { name: 'bad name', url: 'https://x.com/mcp' },
      { name: 'x' },
      { name: 'x', url: 'https://x.com/mcp', command: 'npx' },
      { name: 'x', url: 'https://x.com/sse' },
      { name: 'x', url: 'ftp://x.com/mcp' },
      { name: 'x', command: 'npx -y server' },
      { name: 'x', command: 'npx', headers: { a: 'b' } },
      { name: 'x', url: 'https://x.com/mcp', env: { a: 'b' } },
    ]) {
      expect('error' in mcpEntry(bad)).toBe(true)
    }
  })
})

test('the confirmation masks secrets', () => {
  expect(mask('Bearer abcd-efgh-1234567890')).toBe('Bearer abcd…')
  expect(mask('short')).toBe('••••')
  expect(mask('${GITHUB_TOKEN}')).toBe('${GITHUB_TOKEN}')
  expect(describeMcp('x', { url: 'https://x.com/mcp', headers: { Authorization: 'Bearer abcdefghijkl' } })).toBe('x: https://x.com/mcp\nAuthorization=Bearer abcd…')
  expect(describeMcp('rea', { command: 'npx', args: ['-y', 'rea-mcp'], env: { KEY: 'abcdefghijkl' } })).toBe('rea: npx -y rea-mcp\nKEY=abcd…')
})

test('withServer keeps every other server and key', () => {
  const file = { mcpServers: { engram: { command: 'engram' } }, other: 1 }
  expect(withServer(file, 'rea', { command: 'rea' })).toEqual({ mcpServers: { engram: { command: 'engram' }, rea: { command: 'rea' } }, other: 1 })
  expect(withServer(null, 'rea', { command: 'rea' })).toEqual({ mcpServers: { rea: { command: 'rea' } } })
})

test('the MCP files live in Aria\'s home, or the agent folder outside Aria', () => {
  expect(mcpHome('/h/.aria/agent', '/h/.aria/settings.json')).toBe('/h/.aria')
  expect(mcpHome('/h/.pi/agent', '')).toBe('/h/.pi/agent')
})

test('package sources', () => {
  expect(validPackage('npm:pi-parallel-agents@0.2.0')).toBe(true)
  expect(validPackage('npm:@scope/pkg')).toBe(true)
  expect(validPackage('git:github.com/obra/superpowers@bb92a77')).toBe(true)
  expect(validPackage('https://github.com/acme/pi-pkg')).toBe(true)
  expect(validPackage('npm:x; rm -rf ~')).toBe(false)
  expect(validPackage('--help')).toBe(false)
})
