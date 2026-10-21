// Run: bun test extensions
import { describe, expect, test } from 'bun:test'
import { actionOf, isWrite, serverOf } from './classify.ts'

describe('what a connector call does', () => {
  test('annotations_decide_first', () => {
    expect(isWrite('mcp__notion__notion_create_pages', { readOnlyHint: true })).toBe(false)
    expect(isWrite('mcp__notion__notion_search', { readOnlyHint: false })).toBe(true)
    expect(isWrite('mcp__notion__notion_get_page', { destructiveHint: true })).toBe(true)
    // Only a hint about something else: the name decides.
    expect(isWrite('mcp__notion__notion_search', {})).toBe(false)
  })

  test('names_without_annotations', () => {
    expect(isWrite('mcp__notion__notion_search')).toBe(false)
    expect(isWrite('mcp__notion__notion_create_pages')).toBe(true)
    expect(isWrite('mcp__notion__notion_update_page')).toBe(true)
    expect(isWrite('mcp__pages__createPage')).toBe(true)
    expect(isWrite('mcp__pages__getPage')).toBe(false)
    expect(isWrite('mcp__rea__open_binary')).toBe(true)
    expect(isWrite('mcp__rea__rea_open_binary')).toBe(true)
    expect(isWrite('mcp__tavily__tavily_search')).toBe(false)
    expect(isWrite('mcp__docs__listDocuments')).toBe(false)
    expect(isWrite('list_mcp_resources')).toBe(false)
    expect(isWrite('list_mcp_resource_templates')).toBe(false)
    expect(isWrite('read_mcp_resource')).toBe(false)
    // Bare names, as a server offers them.
    expect(isWrite('getPage')).toBe(false)
    expect(isWrite('createPage')).toBe(true)
    // A name with the collision hash Pi adds.
    expect(isWrite('mcp__notion__notion_fetch_1a2b3c4d')).toBe(false)
  })

  test('the_server_is_the_middle_part', () => {
    expect(serverOf('mcp__notion__notion_search')).toBe('notion')
    expect(serverOf('mcp__my_server__do__thing')).toBe('my_server')
    expect(serverOf('list_mcp_resources')).toBe('*')
    expect(serverOf('read_mcp_resource')).toBe('*')
    expect(serverOf('bash')).toBe(null)
    expect(serverOf('mcp__')).toBe(null)
  })

  test('the action is the name made readable', () => {
    expect(actionOf('mcp__notion__notion_create_pages')).toBe('create pages')
    expect(actionOf('notion-create-pages')).toBe('notion create pages')
    expect(actionOf('mcp__pages__createPage')).toBe('create page')
    expect(actionOf('mcp__rea__open_binary')).toBe('open binary')
    expect(actionOf('mcp__notion__notion_fetch_1a2b3c4d')).toBe('fetch')
  })
})
