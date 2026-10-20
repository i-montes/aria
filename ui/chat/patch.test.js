// Run: bun test ui
import { expect, test } from 'bun:test'
import { patchItems } from './patch.js'

// A container with just what patchItems uses; each child remembers the markup it was made from.
function fakeContainer() {
  const made = []
  const container = {
    children: [],
    appendChild(node) { this.children.push(node) },
    replaceChild(node, old) { this.children[this.children.indexOf(old)] = node },
    removeChild(old) { this.children.splice(this.children.indexOf(old), 1) },
  }
  const make = (html) => {
    made.push(html)
    return { html }
  }
  return { container, made, make }
}

test('only the items whose markup changed are built and swapped in', () => {
  const { container, made, make } = fakeContainer()
  patchItems(container, ['<a>', '<b>', '<c>'], make)
  const [a, b] = container.children
  made.length = 0
  patchItems(container, ['<a>', '<b>', '<c2>', '<d>'], make)
  expect(made).toEqual(['<c2>', '<d>'])
  expect(container.children[0]).toBe(a)
  expect(container.children[1]).toBe(b)
  expect(container.children.map((n) => n.html)).toEqual(['<a>', '<b>', '<c2>', '<d>'])
})

test('items that are gone are removed, and nothing changes when nothing did', () => {
  const { container, made, make } = fakeContainer()
  patchItems(container, ['<a>', '<b>'], make)
  made.length = 0
  expect(patchItems(container, ['<a>', '<b>'], make)).toBe(false)
  expect(made).toEqual([])
  expect(patchItems(container, ['<a>'], make)).toBe(true)
  expect(container.children.map((n) => n.html)).toEqual(['<a>'])
})

test('a child someone else put in or took out makes the container rebuild', () => {
  const { container, made, make } = fakeContainer()
  patchItems(container, ['<a>', '<b>'], make)
  container.children.pop()
  made.length = 0
  patchItems(container, ['<a>', '<b>'], make)
  expect(container.children.map((n) => n.html)).toEqual(['<a>', '<b>'])
})
