// Puts new markup into a container by changing only the top-level children that differ, so a streamed word does not
// rebuild the whole thread: the other messages keep their elements (no flicker, no restarted animations, a text
// selection survives). Markup that did not change at all is not touched.

/** Returns whether anything changed. */
export function patchHtml(container, html) {
  const fresh = document.createElement('template')
  fresh.innerHTML = html
  const next = [...fresh.content.childNodes]
  const old = [...container.childNodes]
  let changed = false
  next.forEach((node, i) => {
    const prev = old[i]
    if (!prev) {
      container.appendChild(node)
      changed = true
    } else if (!same(prev, node)) {
      container.replaceChild(node, prev)
      changed = true
    }
  })
  for (const extra of old.slice(next.length)) {
    extra.remove()
    changed = true
  }
  return changed
}

const same = (a, b) =>
  a.nodeType === b.nodeType && (a.nodeType === Node.ELEMENT_NODE ? a.outerHTML === b.outerHTML : a.nodeValue === b.nodeValue)

/** The markup each container was last patched with, one string per child element. */
const lastItems = new WeakMap()

const element = (html) => {
  const fresh = document.createElement('template')
  fresh.innerHTML = html
  return fresh.content.firstElementChild
}

/** Like `patchHtml`, for markup already split into one string per child element (thread.js `threadItems`): an item
 * whose string did not change is not parsed or compared at all, so a streamed word costs the same in a long
 * conversation as in a short one. A child someone else added or removed makes it rebuild them all. Returns whether
 * anything changed. */
export function patchItems(container, items, make = element) {
  const children = container.children
  const before = lastItems.get(container)
  const prev = before && before.length === children.length ? before : null
  let changed = false
  items.forEach((html, i) => {
    if (prev && prev[i] === html && children[i]) return
    const node = make(html)
    if (children[i]) container.replaceChild(node, children[i])
    else container.appendChild(node)
    changed = true
  })
  while (children.length > items.length) {
    container.removeChild(children[children.length - 1])
    changed = true
  }
  lastItems.set(container, items)
  return changed
}
