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

/**
 * Puts new markup into a container by updating the elements already there wherever they match (same tag, and the
 * same `data-id` when they carry one): their attributes and text change in place, so an animation that is running
 * does not start again and a CSS transition moves a bar instead of redrawing it. Anything that does not match is
 * replaced. For a view redrawn many times a second (the `/debug` trace).
 */
export function morphHtml(container, html) {
  const fresh = document.createElement('template')
  fresh.innerHTML = html
  morphChildren(container, fresh.content)
}

function morphChildren(into, from) {
  const next = [...from.childNodes]
  const old = [...into.childNodes]
  next.forEach((node, i) => {
    const prev = old[i]
    if (!prev) into.appendChild(node)
    else if (!morphNode(prev, node)) into.replaceChild(node, prev)
  })
  for (const extra of old.slice(next.length)) extra.remove()
}

/** Updates `prev` to look like `node`; false when they are not the same kind of node and `prev` must be replaced. */
function morphNode(prev, node) {
  if (prev.nodeType !== node.nodeType) return false
  if (prev.nodeType !== Node.ELEMENT_NODE) {
    if (prev.nodeValue !== node.nodeValue) prev.nodeValue = node.nodeValue
    return true
  }
  if (prev.tagName !== node.tagName || prev.getAttribute('data-id') !== node.getAttribute('data-id')) return false
  for (const { name } of [...prev.attributes]) if (!node.hasAttribute(name)) prev.removeAttribute(name)
  for (const { name, value } of [...node.attributes]) if (prev.getAttribute(name) !== value) prev.setAttribute(name, value)
  morphChildren(prev, node)
  return true
}
