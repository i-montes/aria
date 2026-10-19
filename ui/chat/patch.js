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
