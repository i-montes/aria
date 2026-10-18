// Placeholder shell for the dock (the real one lands with the UI tasks): the open/close classes,
// the `opened` report, and the problem screen. The dock calls `window.aria.receive(message)`.
const post = (message) => window.webkit.messageHandlers.aria.postMessage(message)

window.aria = {
  send: post,
  receive(message) {
    if (message.type === 'open') {
      document.body.classList.remove('open')
      void document.body.offsetWidth
      // Two frames, so the closed state is on screen before the transition starts.
      requestAnimationFrame(() => requestAnimationFrame(() => document.body.classList.add('open')))
    } else if (message.type === 'close') {
      document.body.classList.remove('open')
    } else if (message.type === 'problem') {
      const el = document.getElementById('problem')
      el.textContent = 'Problem: ' + message.code
      el.hidden = false
    }
  },
}

// Tell the dock when the open transition is over (it does not unmap or re-open on its own).
document.querySelector('aside').addEventListener('transitionend', (e) => {
  if (e.propertyName === 'transform' && document.body.classList.contains('open')) post({ type: 'opened' })
})
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') post({ type: 'closeRequest' }) })
post({ type: 'ready' })
