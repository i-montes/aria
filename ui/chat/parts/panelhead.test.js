// Run: bun test ui
import { describe, expect, test } from 'bun:test'
import { renderPanelHead } from './panelhead.js'

describe('the panel head', () => {
  test('carries the back button\'s attributes, its label, the title and the optional lines, escaped', () => {
    const html = renderPanelHead({ back: { attrs: 'data-chat="memory-back"', label: 'Back <to> chat' }, title: 'Memory & co', meta: 'engram · aria', aside: '3 active' })
    expect(html).toContain('data-chat="memory-back"')
    expect(html).toContain('aria-label="Back &lt;to&gt; chat"')
    expect(html).toContain('Memory &amp; co')
    expect(html).toContain('<span class="p-head-meta">engram · aria</span>')
    expect(html).toContain('<span class="p-head-aside">3 active</span>')
  })

  test('leaves out the lines it was not given', () => {
    const html = renderPanelHead({ back: { attrs: 'data-action="settings-exit"', label: 'Back' }, title: 'Settings' })
    expect(html).not.toContain('p-head-meta')
    expect(html).not.toContain('p-head-aside')
  })
})
