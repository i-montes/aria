// Run: bun test ui/chat/parts/
import { describe, expect, test } from 'bun:test'
import { TEXTS, renderUser, splitMentions } from './user.js'

const t = (key) => key
const hostile = '<script>alert("x")</script> & \'q\''

describe('splitMentions', () => {
  test('an @mention at the start or after a space becomes a chip', () => {
    expect(splitMentions('see @src/app.js now')).toEqual([
      { t: 'see ', chip: false },
      { t: '@src/app.js', chip: true },
      { t: ' now', chip: false },
    ])
  })
  test('trailing punctuation stays outside the chip', () => {
    expect(splitMentions('open @a.js, then @b.js.')).toEqual([
      { t: 'open ', chip: false },
      { t: '@a.js', chip: true },
      { t: ', then ', chip: false },
      { t: '@b.js', chip: true },
      { t: '.', chip: false },
    ])
  })
  test('an @ inside a word (an email) is plain text and a lone @ is not a chip', () => {
    expect(splitMentions('mail me@x.com or @ here')).toEqual([{ t: 'mail me@x.com or @ here', chip: false }])
  })
  test('empty text gives one plain part', () => {
    expect(splitMentions('')).toEqual([{ t: '', chip: false }])
  })
})

describe('renderUser', () => {
  test('escapes the text, including inside a mention chip', () => {
    const html = renderUser({ text: hostile }, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).toContain('&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &#39;q&#39;')
    const chip = renderUser({ text: '@<script>' }, {}, t)
    expect(chip).not.toContain('<script>')
    expect(chip).toContain('p-user-chip')
    expect(chip).toContain('@&lt;script&gt;')
  })
  test('escapes the id of a queued message', () => {
    const html = renderUser({ text: 'hi', queued: true, id: '"><script>' }, {}, t)
    expect(html).not.toContain('<script>')
    expect(html).toContain('data-chat="cancel-queued"')
    expect(html).toContain('data-id="&quot;&gt;&lt;script&gt;"')
  })
  test('a sent message has no queue line and no animation class', () => {
    const html = renderUser({ text: 'hi' }, {}, t)
    expect(html).not.toContain('cancel-queued')
    expect(html).not.toContain('is-queued')
    expect(html).not.toContain('chat.user.queued')
  })
  test('a queued message shows the line, the cancel button and the queued styling', () => {
    const html = renderUser({ text: 'hi', queued: true, id: 'q1' }, {}, t)
    expect(html).toContain('is-queued')
    expect(html).toContain('chat.user.queued')
    expect(html).toContain('chat.user.cancel')
    expect(html).toContain('data-id="q1"')
  })
  test('a command uses the mono styling', () => {
    expect(renderUser({ text: '/plan', command: true }, {}, t)).toContain('is-command')
    expect(renderUser({ text: 'hello' }, {}, t)).not.toContain('is-command')
  })
})

describe('texts', () => {
  test('both languages have the same keys, all prefixed', () => {
    expect(Object.keys(TEXTS.es).sort()).toEqual(Object.keys(TEXTS.en).sort())
    for (const key of Object.keys(TEXTS.en)) expect(key.startsWith('chat.user.')).toBe(true)
  })
})

describe('markdown in the user\'s message', () => {
  test('a message is markdown with its mentions as chips; a slash command is shown as typed', () => {
    const html = renderUser({ text: 'fix **this** in @src/a.rs\n- one' }, {}, (key) => key)
    expect(html).toContain('<strong>this</strong>')
    expect(html).toContain('<span class="p-user-chip">@src/a.rs</span>')
    expect(html).toContain('<ul class="p-md-list"><li>one</li></ul>')
    expect(renderUser({ text: '/plan **x**', command: true }, {}, (key) => key)).toContain('**x**')
  })
})
