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
  test('a message waiting for the next gap says so instead of waiting for the end', () => {
    const html = renderUser({ text: 'hi', queued: true, steer: true, id: '0' }, {}, t)
    expect(html).toContain('chat.user.steer')
    expect(html).not.toContain('chat.user.queued')
    expect(html).toContain('is-steer')
    expect(html).toContain('data-chat="cancel-queued"')
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

describe('attachments in the bubble', () => {
  const raw = 'look\n\n[attached: notes.txt (text/plain, 12 KB) at /x/1-notes.txt]\n[attached image: shot.png at /x/2-shot.png]'
  test('attached lines become chips', () => {
    const html = renderUser({ text: raw }, {}, t)
    expect(html).toContain('p-attach')
    expect(html).toContain('notes.txt')
    expect(html).toContain('12 KB')
    expect(html).toContain('shot.png')
    expect(html).toContain('title="/x/2-shot.png"')
    expect(html).not.toContain('attach-remove')
  })
  test('the raw lines are not shown', () => {
    const html = renderUser({ text: raw }, {}, t)
    expect(html).not.toContain('[attached')
    expect(html).not.toContain('/x/1-notes.txt]')
    expect(html).toContain('>look</p>')
  })
  test('files given with the message are drawn the same; only files leaves no empty bubble', () => {
    const html = renderUser({ text: '', files: [{ name: 'a.pdf', size: '3 KB' }] }, {}, t)
    expect(html).toContain('a.pdf')
    expect(html).not.toContain('p-user-bubble')
  })
  test('a queued message keeps its chips', () => {
    const html = renderUser({ text: raw, queued: true, id: '0' }, {}, t)
    expect(html).toContain('notes.txt')
    expect(html).toContain('chat.user.queued')
  })
})

describe('a scheduled run', () => {
  const t = (key, vars) => (vars ? `${key} at=${vars.at}` : key)
  test('says it was scheduled, and when', () => {
    const html = renderUser({ text: 'My tasks', scheduled: { at: '07:30', late: false } }, {}, t)
    expect(html).toContain('chat.user.scheduled at=07:30')
    expect(html).toContain('is-scheduled')
  })
  test('a late one says it ran late', () => {
    expect(renderUser({ text: 'My tasks', scheduled: { at: '07:30', late: true } }, {}, t)).toContain('chat.user.scheduledLate at=07:30')
  })
  test('the texts exist in both languages', () => {
    for (const lang of ['en', 'es']) for (const key of ['chat.user.scheduled', 'chat.user.scheduledLate']) expect(TEXTS[lang][key], `${lang} ${key}`).toBeTruthy()
  })
})
