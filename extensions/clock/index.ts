// The date and time for the model: every message the user sends gets a hidden `[now]` line with the local date, time,
// weekday and time zone (see clock.ts), so "today", deadlines and relative dates in answers and memories are right
// without the model having to run `date`. It is sent after the user's message, like the router's hint, so the
// system prompt and its cache never change.
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { nowLine } from './clock.ts'

export default function (pi: ExtensionAPI) {
  pi.on('before_agent_start', (event: any) => {
    if (event.messageType !== 'user') return
    return { message: { customType: 'clock', content: nowLine(), display: false } }
  })
}
