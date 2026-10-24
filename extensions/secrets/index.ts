// The person's secrets in Aria (docs/superpowers/plans/2026-10-22-secrets.md). The model never sees a value:
// - `request_secret` opens the chat's secret card; the page sends what the person types straight to the dock, which keeps
//   it in the keyring, and only then answers the card with the name it was saved under.
// - `secret_import` asks the dock to read a file and keep one value from it (the value stays in the dock).
// - `secrets_list` names what is kept. A command uses one as `$ARIA_SECRET_<NAME>` (the sandbox adds it at spawn time).
// - Every tool output is masked before the model reads it: stored values by their name, key shapes and secret-named
//   assignments by `[masked …]`, with a note on how to keep one.
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { DOCK_REFUSAL, jobsClient, refusal, textResult } from '../background/client.ts'
import { isChild, object, str } from '../background/run.ts'
import { ariaConfig } from '../lib/config.ts'
import { envName, foldName, listSecrets, maskNote, maskText, storedValues } from '../lib/secrets.ts'

const USE = (name: string) =>
  `Use it in commands as \`$${envName(name)}\` (the command gets it in its environment); never print it, echo it or write it to a file.`

/** The secret card: `ctx.ui.input(card)`; the page answers `saved:<name>` once the dock has kept it. */
export function secretCard(name: string, purpose: string): string {
  return JSON.stringify({ aria: 1, kind: 'secret', name, purpose, env: envName(name) })
}

/** The page's answer to the card: the name it was saved under, or null (cancelled, or anything else). */
export function savedName(reply: unknown): string | null {
  const match = typeof reply === 'string' ? /^saved:([a-z0-9-]+)$/.exec(reply.trim()) : null
  return match ? match[1] : null
}

export default function secrets(pi: ExtensionAPI) {
  pi.registerTool({
    name: 'request_secret',
    label: 'Ask for a secret',
    description: [
      'Ask the person for a password, API key or token without seeing it: the chat shows them a secret field, the value goes straight to the system keyring, and you get back only its name.',
      'Use it whenever a task needs a secret the person has not saved yet (check `secrets_list` first). Never ask them to paste one in the chat.',
      '`name`: short, lower case with dashes (`github-token`, `stripe-test-key`). `purpose`: one line the person reads on the card, what it is for.',
    ].join('\n'),
    parameters: object({ name: str('The name to keep it under.'), purpose: str('What it is for, in one line.') }, ['name', 'purpose']),
    async execute(_id: string, params: any, signal: any, _onUpdate: any, ctx: any) {
      const name = foldName(String(params?.name ?? ''))
      if (!name) return refusal('The name is letters, digits and dashes (`github-token`).')
      if (isChild(ctx)) return refusal('Only the conversation with the person can ask for a secret, not a subagent.')
      if (ariaConfig().noAsk) return refusal('Nobody can answer here: this is a scheduled task running by itself.')
      if (!ctx.hasUI || ctx.mode !== 'rpc') return refusal("Secrets are asked in Aria's chat.")
      const reply = await ctx.ui.input(secretCard(name, String(params?.purpose ?? '').trim()), undefined, { signal })
      const saved = savedName(reply)
      if (!saved) return textResult('The person did not give the secret (they closed the card). Do not ask for it in the chat.', false)
      return textResult(`Saved in the keyring as \`${saved}\`. ${USE(saved)}`, false, { name: saved })
    },
  } as any)

  pi.registerTool({
    name: 'secret_import',
    label: 'Keep a secret from a file',
    description: [
      'Keep a secret that sits in a file (a `.env`, a config) in the system keyring, without you seeing it: Aria reads the file and keeps the value of `key` (the variable or field that holds it; leave it out when the file has one key-shaped value).',
      'Use it when the person asks, or after offering it when an output of yours had a masked secret. Tell them afterwards that the file still holds it (they may want it removed).',
      'Only a file inside this project or the projects folder works (no `..`, and nothing in a hidden folder of the home folder or in Aria\'s own folder); anything else is refused with one sentence that does not say whether the file exists.',
    ].join('\n'),
    parameters: object(
      {
        path: str('The file inside this project or the projects folder, absolute or relative to the working folder.'),
        key: str('The variable or field that holds it (`OPENAI_API_KEY`).'),
        name: str('The name to keep it under (`openai-key`).'),
        purpose: str('What it is for, in one line.'),
      },
      ['path', 'name'],
    ),
    async execute(_id: string, params: any, _signal: any, _onUpdate: any, ctx: any) {
      const name = foldName(String(params?.name ?? ''))
      if (!name) return refusal('The name is letters, digits and dashes (`openai-key`).')
      const client = jobsClient()
      if (!client) return refusal(DOCK_REFUSAL.replace('Background work', 'Secrets'))
      try {
        const answer = await client.request('secret_import', {
          path: String(params.path ?? ''),
          key: params.key ? String(params.key) : undefined,
          name,
          purpose: String(params.purpose ?? ''),
          cwd: ctx.cwd,
        })
        if (!answer?.ok) return refusal(`${answer?.error ?? "Aria's dock did not answer"}`)
        return textResult(`Kept in the keyring as \`${answer.name}\` (the file still holds it). ${USE(answer.name)}`, false, { name: answer.name })
      } catch (e: any) {
        return refusal(`${e?.message ?? e}`)
      }
    },
  } as any)

  pi.registerTool({
    name: 'secrets_list',
    label: 'Saved secrets',
    description: 'The secrets the person keeps in Aria, by name and purpose, and the variable a command reads each from. Never their values.',
    parameters: object({}, []),
    async execute() {
      const list = listSecrets()
      if (!list.length) return textResult('No secrets saved yet.', false)
      return textResult(list.map((s) => `${s.name}${s.purpose ? ` (${s.purpose})` : ''}: $${envName(s.name)}`).join('\n'), false)
    },
  } as any)

  // Every output is masked before the model reads it.
  pi.on('tool_result', (event: any) => {
    const content = Array.isArray(event.content) ? event.content : []
    if (!content.some((c: any) => c?.type === 'text' && c.text)) return
    const stored = storedValues()
    const found: { kind: string; line: number }[] = []
    const masked = content.map((c: any) => {
      if (c?.type !== 'text' || !c.text) return c
      const m = maskText(c.text, stored)
      found.push(...m.found)
      return m.text === c.text ? c : { ...c, text: m.text }
    })
    if (masked.every((c: any, i: number) => c === content[i])) return
    if (found.length) {
      const path = typeof event.input?.path === 'string' ? event.input.path : undefined
      const last = masked.length - 1
      masked[last] = { ...masked[last], text: `${masked[last].text ?? ''}${maskNote(found, path)}` }
    }
    return { content: masked, details: event.details }
  })
}
