// Asking in Aria's dock: the question goes in the dialog's title as the chat's card
// (`{"aria":1,"kind":"sandbox"|"write"|"connector",…}`, see ui/chat/parts/cards.js). The options are the same strings the terminal
// shows, so whatever comes back is handled exactly as before.

/** Is there someone to ask? Pi's terminal, or Aria's dock (RPC). Subagents and `pi -p` have nobody. */
export const canAsk = (ctx: any): boolean => !!ctx?.hasUI && (ctx.mode === 'tui' || ctx.mode === 'rpc')

/** The card for `ctx.ui.select(title, options)`; the first option is the main button. */
export function sandboxCard(kind: 'sandbox' | 'write' | 'connector', fields: Record<string, string>, options: string[]): string {
  return JSON.stringify({
    aria: 1,
    kind,
    ...fields,
    options: options.map((option, i) => (i === 0 ? { value: option, label: option, primary: true } : { value: option, label: option })),
  })
}
