// The setup's smoke test (ARIA-SETUP item 37): the answer to "hola" is short (2-4 sentences, no tables, headings or
// lists) and in Spanish. `smoke.sh` pipes Pi's answer here; run directly, it prints the reasons and fails on any.

const SPANISH = /[¡¿]|\b(hola|qué|en qué|puedo|ayudarte|ayudar|estás|cómo)\b/i

/** The reasons an answer to "hola" fails; empty when it passes. */
export function judge(answer: string): string[] {
  const text = answer.trim()
  if (!text) return ['empty']
  const reasons: string[] = []
  if (/^\s*\|.*\|\s*$/m.test(text)) reasons.push('table')
  if (/^\s*#{1,6}\s/m.test(text)) reasons.push('heading')
  if (/^\s*([-*+]|\d+\.)\s/m.test(text)) reasons.push('bullet list')
  const sentences = text.split(/(?<=[.!?])\s+/).filter((s) => /\p{L}/u.test(s))
  if (sentences.length > 4) reasons.push('more than 4 sentences')
  if (!SPANISH.test(text)) reasons.push('not in Spanish')
  return reasons
}

if (import.meta.main) {
  const answer = await new Response(Bun.stdin.stream()).text()
  const reasons = judge(answer)
  if (reasons.length) {
    console.error(`smoke: FAIL (${reasons.join(', ')})`)
    process.exit(1)
  }
  console.log('smoke: pass')
}
