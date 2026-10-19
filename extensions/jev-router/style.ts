// The answer-style reminder repeated after each user message: a rule buried in a long system prompt gets ignored by
// smaller models. One per style of Aria's setup (assistant.style); `very-short` is the owner's own.

export const STYLE_NOTE =
  'Answer style: very short and direct, the answer first, at most ~80 words. No tables, no headings, no closing summary ("en resumen…") or offer. Longer only if the user asked for detail.'

const NOTES: Record<string, string> = {
  'very-short': STYLE_NOTE,
  brief: 'Answer style: brief, the answer first, a short paragraph or a short list when it helps. No tables or headings unless asked, no closing summary or offer.',
  detailed: 'Answer style: detailed, the answer first, then the explanation that makes it clear, with short sections or lists when they help. No padding.',
  technical: 'Answer style: technical and precise, the answer first, with the exact code, commands and file paths that decide it. No filler, no recap.',
  teaching: 'Answer style: teach step by step, the answer first, then how it works with a small example; explain any term the user may not know.',
}

export const styleNote = (style: string | undefined): string => NOTES[style ?? ''] ?? STYLE_NOTE
