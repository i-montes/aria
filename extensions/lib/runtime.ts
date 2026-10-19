// The pinned Pi's model runtime, for the evals that classify with real models (jev-router/eval.ts,
// sandbox/review-eval.ts). tools/parity/scratch-home.sh sets ARIA_PI_DIST to its pi-coding-agent dist folder.
export async function modelRuntime(): Promise<any> {
  const dist = process.env.ARIA_PI_DIST
  if (!dist) throw new Error('ARIA_PI_DIST is not set: run this through tools/parity/scratch-home.sh')
  const { ModelRuntime } = await import(`${dist}/index.js`)
  return ModelRuntime.create()
}
