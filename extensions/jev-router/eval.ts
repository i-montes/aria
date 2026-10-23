// Acceptance check for the router criteria: classifies real user messages with the free Jev and
// compares them with the expected hints. Not loaded by Pi (only index.ts is).
// Run: sh tools/parity/scratch-home.sh bun $PWD/extensions/jev-router/eval.ts
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { agentDir, projectsDir } from '../lib/agent-dir.ts'
import { modelRuntime } from '../lib/runtime.ts'
import { bashWriteRequest, CLASSIFIER, classifierRequest, expertRequest, type Hints, mayWrite, parseExpert, parseExpertChoice, parseHints, parsePlanning, planContinuationRequest } from './route.ts'

// A neutral stand-in for a project of the user's.
const PROJECT = `${projectsDir()}/acme/checkout-widget`

type Expect = Partial<Omit<Hints, 'routeP'>> & { route: Hints['route'] | Hints['route'][] }
const CASES: [string, Expect, string?][] = [
  ['hola', { route: 'quick', memory: false, web: false, notion: false, readOnly: true }],
  ['Dime el path completo de donde estas ahora mismo', { route: 'quick', web: false, readOnly: true }],
  ['que hay en el home?', { route: 'quick', readOnly: true }],
  ['Sabes cual es mi nombre?', { route: 'quick', memory: true, readOnly: true }],
  ['Recuerdas algo de lo que hablamos?', { route: 'quick', memory: true, readOnly: true }],
  ['oye, dime mis sitios de componente ui favoritos', { route: ['quick', 'think'], memory: true, readOnly: true }],
  ['Mi framework favorito para cuando voy a hacer un proyecto muy grande es Nextjs', { route: 'quick', memory: true }],
  ['puedes buscar las ultimas noticias de elespectador.com', { route: ['quick', 'focused'], web: true, readOnly: true }],
  ['¿Puedes investigar en internet cómo hacer para instalarle skills al arnés pi?', { route: 'focused', web: true, readOnly: true }],
  ['puedes decirme cuantas tareas tengo pendientes en mi notion?', { route: ['quick', 'focused'], notion: true, readOnly: true }],
  ['cuales son las tareas que tengo vencidas?', { route: ['quick', 'focused'], notion: true, readOnly: true }],
  ['si tuvieras la oportunidad de crearte a ti mismo, desde todas tus bases y componentes, que quisieras tener?', { route: 'think', web: false, readOnly: true }],
  ['¿Me conviene usar Zustand o Redux para el estado del widget de donaciones?', { route: 'think', readOnly: true }],
  ['quiero que analices este proyecto ~/Projects/acme/checkout-widget - quiero que detectes vulnerabilidades - que revises la estructura del proyecto - quiero ver oportunidades de rendimiento', { route: 'parallel', readOnly: true }],
  ['revisa los repos de ~/Projects/acme y dime cuáles tienen dependencias desactualizadas y cuáles no tienen tests', { route: 'parallel', readOnly: true }],
  ['compara Wompi, ePayco y Mercado Pago para donaciones recurrentes en Colombia: comisiones, API y soporte de Bre-B', { route: 'parallel', web: true, readOnly: true }],
  ['Quiero que crees un nuevo proyecto, es una pagina web llamada openmovie en la que se podran ver peliculas, el tema por defecto quiero que sea dark', { route: 'focused', readOnly: false, planWorth: true }],
  ['ejecuta el dev server para poder verlo', { route: 'quick', readOnly: false, planWorth: false }],
  ['puedes hacer un archivo que diga hora en la carpeta de downloads?', { route: 'quick', readOnly: false, planWorth: false }],
  ['Crea un archivo en downloads con un resumen de esa transcripcion', { route: ['quick', 'focused'], readOnly: false, planWorth: false }],
  ['arregla el test que falla en src/lib/money.test.ts', { route: 'focused', readOnly: false, planWorth: false }],
  ['Me mentiste, si hiciste cambios en el proyecto, quiero que dejes el repositorio como estaba', { route: 'focused', readOnly: false }],
  ['Si, agrega el analisis a los logs y di que no es posible hacer una solucion a la medida', { route: 'focused', notion: true, readOnly: false }, 'Puedo agregar este análisis a los logs de la tarea Rifatón en Notion. ¿Lo agrego?'],
  ['sí, hazlo', { route: 'focused', readOnly: false, planWorth: false }, 'Puedo cambiar el tema de tu terminal a verde menta editando ~/.config/alacritty/alacritty.toml. ¿Lo hago?'],
]

const BASH: [string, boolean][] = [
  ['rg -n "innerHTML" src 2>/dev/null | head', false],
  [`cd ${PROJECT} && cp src/api/schemas.ts ./_schemas.ts && bun build ./_schemas.ts --outdir /tmp/x`, true],
  [`cd ${PROJECT} && rm -f client.ts guards.ts wompi.ts`, true],
  ['cat > /tmp/zodtest.ts <<\'EOF\'\nimport { z } from "zod";\nEOF\nbun build /tmp/zodtest.ts --outdir /tmp/out', false],
  ['bunx tsc -p tsconfig.app.json --noEmit > /tmp/tsc.log 2>&1; tail /tmp/tsc.log', false],
  ['git stash && bun run test', true],
  [`cd ${PROJECT} && nohup bun run dev --port 5174 > /tmp/pw-dev-5174.log 2>&1 & sleep 5; cat /tmp/pw-dev-5174.log`, false],
  [`cd ${PROJECT} && nohup bun run smoke:mock > /tmp/pw-mock.log 2>&1 & sleep 3; ss -ltnp | rg :3000`, false],
  [`cd ${PROJECT}/../checkout-api && pnpm install && pnpm dev`, true],
  ['bun run dev > dev.log 2>&1 &', true],
]

const EXPERT_CASES: [string, string | null][] = [
  ['Busca vulnerabilidades en ~/Projects/acme/checkout-widget: XSS, secretos, PII en telemetría y dependencias.', 'security'],
  ['Revisa la estructura del proyecto checkout-widget: organización de carpetas, acoplamiento, tests y código muerto.', 'reviewer'],
  ['Revisa el diff de la rama actual y dime si hay errores o casos borde sin manejar.', 'reviewer'],
  ['Busca oportunidades de rendimiento en el bundle y en el código de checkout-widget.', 'reviewer'],
  ['Crea un componente React de tarjeta de película con póster, título y año, responsive y accesible, en src/components.', 'frontend'],
  ['Haz que el formulario de donación muestre estados de carga y error, y que funcione con teclado.', 'frontend'],
  ['Crea un endpoint POST /api/donations que valide el body, guarde en Postgres y sea idempotente.', 'backend'],
  ['Escribe la migración para añadir la tabla subscriptions y su índice por user_id.', 'backend'],
  ['Investiga en internet las diferencias entre Vite 7 y Rspack para un proyecto React grande, con fuentes.', 'researcher'],
  ['Averigua la última versión estable de Next.js y qué cambió respecto a la anterior.', 'researcher'],
  ['Cuenta cuántos archivos .ts hay en ~/.aria/agent/extensions.', null],
  ['Dime qué versión de bun está instalada.', null],
]
// The experts as Aria wrote them for its settings (the scratch home's agent directory).
const AGENTS = join(agentDir(), 'agents')
const experts = readdirSync(AGENTS).flatMap((f) => {
  const e = parseExpert(readFileSync(`${AGENTS}/${f}`, 'utf8'))
  return e ? [e] : []
})
const runtime = await modelRuntime()
const jev = runtime.getModelOfType('classifier', CLASSIFIER.provider, CLASSIFIER.id)
if (!jev) throw new Error('classifier not found')

let ok = 0
let total = 0
const fails: string[] = []
await Promise.all(
  CASES.map(async ([message, expect, previous]) => {
    const t0 = Date.now()
    const result = await runtime.classify(jev, classifierRequest(message, previous ?? '') as any)
    const ms = Date.now() - t0
    const h = parseHints(result)
    const bad: string[] = []
    if (!h) bad.push(`no hints (${result.stopReason} ${result.errorMessage ?? ''})`)
    else {
      const routes = Array.isArray(expect.route) ? expect.route : [expect.route]
      if (!routes.includes(h.route)) bad.push(`route=${h.route}(${h.routeP.toFixed(2)}) want ${routes.join('|')} ${JSON.stringify(result.answers.route.probabilities)}`)
      for (const key of ['memory', 'web', 'notion', 'readOnly', 'planWorth'] as const) {
        if (expect[key] !== undefined && expect[key] !== h[key]) {
          const q = key === 'readOnly' ? 'changes' : key === 'planWorth' ? 'sizable' : key
          bad.push(`${key}=${h[key]} (p=${result.answers[q]?.probability?.toFixed(2)})`)
        }
      }
    }
    const checks = 1 + (['memory', 'web', 'notion', 'readOnly', 'planWorth'] as const).filter((k) => expect[k] !== undefined).length
    total += checks
    ok += checks - bad.length
    if (bad.length) fails.push(`✗ [${ms}ms] ${message.slice(0, 70)}\n    ${bad.join('\n    ')}`)
  }),
)
await Promise.all(
  BASH.map(async ([command, writes]) => {
    total += 1
    const pre = mayWrite(command)
    if (!pre) {
      if (writes) fails.push(`✗ bash prefilter missed a write: ${command.slice(0, 70)}`)
      else ok += 1
      return
    }
    const r = await runtime.classify(jev, bashWriteRequest(command, PROJECT, ['/tmp']) as any)
    const p = r.answers?.writes?.probability ?? NaN
    if (p >= 0.5 === writes) ok += 1
    else fails.push(`✗ bash writes=${p.toFixed(2)} want ${writes}: ${command.slice(0, 70)}`)
  }),
)
// Read-only turns choose among read-only experts only.
const READ_ONLY_CASES: [string, string][] = [
  ['Busca oportunidades de rendimiento en el bundle y en el código de checkout-widget.', 'reviewer'],
  ['Revisa los componentes de UI del proyecto: accesibilidad, estados de error y responsive.', 'reviewer'],
  ['Analiza la API del backend: validación de entradas, manejo de errores y consultas lentas.', 'reviewer'],
]
const readOnlyExperts = experts.filter((e) => e.readOnly)
await Promise.all(
  READ_ONLY_CASES.map(async ([task, want]) => {
    total += 1
    const r = await runtime.classify(jev, expertRequest(task, readOnlyExperts) as any)
    const got = parseExpertChoice(r)
    if (got === want) ok += 1
    else fails.push(`✗ (read-only) expert=${got} want ${want} ${JSON.stringify(r.answers?.expert?.probabilities)}: ${task.slice(0, 60)}`)
  }),
)
await Promise.all(
  EXPERT_CASES.map(async ([task, want]) => {
    total += 1
    const r = await runtime.classify(jev, expertRequest(task, experts) as any)
    const got = parseExpertChoice(r)
    if (got === want) ok += 1
    else fails.push(`✗ experto=${got} want ${want} ${JSON.stringify(r.answers?.expert?.probabilities)}: ${task.slice(0, 60)}`)
  }),
)
// In plan mode, a message below 0.3 ends planning (index.ts): planning answers must stay above it.
const ASK = 'Antes de proponer el diseño: ¿el consentimiento de WhatsApp debe guardarse solo en la orden o también en el usuario? (a) solo orden (b) ambos'
const SPEC = 'Escribí el spec en docs/superpowers/specs/2026-10-14-whatsapp-consent-design.md. Revísalo y dime si apruebas o quieres cambios antes de escribir el plan.'
const PLAN_CASES: [string, string, boolean][] = [
  ['a', ASK, true],
  ['solo en la orden, la info ya vive ahí', ASK, true],
  ['la b, pero sin tocar Mailchimp', ASK, true],
  ['¿qué pasa si el usuario ya tenía el checkbox marcado?', ASK, true],
  ['aprobado, escribe el plan', SPEC, true],
  ['cambia la sección de riesgos: el caso más común es que el usuario repita la donación', SPEC, true],
  ['quita la parte de los tests e2e del alcance', SPEC, true],
  ['oye, levanta el dev server del widget que quiero ver una cosa', ASK, false],
  ['corre los tests de checkout-api y dime si pasan', SPEC, false],
  ['¿cuántas tareas tengo vencidas en Notion?', SPEC, false],
  ['arregla el error de tipos en src/lib/money.ts', ASK, false],
]
await Promise.all(
  PLAN_CASES.map(async ([message, previous, planning]) => {
    total += 1
    const r = await runtime.classify(jev, planContinuationRequest(message, previous) as any)
    const p = parsePlanning(r)
    if (p !== null && p >= 0.3 === planning) ok += 1
    else fails.push(`✗ planning=${p?.toFixed(2)} want ${planning}: ${message.slice(0, 60)}`)
  }),
)
console.log(fails.join('\n'))
console.log(`\n${ok}/${total} checks passed`)
