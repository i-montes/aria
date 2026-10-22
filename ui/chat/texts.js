// Every text of the chat in both languages: each part's own `TEXTS` merged with the few that belong
// to the composition (the empty thread, results of cards, the header, the toast).
import * as answers from './parts/answers.js'
import * as assistant from './parts/assistant.js'
import * as attachments from './parts/attachments.js'
import * as background from './parts/background.js'
import * as commands from './parts/commands.js'
import * as composer from './parts/composer.js'
import * as connect from './parts/connect.js'
import * as divider from './parts/divider.js'
import * as doc from './parts/document.js'
import * as files from './parts/files.js'
import * as index from './parts/index.js'
import * as jev from './parts/jev.js'
import * as memory from './parts/memory.js'
import * as schedules from './parts/schedules.js'
import * as models from './parts/models.js'
import * as note from './parts/note.js'
import * as permission from './parts/permission.js'
import * as pill from './parts/pill.js'
import * as phases from './parts/phases.js'
import * as question from './parts/question.js'
import * as quota from './parts/quota.js'
import * as quotabars from './parts/quotabars.js'
import * as session from './parts/session.js'
import * as subagents from './parts/subagents.js'
import * as tabs from './parts/tabs.js'
import * as tasks from './parts/tasks.js'
import * as tools from './parts/tools.js'
import * as user from './parts/user.js'
import * as working from './parts/working.js'

export const PARTS = [answers, assistant, attachments, background, commands, composer, connect, divider, doc, files, index, jev, memory, models, note, permission, phases, pill, question, quota, quotabars, schedules, session, subagents, tabs, tasks, tools, user, working]

const OWN = {
  en: {
    'chat.empty.title': 'NEW SESSION',
    'chat.loading': 'Loading the conversation',
    'chat.empty.text': 'Ask for something, or type / to see the commands.',
    'chat.stoppedByYou': 'stopped by you',
    'chat.compacted': 'context compacted',
    'chat.earlier.divider': 'Earlier in this conversation',
    'chat.earlier.summary': 'Summary',
    'chat.earlier.show': 'Show earlier',
    'chat.earlier.loading': 'Loading earlier messages…',
    'chat.askSkipped': 'skipped the questions',
    'chat.result.allow': 'allowed once',
    'chat.result.always': 'allowed for this session',
    'chat.result.deny': 'blocked',
    'chat.result.free': 'switched to the free model',
    'chat.result.keep': 'kept the current model',
    'chat.close': 'Close Aria',
    'chat.agents.title': 'Subagents',
    'chat.agents.back': 'Back to chat',
    'chat.agents.footnote': 'select an agent to open its session',
    'chat.agents.empty': 'No subagents are running. When Aria delegates work, they appear here.',
    'chat.toast.open': 'Open',
    'chat.toast.needs': 'needs your answer',
    'chat.memory.open': 'Memory',
    'chat.cmd.memory': 'Open the memory panel',
    'chat.cmd.schedules': 'Scheduled tasks: when each runs, pause or delete them',
    'chat.cmd.model': 'Switch the model',
    'chat.cmd.connect': 'Connect a provider (Anthropic, OpenCode Go)',
    'chat.cmd.compact': 'Compact the context (optionally say what to keep)',
    'chat.cmd.session': 'This session in numbers: messages, tokens, cost and context',
    'chat.cmd.copy': 'Copy the last answer',
    'chat.cmd.export': 'Export this session as an HTML page',
    'chat.cmd.reload': "Restart this tab's Pi to read its extensions, skills and templates again",
    'chat.compact.failed': 'Could not compact: {error}',
    'chat.session.stats': '{messages} messages · {tools} tool calls · {tokens} tokens · ${cost}',
    'chat.session.context': 'context {percent}% of {window} tokens',
    'chat.session.failed': 'Could not read the session: {error}',
    'chat.copy.done': 'Last answer copied',
    'chat.copy.none': 'There is no answer to copy yet',
    'chat.copy.failed': 'Could not copy to the clipboard',
    'chat.export.done': 'Session exported to {path}',
    'chat.export.failed': 'Could not export: {error}',
    'chat.reload.done': "This tab's Pi restarts (after the current run, if one is going) and reads its extensions, skills and templates again",
  },
  es: {
    'chat.empty.title': 'SESIÓN NUEVA',
    'chat.loading': 'Cargando la conversación',
    'chat.empty.text': 'Pídele algo, o escribe / para ver los comandos.',
    'chat.stoppedByYou': 'detenido por ti',
    'chat.compacted': 'contexto compactado',
    'chat.earlier.divider': 'Antes en esta conversación',
    'chat.earlier.summary': 'Resumen',
    'chat.earlier.show': 'Ver lo anterior',
    'chat.earlier.loading': 'Cargando lo anterior…',
    'chat.askSkipped': 'omitió las preguntas',
    'chat.result.allow': 'permitido una vez',
    'chat.result.always': 'permitido en esta sesión',
    'chat.result.deny': 'bloqueado',
    'chat.result.free': 'cambió al modelo gratuito',
    'chat.result.keep': 'mantuvo el modelo actual',
    'chat.close': 'Cerrar Aria',
    'chat.agents.title': 'Subagentes',
    'chat.agents.back': 'Volver al chat',
    'chat.agents.footnote': 'elige un agente para abrir su sesión',
    'chat.agents.empty': 'No hay subagentes en marcha. Cuando Aria delegue trabajo, aparecerán aquí.',
    'chat.toast.open': 'Abrir',
    'chat.toast.needs': 'necesita tu respuesta',
    'chat.memory.open': 'Memoria',
    'chat.cmd.memory': 'Abrir el panel de memoria',
    'chat.cmd.schedules': 'Tareas programadas: cuándo corre cada una, pausarlas o borrarlas',
    'chat.cmd.model': 'Cambiar de modelo',
    'chat.cmd.connect': 'Conectar un proveedor (Anthropic, OpenCode Go)',
    'chat.cmd.compact': 'Compactar el contexto (opcional: qué conservar)',
    'chat.cmd.session': 'Esta sesión en números: mensajes, tokens, costo y contexto',
    'chat.cmd.copy': 'Copiar la última respuesta',
    'chat.cmd.export': 'Exportar esta sesión como página HTML',
    'chat.cmd.reload': 'Reiniciar el Pi de esta pestaña para leer de nuevo sus extensiones, skills y plantillas',
    'chat.compact.failed': 'No se pudo compactar: {error}',
    'chat.session.stats': '{messages} mensajes · {tools} llamadas a herramientas · {tokens} tokens · ${cost}',
    'chat.session.context': 'contexto al {percent}% de {window} tokens',
    'chat.session.failed': 'No se pudo leer la sesión: {error}',
    'chat.copy.done': 'Última respuesta copiada',
    'chat.copy.none': 'Todavía no hay una respuesta para copiar',
    'chat.copy.failed': 'No se pudo copiar al portapapeles',
    'chat.export.done': 'Sesión exportada en {path}',
    'chat.export.failed': 'No se pudo exportar: {error}',
    'chat.reload.done': 'El Pi de esta pestaña se reinicia (al terminar el run en curso, si hay uno) y lee de nuevo sus extensiones, skills y plantillas',
  },
}

/** `{ en, es }`: the parts' texts and the composition's, ready to merge into the page's catalogs. */
export function chatTexts() {
  const out = { en: { ...OWN.en }, es: { ...OWN.es } }
  for (const part of PARTS) {
    for (const lang of ['en', 'es']) Object.assign(out[lang], part.TEXTS?.[lang] ?? {})
  }
  return out
}
