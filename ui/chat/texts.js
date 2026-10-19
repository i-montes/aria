// Every text of the chat in both languages: each part's own `TEXTS` merged with the few that belong
// to the composition (the empty thread, results of cards, the header, the toast).
import * as answers from './parts/answers.js'
import * as assistant from './parts/assistant.js'
import * as commands from './parts/commands.js'
import * as composer from './parts/composer.js'
import * as connect from './parts/connect.js'
import * as divider from './parts/divider.js'
import * as doc from './parts/document.js'
import * as files from './parts/files.js'
import * as index from './parts/index.js'
import * as jev from './parts/jev.js'
import * as memory from './parts/memory.js'
import * as meter from './parts/meter.js'
import * as models from './parts/models.js'
import * as note from './parts/note.js'
import * as permission from './parts/permission.js'
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

export const PARTS = [answers, assistant, commands, composer, connect, divider, doc, files, index, jev, memory, meter, models, note, permission, phases, question, quota, quotabars, session, subagents, tabs, tasks, tools, user, working]

const OWN = {
  en: {
    'chat.empty.title': 'NEW SESSION',
    'chat.empty.text': 'Ask for something, or type / to see the commands.',
    'chat.stoppedByYou': 'stopped by you',
    'chat.compacted': 'context compacted',
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
    'chat.meter.noModel': 'no model',
    'chat.cmd.memory': 'Open the memory panel',
    'chat.cmd.model': 'Switch the model',
    'chat.cmd.connect': 'Connect a provider (Anthropic, OpenCode Go)',
  },
  es: {
    'chat.empty.title': 'SESIÓN NUEVA',
    'chat.empty.text': 'Pídele algo, o escribe / para ver los comandos.',
    'chat.stoppedByYou': 'detenido por ti',
    'chat.compacted': 'contexto compactado',
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
    'chat.meter.noModel': 'sin modelo',
    'chat.cmd.memory': 'Abrir el panel de memoria',
    'chat.cmd.model': 'Cambiar de modelo',
    'chat.cmd.connect': 'Conectar un proveedor (Anthropic, OpenCode Go)',
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
