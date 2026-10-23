//! The client's words. The dock's own texts (`ui/i18n/<lang>.json`, built into the binary) come first, so a notice says
//! in the terminal what it says in the dock; then the chat's texts that live in `ui/chat/texts.js` and the client's own
//! (`term.*`). Spanish when the assistant speaks it, English otherwise. `{name}` takes the value of `vars.name`.

use serde_json::{Map, Value};
use std::sync::OnceLock;

/// (key, English, Spanish).
const OWN: &[(&str, &str, &str)] = &[
    ("chat.stoppedByYou", "stopped by you", "detenido por ti"),
    ("chat.compacted", "context compacted", "contexto compactado"),
    ("chat.askSkipped", "skipped the questions", "omitió las preguntas"),
    ("chat.result.allow", "allowed once", "permitido una vez"),
    ("chat.result.always", "allowed for this session", "permitido en esta sesión"),
    ("chat.result.deny", "blocked", "bloqueado"),
    ("chat.result.free", "switched to the free model", "cambió al modelo gratuito"),
    ("chat.result.keep", "kept the current model", "mantuvo el modelo actual"),
    ("chat.working.phase.thinking", "Thinking", "Pensando"),
    ("chat.working.phase.working", "Working", "Trabajando"),
    ("chat.working.phase.searching_memory", "Searching memory", "Buscando en la memoria"),
    ("chat.working.phase.searching_the_web", "Searching the web", "Buscando en la web"),
    ("chat.working.phase.checking_Notion", "Checking Notion", "Revisando Notion"),
    ("chat.working.phase.reading_files", "Reading files", "Leyendo archivos"),
    ("chat.working.phase.writing_files", "Writing files", "Escribiendo archivos"),
    ("chat.working.phase.running_command", "Running command", "Ejecutando un comando"),
    ("chat.working.phase.delegating", "Delegating", "Delegando"),
    ("chat.working.phase.compacting_context", "Compacting context", "Compactando el contexto"),
    ("chat.tools.run", "Run command", "Ejecutar comando"),
    ("chat.tools.read", "Read file", "Leer archivo"),
    ("chat.tools.write_file", "Write file", "Escribir archivo"),
    ("chat.tools.write_one", "Write 1 line", "Escribir 1 línea"),
    ("chat.tools.write_n", "Write {n} lines", "Escribir {n} líneas"),
    ("chat.tools.edit", "Edit file", "Editar archivo"),
    ("chat.tools.grep", "Search in files", "Buscar en archivos"),
    ("chat.tools.find", "Find files", "Buscar archivos"),
    ("chat.tools.ls", "List folder", "Listar carpeta"),
    ("chat.tools.tool", "Tool", "Herramienta"),
    ("term.untitled", "untitled", "sin título"),
    (
        "term.undo.busy",
        "Not now: Aria is working. Undo once it is quiet.",
        "Ahora no: Aria está trabajando. Deshaz cuando termine.",
    ),
    ("term.undo.failed", "Could not go back there: nothing was changed.", "No se pudo volver ahí: no se cambió nada."),
    ("term.rewind.title", "Rewind", "Rebobinar"),
    (
        "term.rewind.lead",
        "Restore the code and/or conversation to the point before…",
        "Restaurar el código y/o la conversación al punto anterior a…",
    ),
    (
        "term.rewind.leadConversation",
        "Restore the conversation to the point before…",
        "Restaurar la conversación al punto anterior a…",
    ),
    (
        "term.rewind.off.projects",
        "Code is not kept in the projects folder.",
        "El código no se guarda en la carpeta de proyectos.",
    ),
    (
        "term.rewind.off.system",
        "Code is not kept in the system's folders.",
        "El código no se guarda en las carpetas del sistema.",
    ),
    (
        "term.rewind.off.big",
        "Code is not kept here: the folder has more than 20,000 files.",
        "El código no se guarda aquí: la carpeta tiene más de 20.000 archivos.",
    ),
    ("term.rewind.off.folder", "Code is not kept in this folder.", "El código no se guarda en esta carpeta."),
    ("term.rewind.current", "(current)", "(actual)"),
    ("term.rewind.none", "No code changes", "Sin cambios de código"),
    ("term.rewind.above", "↑ {n} more", "↑ {n} más"),
    ("term.rewind.below", "↓ {n} more", "↓ {n} más"),
    ("term.rewind.keys", "Enter to continue · Esc to cancel", "Enter para continuar · Esc para cancelar"),
    ("term.rewind.keysBack", "Enter to continue · Esc to go back", "Enter para continuar · Esc para volver"),
    ("term.rewind.from", "Code changed from here on:", "Código cambiado desde aquí:"),
    ("term.rewind.both", "Restore code and conversation", "Restaurar código y conversación"),
    ("term.rewind.conversation", "Restore conversation", "Restaurar conversación"),
    ("term.rewind.code", "Restore code", "Restaurar código"),
    ("term.rewind.never", "Never mind", "Mejor no"),
    ("term.cmd.context", "what fills the context", "qué llena el contexto"),
    (
        "term.ctx.title",
        "Context  {used} / {window} tokens ({percent}%)",
        "Contexto  {used} / {window} tokens ({percent}%)",
    ),
    ("term.ctx.system", "System", "Sistema"),
    ("term.ctx.tools", "Tools", "Herramientas"),
    ("term.ctx.skills", "Skills", "Skills"),
    ("term.ctx.memory", "Memory", "Memoria"),
    ("term.ctx.messages", "Messages", "Mensajes"),
    ("term.ctx.free", "Free", "Libre"),
    ("term.ctx.reserve", "Compaction reserve", "Reserva de compactación"),
    ("term.tasks", "tasks", "tareas"),
    ("term.changes.one", "1 file changed", "1 archivo cambiado"),
    ("term.changes.many", "{n} files changed", "{n} archivos cambiados"),
    ("term.cmd.undo", "back to before a message", "volver a antes de un mensaje"),
    ("term.files", "{n} files", "{n} archivos"),
    ("term.reconnected", "Connected to the dock again.", "Conectado de nuevo al dock."),
    (
        "term.oldDock",
        "Aria's dock is older than this aria: restart it with `aria quit` and `aria`.",
        "El dock de Aria es más viejo que este aria: reinícialo con `aria quit` y `aria`.",
    ),
    ("term.new", "+ new in {folder}", "+ nueva en {folder}"),
    ("term.newRemote", "+ new on {host}", "+ nueva en {host}"),
    (
        "term.noSsh",
        "ssh is not installed here (or ARIA_SSH names a program that is not there).",
        "ssh no está instalado aquí (o ARIA_SSH nombra un programa que no existe).",
    ),
    (
        "term.hubDown",
        "Aria is not running on that computer: start it there with `aria start`.",
        "Aria no está corriendo en ese computador: inícialo allá con `aria start`.",
    ),
    (
        "term.hubTooOld",
        "Aria on that computer is older than this aria: update it there.",
        "Aria en ese computador es más viejo que este aria: actualízalo allá.",
    ),
    (
        "term.protocolMismatch",
        "This aria speaks protocol {ours} and Aria on that computer speaks {theirs}: update the older one.",
        "Este aria habla el protocolo {ours} y Aria en ese computador habla el {theirs}: actualiza el más viejo.",
    ),
    ("term.working", "working", "trabajando"),
    ("term.waiting", "waiting for you", "esperando tu respuesta"),
    ("term.done", "done", "listo"),
    ("term.ago.now", "just now", "ahora"),
    ("term.ago.minutes", "{n} min ago", "hace {n} min"),
    ("term.ago.hours", "{n} h ago", "hace {n} h"),
    ("term.ago.days", "{n} d ago", "hace {n} d"),
    ("term.closedElsewhere", "This session was closed.", "Esta sesión se cerró."),
    (
        "term.gone",
        "Aria's dock stopped. Run `aria` again to start it.",
        "El dock de Aria se detuvo. Ejecuta `aria` de nuevo para iniciarlo.",
    ),
    ("term.cannotStart", "Aria's dock could not start: {error}", "El dock de Aria no pudo iniciar: {error}"),
    ("term.tooMany", "There are already 8 sessions: close one first.", "Ya hay 8 sesiones: cierra una primero."),
    ("term.more", "… {n} more lines", "… {n} líneas más"),
    (
        "term.input",
        "write, @ for a file, ! runs a command, / for commands",
        "escribe, @ adjunta un archivo, ! corre un comando, / abre comandos",
    ),
    ("term.queued", "queued", "en cola"),
    ("term.free", "free", "gratis"),
    ("term.week", "wk", "sem"),
    ("term.month", "mo", "mes"),
    ("term.verbose.on", "showing everything", "mostrando todo"),
    ("term.verbose.off", "showing one line per step", "una línea por paso"),
    ("term.cmd.exit", "close this session", "cerrar esta sesión"),
    ("term.cmd.verbose", "show everything or one line per step", "mostrar todo o una línea por paso"),
    ("term.cmd.model", "change the model", "cambiar el modelo"),
    ("term.cmd.sessions", "back to the sessions", "volver a las sesiones"),
    ("term.secret.placeholder", "the value (hidden)", "el valor (oculto)"),
    ("term.answer.placeholder", "or write your answer", "o escribe tu respuesta"),
    ("term.permission.keys", "y once · a always · n no", "y una vez · a siempre · n no"),
];

fn dock_texts(lang: &str) -> &'static Map<String, Value> {
    static EN: OnceLock<Map<String, Value>> = OnceLock::new();
    static ES: OnceLock<Map<String, Value>> = OnceLock::new();
    let (slot, file) = if lang == "es" { (&ES, "i18n/es.json") } else { (&EN, "i18n/en.json") };
    slot.get_or_init(|| {
        crate::dock::UI
            .get_file(file)
            .and_then(|f| f.contents_utf8())
            .and_then(|text| serde_json::from_str::<Map<String, Value>>(text).ok())
            .unwrap_or_default()
    })
}

fn raw(lang: &str, key: &str) -> Option<String> {
    let spanish = lang == "es";
    if let Some(text) = dock_texts(lang).get(key).and_then(Value::as_str) {
        return Some(text.to_string());
    }
    if let Some((_, en, es)) = OWN.iter().find(|(k, _, _)| *k == key) {
        return Some(if spanish { es } else { en }.to_string());
    }
    if spanish {
        return raw("en", key);
    }
    None
}

/// The text of `key` in `lang` with its `{vars}`; the key itself when nothing says it.
pub fn t(lang: &str, key: &str, vars: &Value) -> String {
    let mut text = raw(lang, key).unwrap_or_else(|| key.to_string());
    if let Some(vars) = vars.as_object() {
        for (name, value) in vars {
            let value = value.as_str().map(str::to_string).unwrap_or_else(|| value.to_string());
            text = text.replace(&format!("{{{name}}}"), &value);
        }
    }
    text
}
