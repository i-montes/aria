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
    ("term.untitled", "untitled", "sin título"),
    ("term.new", "+ new in {folder}", "+ nueva en {folder}"),
    ("term.working", "working", "trabajando"),
    ("term.waiting", "waiting for you", "esperando tu respuesta"),
    ("term.done", "done", "listo"),
    ("term.ago.now", "just now", "ahora"),
    ("term.ago.minutes", "{n} min ago", "hace {n} min"),
    ("term.ago.hours", "{n} h ago", "hace {n} h"),
    ("term.ago.days", "{n} d ago", "hace {n} d"),
    ("term.closedElsewhere", "This session was closed in the dock.", "Esta sesión se cerró en el dock."),
    (
        "term.gone",
        "Aria's dock stopped. Run `aria` again to start it.",
        "El dock de Aria se detuvo. Ejecuta `aria` de nuevo para iniciarlo.",
    ),
    ("term.cannotStart", "Aria's dock could not start: {error}", "El dock de Aria no pudo iniciar: {error}"),
    ("term.tooMany", "There are already 8 sessions: close one first.", "Ya hay 8 sesiones: cierra una primero."),
    (
        "term.selectHint",
        "Hold Shift to select text with the mouse.",
        "Mantén Shift para seleccionar texto con el ratón.",
    ),
    ("term.more", "… {n} more lines", "… {n} líneas más"),
    ("term.summary.reads", "{n} reads", "{n} lecturas"),
    ("term.summary.read", "1 read", "1 lectura"),
    ("term.summary.edits", "{n} edits", "{n} ediciones"),
    ("term.summary.edit", "1 edit", "1 edición"),
    ("term.summary.writes", "{n} new files", "{n} archivos nuevos"),
    ("term.summary.write", "1 new file", "1 archivo nuevo"),
    ("term.summary.commands", "{n} commands", "{n} comandos"),
    ("term.summary.command", "1 command", "1 comando"),
    ("term.summary.tests", "tests {passed}/{total}", "tests {passed}/{total}"),
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
    ("term.verbose.off", "showing one line per turn", "una línea por turno"),
    ("term.cmd.exit", "close this session", "cerrar esta sesión"),
    ("term.cmd.verbose", "show everything or one line per turn", "mostrar todo o una línea por turno"),
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
