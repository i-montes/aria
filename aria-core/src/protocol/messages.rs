use serde::{Deserialize, Serialize};
use crate::whiteboard::schema::TaskNoteType;

/// Mensajes que un agente envía al orquestador
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentMessage {
    /// Actualización de progreso (no bloquea al orquestador)
    Progress {
        task_id: String,
        agent_name: String,
        message: String,
        note_type: TaskNoteType,
    },
    /// El agente terminó y pone su tarea en REVIEW
    Done {
        task_id: String,
        agent_name: String,
        result_summary: String,
    },
    /// El agente no puede continuar sin intervención
    Blocked {
        task_id: String,
        agent_name: String,
        reason: String,
    },
    /// El agente necesita que el orquestador resuelva una dependencia
    NeedsInput {
        task_id: String,
        agent_name: String,
        question: String,
    },
}

/// Mensajes que el orquestador envía a un agente
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrchestratorMessage {
    /// Respuesta a NeedsInput
    Input {
        task_id: String,
        content: String,
    },
    /// El orquestador cancela la task (ej: el proyecto falló)
    Cancel {
        task_id: String,
        reason: String,
    },
}
