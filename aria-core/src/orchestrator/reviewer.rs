use anyhow::Result;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::whiteboard::storage::WhiteboardStorage;

pub struct OrchestratorReviewer {
    #[allow(dead_code)]
    storage: Arc<WhiteboardStorage>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReviewReport {
    pub project_goal: String,
    pub task_summaries: Vec<TaskSummary>,
    pub critical_notes: Vec<CriticalNote>,
    pub locked_tasks: Vec<LockedTaskInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TaskSummary {
    pub task_id: String,
    pub agent: String,
    pub summary: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CriticalNote {
    pub task_id: String,
    pub author: String,
    pub note_type: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LockedTaskInfo {
    pub task_id: String,
    pub agent: String,
    pub reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ReviewOutcome {
    Approved { final_response: String },
    CorrectionRequired { corrections: Vec<CorrectionTask> },
    Blocked { reason: String },
}

#[derive(Debug, Deserialize)]
pub struct CorrectionTask {
    pub agent_name: String,
    pub description: String,
    pub priority: String,
}

impl OrchestratorReviewer {
    pub fn new(storage: Arc<WhiteboardStorage>) -> Self {
        Self { storage }
    }

    /// Recopila toda la información del proyecto para la revisión
    pub async fn prepare_report(&self, _project_id: &str, goal: &str) -> Result<ReviewReport> {
        // TODO: Implementar queries en storage.rs para obtener:
        // - Tasks filtradas por project_id
        // - Notas filtradas por tipos CRITICAL (DECISION, WARNING, BLOCKER)
        
        Ok(ReviewReport {
            project_goal: goal.to_string(),
            task_summaries: vec![],
            critical_notes: vec![],
            locked_tasks: vec![],
        })
    }

    /// Ejecuta la revisión con el LLM (esta lógica se moverá al Engine o usará el LLM Client)
    pub async fn review(&self, _report: ReviewReport) -> Result<ReviewOutcome> {
        // Pseudocódigo:
        // 1. Build prompt con ReviewReport
        // 2. LLM Call con response_format: json
        // 3. Parse ReviewOutcome
        
        Ok(ReviewOutcome::Approved { 
            final_response: "Revisión completada (simulada)".to_string() 
        })
    }
}
