use anyhow::Result;
use std::sync::Arc;
use tokio::sync::mpsc;
use crate::agent::definition::AgentDefinition;
use crate::whiteboard::schema::{ContextSnapshot, TaskStatus, TaskNote, TaskNoteType};
use crate::whiteboard::storage::WhiteboardStorage;
use crate::protocol::messages::{AgentMessage, OrchestratorMessage};
use crate::tools::tool_trait::Tool;

pub struct AgentRunner {
    #[allow(dead_code)]
    definition: AgentDefinition,
    context: ContextSnapshot,
    storage: Arc<WhiteboardStorage>,
    #[allow(dead_code)]
    tools: Vec<Arc<dyn Tool>>,
    // Canales
    orchestrator_tx: mpsc::Sender<AgentMessage>,
    #[allow(dead_code)]
    agent_rx: mpsc::Receiver<OrchestratorMessage>,
}

impl AgentRunner {
    pub fn new(
        definition: AgentDefinition,
        context: ContextSnapshot,
        storage: Arc<WhiteboardStorage>,
        tools: Vec<Arc<dyn Tool>>,
        orchestrator_tx: mpsc::Sender<AgentMessage>,
        agent_rx: mpsc::Receiver<OrchestratorMessage>,
    ) -> Self {
        Self {
            definition,
            context,
            storage,
            tools,
            orchestrator_tx,
            agent_rx,
        }
    }

    pub async fn run(self) -> Result<()> {
        let task_id = self.context.task.id.clone();
        let agent_name = self.definition.name.clone();
        
        // 1. Marcar tarea como IN_PROGRESS
        self.storage.update_task_status(&task_id, TaskStatus::InProgress)?;
        
        let tool_calls_count = 0;
        let max_tool_calls = self.context.constraints.max_tool_calls;

        // 3. Loop ReAct
        loop {
            // Verificar límites de seguridad
            if tool_calls_count >= max_tool_calls {
                self.handle_blocked(&task_id, &agent_name, "Max tool calls reached").await?;
                break;
            }

            // LLM Call (Implementar cliente después)
            // let response = llm.call(...).await?;

            // Simulación de stop_reason == "end_turn" (por ahora)
            // ...
            
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            break; 
        }

        Ok(())
    }

    async fn handle_blocked(&self, task_id: &str, agent_name: &str, reason: &str) -> Result<()> {
        self.storage.update_task_status(task_id, TaskStatus::Locked)?;
        
        let note = TaskNote {
            id: nanoid::nanoid!(8),
            task_id: task_id.to_string(),
            author: agent_name.to_string(),
            note_type: TaskNoteType::Blocker,
            content: reason.to_string(),
            metadata: None,
            created_at: chrono::Utc::now(),
        };
        self.storage.add_note(&note)?;

        self.orchestrator_tx.send(AgentMessage::Blocked {
            task_id: task_id.to_string(),
            agent_name: agent_name.to_string(),
            reason: reason.to_string(),
        }).await?;

        Ok(())
    }

    #[allow(dead_code)]
    pub async fn handle_progress(&self, task_id: &str, agent_name: &str, message: &str, note_type: TaskNoteType) -> Result<()> {
        let note = TaskNote {
            id: nanoid::nanoid!(8),
            task_id: task_id.to_string(),
            author: agent_name.to_string(),
            note_type: note_type.clone(),
            content: message.to_string(),
            metadata: None,
            created_at: chrono::Utc::now(),
        };
        self.storage.add_note(&note)?;

        self.orchestrator_tx.send(AgentMessage::Progress {
            task_id: task_id.to_string(),
            agent_name: agent_name.to_string(),
            message: message.to_string(),
            note_type,
        }).await?;

        Ok(())
    }
}
