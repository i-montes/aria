use anyhow::Result;
use std::collections::HashMap;
use tokio::sync::mpsc;
use crate::agent::definition::AgentDefinition;
use crate::whiteboard::storage::WhiteboardStorage;
use crate::protocol::messages::OrchestratorMessage;

pub struct OrchestratorEngine {
    #[allow(dead_code)]
    storage: WhiteboardStorage,
    agents: HashMap<String, AgentDefinition>,
    // Para enviar mensajes de vuelta a los hilos de los agentes: task_id -> sender
    #[allow(dead_code)]
    agent_txs: HashMap<String, mpsc::Sender<OrchestratorMessage>>,
}

use crate::agent::loader::AgentLoader;
use std::path::Path;

impl OrchestratorEngine {
    pub fn new(storage: WhiteboardStorage) -> Self {
        Self {
            storage,
            agents: HashMap::new(),
            agent_txs: HashMap::new(),
        }
    }

    /// Carga todos los agentes desde el directorio /agents
    pub fn load_agents(&mut self, path: &str) -> Result<()> {
        let agents = AgentLoader::load_all(Path::new(path))?;
        for agent in agents {
            self.agents.insert(agent.name.clone(), agent);
        }
        Ok(())
    }

    /// El loop principal que orquesta la ejecución
    pub async fn run(&mut self, goal: &str) -> Result<()> {
        println!("Orchestrator: starting goal -> {}", goal);
        
        // 1. Planificar subtareas (OrchestratorPlanner)
        
        // 2. Crear las tareas en el whiteboard
        
        // 3. Spawn de agentes (AgentRunner)
        
        // 4. Message Loop (select!)
        
        // 5. Revisión final (OrchestratorReviewer)
        
        Ok(())
    }
}
