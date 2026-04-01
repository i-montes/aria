use async_trait::async_trait;
use serde_json::Value;
use std::path::PathBuf;
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("Execution failed: {0}")]
    ExecutionError(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Access denied: {0}")]
    AccessDenied(String),
    #[error("Timeout: {0}")]
    Timeout(String),
}

pub struct ToolContext {
    pub task_id: String,
    pub agent_name: String,
    pub allowed_paths: Vec<PathBuf>,
    pub max_output_bytes: usize,
}

pub struct ToolResult {
    pub content: String,
    pub metadata: Option<Value>,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> Value;
    async fn execute(
        &self,
        args: Value,
        context: &ToolContext,
    ) -> Result<ToolResult, ToolError>;
}

impl fmt::Debug for dyn Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tool")
         .field("name", &self.name())
         .finish()
    }
}
