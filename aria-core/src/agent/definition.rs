use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDefinition {
    pub name: String,
    pub role: String,
    pub soul: String,
    pub params: HashMap<String, serde_json::Value>,
    pub tools: Vec<String>,
    pub workflow_rules: Vec<String>,
    pub memory_boundaries: MemoryBoundaries,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBoundaries {
    pub can_read_global_context: bool,
    pub can_write_global_context: bool,
    pub namespaces: Vec<String>,
}

impl AgentDefinition {
    pub fn from_yaml(content: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(content)
    }
}
