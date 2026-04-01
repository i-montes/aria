use std::fs;
use std::path::{Path, PathBuf};
use std::ffi::OsStr;
use anyhow::Result;
use thiserror::Error;
use crate::agent::definition::AgentDefinition;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("IO error reading agents directory: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Failed to parse agent YAML at {path}: {source}")]
    ParseError {
        path: PathBuf,
        source: serde_yaml::Error,
    },
    #[error("No agents found in directory: {0}")]
    NoAgentsFound(PathBuf),
    #[error("Agent not found: {0}")]
    AgentNotFound(String),
}

pub struct AgentLoader;

impl AgentLoader {
    /// Lee todos los .yaml de la carpeta agents_dir y los parsea.
    pub fn load_all(agents_dir: &Path) -> Result<Vec<AgentDefinition>, LoadError> {
        let mut agents = Vec::new();
        
        if !agents_dir.exists() {
            return Err(LoadError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Agents directory not found: {:?}", agents_dir)
            )));
        }

        for entry in fs::read_dir(agents_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension() == Some(OsStr::new("yaml")) {
                let content = fs::read_to_string(&path)?;
                let agent: AgentDefinition = serde_yaml::from_str(&content)
                    .map_err(|e| LoadError::ParseError { 
                        path: path.clone(), 
                        source: e 
                    })?;
                
                tracing::info!("Loaded agent: {} ({})", agent.name, agent.role);
                agents.push(agent);
            }
        }

        if agents.is_empty() {
            return Err(LoadError::NoAgentsFound(agents_dir.to_path_buf()));
        }

        Ok(agents)
    }

    /// Lista los nombres de los agentes disponibles
    pub fn list_available(agents_dir: &Path) -> Vec<String> {
        match fs::read_dir(agents_dir) {
            Ok(entries) => entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension() == Some(OsStr::new("yaml")))
                .filter_map(|p| p.file_stem().and_then(|s| s.to_str()).map(|s| s.to_string()))
                .collect(),
            Err(_) => vec![],
        }
    }

    /// Carga un agente específico por nombre
    pub fn load_by_name(agents_dir: &Path, name: &str) -> Result<AgentDefinition, LoadError> {
        let path = agents_dir.join(format!("{}.yaml", name));
        if !path.exists() {
            return Err(LoadError::AgentNotFound(name.to_string()));
        }

        let content = fs::read_to_string(&path)?;
        let agent: AgentDefinition = serde_yaml::from_str(&content)
            .map_err(|e| LoadError::ParseError { 
                path, 
                source: e 
            })?;
        
        Ok(agent)
    }
}
