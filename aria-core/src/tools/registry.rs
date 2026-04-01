use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use crate::tools::tool_trait::Tool;

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Resuelve categorías del YAML (ej: "read", "shell") a herramientas concretas.
    pub fn resolve_categories(&self, categories: &[String]) -> Vec<Arc<dyn Tool>> {
        let mut resolved_names = HashSet::new();
        
        for category in categories {
            match category.as_str() {
                "read" => {
                    resolved_names.insert("file_read");
                    resolved_names.insert("file_list");
                    resolved_names.insert("folder_read");
                }
                "write" => {
                    resolved_names.insert("file_write");
                    resolved_names.insert("file_create");
                    resolved_names.insert("folder_create");
                }
                "shell" => {
                    resolved_names.insert("shell");
                }
                "search" => {
                    resolved_names.insert("search_memory");
                    resolved_names.insert("file_search");
                    resolved_names.insert("folder_search");
                }
                "web" => {
                    resolved_names.insert("web_fetch");
                    resolved_names.insert("web_search");
                }
                "memory" => {
                    resolved_names.insert("memory_read");
                    resolved_names.insert("memory_write");
                    resolved_names.insert("memory_link");
                    resolved_names.insert("memory_delete");
                }
                _ => {
                    // Si es un nombre de herramienta específico en lugar de una categoría
                    if self.tools.contains_key(category) {
                        resolved_names.insert(category.as_str());
                    }
                }
            }
        }

        resolved_names
            .into_iter()
            .filter_map(|name| self.tools.get(name).cloned())
            .collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
