use crate::core::{Memory, Edge, MemoryQuery};
use crate::vector::SearchResult;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("Memory not found: {0}")]
    NotFound(String),
    #[error("Edge not found: {0}")]
    EdgeNotFound(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
}

pub type Result<T> = std::result::Result<T, StorageError>;

pub trait Storage: Send + Sync {
    fn save_memory(&self, memory: &Memory) -> Result<()>;
    fn load_memory(&self, id: &String) -> Result<Memory>;
    fn update_memory(&self, memory: &Memory) -> Result<()>;
    fn delete_memory(&self, id: &String) -> Result<()>;
    fn list_memories(&self, query: &MemoryQuery) -> Result<Vec<Memory>>;
    fn count_memories(&self) -> Result<usize>;
    fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>>;

    fn save_edge(&self, edge: &Edge) -> Result<()>;
    fn load_edge(&self, id: &String) -> Result<Edge>;
    fn delete_edge(&self, id: &String) -> Result<()>;
    fn query_edges(&self, source_id: &String) -> Result<Vec<Edge>>;
    fn query_edges_by_target(&self, target_id: &String) -> Result<Vec<Edge>>;
}
