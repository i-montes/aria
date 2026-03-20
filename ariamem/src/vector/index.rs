use thiserror::Error;

#[derive(Error, Debug)]
pub enum IndexError {
    #[error("Index error: {0}")]
    Index(String),
}

pub type Result<T> = std::result::Result<T, IndexError>;

pub trait VectorIndex: Send + Sync {
    fn add(&mut self, id: u64, vector: &[f32]) -> Result<()>;
    fn remove(&mut self, id: u64) -> Result<()>;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;
    fn dimension(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
}
