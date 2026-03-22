use thiserror::Error;

#[derive(Error, Debug)]
pub enum IndexError {
    #[error("Index error: {0}")]
    Index(String),
}

pub type Result<T> = std::result::Result<T, IndexError>;

pub trait VectorIndex: Send + Sync {
    fn add(&self, id: String, vector: &[f32]) -> Result<()>;
    fn remove(&self, id: String) -> Result<()>;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;
    fn dimension(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
}
