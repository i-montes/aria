use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbedderError {
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),
    #[error("Inference error: {0}")]
    Inference(String),
    #[error("Download error: {0}")]
    Download(String),
}

pub type Result<T> = std::result::Result<T, EmbedderError>;

pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
    fn name(&self) -> &str;
}
