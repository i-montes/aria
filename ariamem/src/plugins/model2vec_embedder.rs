use crate::plugins::embedder::{Embedder, EmbedderError, Result};
use model2vec::Model2Vec;
use ndarray::Array2;
use hf_hub::api::sync::ApiBuilder;
use std::path::PathBuf;

use std::sync::Arc;

pub struct Model2VecEmbedder {
    model: Arc<Model2Vec>,
    dimension: usize,
}

impl Model2VecEmbedder {
    pub fn new(model_path: &str) -> Result<Self> {
        let model = Model2Vec::from_pretrained(model_path, None, None)
            .map_err(|e| EmbedderError::ModelNotLoaded(e.to_string()))?;

        let embeddings = model.encode(&["test"]).unwrap();
        let dimension = embeddings.ncols();

        Ok(Self { model: Arc::new(model), dimension })
    }

    pub fn from_hub(repo_id: &str) -> Result<Self> {
        // If repo_id is a valid path, use new()
        let path_obj = PathBuf::from(repo_id);
        if path_obj.exists() && path_obj.is_dir() {
            return Self::new(repo_id);
        }

        // Otherwise, download from HuggingFace
        // We use an explicit ApiBuilder without tokens to avoid 401s on public models
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(None) 
            .build()
            .map_err(|e| EmbedderError::Download(format!("API Build error: {}", e)))?;
        
        let repo = api.model(repo_id.to_string());

        // Download required files
        let files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ];

        let mut last_parent: Option<PathBuf> = None;

        for file in files {
            match repo.get(file) {
                Ok(path) => {
                    last_parent = Some(path.parent().unwrap().to_path_buf());
                },
                Err(e) => {
                    return Err(EmbedderError::Download(format!(
                        "Failed to download '{}' from '{}': {}. Check your internet or if the model ID is correct.", 
                        file, repo_id, e
                    )));
                }
            }
        }

        let model_path = last_parent.ok_or_else(|| EmbedderError::Download("No files downloaded".to_string()))?;
        
        // Model2Vec expects the directory path
        let model = Model2Vec::from_pretrained(model_path.to_string_lossy().to_string(), None, None)
            .map_err(|e| EmbedderError::ModelNotLoaded(format!("Model2Vec init error: {}", e)))?;

        let embeddings = model.encode(&["test"]).unwrap();
        let dimension = embeddings.ncols();

        Ok(Self { model: Arc::new(model), dimension })
    }
}

use async_trait::async_trait;

#[async_trait]
impl Embedder for Model2VecEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let text = text.to_string();
        let model = self.model.clone();
        
        let embeddings = tokio::task::spawn_blocking(move || {
            model.encode(&[&text])
        }).await.map_err(|e: tokio::task::JoinError| EmbedderError::Inference(e.to_string()))?
          .map_err(|e| EmbedderError::Inference(e.to_string()))?;

        let row = embeddings.row(0);
        Ok(row.to_vec())
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let model = self.model.clone();

        let embeddings: Array2<f32> = tokio::task::spawn_blocking(move || {
            let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            model.encode(&texts_ref)
        }).await.map_err(|e: tokio::task::JoinError| EmbedderError::Inference(e.to_string()))?
          .map_err(|e| EmbedderError::Inference(e.to_string()))?;

        let mut result = Vec::with_capacity(embeddings.nrows());
        for i in 0..embeddings.nrows() {
            result.push(embeddings.row(i).to_vec());
        }
        Ok(result)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "Model2Vec"
    }
}
