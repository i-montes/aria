use crate::plugins::embedder::{Embedder, EmbedderError, Result};
use model2vec::Model2Vec;
use ndarray::Array2;

pub struct Model2VecEmbedder {
    model: Model2Vec,
    dimension: usize,
}

impl Model2VecEmbedder {
    pub fn new(model_path: &str) -> Result<Self> {
        let model = Model2Vec::from_pretrained(model_path, None, None)
            .map_err(|e| EmbedderError::ModelNotLoaded(e.to_string()))?;

        let embeddings = model.encode(&["test"]).unwrap();
        let dimension = embeddings.ncols();

        Ok(Self { model, dimension })
    }

    pub fn from_hub(repo_id: &str) -> Result<Self> {
        let model = Model2Vec::from_pretrained(repo_id, None, None)
            .map_err(|e| EmbedderError::ModelNotLoaded(e.to_string()))?;

        let embeddings = model.encode(&["test"]).unwrap();
        let dimension = embeddings.ncols();

        Ok(Self { model, dimension })
    }
}

impl Embedder for Model2VecEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self
            .model
            .encode(&[text])
            .map_err(|e| EmbedderError::Inference(e.to_string()))?;

        let row = embeddings.row(0);
        Ok(row.to_vec())
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let embeddings: Array2<f32> = self
            .model
            .encode(texts)
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
