use crate::vector::index::{VectorIndex, SearchResult, Result, IndexError};
use dashmap::DashMap;

pub struct NaiveIndex {
    dimension: usize,
    vectors: DashMap<String, Vec<f32>>,
}

impl NaiveIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vectors: DashMap::new(),
        }
    }

    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot / (norm_a * norm_b)
    }
}

impl VectorIndex for NaiveIndex {
    fn add(&self, id: String, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(IndexError::Index(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            )));
        }
        self.vectors.insert(id, vector.to_vec());
        Ok(())
    }

    fn remove(&self, id: String) -> Result<()> {
        self.vectors.remove(&id);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let mut results: Vec<SearchResult> = self.vectors
            .iter()
            .map(|entry| {
                let score = Self::cosine_similarity(query, entry.value());
                SearchResult { id: entry.key().clone(), score }
            })
            .collect();
        
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(k);
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
