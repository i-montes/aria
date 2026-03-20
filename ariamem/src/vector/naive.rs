use crate::vector::index::{VectorIndex, SearchResult, Result, IndexError};
use std::collections::HashMap;

pub struct NaiveIndex {
    dimension: usize,
    vectors: HashMap<u64, Vec<f32>>,
}

impl NaiveIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vectors: HashMap::new(),
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
    fn add(&mut self, id: u64, vector: &[f32]) -> Result<()> {
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

    fn remove(&mut self, id: u64) -> Result<()> {
        self.vectors.remove(&id);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let mut results: Vec<SearchResult> = self.vectors
            .iter()
            .map(|(id, vector)| {
                let score = Self::cosine_similarity(query, vector);
                SearchResult { id: *id, score }
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
