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
}

use crate::relevance;

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
                let score = relevance::cosine_similarity(query, entry.value());
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

    fn count(&self) -> usize {
        self.vectors.len()
    }
}
