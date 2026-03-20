use crate::core::{Memory, MemoryResult};
use crate::plugins::Embedder;
use crate::relevance;

pub struct RetrievalEngine<E: Embedder> {
    embedder: E,
}

impl<E: Embedder> RetrievalEngine<E> {
    pub fn new(embedder: E) -> Self {
        Self { embedder }
    }

    pub fn search(&self, query: &str, memories: &[Memory]) -> Vec<MemoryResult> {
        let query_embedding = match self.embedder.embed(query) {
            Ok(emb) => emb,
            Err(_) => return Vec::new(),
        };

        memories
            .iter()
            .filter(|m| !m.embedding.is_empty())
            .map(|memory| {
                let similarity = cosine_similarity(&query_embedding, &memory.embedding);
                let relevance = relevance::calculate_relevance(memory);
                
                MemoryResult {
                    memory: memory.clone(),
                    relevance_score: relevance,
                    coherence_score: Some(similarity),
                }
            })
            .collect()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot / (norm_a * norm_b)
}
