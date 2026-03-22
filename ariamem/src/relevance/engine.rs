use crate::core::Memory;
use ndarray::ArrayView1;

pub fn calculate_relevance(memory: &Memory, lambda: f32) -> f32 {
    let access_factor = 1.0 + (1.0 + memory.access_count as f32).ln();
    let recency_decay = calculate_recency_decay(&memory.last_accessed, lambda);
    let base_weight = memory.metadata
        .get("base_weight")
        .and_then(|w| w.parse::<f32>().ok())
        .unwrap_or(1.0);
    
    base_weight * access_factor * recency_decay
}

pub fn calculate_recency_decay(last_accessed: &Option<chrono::DateTime<chrono::Utc>>, lambda: f32) -> f32 {
    match last_accessed {
        Some(time) => {
            let days = (chrono::Utc::now() - *time).num_days() as f32;
            // Exponential decay: e^(-lambda * t)
            // lambda = 0.1 means ~50% relevance after 7 days, ~5% after 30 days
            (-lambda * days).exp()
        }
        None => 1.0,
    }
}

pub fn calculate_coherence(memory1: &Memory, memory2: &Memory) -> f32 {
    if memory1.embedding.is_empty() || memory2.embedding.is_empty() {
        return 0.0;
    }
    
    cosine_similarity(&memory1.embedding, &memory2.embedding)
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let av = ArrayView1::from(a);
    let bv = ArrayView1::from(b);

    let dot = av.dot(&bv);
    let norm_a = av.dot(&av).sqrt();
    let norm_b = bv.dot(&bv).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

pub fn calculate_retrieval_score(
    similarity: f32,
    recency: f32,
    relevance: f32,
    alpha: f32,
    beta: f32,
    gamma: f32,
) -> f32 {
    alpha * similarity + beta * recency + gamma * relevance
}
