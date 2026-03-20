use crate::core::Memory;

pub fn calculate_relevance(memory: &Memory) -> f32 {
    let access_factor = (1.0 + memory.access_count as f32).ln();
    let recency_decay = calculate_recency_decay(&memory.last_accessed);
    let base_weight = memory.metadata
        .get("base_weight")
        .and_then(|w| w.parse::<f32>().ok())
        .unwrap_or(1.0);
    
    base_weight * access_factor * recency_decay
}

pub fn calculate_recency_decay(last_accessed: &Option<chrono::DateTime<chrono::Utc>>) -> f32 {
    match last_accessed {
        Some(time) => {
            let days = (chrono::Utc::now() - *time).num_days() as f32;
            1.0 / (1.0 + days)
        }
        None => 1.0,
    }
}

pub fn calculate_coherence(memory1: &Memory, memory2: &Memory) -> f32 {
    if memory1.embedding.is_empty() || memory2.embedding.is_empty() {
        return 0.0;
    }
    
    let distance = euclidean_distance(&memory1.embedding, &memory2.embedding);
    (-distance).exp()
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
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
