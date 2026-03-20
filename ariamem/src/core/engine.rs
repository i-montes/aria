use crate::core::{Memory, Edge, MemoryType, RelationType};
use crate::plugins::{Storage, Embedder};
use crate::relevance;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub struct MemoryEngine<S: Storage, E: Embedder> {
    storage: Arc<S>,
    embedder: Arc<E>,
    embeddings: RwLock<HashMap<uuid::Uuid, Vec<f32>>>,
}

impl<S: Storage, E: Embedder> MemoryEngine<S, E> {
    pub fn new(storage: S, embedder: E, _embedding_dimension: usize) -> Self {
        Self {
            storage: Arc::new(storage),
            embedder: Arc::new(embedder),
            embeddings: RwLock::new(HashMap::new()),
        }
    }

    pub fn store(&self, mut memory: Memory) -> Result<Memory, String> {
        let embedding = self.embedder
            .embed(&memory.content)
            .map_err(|e| format!("Embedder error: {:?}", e))?;
        
        memory.embedding = embedding.clone();
        
        self.storage.save_memory(&memory)
            .map_err(|e| format!("Storage error: {:?}", e))?;

        let mut embeddings = self.embeddings.write()
            .map_err(|_| "Lock error")?;
        embeddings.insert(memory.id, embedding);

        Ok(memory)
    }

    pub fn store_with_edge(&self, source: Memory, target: Memory, relation: RelationType) -> Result<(Memory, Edge, Memory), String> {
        let source = self.store(source)?;
        let target = self.store(target)?;
        
        let edge = Edge::new(source.id, target.id, relation);
        
        self.storage.save_edge(&edge)
            .map_err(|e| format!("Storage error: {:?}", e))?;

        Ok((source, edge, target))
    }

    pub fn get(&self, id: &uuid::Uuid) -> Result<Memory, String> {
        let mut memory = self.storage.load_memory(id)
            .map_err(|e| format!("Storage error: {:?}", e))?;
        
        memory.record_access();
        let _ = self.storage.update_memory(&memory);
        
        Ok(memory)
    }

    pub fn update(&self, memory: &Memory) -> Result<(), String> {
        self.storage.update_memory(memory)
            .map_err(|e| format!("Storage error: {:?}", e))
    }

    pub fn delete(&self, id: &uuid::Uuid) -> Result<(), String> {
        if let Ok(mut embeddings) = self.embeddings.write() {
            embeddings.remove(id);
        }
        
        self.storage.delete_memory(id)
            .map_err(|e| format!("Storage error: {:?}", e))
    }

    pub fn search_by_text(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, String> {
        let query_embedding = self.embedder
            .embed(query)
            .map_err(|e| format!("Embedder error: {:?}", e))?;

        let all_memories = self.storage.list_memories(&Default::default())
            .map_err(|e| format!("Storage error: {:?}", e))?;

        let mut search_results: Vec<SearchResult> = Vec::new();
        
        for memory in &all_memories {
            if !memory.embedding.is_empty() {
                let score = cosine_similarity(&query_embedding, &memory.embedding);
                if score > 0.0 {
                    let relevance_score = relevance::calculate_relevance(memory);
                    search_results.push(SearchResult {
                        memory: memory.clone(),
                        score,
                        relevance_score,
                    });
                }
            }
        }

        search_results.sort_by(|a, b| {
            let combined_a = 0.7 * a.score + 0.3 * a.relevance_score;
            let combined_b = 0.7 * b.score + 0.3 * b.relevance_score;
            combined_b.partial_cmp(&combined_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        search_results.truncate(limit);
        Ok(search_results)
    }

    pub fn get_related(&self, id: &uuid::Uuid) -> Result<Vec<(Edge, Memory)>, String> {
        let edges = self.storage.query_edges(id)
            .map_err(|e| format!("Storage error: {:?}", e))?;

        let mut related = Vec::new();
        for edge in edges {
            if let Ok(memory) = self.storage.load_memory(&edge.target_id) {
                related.push((edge, memory));
            }
        }

        Ok(related)
    }

    pub fn count(&self) -> Result<usize, String> {
        self.storage.count_memories()
            .map_err(|e| format!("Storage error: {:?}", e))
    }

    pub fn list_by_type(&self, memory_type: MemoryType) -> Result<Vec<Memory>, String> {
        let query = crate::core::MemoryQuery {
            memory_type: Some(memory_type),
            ..Default::default()
        };
        
        self.storage.list_memories(&query)
            .map_err(|e| format!("Storage error: {:?}", e))
    }

    pub fn all_memories(&self) -> Result<Vec<Memory>, String> {
        self.storage.list_memories(&Default::default())
            .map_err(|e| format!("Storage error: {:?}", e))
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot / (norm_a * norm_b)
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub memory: Memory,
    pub score: f32,
    pub relevance_score: f32,
}
