use crate::core::{Memory, Edge, MemoryType, RelationType, MemoryQuery, CoreError, Result};
use crate::plugins::{Storage, Embedder};
use crate::relevance;
use crate::vector::hnsw::HnswIndex;
use crate::vector::index::VectorIndex;
use std::sync::Arc;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum RetrievalSource {
    Direct,
    Graph(String, RelationType), // origin_id, relation_type
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub memory: Memory,
    pub score: f32,
    pub relevance_score: f32,
    pub source: RetrievalSource,
}

pub struct MemoryEngine {
    storage: Arc<dyn Storage>,
    embedder: Arc<dyn Embedder>,
    index: Arc<HnswIndex>,
    index_path: Option<std::path::PathBuf>,
    config: crate::Config,
}

impl MemoryEngine {
    pub fn new<S, E>(storage: S, embedder: E, dimension: usize) -> Result<Self> 
    where 
        S: Storage + 'static,
        E: Embedder + 'static
    {
        Self::new_with_path(storage, embedder, dimension, None, crate::Config::default())
    }

    pub fn new_with_path<S, E>(storage: S, embedder: E, dimension: usize, index_path: Option<std::path::PathBuf>, config: crate::Config) -> Result<Self> 
    where 
        S: Storage + 'static,
        E: Embedder + 'static
    {
        let storage = Arc::new(storage);
        let embedder = Arc::new(embedder);
        
        let index = if let Some(ref path) = index_path {
            if path.exists() {
                match HnswIndex::load(path, dimension) {
                    Ok(idx) => {
                        tracing::info!("Loaded HNSW index from disk: {:?}", path);
                        Arc::new(idx)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load HNSW index from disk ({}), reconstructing...", e);
                        Arc::new(HnswIndex::new(dimension))
                    }
                }
            } else {
                Arc::new(HnswIndex::new(dimension))
            }
        } else {
            Arc::new(HnswIndex::new(dimension))
        };

        // If index is empty, reconstruct from storage
        if index.count() == 0 {
            let query_all = MemoryQuery {
                limit: 1_000_000,
                ..Default::default()
            };
            
            let memories = storage.list_memories(&query_all)?;
            let count = memories.len();
            for mem in memories {
                if !mem.embedding.is_empty() {
                    index.add(mem.id.clone(), &mem.embedding)?;
                }
            }
            if count > 0 {
                tracing::info!("Reconstructed HNSW index with {} memories", count);
            }
        }

        Ok(Self {
            storage,
            embedder,
            index,
            index_path,
            config,
        })
    }

    pub fn save_index(&self) -> Result<()> {
        if let Some(ref path) = self.index_path {
            self.index.save(path)?;
            tracing::info!("Saved HNSW index to disk: {:?}", path);
        }
        Ok(())
    }

    #[tracing::instrument(skip(self, memory))]
    pub async fn store(&self, mut memory: Memory) -> Result<Memory> {
        // Generate embedding
        let embedding = self.embedder.embed(&memory.content).await?;
        
        memory.embedding = embedding;

        // Save to storage
        self.storage.save_memory(&memory)?;

        // Add to vector index
        self.index.add(memory.id.clone(), &memory.embedding)?;

        Ok(memory)
    }

    pub async fn store_contextual(&self, memory: Memory, links: Vec<(String, RelationType)>) -> Result<Memory> {
        // 1. Store the primary memory
        let stored = self.store(memory).await?;
        
        // 2. Create edges for each link provided
        for (target_id, relation) in links {
            // We ignore errors on individual links to ensure the primary memory persists,
            // but in a production environment we might want more granular error reporting.
            let _ = self.link_by_ids(&stored.id, &target_id, relation);
        }
        
        Ok(stored)
    }

    pub async fn store_with_edge(&self, source: Memory, target: Memory, relation: RelationType) -> Result<(Memory, Edge, Memory)> {
        let source = self.store(source).await?;
        let target = self.store(target).await?;
        
        let edge = Edge::new(source.id.clone(), target.id.clone(), relation);
        
        self.storage.save_edge(&edge)?;

        Ok((source, edge, target))
    }

    pub fn link_by_ids(&self, source_id: &str, target_id: &str, relation: RelationType) -> Result<Edge> {
        // Validate nodes exist
        let _ = self.storage.load_memory(&source_id.to_string())
            .map_err(|_| CoreError::NotFound(format!("Source memory {} not found", source_id)))?;
        let _ = self.storage.load_memory(&target_id.to_string())
            .map_err(|_| CoreError::NotFound(format!("Target memory {} not found", target_id)))?;
            
        let edge = Edge::new(source_id.to_string(), target_id.to_string(), relation);
        self.storage.save_edge(&edge)?;
            
        Ok(edge)
    }

    #[tracing::instrument(skip(self))]
    pub fn remove(&self, id: &str) -> Result<()> {
        // 1. Storage delete (logical)
        self.storage.delete_memory(&id.to_string())?;

        // 2. Index delete
        self.index.remove(id.to_string())?;

        // 3. Auto-reindex if threshold reached (e.g., 20% tombstones)
        let tombstones = self.index.tombstone_count();
        let total = self.index.count() + tombstones;
        
        if total > 100 && (tombstones as f32 / total as f32) > 0.20 {
            tracing::info!("Reindexing HNSW ({}% tombstones)...", (tombstones as f32 / total as f32) * 100.0);
            self.index.reindex()?;
        }

        Ok(())
    }

    pub fn get(&self, id: &str) -> Result<Memory> {
        let mut memory = self.storage.load_memory(&id.to_string())?;
        
        memory.record_access();
        let _ = self.storage.update_memory(&memory);
        
        Ok(memory)
    }

    #[tracing::instrument(skip(self, query))]
    pub async fn search_by_text(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let start_time = std::time::Instant::now();
        
        let query_embedding = self.embedder.embed(query).await?;
        let embed_time = start_time.elapsed();
        tracing::debug!(latency_ms = embed_time.as_millis(), "Generated query embedding");

        // Constant for RRF smoothing
        const K_RRF: f32 = 60.0;
        // Over-fetch depth for better fusion quality
        let fetch_depth = 60.max(limit * 2);

        // 1. Vector Search (Semantic)
        let vector_start = std::time::Instant::now();
        let vector_results = self.index.search(&query_embedding, fetch_depth)
            .unwrap_or_default();
        let vector_time = vector_start.elapsed();
        tracing::debug!(latency_ms = vector_time.as_millis(), count = vector_results.len(), "HNSW vector search completed");

        // 2. FTS5 Search (Keywords)
        let fts_start = std::time::Instant::now();
        let fts_results = self.storage.search_fts(query, fetch_depth)
            .unwrap_or_default();
        let fts_time = fts_start.elapsed();
        tracing::debug!(latency_ms = fts_time.as_millis(), count = fts_results.len(), "FTS5 keyword search completed");

        // 3. RRF Fusion Logic
        let mut combined_scores: HashMap<String, f32> = HashMap::new();
        let mut final_similarities: HashMap<String, f32> = HashMap::new();

        // Add vector contributions
        for (rank, res) in vector_results.iter().enumerate() {
            let score = 1.0 / (K_RRF + rank as f32 + 1.0);
            combined_scores.insert(res.id.clone(), score);
            final_similarities.insert(res.id.clone(), res.score);
        }

        // Add FTS contributions (Summing if already present)
        for (rank, res) in fts_results.iter().enumerate() {
            let rrf_score = 1.0 / (K_RRF + rank as f32 + 1.0);
            // Lexical matches get a massive boost (x4.0) to ensure keyword matches win 
            *combined_scores.entry(res.id.clone()).or_insert(0.0) += rrf_score * 4.0;
        }

        // Fallback for tests/small datasets ONLY if main engines found ABSOLUTELY nothing
        if vector_results.is_empty() && fts_results.is_empty() {
            if let Ok(all) = self.storage.list_memories(&crate::core::MemoryQuery { ..Default::default() }) {
                for mem in all {
                    if !mem.embedding.is_empty() {
                        let score = relevance::cosine_similarity(&query_embedding, &mem.embedding);
                        // Only add if it's a "plausible" match for a test fallback
                        if score >= 0.0 { 
                            combined_scores.insert(mem.id.clone(), (score * 0.01).max(0.0001)); 
                            final_similarities.insert(mem.id.clone(), score);
                        }
                    }
                }
            }
        }

        // 4. Fetch Memories and Calculate Base Relevance
        let mut search_results: Vec<SearchResult> = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();
        
        for (id, rrf_score) in combined_scores {
            if let Ok(memory) = self.storage.load_memory(&id) {
                let static_relevance = relevance::calculate_relevance(&memory, self.config.engine.recency_lambda);
                seen_ids.insert(memory.id.clone());
                search_results.push(SearchResult {
                    memory,
                    score: *final_similarities.get(&id).unwrap_or(&0.0),
                    relevance_score: rrf_score + (0.1 * static_relevance), 
                    source: RetrievalSource::Direct,
                });
            }
        }

        // 5. Spreading Activation (Exact Graph Math) - BFS with 3 hops and decay
        let graph_start = std::time::Instant::now();
        let mut graph_boosts: HashMap<String, (f32, String, RelationType)> = HashMap::new();
        let mut frontier: Vec<(String, f32, usize)> = Vec::new();
        
        // Initialize frontier with direct search results
        for res in &search_results {
            frontier.push((res.memory.id.clone(), res.relevance_score, 0));
        }

        let mut current_hop = 0;
        let max_hops = 3;
        let hop_decay = 0.45; // Energy loss per hop

        while !frontier.is_empty() && current_hop < max_hops {
            let mut next_frontier = Vec::new();
            
            for (source_id, source_relevance, _hop) in frontier {
                if let Ok(edges) = self.storage.query_edges(&source_id) {
                    for edge in edges {
                        let relation_multiplier = match edge.relation_type {
                            RelationType::WorksOn => 0.40,
                            RelationType::Causal => 0.35,
                            RelationType::Entity => 0.25,
                            RelationType::Semantic => 0.15,
                            RelationType::Temporal => 0.15,
                            RelationType::Related => 0.10,
                        };
                        
                        // New energy to transmit: current relevance * relation * decay
                        let boost = source_relevance * relation_multiplier * hop_decay;
                        
                        if boost > 0.001 { // Cutoff for efficiency
                            let entry = graph_boosts.entry(edge.target_id.clone())
                                .or_insert((0.0, source_id.clone(), edge.relation_type.clone()));
                            
                            entry.0 += boost;
                            
                            // We only explore further if it's not a node we already explored in THIS search
                            // to avoid infinite loops and keep it a DAG exploration per search.
                            if !seen_ids.contains(&edge.target_id) {
                                next_frontier.push((edge.target_id.clone(), boost, current_hop + 1));
                            }
                        }
                    }
                }
            }
            frontier = next_frontier;
            current_hop += 1;
        }

        // Introduce Graph-discovered nodes that weren't found by vectors/fts
        for (target_id, (boost, origin_id, rel_type)) in &graph_boosts {
            if !seen_ids.contains(target_id) {
                if let Ok(memory) = self.storage.load_memory(target_id) {
                    let r_score = relevance::calculate_relevance(&memory, self.config.engine.recency_lambda);
                    search_results.push(SearchResult {
                        memory,
                        score: 0.0, 
                        relevance_score: *boost + (0.1 * r_score),
                        source: RetrievalSource::Graph(origin_id.clone(), rel_type.clone()),
                    });
                }
            }
        }
        let graph_time = graph_start.elapsed();
        tracing::debug!(latency_ms = graph_time.as_millis(), "Spreading activation completed");

        // 6. Final Ranking - Add boosts to existing results
        for sr in &mut search_results {
            if let Some((boost, _, _)) = graph_boosts.get(&sr.memory.id) {
                sr.relevance_score += boost;
            }
        }

        search_results.sort_by(|a, b| {
            b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        search_results.truncate(limit);
        
        let total_time = start_time.elapsed();
        tracing::info!(latency_ms = total_time.as_millis(), count = search_results.len(), "Hybrid search completed");
        
        Ok(search_results)
    }

    pub fn format_search_results(&self, results: &[SearchResult]) -> String {
        let mut yaml = String::new();
        yaml.push_str("inst: Responde solo con esta memoria. Usa 'direct' como hechos primarios y 'graph' para el contexto lógico/causal basado en 'rel'. Sintetiza ambos sin alucinar información externa.\n");
        yaml.push_str("mem:\n");

        let mut direct_results = Vec::new();
        let mut graph_results = Vec::new();

        for r in results {
            match &r.source {
                RetrievalSource::Direct => direct_results.push(r),
                RetrievalSource::Graph(_, _) => graph_results.push(r),
            }
        }

        if !direct_results.is_empty() {
            yaml.push_str("  direct:\n");
            for r in direct_results {
                yaml.push_str(&format!("    - id: {}\n", r.memory.id));
                let content = r.memory.content.replace('\n', " ");
                yaml.push_str(&format!("      sum: \"{}\"\n", content));
                yaml.push_str(&format!("      type: {:?}\n", r.memory.memory_type));
                yaml.push_str(&format!("      time: \"{}\"\n", r.memory.temporal.occurrence_start.format("%Y-%m-%d %H:%M:%S UTC")));
                
                // Relaciones salientes (Grafo)
                if let Ok(edges) = self.storage.query_edges(&r.memory.id) {
                    if !edges.is_empty() {
                        yaml.push_str("      links:\n");
                        for edge in edges {
                            yaml.push_str(&format!("        - rel: {:?}\n", edge.relation_type));
                            yaml.push_str(&format!("          target: {}\n", edge.target_id));
                        }
                    }
                }

                if let Some(conf) = r.memory.confidence {
                    yaml.push_str(&format!("      conf: {:.2}\n", conf));
                }
                if !r.memory.metadata.is_empty() {
                    yaml.push_str("      meta:\n");
                    for (k, v) in &r.memory.metadata {
                        yaml.push_str(&format!("        {}: \"{}\"\n", k, v.replace('"', "\\\"")));
                    }
                }
            }
        }

        if !graph_results.is_empty() {
            yaml.push_str("  graph:\n");
            for r in graph_results {
                if let RetrievalSource::Graph(origin, rel) = &r.source {
                    yaml.push_str(&format!("    - id: {}\n", r.memory.id));
                    let content = r.memory.content.replace('\n', " ");
                    yaml.push_str(&format!("      sum: \"{}\"\n", content));
                    yaml.push_str(&format!("      rel: {:?}->{}\n", rel, origin));
                    yaml.push_str(&format!("      type: {:?}\n", r.memory.memory_type));
                    yaml.push_str(&format!("      time: \"{}\"\n", r.memory.temporal.occurrence_start.format("%Y-%m-%d %H:%M:%S UTC")));
                    
                    // También mostramos enlaces salientes para nodos del grafo
                    if let Ok(edges) = self.storage.query_edges(&r.memory.id) {
                        if !edges.is_empty() {
                            yaml.push_str("      links:\n");
                            for edge in edges {
                                yaml.push_str(&format!("        - rel: {:?}\n", edge.relation_type));
                                yaml.push_str(&format!("          target: {}\n", edge.target_id));
                            }
                        }
                    }

                    if !r.memory.metadata.is_empty() {
                        yaml.push_str("      meta:\n");
                        for (k, v) in &r.memory.metadata {
                            yaml.push_str(&format!("        {}: \"{}\"\n", k, v.replace('"', "\\\"")));
                        }
                    }
                }
            }
        }
        yaml
    }

    pub fn get_related(&self, id: &str) -> Result<Vec<(Edge, Memory)>> {
        let edges = self.storage.query_edges(&id.to_string())?;
        
        let mut results = Vec::new();
        for edge in edges {
            if let Ok(target) = self.storage.load_memory(&edge.target_id) {
                results.push((edge, target));
            }
        }
        
        Ok(results)
    }

    pub fn count(&self) -> Result<usize> {
        Ok(self.storage.count_memories()?)
    }

    pub fn list_by_type(&self, mem_type: MemoryType) -> Result<Vec<Memory>> {
        Ok(self.storage.list_memories(&crate::core::MemoryQuery {
            memory_type: Some(mem_type),
            ..Default::default()
        })?)
    }
}
