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

#[derive(Clone)]
pub struct MemoryEngine {
    storage: Arc<dyn Storage>,
    embedder: Arc<dyn Embedder>,
    index: Arc<HnswIndex>,
    index_path: Option<std::path::PathBuf>,
    recency_lambda: f32,
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
        let recency_lambda = config.engine.recency_lambda;
        
        let mut needs_reconstruction = false;
        let index = if let Some(ref path) = index_path {
            if path.exists() {
                match HnswIndex::load(path, dimension) {
                    Ok(idx) => {
                        tracing::info!("Loaded HNSW index from disk: {:?}", path);
                        Arc::new(idx)
                    }
                    Err(e) => {
                        // CRITICAL: File exists but failed to load (corruption or version mismatch)
                        tracing::error!("CRITICAL: Failed to load existing HNSW index at {:?} (Error: {}). Index will be reconstructed from storage.", path, e);
                        needs_reconstruction = true;
                        Arc::new(HnswIndex::new(dimension))
                    }
                }
            } else {
                // NORMAL: First run, file doesn't exist yet
                tracing::info!("No index file found at {:?}, initializing new index", path);
                needs_reconstruction = true;
                Arc::new(HnswIndex::new(dimension))
            }
        } else {
            // Memory-only index
            needs_reconstruction = true;
            Arc::new(HnswIndex::new(dimension))
        };

        // Reconstruct from storage ONLY if explicitly needed (failed load or new index)
        // AND the index is actually empty.
        if needs_reconstruction && index.count() == 0 {
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
                tracing::info!("Successfully reconstructed HNSW index with {} memories from storage", count);
            }
        }

        Ok(Self {
            storage,
            embedder,
            index,
            index_path,
            recency_lambda,
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

    pub async fn store_batch(&self, mut memories: Vec<Memory>) -> Result<Vec<Memory>> {
        if memories.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Generate all embeddings in one batch (Optimized SIMD/Parallelism)
        let texts: Vec<&str> = memories.iter().map(|m| m.content.as_str()).collect();
        let embeddings = self.embedder.embed_batch(&texts).await?;

        for (i, embedding) in embeddings.into_iter().enumerate() {
            memories[i].embedding = embedding;
        }

        // 2. Save all to storage in a single transaction (Atomic and Fast)
        self.storage.save_memories_batch(&memories)?;

        // 3. Add all to vector index
        for memory in &memories {
            self.index.add(memory.id.clone(), &memory.embedding)?;
        }

        Ok(memories)
    }

    pub async fn store_contextual(&self, memory: Memory, links: Vec<(String, RelationType)>) -> Result<Memory> {
        // 1. Pre-validate all target IDs to ensure graph integrity
        for (target_id, _) in &links {
            if !self.storage.exists_memory(target_id)? {
                return Err(CoreError::NotFound(format!("Target memory {} for context link not found", target_id)));
            }
        }

        // 2. Store the primary memory
        let stored = self.store(memory).await?;
        
        // 3. Create edges for each link provided
        for (target_id, relation) in links {
            self.link_by_ids(&stored.id, &target_id, relation)?;
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
    pub fn delete(&self, id: &str) -> Result<()> {
        // 1. Index delete (first, to ensure consistency)
        self.index.remove(id.to_string())?;

        // 2. Storage delete (includes edges cleanup in sqlite.rs)
        self.storage.delete_memory(&id.to_string())?;

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
        let memory = self.storage.load_memory(&id.to_string())?;
        
        // Atomic increment in DB to avoid lost updates under high concurrency
        let _ = self.storage.increment_access_count(id);
        
        Ok(memory)
    }

    pub fn exists(&self, id: &str) -> Result<bool> {
        Ok(self.storage.exists_memory(id)?)
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
        
        // Collect IDs to fetch in batch
        let ids_to_fetch: Vec<String> = combined_scores.keys().cloned().collect();
        let fetched_memories = self.storage.load_memories_by_ids(&ids_to_fetch)?;

        for memory in fetched_memories {
            let id = &memory.id;
            if let Some(&rrf_score) = combined_scores.get(id) {
                let static_relevance = relevance::calculate_relevance(&memory, self.recency_lambda);
                seen_ids.insert(id.clone());
                search_results.push(SearchResult {
                    memory,
                    score: *final_similarities.get(id).unwrap_or(&0.0),
                    relevance_score: rrf_score + (0.1 * static_relevance), 
                    source: RetrievalSource::Direct,
                });
            }
        }

        // 5. Spreading Activation (Budgeted Graph Search)
        let graph_start = std::time::Instant::now();
        let mut graph_boosts: HashMap<String, (f32, String, RelationType)> = HashMap::new();
        let mut frontier: Vec<(String, f32)> = Vec::new();
        
        let hop_cap = 50;
        let global_budget = 450;
        let energy_cutoff = 0.01;
        let mut total_processed = 0;

        // Initialize frontier with direct search results
        for res in &search_results {
            frontier.push((res.memory.id.clone(), res.relevance_score));
        }

        for _hop in 0..3 {
            if frontier.is_empty() || total_processed >= global_budget { break; }
            
            let mut next_frontier_map: HashMap<String, f32> = HashMap::new();

            for (source_id, source_relevance) in frontier {
                if total_processed >= global_budget { break; }
                total_processed += 1;

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
                        
                        // Energy: current * relation * decay (0.45)
                        let boost = source_relevance * relation_multiplier * 0.45;
                        
                        if boost > energy_cutoff {
                            let current_max = graph_boosts.get(&edge.target_id).map(|(b, _, _)| *b).unwrap_or(0.0);
                            
                            // Mechanism 1: Only propagate if this is a better path (more energy)
                            if boost > current_max {
                                // Record the path with most energy
                                graph_boosts.insert(edge.target_id.clone(), (boost, source_id.clone(), edge.relation_type.clone()));
                                
                                // Only add to next frontier if not already a direct result
                                // Direct results already "seeded" the first hop.
                                if !seen_ids.contains(&edge.target_id) {
                                    let entry = next_frontier_map.entry(edge.target_id.clone()).or_insert(0.0);
                                    if boost > *entry { *entry = boost; }
                                }
                            }
                        }
                    }
                }
            }

            // Mechanism 2: Cap per hop. Sort by energy and truncate to prevent exponential fan-out.
            let mut next_frontier: Vec<(String, f32)> = next_frontier_map.into_iter().collect();
            next_frontier.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            next_frontier.truncate(hop_cap);
            frontier = next_frontier;
        }

        // 5b. Batch Load Graph-discovered nodes (Optimization to eliminate final N+1)
        let graph_discovered_ids: Vec<String> = graph_boosts.keys()
            .filter(|id| !seen_ids.contains(*id))
            .cloned()
            .collect();
            
        if !graph_discovered_ids.is_empty() {
            let memories = self.storage.load_memories_by_ids(&graph_discovered_ids)?;
            for memory in memories {
                if let Some((_boost, origin_id, rel_type)) = graph_boosts.get(&memory.id) {
                    let r_score = relevance::calculate_relevance(&memory, self.recency_lambda);
                    search_results.push(SearchResult {
                        memory,
                        score: 0.0, 
                        relevance_score: (0.1 * r_score), // Initial score, boost added in Step 6
                        source: RetrievalSource::Graph(origin_id.clone(), rel_type.clone()),
                    });
                    seen_ids.insert(memory.id.clone()); 
                }
            }
        }
        let graph_time = graph_start.elapsed();
        tracing::debug!(latency_ms = graph_time.as_millis(), total_processed, "Budgeted spreading activation completed");

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

        if results.is_empty() {
            return yaml;
        }

        // Batch fetch all edges for all results to avoid N+1 queries
        let ids: Vec<String> = results.iter().map(|r| r.memory.id.clone()).collect();
        let all_edges = self.storage.query_edges_batch(&ids).unwrap_or_default();

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
                self.format_single_result(&mut yaml, r, &all_edges, false);
            }
        }

        if !graph_results.is_empty() {
            yaml.push_str("  graph:\n");
            for r in graph_results {
                self.format_single_result(&mut yaml, r, &all_edges, true);
            }
        }
        yaml
    }

    fn format_single_result(&self, yaml: &mut String, r: &SearchResult, all_edges: &HashMap<String, Vec<Edge>>, is_graph: bool) {
        yaml.push_str(&format!("    - id: {}\n", r.memory.id));
        let content = r.memory.content.replace('\n', " ");
        yaml.push_str(&format!("      sum: \"{}\"\n", content));
        
        if is_graph {
            if let RetrievalSource::Graph(origin, rel) = &r.source {
                yaml.push_str(&format!("      rel: {:?}->{}\n", rel, origin));
            }
        }

        yaml.push_str(&format!("      type: {:?}\n", r.memory.memory_type));
        yaml.push_str(&format!("      time: \"{}\"\n", r.memory.temporal.occurrence_start.format("%Y-%m-%d %H:%M:%S UTC")));
        
        // Use pre-fetched edges
        if let Some(edges) = all_edges.get(&r.memory.id) {
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
