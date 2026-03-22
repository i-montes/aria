use ariamem::{
    Memory, MemoryEngine, MemoryType, RelationType, SqliteStorage, WordCountEmbedder, RetrievalSource,
};

#[tokio::test]
async fn test_memory_engine_graph_activation() {
    let storage = SqliteStorage::in_memory().unwrap();
    let embedder = WordCountEmbedder::new(384);
    let engine = MemoryEngine::new(storage, embedder, 384).expect("Failed to create engine");

    let root_memory = Memory::new("rust memory engine".to_string(), MemoryType::World);
    let leaf_memory = Memory::new("safety and performance guarantees".to_string(), MemoryType::World);

    let (_, _, leaf) = engine.store_with_edge(root_memory, leaf_memory, RelationType::WorksOn).await.unwrap();

    let results = engine.search_by_text("rust", 5).await.unwrap();

    println!("Total results: {}", results.len());
    for res in &results {
        println!("Result: ID={}, Content='{}', Score={}, Relevance={}", res.memory.id, res.memory.content, res.score, res.relevance_score);
    }

    let leaf_result = results.iter().find(|r| r.memory.id == leaf.id);
    
    assert!(leaf_result.is_some(), "Graph spreading activation should pull the leaf node into results");
    
    if let Some(res) = leaf_result {
        assert!(res.relevance_score > 0.0, "The relevance score must have been boosted by the graph edge");
    }
}

#[tokio::test]
async fn test_memory_engine_multi_hop_activation() {
    let storage = SqliteStorage::in_memory().unwrap();
    let embedder = WordCountEmbedder::new(384);
    let engine = MemoryEngine::new(storage, embedder, 384).expect("Failed to create engine");

    // Chain: Python -> Guido van Rossum -> Netherlands
    let m1 = Memory::new("Python is a programming language".to_string(), MemoryType::World);
    let m2 = Memory::new("Guido van Rossum created Python".to_string(), MemoryType::World);
    let m3 = Memory::new("Guido was born in the Netherlands".to_string(), MemoryType::World);

    let s1 = engine.store(m1).await.unwrap();
    let s2 = engine.store(m2).await.unwrap();
    let s3 = engine.store(m3).await.unwrap();

    // Link them: s1 -> s2 -> s3
    engine.link_by_ids(&s1.id, &s2.id, RelationType::Entity).unwrap();
    engine.link_by_ids(&s2.id, &s3.id, RelationType::Entity).unwrap();

    // Search for "Python". Netherlands (s3) is 2 hops away.
    // It should NOT be found by vector/fts because it doesn't contain "Python".
    let results = engine.search_by_text("Python", 10).await.unwrap();

    let netherlands_result = results.iter().find(|r| r.memory.id == s3.id);
    
    assert!(netherlands_result.is_some(), "Multi-hop activation should find nodes 2 hops away");
    
    if let Some(res) = netherlands_result {
        println!("Multi-hop found: {} with relevance {}", res.memory.content, res.relevance_score);
        assert!(res.relevance_score > 0.0);
        if let RetrievalSource::Graph(origin, rel) = &res.source {
             assert_eq!(origin, &s2.id); // It came from s2 in the last hop
             assert_eq!(rel, &RelationType::Entity);
        }
    }
}
