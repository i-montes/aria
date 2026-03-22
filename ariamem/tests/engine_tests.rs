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

    let leaf_result = results.iter().find(|r| r.memory.id == leaf.id);
    assert!(leaf_result.is_some());
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

    let results = engine.search_by_text("Python", 10).await.unwrap();
    let netherlands_result = results.iter().find(|r| r.memory.id == s3.id);
    
    assert!(netherlands_result.is_some(), "Multi-hop activation should find nodes 2 hops away");
}

#[tokio::test]
async fn test_memory_engine_store_batch() {
    let storage = SqliteStorage::in_memory().unwrap();
    let embedder = WordCountEmbedder::new(384);
    let engine = MemoryEngine::new(storage, embedder, 384).expect("Failed to create engine");

    let m1 = Memory::new("Batch memory one".to_string(), MemoryType::World);
    let m2 = Memory::new("Batch memory two".to_string(), MemoryType::Experience);
    let m3 = Memory::new("Batch memory three".to_string(), MemoryType::Observation);

    let batch = vec![m1, m2, m3];
    let results = engine.store_batch(batch).await.unwrap();

    assert_eq!(results.len(), 3);
    for res in &results {
        assert!(!res.embedding.is_empty(), "Each memory must have an embedding generated");
        let loaded = engine.get(&res.id).unwrap();
        assert_eq!(loaded.content, res.content);
    }

    let search_res = engine.search_by_text("batch", 5).await.unwrap();
    assert!(search_res.len() >= 3, "All batch memories should be searchable");
}
