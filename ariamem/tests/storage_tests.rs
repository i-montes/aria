use ariamem::plugins::Storage;
use ariamem::{
    Edge, Memory, MemoryEngine, MemoryType, RelationType, SqliteStorage, WordCountEmbedder,
};

#[test]
fn test_sqlite_storage_save_and_load_memory() {
    let storage = SqliteStorage::in_memory().unwrap();

    let memory = Memory::new("Test memory content".to_string(), MemoryType::World);

    storage.save_memory(&memory).unwrap();

    let loaded = storage.load_memory(&memory.id).unwrap();

    assert_eq!(loaded.id, memory.id);
    assert_eq!(loaded.content, memory.content);
    assert_eq!(loaded.memory_type, MemoryType::World);
}

#[test]
fn test_sqlite_storage_update_memory() {
    let storage = SqliteStorage::in_memory().unwrap();

    let mut memory = Memory::new("Original content".to_string(), MemoryType::World);

    storage.save_memory(&memory).unwrap();

    memory.content = "Updated content".to_string();
    memory.record_access();

    storage.update_memory(&memory).unwrap();

    let loaded = storage.load_memory(&memory.id).unwrap();

    assert_eq!(loaded.content, "Updated content");
    assert_eq!(loaded.access_count, 1);
}

#[test]
fn test_sqlite_storage_delete_memory() {
    let storage = SqliteStorage::in_memory().unwrap();

    let memory = Memory::new("To be deleted".to_string(), MemoryType::World);

    storage.save_memory(&memory).unwrap();
    storage.delete_memory(&memory.id).unwrap();

    let result = storage.load_memory(&memory.id);
    assert!(result.is_err());
}

#[test]
fn test_sqlite_storage_delete_memory_cascade() {
    let storage = SqliteStorage::in_memory().unwrap();

    let source = Memory::new("Source".to_string(), MemoryType::World);
    let target = Memory::new("Target".to_string(), MemoryType::World);

    storage.save_memory(&source).unwrap();
    storage.save_memory(&target).unwrap();

    let edge = Edge::new(source.id, target.id, RelationType::Semantic);
    storage.save_edge(&edge).unwrap();

    // Verify edge exists
    let edges = storage.query_edges(&source.id).unwrap();
    assert_eq!(edges.len(), 1);

    // Delete source memory
    storage.delete_memory(&source.id).unwrap();

    // Verify edge is gone
    let edges = storage.query_edges(&source.id).unwrap();
    assert_eq!(edges.len(), 0);
    
    // Also check from target side if we had such query (we have query_edges_by_target)
    let edges_target = storage.query_edges_by_target(&target.id).unwrap();
    assert_eq!(edges_target.len(), 0);
}

#[test]
fn test_sqlite_storage_save_and_load_edge() {
    let storage = SqliteStorage::in_memory().unwrap();

    let source = Memory::new("Source".to_string(), MemoryType::World);
    let target = Memory::new("Target".to_string(), MemoryType::World);

    storage.save_memory(&source).unwrap();
    storage.save_memory(&target).unwrap();

    let edge = Edge::new(source.id, target.id, RelationType::Semantic).with_weight(0.85);

    storage.save_edge(&edge).unwrap();

    let loaded = storage.load_edge(&edge.id).unwrap();

    assert_eq!(loaded.source_id, source.id);
    assert_eq!(loaded.target_id, target.id);
    assert_eq!(loaded.weight, 0.85);
}

#[test]
fn test_sqlite_storage_query_edges() {
    let storage = SqliteStorage::in_memory().unwrap();

    let source = Memory::new("Source".to_string(), MemoryType::World);
    let target1 = Memory::new("Target 1".to_string(), MemoryType::World);
    let target2 = Memory::new("Target 2".to_string(), MemoryType::World);

    storage.save_memory(&source).unwrap();
    storage.save_memory(&target1).unwrap();
    storage.save_memory(&target2).unwrap();

    let edge1 = Edge::new(source.id, target1.id, RelationType::Semantic);
    let edge2 = Edge::new(source.id, target2.id, RelationType::Temporal);

    storage.save_edge(&edge1).unwrap();
    storage.save_edge(&edge2).unwrap();

    let edges = storage.query_edges(&source.id).unwrap();

    assert_eq!(edges.len(), 2);
}

#[test]
fn test_sqlite_storage_count() {
    let storage = SqliteStorage::in_memory().unwrap();

    assert_eq!(storage.count_memories().unwrap(), 0);

    let memory1 = Memory::new("Memory 1".to_string(), MemoryType::World);
    let memory2 = Memory::new("Memory 2".to_string(), MemoryType::Experience);

    storage.save_memory(&memory1).unwrap();
    storage.save_memory(&memory2).unwrap();

    assert_eq!(storage.count_memories().unwrap(), 2);
}

#[test]
fn test_memory_engine_store() {
    let storage = SqliteStorage::in_memory().unwrap();
    let embedder = WordCountEmbedder::new(384);
    let engine = MemoryEngine::new(storage, embedder, 384);

    let memory = Memory::new("Test memory with embedding".to_string(), MemoryType::World);

    let stored = engine.store(memory).unwrap();

    assert!(!stored.embedding.is_empty());
    assert_eq!(stored.embedding.len(), 384);
}

#[test]
fn test_memory_engine_get() {
    let storage = SqliteStorage::in_memory().unwrap();
    let embedder = WordCountEmbedder::new(384);
    let engine = MemoryEngine::new(storage, embedder, 384);

    let memory = Memory::new("Memory to retrieve".to_string(), MemoryType::World);

    let stored = engine.store(memory).unwrap();

    let retrieved = engine.get(&stored.id).unwrap();

    assert_eq!(retrieved.id, stored.id);
    assert_eq!(retrieved.access_count, 1);
}

#[test]
fn test_memory_engine_search() {
    let storage = SqliteStorage::in_memory().unwrap();
    let embedder = WordCountEmbedder::new(384);
    let engine = MemoryEngine::new(storage, embedder, 384);

    let memory1 = Memory::new("Python programming language".to_string(), MemoryType::World);
    let memory2 = Memory::new("Rust programming language".to_string(), MemoryType::World);
    let memory3 = Memory::new("Coffee beans from Colombia".to_string(), MemoryType::World);

    engine.store(memory1).unwrap();
    engine.store(memory2).unwrap();
    engine.store(memory3).unwrap();

    let results = engine.search_by_text("programming", 3).unwrap();

    assert!(results.len() <= 3);
    for result in &results {
        println!(
            "Found: {} (score: {:.3})",
            result.memory.content, result.score
        );
    }
}

#[test]
fn test_memory_engine_store_with_edge() {
    let storage = SqliteStorage::in_memory().unwrap();
    let embedder = WordCountEmbedder::new(384);
    let engine = MemoryEngine::new(storage, embedder, 384);

    let source = Memory::new("User prefers dark mode".to_string(), MemoryType::Experience);
    let target = Memory::new("Theme configuration".to_string(), MemoryType::World);

    let (stored_source, edge, stored_target) = engine
        .store_with_edge(source, target, RelationType::Related)
        .unwrap();

    assert_eq!(edge.source_id, stored_source.id);
    assert_eq!(edge.target_id, stored_target.id);
}

#[test]
fn test_memory_engine_get_related() {
    let storage = SqliteStorage::in_memory().unwrap();
    let embedder = WordCountEmbedder::new(384);
    let engine = MemoryEngine::new(storage, embedder, 384);

    let source = Memory::new("Main topic".to_string(), MemoryType::World);
    let target = Memory::new("Related topic".to_string(), MemoryType::World);

    let (stored_source, edge, _) = engine
        .store_with_edge(source, target, RelationType::Related)
        .unwrap();

    let related = engine.get_related(&stored_source.id).unwrap();

    assert_eq!(related.len(), 1);
    assert_eq!(related[0].0.id, edge.id);
}

#[test]
fn test_memory_engine_count() {
    let storage = SqliteStorage::in_memory().unwrap();
    let embedder = WordCountEmbedder::new(384);
    let engine = MemoryEngine::new(storage, embedder, 384);

    assert_eq!(engine.count().unwrap(), 0);

    let memory = Memory::new("Test".to_string(), MemoryType::World);
    engine.store(memory).unwrap();

    assert_eq!(engine.count().unwrap(), 1);
}

#[test]
fn test_memory_engine_graph_boost() {
    let storage = SqliteStorage::in_memory().unwrap();
    let embedder = WordCountEmbedder::new(384);
    let engine = MemoryEngine::new(storage, embedder, 384);

    let root_memory = Memory::new("rust memory engine".to_string(), MemoryType::World);
    let leaf_memory = Memory::new("safety and performance guarantees".to_string(), MemoryType::World);

    let (_, _, leaf) = engine.store_with_edge(root_memory, leaf_memory, RelationType::WorksOn).unwrap();

    let results = engine.search_by_text("rust", 5).unwrap();

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
