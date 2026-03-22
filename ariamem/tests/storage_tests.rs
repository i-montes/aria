use ariamem::plugins::Storage;
use ariamem::{
    Edge, Memory, MemoryType, RelationType, SqliteStorage,
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

    let edge = Edge::new(source.id.clone(), target.id.clone(), RelationType::Semantic);
    storage.save_edge(&edge).unwrap();

    // Verify edge exists
    let edges = storage.query_edges(&source.id).unwrap();
    assert_eq!(edges.len(), 1);

    // Delete source memory
    storage.delete_memory(&source.id).unwrap();

    // Verify edge is gone
    let edges = storage.query_edges(&source.id).unwrap();
    assert_eq!(edges.len(), 0);
    
    // Also check from target side
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

    let edge = Edge::new(source.id.clone(), target.id.clone(), RelationType::Semantic).with_weight(0.85);

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
    let t1 = Memory::new("Target 1".to_string(), MemoryType::World);
    let t2 = Memory::new("Target 2".to_string(), MemoryType::World);

    storage.save_memory(&source).unwrap();
    storage.save_memory(&t1).unwrap();
    storage.save_memory(&t2).unwrap();

    let e1 = Edge::new(source.id.clone(), t1.id.clone(), RelationType::Semantic);
    let e2 = Edge::new(source.id.clone(), t2.id.clone(), RelationType::Temporal);

    storage.save_edge(&e1).unwrap();
    storage.save_edge(&e2).unwrap();

    let edges = storage.query_edges(&source.id).unwrap();
    assert_eq!(edges.len(), 2);
}

#[test]
fn test_sqlite_storage_search_fts() {
    let storage = SqliteStorage::in_memory().unwrap();

    let m1 = Memory::new("The quick brown fox".to_string(), MemoryType::World);
    let m2 = Memory::new("Jumps over the lazy dog".to_string(), MemoryType::World);

    storage.save_memory(&m1).unwrap();
    storage.save_memory(&m2).unwrap();

    let results = storage.search_fts("fox", 5).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, m1.id);

    let results = storage.search_fts("dog", 5).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, m2.id);
}
