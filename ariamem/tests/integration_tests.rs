use ariamem::core::{Memory, Edge, MemoryType, RelationType, TemporalMetadata};
use ariamem::relevance::calculate_relevance;
use ariamem::vector::{NaiveIndex, VectorIndex};
use uuid::Uuid;

#[test]
fn test_memory_creation() {
    let memory = Memory::new(
        "Test memory content".to_string(),
        MemoryType::World,
    );
    
    assert!(!memory.content.is_empty());
    assert_eq!(memory.memory_type, MemoryType::World);
    assert_eq!(memory.access_count, 0);
}

#[test]
fn test_memory_with_embedding() {
    let embedding = vec![0.1, 0.2, 0.3, 0.4];
    let memory = Memory::new(
        "Test with embedding".to_string(),
        MemoryType::Experience,
    )
    .with_embedding(embedding.clone());
    
    assert_eq!(memory.embedding, embedding);
}

#[test]
fn test_memory_with_confidence() {
    let memory = Memory::new(
        "Opinion memory".to_string(),
        MemoryType::Opinion,
    )
    .with_confidence(0.85);
    
    assert_eq!(memory.confidence, Some(0.85));
}

#[test]
fn test_memory_record_access() {
    let mut memory = Memory::new(
        "Access test".to_string(),
        MemoryType::World,
    );
    
    assert_eq!(memory.access_count, 0);
    
    memory.record_access();
    assert_eq!(memory.access_count, 1);
    
    memory.record_access();
    assert_eq!(memory.access_count, 2);
    
    assert!(memory.last_accessed.is_some());
}

#[test]
fn test_edge_creation() {
    let source_id = uuid::Uuid::new_v4();
    let target_id = uuid::Uuid::new_v4();
    
    let edge = Edge::new(source_id.to_string(), target_id.to_string(), RelationType::Semantic)
        .with_weight(0.8);

    assert_eq!(edge.source_id, source_id.to_string());
    assert_eq!(edge.target_id, target_id.to_string());
    assert_eq!(edge.relation_type, RelationType::Semantic);
    assert_eq!(edge.weight, 0.8);
}

#[test]
fn test_calculate_relevance_new_memory() {
    let memory = Memory::new(
        "New memory".to_string(),
        MemoryType::World,
    );
    
    let relevance = calculate_relevance(&memory);
    
    assert!(relevance >= 0.0);
}

#[test]
fn test_calculate_relevance_accessed_memory() {
    let mut memory = Memory::new(
        "Accessed memory".to_string(),
        MemoryType::World,
    );
    
    for _ in 0..10 {
        memory.record_access();
    }
    
    let relevance = calculate_relevance(&memory);
    
    assert!(relevance > 1.0);
}

#[test]
fn test_naive_vector_index() {
    let index = NaiveIndex::new(4);
    
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    
    index.add(id1.to_string(), &[0.1, 0.2, 0.3, 0.4]).unwrap();
    index.add(id2.to_string(), &[0.4, 0.3, 0.2, 0.1]).unwrap();
    
    let results = index.search(&[0.1, 0.2, 0.3, 0.4], 2).unwrap();
    
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, id1.to_string());
}

#[test]
fn test_temporal_metadata() {
    let now = chrono::Utc::now();
    let metadata = TemporalMetadata::new(now);
    
    assert_eq!(metadata.occurrence_start, now);
    assert!(metadata.occurrence_end.is_none());
}

#[test]
fn test_memory_type_default() {
    let memory_type = MemoryType::default();
    assert_eq!(memory_type, MemoryType::World);
}

#[test]
fn test_relation_type_default() {
    let relation_type = RelationType::default();
    assert_eq!(relation_type, RelationType::Semantic);
}
