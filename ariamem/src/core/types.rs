use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use crate::plugins::{StorageError, EmbedderError};
use crate::vector::index::IndexError;

pub fn generate_id() -> String {
    nanoid::nanoid!(8)
}

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),
    #[error("Embedder error: {0}")]
    Embedder(#[from] EmbedderError),
    #[error("Index error: {0}")]
    Index(#[from] IndexError),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, CoreError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    World,
    Experience,
    Opinion,
    Observation,
}

impl Default for MemoryType {
    fn default() -> Self {
        MemoryType::World
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    Temporal,
    Semantic,
    Entity,
    Causal,
    Related,
    WorksOn,
}

impl Default for RelationType {
    fn default() -> Self {
        RelationType::Semantic
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMetadata {
    pub occurrence_start: DateTime<Utc>,
    pub occurrence_end: Option<DateTime<Utc>>,
    pub mention_time: DateTime<Utc>,
}

impl TemporalMetadata {
    pub fn new(occurrence_start: DateTime<Utc>) -> Self {
        Self {
            occurrence_start,
            occurrence_end: None,
            mention_time: Utc::now(),
        }
    }

    pub fn with_occurrence_end(mut self, end: DateTime<Utc>) -> Self {
        self.occurrence_end = Some(end);
        self
    }
}

impl Default for TemporalMetadata {
    fn default() -> Self {
        Self::new(Utc::now())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub memory_type: MemoryType,
    pub content: String,
    pub embedding: Vec<f32>,
    pub temporal: TemporalMetadata,
    pub metadata: HashMap<String, String>,
    pub confidence: Option<f32>,
    pub access_count: u32,
    pub last_accessed: Option<DateTime<Utc>>,
    pub is_active: bool,
}

impl Memory {
    pub fn new(content: String, memory_type: MemoryType) -> Self {
        Self {
            id: generate_id(),
            memory_type,
            content,
            embedding: Vec::new(),
            temporal: TemporalMetadata::default(),
            metadata: HashMap::new(),
            confidence: None,
            access_count: 0,
            last_accessed: None,
            is_active: true,
        }
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Some(Utc::now());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub relation_type: RelationType,
    pub weight: f32,
    pub metadata: HashMap<String, String>,
}

impl Edge {
    pub fn new(source_id: String, target_id: String, relation_type: RelationType) -> Self {
        Self {
            id: generate_id(),
            source_id,
            target_id,
            relation_type,
            weight: 1.0,
            metadata: HashMap::new(),
        }
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQuery {
    pub text: Option<String>,
    pub memory_type: Option<MemoryType>,
    pub time_range: Option<TimeRange>,
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl Default for MemoryQuery {
    fn default() -> Self {
        Self {
            text: None,
            memory_type: None,
            time_range: None,
            limit: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryResult {
    pub memory: Memory,
    pub relevance_score: f32,
    pub coherence_score: Option<f32>,
}
