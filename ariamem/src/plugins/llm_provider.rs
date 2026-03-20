use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub content: String,
    pub fact_type: FactType,
    pub entities: Vec<String>,
    pub temporal_range: Option<TemporalRange>,
    pub confidence: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactType {
    World,
    Experience,
    Opinion,
    Observation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRange {
    pub start: DateTime<Utc>,
    pub end: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub facts: Vec<Fact>,
    pub summary: Option<String>,
    pub entities: Vec<Entity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub entity_type: String,
    pub mentions: usize,
}

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LLMProviderError {
    #[error("API error: {0}")]
    Api(String),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Configuration error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, LLMProviderError>;

pub trait LLMProvider: Send + Sync {
    fn extract_facts(&self, text: &str) -> Result<ExtractionResult>;
    fn extract_facts_batch(&self, texts: &[&str]) -> Result<Vec<ExtractionResult>>;
    fn summarize(&self, text: &str, max_tokens: usize) -> Result<String>;
}
