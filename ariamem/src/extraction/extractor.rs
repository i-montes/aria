use crate::plugins::{Fact, FactType, ExtractionResult};
use std::collections::HashMap;

pub struct SimpleExtractor;

impl SimpleExtractor {
    pub fn new() -> Self {
        Self
    }

    pub fn extract(&self, text: &str) -> ExtractionResult {
        let mut facts = Vec::new();
        let entities = Vec::new();
        
        for sentence in text.split(['.', '!', '?']) {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            
            facts.push(Fact {
                content: sentence.to_string(),
                fact_type: FactType::World,
                entities: Vec::new(),
                temporal_range: None,
                confidence: 0.8,
                metadata: HashMap::new(),
            });
        }
        
        ExtractionResult {
            facts,
            summary: None,
            entities,
        }
    }
}

impl Default for SimpleExtractor {
    fn default() -> Self {
        Self::new()
    }
}
