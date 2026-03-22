use crate::plugins::{Fact, FactType, ExtractionResult, Entity};
use std::collections::HashMap;

pub struct SimpleExtractor;

impl SimpleExtractor {
    pub fn new() -> Self {
        Self
    }

    pub fn extract(&self, text: &str) -> ExtractionResult {
        let mut facts = Vec::new();
        let mut entity_map: HashMap<String, usize> = HashMap::new();
        
        // Basic entity detection: look for capitalized words (2+ chars) not at start of sentence
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if clean_word.len() > 2 && clean_word.chars().next().map_or(false, |c| c.is_uppercase()) {
                // If not first word, or if first word but looks like a proper noun
                if i > 0 || clean_word.len() > 3 {
                    *entity_map.entry(clean_word.to_string()).or_insert(0) += 1;
                }
            }
        }

        let entities: Vec<Entity> = entity_map.into_iter().map(|(name, count)| {
            Entity {
                name,
                entity_type: "Concept".to_string(), // Default until we have a real NER
                mentions: count,
            }
        }).collect();
        
        for sentence in text.split(['.', '!', '?']) {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            
            // Simple type heuristic
            let fact_type = if sentence.contains("creo") || sentence.contains("pienso") || sentence.contains("me parece") {
                FactType::Opinion
            } else if sentence.contains("vi") || sentence.contains("escuche") || sentence.contains("note") {
                FactType::Observation
            } else if sentence.contains("hice") || sentence.contains("fui") || sentence.contains("estuve") {
                FactType::Experience
            } else {
                FactType::World
            };

            facts.push(Fact {
                content: sentence.to_string(),
                fact_type,
                entities: entities.iter().filter(|e| sentence.contains(&e.name)).map(|e| e.name.clone()).collect(),
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
