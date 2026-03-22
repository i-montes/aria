use crate::plugins::{Fact, FactType, ExtractionResult, Entity};
use std::collections::HashMap;

pub struct SimpleExtractor;

impl SimpleExtractor {
    pub fn new() -> Self {
        Self
    }

    fn segment_sentences(&self, text: &str) -> Vec<String> {
        // A bit smarter than just split('.') - avoids some common abbreviations
        let abbreviations = ["id.", "etc.", "mr.", "mrs.", "dr.", "ms.", "prof.", "sr.", "sra.", "sta.", "av.", "ca.", "pág.", "vol.", "vs."];
        let mut sentences = Vec::new();
        let mut current = String::new();
        
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            current.push_str(word);
            
            let is_last = i == words.len() - 1;
            let ends_with_punct = word.ends_with('.') || word.ends_with('!') || word.ends_with('?');
            let is_abbreviation = abbreviations.iter().any(|&abbr| word.to_lowercase().ends_with(abbr));
            
            if (ends_with_punct && !is_abbreviation) || is_last {
                sentences.push(current.trim().to_string());
                current = String::new();
            } else {
                current.push(' ');
            }
        }
        
        sentences.into_iter().filter(|s| !s.is_empty()).collect()
    }

    pub fn extract(&self, text: &str) -> ExtractionResult {
        let mut facts = Vec::new();
        let mut entity_map: HashMap<String, usize> = HashMap::new();
        
        // Better entity detection: multi-word proper nouns
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut i = 0;
        while i < words.len() {
            let mut entity_candidate = Vec::new();
            
            // Check if word starts with uppercase
            while i < words.len() {
                let word = words[i].trim_matches(|c: char| !c.is_alphanumeric());
                if word.len() >= 2 && word.chars().next().unwrap().is_uppercase() {
                    // Filter out common sentence starters that aren't proper nouns
                    let common_starters = ["The", "This", "That", "El", "La", "Los", "Las", "Un", "Una", "My", "Your", "Mi", "Su"];
                    if entity_candidate.is_empty() && common_starters.contains(&word) {
                        break;
                    }
                    entity_candidate.push(word.to_string());
                    i += 1;
                } else {
                    break;
                }
            }

            if !entity_candidate.is_empty() {
                let full_name = entity_candidate.join(" ");
                *entity_map.entry(full_name).or_insert(0) += 1;
            } else {
                i += 1;
            }
        }

        let entities: Vec<Entity> = entity_map.into_iter().map(|(name, count)| {
            Entity {
                name,
                entity_type: "Concept".to_string(),
                mentions: count,
            }
        }).collect();
        
        let sentences = self.segment_sentences(text);
        for sentence in sentences {
            let s_lower = sentence.to_lowercase();
            
            // English and Spanish intent keywords
            let fact_type = if s_lower.contains("creo") || s_lower.contains("pienso") || 
                               s_lower.contains("me parece") || s_lower.contains("i think") || 
                               s_lower.contains("i believe") || s_lower.contains("opinion") {
                FactType::Opinion
            } else if s_lower.contains("vi") || s_lower.contains("escuche") || 
                      s_lower.contains("note") || s_lower.contains("observed") || 
                      s_lower.contains("noticed") || s_lower.contains("saw") {
                FactType::Observation
            } else if s_lower.contains("hice") || s_lower.contains("fui") || 
                      s_lower.contains("estuve") || s_lower.contains("i did") || 
                      s_lower.contains("i went") || s_lower.contains("i worked") {
                FactType::Experience
            } else {
                FactType::World
            };

            facts.push(Fact {
                content: sentence.clone(),
                fact_type,
                entities: entities.iter()
                    .filter(|e| sentence.contains(&e.name))
                    .map(|e| e.name.clone())
                    .collect(),
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
