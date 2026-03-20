use crate::plugins::Embedder;
use std::collections::{HashMap, HashSet};

pub struct TfIdfEmbedder {
    dimension: usize,
    vocabulary: HashMap<String, usize>,
    idf: Vec<f32>,
}

impl TfIdfEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vocabulary: HashMap::new(),
            idf: Vec::new(),
        }
    }

    pub fn from_vocabulary(vocabulary: HashMap<String, usize>, idf: Vec<f32>, dimension: usize) -> Self {
        Self {
            dimension,
            vocabulary,
            idf,
        }
    }

    pub fn fit(&mut self, documents: &[&str]) {
        let mut doc_freq: HashMap<String, f32> = HashMap::new();
        let num_docs = documents.len() as f32;
        
        let mut all_terms: HashSet<String> = HashSet::new();
        
        for doc in documents {
            let terms = self.tokenize(doc);
            for term in &terms {
                all_terms.insert(term.clone());
            }
        }
        
        for term in all_terms {
            doc_freq.insert(term, 1.0);
        }
        
        for doc in documents {
            let terms = self.tokenize(doc);
            let mut seen: HashSet<String> = HashSet::new();
            for term in terms {
                if !seen.contains(&term) {
                    *doc_freq.entry(term.clone()).or_insert(0.0) += 1.0;
                    seen.insert(term);
                }
            }
        }
        
        let mut sorted_terms: Vec<String> = doc_freq.keys().cloned().collect();
        sorted_terms.sort();
        
        self.vocabulary.clear();
        for (i, term) in sorted_terms.iter().enumerate() {
            if i < self.dimension {
                self.vocabulary.insert(term.clone(), i);
            }
        }
        
        self.idf = vec![0.0; self.vocabulary.len()];
        for (term, &df) in &doc_freq {
            if let Some(&idx) = self.vocabulary.get(term) {
                self.idf[idx] = (num_docs / df).ln() + 1.0;
            }
        }
        
        self.dimension = self.vocabulary.len().min(self.dimension);
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 2)
            .map(|s| s.to_string())
            .collect()
    }

    fn compute_tf(&self, terms: &[String]) -> Vec<f32> {
        let mut tf: HashMap<String, f32> = HashMap::new();
        for term in terms {
            *tf.entry(term.clone()).or_insert(0.0) += 1.0;
        }
        let max_tf = tf.values().cloned().fold(0.0f32, f32::max);
        tf.iter()
            .map(|(_term, &count)| {
                if max_tf > 0.0 {
                    (0.5 + 0.5 * count / max_tf) as f32
                } else {
                    0.0
                }
            })
            .collect()
    }
}

impl Embedder for TfIdfEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, crate::plugins::EmbedderError> {
        let terms = self.tokenize(text);
        let tf = self.compute_tf(&terms);
        
        let mut vector = vec![0.0; self.dimension];
        
        for (term, tf_val) in terms.iter().zip(tf.iter()) {
            if let Some(&idx) = self.vocabulary.get(term) {
                if idx < self.dimension {
                    let idf_val = if idx < self.idf.len() { self.idf[idx] } else { 1.0 };
                    vector[idx] = tf_val * idf_val;
                }
            }
        }
        
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vector {
                *v /= norm;
            }
        }
        
        Ok(vector)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, crate::plugins::EmbedderError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "tfidf"
    }
}

pub struct WordCountEmbedder {
    dimension: usize,
    vocabulary: HashMap<String, usize>,
}

impl WordCountEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vocabulary: HashMap::new(),
        }
    }

    pub fn fit(&mut self, documents: &[&str]) {
        let mut term_freq: HashMap<String, f32> = HashMap::new();
        
        for doc in documents {
            let terms = self.tokenize(doc);
            for term in terms {
                *term_freq.entry(term).or_insert(0.0) += 1.0;
            }
        }
        
        let mut sorted: Vec<(String, f32)> = term_freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        self.vocabulary.clear();
        for (i, (term, _)) in sorted.iter().enumerate() {
            if i < self.dimension {
                self.vocabulary.insert(term.clone(), i);
            }
        }
        
        self.dimension = self.vocabulary.len().min(self.dimension);
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 2)
            .map(|s| s.to_string())
            .collect()
    }
}

impl Embedder for WordCountEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, crate::plugins::EmbedderError> {
        let terms = self.tokenize(text);
        let mut counts: HashMap<String, f32> = HashMap::new();
        for term in &terms {
            *counts.entry(term.clone()).or_insert(0.0) += 1.0;
        }
        
        let mut vector = vec![0.0; self.dimension];
        
        for (term, count) in &counts {
            if let Some(&idx) = self.vocabulary.get(term) {
                if idx < self.dimension {
                    vector[idx] = *count;
                }
            }
        }
        
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vector {
                *v /= norm;
            }
        }
        
        Ok(vector)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, crate::plugins::EmbedderError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "wordcount"
    }
}
