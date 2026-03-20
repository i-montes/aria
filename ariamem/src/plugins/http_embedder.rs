use crate::plugins::{Embedder, EmbedderError};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct HttpEmbedder {
    url: String,
    model: String,
    dimension: usize,
    client: reqwest::Client,
    cache: Arc<RwLock<std::collections::HashMap<String, Vec<f32>>>>,
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

impl HttpEmbedder {
    pub fn new(url: String, model: String, dimension: usize) -> Self {
        Self {
            url,
            model,
            dimension,
            client: reqwest::Client::new(),
            cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    pub async fn from_ollama(url: &str, model: &str) -> Result<Self, EmbedderError> {
        let url = format!("{}/api/embeddings", url.trim_end_matches('/'));
        
        let test_req = EmbedRequest {
            model: model.to_string(),
            prompt: "test".to_string(),
        };
        
        let client = reqwest::Client::new();
        let resp = client.post(&url).json(&test_req).send().await
            .map_err(|e| EmbedderError::Inference(e.to_string()))?;
        
        if !resp.status().is_success() {
            return Err(EmbedderError::Inference(format!("Ollama returned: {}", resp.status())));
        }
        
        let body: serde_json::Value = resp.json().await
            .map_err(|e| EmbedderError::Inference(e.to_string()))?;
        
        let dim = body["embedding"].as_array()
            .map(|a| a.len())
            .unwrap_or(384);
        
        Ok(Self::new(url, model.to_string(), dim))
    }
}

#[async_trait::async_trait]
impl Embedder for HttpEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(self.embed_async(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut results = Vec::new();
            for text in texts {
                results.push(self.embed_async(text).await?);
            }
            Ok(results)
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "http"
    }
}

impl HttpEmbedder {
    async fn embed_async(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        let cache_key = text.to_string();
        
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }
        
        let request = EmbedRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
        };
        
        let response = self.client.post(&self.url)
            .json(&request)
            .timeout(std::time::Duration::from_secs(30))
            .send().await
            .map_err(|e| EmbedderError::Inference(e.to_string()))?;
        
        if !response.status().is_success() {
            return Err(EmbedderError::Inference(format!(
                "HTTP {}: {}", 
                response.status(),
                response.text().await.unwrap_or_default()
            )));
        }
        
        let body: serde_json::Value = response.json().await
            .map_err(|e| EmbedderError::Inference(e.to_string()))?;
        
        let embedding: Vec<f32> = body["embedding"].as_array()
            .ok_or_else(|| EmbedderError::Inference("No embedding in response".to_string()))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        
        let mut cache = self.cache.write().await;
        if cache.len() < 10000 {
            cache.insert(cache_key, embedding.clone());
        }
        
        Ok(embedding)
    }
}
