use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use thiserror::Error;
use reqwest::Client;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub content: String,
    pub fact_type: FactType,
    pub entities: Vec<String>,
    pub temporal_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    pub confidence: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactType {
    World,
    Experience,
    Opinion,
    Observation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub entity_type: String,
    pub mentions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub facts: Vec<Fact>,
    pub entities: Vec<Entity>,
    pub summary: Option<String>,
}

use async_trait::async_trait;

#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn extract_facts(&self, text: &str) -> Result<ExtractionResult>;
    async fn extract_facts_batch(&self, texts: &[&str]) -> Result<Vec<ExtractionResult>>;
    async fn summarize(&self, text: &str, max_tokens: usize) -> Result<String>;
}

pub struct HttpLLMProvider {
    url: String,
    model: String,
    api_key: Option<String>,
    client: Client,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Deserialize)]
struct ChatMessageResponse {
    content: String,
}

impl HttpLLMProvider {
    pub fn new(url: String, model: String, api_key: Option<String>) -> Self {
        Self {
            url,
            model,
            api_key,
            client: Client::new(),
        }
    }

    pub fn from_ollama(model: &str) -> Self {
        Self::new(
            "http://localhost:11434/v1/chat/completions".to_string(),
            model.to_string(),
            None,
        )
    }

    pub fn from_openai(model: &str, api_key: String) -> Self {
        Self::new(
            "https://api.openai.com/v1/chat/completions".to_string(),
            model.to_string(),
            Some(api_key),
        )
    }

    async fn call_llm(&self, prompt: &str, system_prompt: &str) -> Result<String> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                },
            ],
            response_format: Some(ResponseFormat {
                format_type: "json_object".to_string(),
            }),
        };

        let mut req_builder = self.client.post(&self.url).json(&request);
        
        if let Some(key) = &self.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
        }

        let response = req_builder.send().await
            .map_err(|e| LLMProviderError::Api(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMProviderError::Api(format!("LLM API error: {} - {}", status, error_text)));
        }

        let chat_res: ChatResponse = response.json().await
            .map_err(|e| LLMProviderError::Parse(e.to_string()))?;

        let content = chat_res.choices.get(0)
            .ok_or_else(|| LLMProviderError::Api("No choices in LLM response".to_string()))?
            .message.content.clone();

        Ok(content)
    }
}

#[async_trait]
impl LLMProvider for HttpLLMProvider {
    async fn extract_facts(&self, text: &str) -> Result<ExtractionResult> {
        let system_prompt = r#"You are an expert information extractor. 
Extract facts, entities, and a summary from the text. 
Respond ONLY with a valid JSON object in this format:
{
  "facts": [{"content": "string", "type": "world|experience|opinion|observation", "entities": ["string"], "confidence": 0.0-1.0}],
  "entities": [{"name": "string", "type": "Person|Place|Organization|Concept", "mentions": 1}],
  "summary": "string"
}"#;

        let json_str = self.call_llm(text, system_prompt).await?;
        
        let raw_res: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| LLMProviderError::Parse(e.to_string()))?;

        // Map to ExtractionResult
        let facts = raw_res["facts"].as_array().unwrap_or(&vec![]).iter().map(|f| {
            let f_type = match f["type"].as_str().unwrap_or("world") {
                "experience" => FactType::Experience,
                "opinion" => FactType::Opinion,
                "observation" => FactType::Observation,
                _ => FactType::World,
            };
            Fact {
                content: f["content"].as_str().unwrap_or("").to_string(),
                fact_type: f_type,
                entities: f["entities"].as_array().unwrap_or(&vec![]).iter().map(|e| e.as_str().unwrap_or("").to_string()).collect(),
                temporal_range: None,
                confidence: f["confidence"].as_f64().unwrap_or(0.8) as f32,
                metadata: HashMap::new(),
            }
        }).collect();

        let entities = raw_res["entities"].as_array().unwrap_or(&vec![]).iter().map(|e| {
            Entity {
                name: e["name"].as_str().unwrap_or("").to_string(),
                entity_type: e["type"].as_str().unwrap_or("Concept").to_string(),
                mentions: e["mentions"].as_u64().unwrap_or(1) as usize,
            }
        }).collect();

        Ok(ExtractionResult {
            facts,
            entities,
            summary: raw_res["summary"].as_str().map(|s| s.to_string()),
        })
    }

    async fn extract_facts_batch(&self, texts: &[&str]) -> Result<Vec<ExtractionResult>> {
        use tokio::task::JoinSet;
        use std::sync::Arc;

        let mut set = JoinSet::new();
        let provider = Arc::new(HttpLLMProvider {
            url: self.url.clone(),
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            client: self.client.clone(),
        });

        for (i, text) in texts.iter().enumerate() {
            let text = text.to_string();
            let p = provider.clone();
            set.spawn(async move {
                (i, p.extract_facts(&text).await)
            });
        }

        let mut results = vec![None; texts.len()];
        while let Some(res) = set.join_next().await {
            let (i, extraction_res) = res.map_err(|e| LLMProviderError::Api(e.to_string()))?;
            results[i] = Some(extraction_res?);
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    async fn summarize(&self, text: &str, _max_tokens: usize) -> Result<String> {
        let system_prompt = "Summarize the following text concisely.";
        self.call_llm(text, system_prompt).await
    }
}
