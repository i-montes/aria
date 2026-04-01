use serde::{Deserialize, Serialize};
use async_trait::async_trait;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("API request failed: {0}")]
    ApiError(String),
    #[error("Parsing error: {0}")]
    ParseError(String),
    #[error("Rate limited")]
    RateLimit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: String, // system, user, assistant
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    pub messages: Vec<LlmMessage>,
    pub temperature: f32,
    pub max_tokens: u32,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub content: String,
    pub raw_response: Option<serde_json::Value>,
}

#[async_trait]
pub trait LlmConnector: Send + Sync {
    async fn completion(&self, request: LlmRequest) -> anyhow::Result<LlmResponse>;
    fn name(&self) -> &str;
}
