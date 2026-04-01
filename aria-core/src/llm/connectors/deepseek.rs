use crate::llm::connectors::trait_base::{LlmConnector, LlmRequest, LlmResponse};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;

pub struct DeepSeekConnector {
    client: Client,
    api_key: String,
}

impl DeepSeekConnector {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }
}

#[async_trait]
impl LlmConnector for DeepSeekConnector {
    fn name(&self) -> &str { "deepseek" }

    async fn completion(&self, request: LlmRequest) -> Result<LlmResponse> {
        let url = "https://api.deepseek.com/chat/completions";

        let body = json!({
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        });

        let response = self.client.post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await?;

        let data: serde_json::Value = response.json().await?;
        
        let content = data["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek API error: {:?}", data))?
            .to_string();

        Ok(LlmResponse {
            content,
            raw_response: Some(data),
        })
    }
}
