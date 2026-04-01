use crate::llm::connectors::trait_base::{LlmConnector, LlmRequest, LlmResponse};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;

pub struct GoogleConnector {
    client: Client,
    api_key: String,
}

impl GoogleConnector {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }
}

#[async_trait]
impl LlmConnector for GoogleConnector {
    fn name(&self) -> &str { "google" }

    async fn completion(&self, request: LlmRequest) -> Result<LlmResponse> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            request.model, self.api_key
        );

        let body = json!({
            "contents": request.messages.iter().map(|m| {
                json!({
                    "role": if m.role == "assistant" { "model" } else { "user" },
                    "parts": [{ "text": m.content }]
                })
            }).collect::<Vec<_>>(),
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            }
        });

        let response = self.client.post(&url)
            .json(&body)
            .send()
            .await?;

        let data: serde_json::Value = response.json().await?;
        
        let content = data["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Google API error: {:?}", data))?
            .to_string();

        Ok(LlmResponse {
            content,
            raw_response: Some(data),
        })
    }
}
