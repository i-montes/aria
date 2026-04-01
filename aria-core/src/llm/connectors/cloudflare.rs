use crate::llm::connectors::trait_base::{LlmConnector, LlmRequest, LlmResponse};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;

pub struct CloudflareConnector {
    client: Client,
    account_id: String,
    api_token: String,
}

impl CloudflareConnector {
    pub fn new(account_id: String, api_token: String) -> Self {
        Self {
            client: Client::new(),
            account_id,
            api_token,
        }
    }
}

#[async_trait]
impl LlmConnector for CloudflareConnector {
    fn name(&self) -> &str { "cloudflare" }

    async fn completion(&self, request: LlmRequest) -> Result<LlmResponse> {
        let url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/{}",
            self.account_id, request.model
        );

        let body = json!({
            "messages": request.messages.iter().map(|m| {
                json!({ "role": m.role, "content": m.content })
            }).collect::<Vec<_>>(),
            "max_tokens": request.max_tokens,
        });

        let response = self.client.post(&url)
            .header("Authorization", format!("Bearer {}", self.api_token))
            .json(&body)
            .send()
            .await?;

        let data: serde_json::Value = response.json().await?;
        
        let content = data["result"]["response"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Cloudflare API error: {:?}", data))?
            .to_string();

        Ok(LlmResponse {
            content,
            raw_response: Some(data),
        })
    }
}
