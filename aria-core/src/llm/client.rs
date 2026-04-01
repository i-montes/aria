use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use crate::llm::connectors::trait_base::{LlmConnector, LlmRequest, LlmResponse};
use crate::llm::connectors::cloudflare::CloudflareConnector;
use crate::llm::connectors::google::GoogleConnector;
use crate::llm::connectors::deepseek::DeepSeekConnector;

pub struct LlmClient {
    connectors: HashMap<String, Arc<dyn LlmConnector>>,
    default_connector: String,
}

impl LlmClient {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();
        let mut connectors: HashMap<String, Arc<dyn LlmConnector>> = HashMap::new();
        let mut default_connector = String::new();

        // 1. Cloudflare
        if let (Ok(id), Ok(token)) = (std::env::var("CLOUDFLARE_ACCOUNT_ID"), std::env::var("CLOUDFLARE_API_TOKEN")) {
            connectors.insert("cloudflare".to_string(), Arc::new(CloudflareConnector::new(id, token)));
            default_connector = "cloudflare".to_string();
        }

        // 2. Google
        if let Ok(key) = std::env::var("GOOGLE_AI_API_KEY") {
            connectors.insert("google".to_string(), Arc::new(GoogleConnector::new(key)));
            if default_connector.is_empty() { default_connector = "google".to_string(); }
        }

        // 3. DeepSeek
        if let Ok(key) = std::env::var("DEEPSEEK_API_KEY") {
            connectors.insert("deepseek".to_string(), Arc::new(DeepSeekConnector::new(key)));
            if default_connector.is_empty() { default_connector = "deepseek".to_string(); }
        }

        if connectors.is_empty() {
            anyhow::bail!("No LLM connectors configured. Set CLOUDFLARE_API_TOKEN, GOOGLE_AI_API_KEY, or DEEPSEEK_API_KEY.");
        }

        Ok(Self { connectors, default_connector })
    }

    pub async fn completion(&self, request: LlmRequest) -> Result<LlmResponse> {
        let connector_name = std::env::var("ARIA_LLM_CONNECTOR").unwrap_or_else(|_| self.default_connector.clone());
        
        let connector = self.connectors.get(&connector_name)
            .ok_or_else(|| anyhow::anyhow!("Connector '{}' not found or not configured", connector_name))?;
        
        connector.completion(request).await
    }
}
