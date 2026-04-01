use anyhow::Result;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use std::sync::Arc;
use std::fs;
use std::path::PathBuf;
use aria_core::orchestrator::engine::OrchestratorEngine;
use aria_core::whiteboard::storage::WhiteboardStorage;
use aria_core::llm::client::LlmClient;
use aria_core::llm::connectors::trait_base::{LlmRequest, LlmMessage};
use axum::{routing::post, Json, Router};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize, Debug)]
struct AriaConfig {
    #[allow(dead_code)]
    workspace_root: String,
    database_path: String,
    #[allow(dead_code)]
    memory_port: u16,
    core_port: u16,
    // Add other fields to ignore them during deserialization if needed
    #[serde(flatten)]
    extra: serde_json::Value,
}

impl AriaConfig {
    fn load() -> Self {
        let config_path = PathBuf::from("aria.config.json");
        if config_path.exists() {
            if let Ok(content) = fs::read_to_string(config_path) {
                if let Ok(config) = serde_json::from_str(&content) {
                    return config;
                }
            }
        }
        // Fallback defaults
        Self {
            workspace_root: ".".into(),
            database_path: "aria_whiteboard.db".into(),
            memory_port: 8080,
            core_port: 3000,
        }
    }
}

#[derive(Deserialize)]
struct GoalRequest {
    goal: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    println!("ARIA Core Orchestrator starting...");
    
    let config = AriaConfig::load();
    println!("✓ Config loaded from aria.config.json");

    // 1. Inicializar Almacenamiento
    let manager = SqliteConnectionManager::file(&config.database_path);
    let pool = Pool::new(manager)?;
    let storage = WhiteboardStorage::new(pool);
    storage.init()?;
    
    // 2. Inicializar Cliente LLM
    let llm_client = Arc::new(LlmClient::from_env()?);
    println!("✓ LLM Client initialized.");

    // 3. Inicializar Motor
    let mut engine = OrchestratorEngine::new(storage.clone());
    engine.load_agents("./agents")?;
    let engine = Arc::new(tokio::sync::Mutex::new(engine));

    // 4. Servidor de Control (para el CLI)
    let app = Router::new()
        .route("/goal", post(move |Json(payload): Json<GoalRequest>| {
            let _engine = engine.clone();
            let llm = llm_client.clone();
            async move {
                println!("--> Processing Goal: {}", payload.goal);
                
                let model = std::env::var("ARIA_LLM_MODEL")
                    .unwrap_or_else(|_| "meta/llama-3.1-70b-instruct".into());

                let request = LlmRequest {
                    messages: vec![
                        LlmMessage { 
                            role: "system".into(), 
                            content: "Eres ARIA, el orquestador central de un sistema de agentes. Responde de forma clara y profesional.".into() 
                        },
                        LlmMessage { 
                            role: "user".into(), 
                            content: payload.goal.clone() 
                        },
                    ],
                    temperature: 0.7,
                    max_tokens: 1024,
                    model,
                };

                match llm.completion(request).await {
                    Ok(resp) => {
                        println!("✓ LLM Response received.");
                        Json(json!({ 
                            "status": "completed", 
                            "goal": payload.goal,
                            "response": resp.content 
                        }))
                    },
                    Err(e) => {
                        eprintln!("✗ LLM Error: {}", e);
                        Json(json!({ 
                            "status": "error", 
                            "message": e.to_string() 
                        }))
                    }
                }
            }
        }));

    let addr = format!("127.0.0.1:{}", config.core_port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("✓ Core listening on http://{}", addr);
    
    axum::serve(listener, app).await?;

    Ok(())
}
