use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
    http::StatusCode,
    response::Html,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::core::{Memory, MemoryType, RelationType, engine::MemoryEngine};

pub struct RestApi {
    port: u16,
    engine: Arc<MemoryEngine>,
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct StoreRequest {
    content: String,
    #[serde(rename = "type")]
    memory_type: Option<String>,
}

#[derive(Deserialize)]
struct LinkRequest {
    source: String,
    target: String,
    relation: Option<String>,
}

#[derive(Serialize)]
struct StatsResponse {
    total_memories: usize,
}

impl RestApi {
    pub fn new(port: u16, engine: Arc<MemoryEngine>) -> Self {
        Self { port, engine }
    }

    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        let app = Router::new()
            .route("/", get(handle_dashboard))
            .route("/search", post(handle_search))
            .route("/memories", post(handle_store))
            .route("/memories/{id}", get(handle_get))
            .route("/links", post(handle_link))
            .route("/stats", get(handle_stats))
            .with_state(self.engine.clone());

        let addr = format!("0.0.0.0:{}", self.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        
        tracing::info!("REST API listening on {}", addr);
        axum::serve(listener, app).await?;
        
        Ok(())
    }
}

async fn handle_dashboard() -> Html<&'static str> {
    Html(include_str!("dashboard.html"))
}

async fn handle_search(
    State(engine): State<Arc<MemoryEngine>>,
    Json(payload): Json<SearchRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let limit = payload.limit.unwrap_or(10);
    match engine.search_by_text(&payload.query, limit).await {
        Ok(results) => {
            // We return the raw search results as JSON
            let json_results: Vec<serde_json::Value> = results.iter().map(|r| {
                serde_json::json!({
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "type": r.memory.memory_type,
                    "score": r.score,
                    "relevance_score": r.relevance_score,
                    "source": format!("{:?}", r.source)
                })
            }).collect();
            Ok(Json(serde_json::json!(json_results)))
        },
        Err(e) => {
            tracing::error!("Search error: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn handle_store(
    State(engine): State<Arc<MemoryEngine>>,
    Json(payload): Json<StoreRequest>,
) -> Result<(StatusCode, Json<Memory>), StatusCode> {
    let mem_type = match payload.memory_type.as_deref() {
        Some("experience") => MemoryType::Experience,
        Some("opinion") => MemoryType::Opinion,
        Some("observation") => MemoryType::Observation,
        _ => MemoryType::World,
    };

    let memory = Memory::new(payload.content, mem_type);
    match engine.store(memory).await {
        Ok(stored) => {
            let _ = engine.save_index();
            Ok((StatusCode::CREATED, Json(stored)))
        },
        Err(e) => {
            tracing::error!("Store error: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn handle_get(
    State(engine): State<Arc<MemoryEngine>>,
    Path(id): Path<String>,
) -> Result<Json<Memory>, StatusCode> {
    match engine.get(&id) {
        Ok(memory) => Ok(Json(memory)),
        Err(_) => Err(StatusCode::NOT_FOUND),
    }
}

async fn handle_link(
    State(engine): State<Arc<MemoryEngine>>,
    Json(payload): Json<LinkRequest>,
) -> Result<StatusCode, StatusCode> {
    let rel_type = match payload.relation.as_deref() {
        Some("temporal") => RelationType::Temporal,
        Some("entity") => RelationType::Entity,
        Some("causal") => RelationType::Causal,
        Some("works_on") => RelationType::WorksOn,
        _ => RelationType::Related,
    };

    match engine.link_by_ids(&payload.source, &payload.target, rel_type) {
        Ok(_) => Ok(StatusCode::CREATED),
        Err(_) => Err(StatusCode::BAD_REQUEST),
    }
}

async fn handle_stats(
    State(engine): State<Arc<MemoryEngine>>,
) -> Json<StatsResponse> {
    let count = engine.count().unwrap_or(0);
    Json(StatsResponse { total_memories: count })
}
