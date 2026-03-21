use crate::core::{Memory, MemoryType, RelationType};
use crate::core::engine::{MemoryEngine, RetrievalSource};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use uuid::Uuid;
use axum::{
    extract::State,
    routing::post,
    Json, Router,
};
use std::net::SocketAddr;

#[derive(Debug, Deserialize, Serialize)]
struct RpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct RpcResponse {
    jsonrpc: String,
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(Debug, Serialize)]
struct RpcError {
    code: i32,
    message: String,
}

pub struct McpServer {
    engine: Arc<MemoryEngine>,
}

impl McpServer {
    pub fn new(engine: MemoryEngine) -> Self {
        Self {
            engine: Arc::new(engine),
        }
    }

    pub async fn run_stdio(&self) -> Result<(), Box<dyn std::error::Error>> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin).lines();

        while let Some(line) = reader.next_line().await? {
            if let Ok(req) = serde_json::from_str::<RpcRequest>(&line) {
                let id = req.id.unwrap_or(Value::Null);
                let response = handle_request(&req.method, req.params, self.engine.clone(), id).await;
                let response_json = serde_json::to_string(&response)?;
                stdout.write_all(response_json.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
            }
        }

        Ok(())
    }

    pub async fn run_http(&self, port: u16) -> Result<(), Box<dyn std::error::Error>> {
        let app = Router::new()
            .route("/", post(handle_http_post))
            .with_state(self.engine.clone());

        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

async fn handle_http_post(
    State(engine): State<Arc<MemoryEngine>>,
    Json(req): Json<RpcRequest>,
) -> Json<RpcResponse> {
    let id = req.id.unwrap_or(Value::Null);
    let response = handle_request(&req.method, req.params, engine, id).await;
    Json(response)
}

async fn handle_request(
    method: &str,
    params: Option<Value>,
    engine: Arc<MemoryEngine>,
    id: Value,
) -> RpcResponse {
    match method {
        "initialize" => make_tool_result(id, serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {
                "name": "ariamem",
                "version": "0.1.0"
            }
        })),
        "list_tools" => make_tool_result(id, serde_json::json!({
            "tools": [
                {
                    "name": "store_memory",
                    "description": "Store a new memory in ARIA",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string" },
                            "type": { "type": "string", "enum": ["world", "experience", "opinion", "observation"] }
                        },
                        "required": ["content"]
                    }
                },
                {
                    "name": "search_memories",
                    "description": "Search for memories using hybrid retrieval",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" },
                            "limit": { "type": "number", "default": 5 }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "link_memories",
                    "description": "Create a relation between two existing memories",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "source_id": { "type": "string" },
                            "target_id": { "type": "string" },
                            "relation": { "type": "string" }
                        },
                        "required": ["source_id", "target_id"]
                    }
                },
                {
                    "name": "get_stats",
                    "description": "Get memory engine statistics",
                    "inputSchema": { "type": "object", "properties": {} }
                }
            ]
        })),
        "call_tool" => {
            let params = params.unwrap_or(Value::Null);
            let name = params["name"].as_str().unwrap_or("");
            let args = &params["arguments"];
            handle_tool_call(name, args, engine, id).await
        }
        _ => make_tool_error(id, -32601, "Method not found"),
    }
}

async fn handle_tool_call(
    name: &str,
    args: &Value,
    engine: Arc<MemoryEngine>,
    id: Value,
) -> RpcResponse {
    match name {
        "store_memory" => {
            let content = args["content"].as_str().unwrap_or("");
            let type_str = args["type"].as_str().unwrap_or("world");
            
            let mem_type = match type_str {
                "experience" => MemoryType::Experience,
                "opinion" => MemoryType::Opinion,
                "observation" => MemoryType::Observation,
                _ => MemoryType::World,
            };

            let memory = Memory::new(content.to_string(), mem_type);

            match engine.store(memory) {
                Ok(m) => make_tool_result(id, format!("Memory stored successfully with ID: {}", m.id).into()),
                Err(e) => make_tool_error(id, -1, &format!("Failed to store memory: {}", e)),
            }
        }
        "search_memories" => {
            let query = args["query"].as_str().unwrap_or("");
            let limit = args["limit"].as_u64().unwrap_or(5) as usize;

            match engine.search_by_text(query, limit) {
                Ok(results) => {
                    let mut yaml = String::new();
                    yaml.push_str("inst: Responde solo con esta memoria. Usa 'direct' como hechos primarios y 'graph' para el contexto lógico/causal basado en 'rel'. Sintetiza ambos sin alucinar información externa.\n");
                    yaml.push_str("mem:\n");
                    
                    let mut direct_results = Vec::new();
                    let mut graph_results = Vec::new();
                    
                    for r in &results {
                        match &r.source {
                            RetrievalSource::Direct => direct_results.push(r),
                            RetrievalSource::Graph(_, _) => graph_results.push(r),
                        }
                    }

                    if !direct_results.is_empty() {
                        yaml.push_str("  direct:\n");
                        for r in direct_results {
                            yaml.push_str(&format!("    - id: {}\n", r.memory.id));
                            let content = r.memory.content.replace('\n', " ");
                            yaml.push_str(&format!("      sum: {}\n", content));
                        }
                    }

                    if !graph_results.is_empty() {
                        yaml.push_str("  graph:\n");
                        for r in graph_results {
                            if let RetrievalSource::Graph(origin, rel) = &r.source {
                                yaml.push_str(&format!("    - id: {}\n", r.memory.id));
                                let content = r.memory.content.replace('\n', " ");
                                yaml.push_str(&format!("      sum: {}\n", content));
                                yaml.push_str(&format!("      rel: {:?}->{}\n", rel, origin));
                            }
                        }
                    }

                    make_tool_result(id, Value::String(yaml))
                }
                Err(e) => make_tool_error(id, -1, &format!("Search failed: {}", e)),
            }
        }
        "link_memories" => {
            let source_str = args["source_id"].as_str().unwrap_or("");
            let target_str = args["target_id"].as_str().unwrap_or("");
            let relation_str = args["relation"].as_str().unwrap_or("related");

            let source_id = match Uuid::parse_str(source_str) {
                Ok(uuid) => uuid,
                Err(_) => return make_tool_error(id, -1, "Invalid source UUID format"),
            };

            let target_id = match Uuid::parse_str(target_str) {
                Ok(uuid) => uuid,
                Err(_) => return make_tool_error(id, -1, "Invalid target UUID format"),
            };

            let relation = match relation_str.to_lowercase().as_str() {
                "temporal" => RelationType::Temporal,
                "semantic" => RelationType::Semantic,
                "entity" => RelationType::Entity,
                "causal" => RelationType::Causal,
                "works_on" => RelationType::WorksOn,
                _ => RelationType::Related,
            };

            match engine.link_by_ids(&source_id, &target_id, relation) {
                Ok(edge) => make_tool_result(id, format!("Successfully linked {} -> {} with relation {:?}. Edge ID: {}", source_id, target_id, relation, edge.id).into()),
                Err(e) => make_tool_error(id, -1, &format!("Failed to link memories: {}", e)),
            }
        }
        "get_stats" => {
            match engine.count() {
                Ok(count) => {
                    let mut stats = format!("Total memories: {}\n", count);
                    let types = [MemoryType::World, MemoryType::Experience, MemoryType::Opinion, MemoryType::Observation];
                    for mt in types {
                        if let Ok(mems) = engine.list_by_type(mt) {
                            stats.push_str(&format!("  {:?}: {}\n", mt, mems.len()));
                        }
                    }
                    make_tool_result(id, Value::String(stats))
                },
                Err(e) => make_tool_error(id, -1, &format!("Failed to get stats: {}", e)),
            }
        }
        _ => make_tool_error(id, -32601, "Tool not found"),
    }
}

fn make_tool_result(id: Value, result: Value) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0".to_string(),
        id: Some(id),
        result: Some(result),
        error: None,
    }
}

fn make_tool_error(id: Value, code: i32, message: &str) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0".to_string(),
        id: Some(id),
        result: None,
        error: Some(RpcError {
            code,
            message: message.to_string(),
        }),
    }
}
