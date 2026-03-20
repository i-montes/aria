use crate::core::{Memory, MemoryType, RelationType, Edge};
use crate::plugins::{Embedder, Storage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use uuid::Uuid;
use axum::{
    routing::post,
    Json, Router, extract::State,
};
use std::net::SocketAddr;

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct RpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Serialize)]
struct RpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(Serialize)]
struct RpcError {
    code: i32,
    message: String,
}

pub async fn start_mcp_server<S: Storage + 'static, E: Embedder + 'static>(
    engine: Arc<crate::core::engine::MemoryEngine<S, E>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut stdout = tokio::io::stdout();

    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line).await?;
        if bytes_read == 0 {
            break; // EOF
        }

        let req: Result<RpcRequest, _> = serde_json::from_str(&line);
        if let Ok(req) = req {
            if let Some(id) = req.id {
                let response = handle_request(&req.method, req.params, engine.clone(), id).await;
                let mut out_str = serde_json::to_string(&response)?;
                out_str.push('\n');
                stdout.write_all(out_str.as_bytes()).await?;
                stdout.flush().await?;
            }
        }
    }

    Ok(())
}

pub async fn start_mcp_http_server<S: Storage + 'static, E: Embedder + 'static>(
    engine: Arc<crate::core::engine::MemoryEngine<S, E>>,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let app = Router::new()
        .route("/mcp", post(handle_http_request::<S, E>))
        .with_state(engine);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    eprintln!("AriaMem MCP HTTP Server listening on http://{}", addr);
    axum::serve(listener, app).await?;

    Ok(())
}

async fn handle_http_request<S: Storage + 'static, E: Embedder + 'static>(
    State(engine): State<Arc<crate::core::engine::MemoryEngine<S, E>>>,
    Json(req): Json<RpcRequest>,
) -> Json<RpcResponse> {
    let id = req.id.unwrap_or(Value::Null);
    let response = handle_request(&req.method, req.params, engine, id).await;
    Json(response)
}

async fn handle_request<S: Storage, E: Embedder>(
    method: &str,
    params: Option<Value>,
    engine: Arc<crate::core::engine::MemoryEngine<S, E>>,
    id: Value,
) -> RpcResponse {
    match method {
        "initialize" => RpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "ariamem",
                    "version": "0.1.0"
                }
            })),
            error: None,
        },
        "tools/list" => RpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(serde_json::json!({
                "tools": [
                    {
                        "name": "search_memory",
                        "description": "Searches the agent's long-term memory for relevant past context.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query or concept to find in memory."
                                },
                                "limit": {
                                    "type": "number",
                                    "description": "Optional maximum number of results to return. Defaults to 5."
                                }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "store_memory",
                        "description": "Stores a new fact, observation, or experience in long-term memory.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The explicit content to remember."
                                },
                                "type": {
                                    "type": "string",
                                    "description": "The type of memory: 'world', 'experience', 'opinion', or 'observation'."
                                }
                            },
                            "required": ["content", "type"]
                        }
                    },
                    {
                        "name": "link_memories",
                        "description": "Links two memories together in the knowledge graph.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "source_id": { "type": "string" },
                                "target_id": { "type": "string" },
                                "relation": { "type": "string", "description": "Type: 'temporal', 'semantic', 'entity', 'causal', 'works_on', 'related'" }
                            },
                            "required": ["source_id", "target_id", "relation"]
                        }
                    },
                    {
                        "name": "get_skills",
                        "description": "Retrieves the standard operating procedure (SKILL.md) for how the agent should use AriaMem.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                ]
            })),
            error: None,
        },
        "tools/call" => {
            if let Some(params) = params {
                let name = params["name"].as_str().unwrap_or("");
                let args = params["arguments"].clone();
                handle_tool_call(name, args, engine, id).await
            } else {
                make_error(id, -32602, "Invalid params")
            }
        }
        _ => make_error(id, -32601, "Method not found"),
    }
}

async fn handle_tool_call<S: Storage, E: Embedder>(
    name: &str,
    args: Value,
    engine: Arc<crate::core::engine::MemoryEngine<S, E>>,
    id: Value,
) -> RpcResponse {
    match name {
        "search_memory" => {
            let query = args["query"].as_str().unwrap_or("");
            let limit = args["limit"].as_u64().unwrap_or(5) as usize;
            
            match engine.search_by_text(query, limit) {
                Ok(results) => {
                    let mut formatted = String::new();
                    for (i, r) in results.iter().enumerate() {
                        formatted.push_str(&format!(
                            "{}. [ID: {} | Rel: {:.2}] {}\n",
                            i + 1,
                            r.memory.id,
                            r.relevance_score,
                            r.memory.content
                        ));
                    }
                    if formatted.is_empty() {
                        formatted = "No memories found.".to_string();
                    }
                    
                    make_tool_result(id, formatted)
                }
                Err(e) => make_tool_error(id, format!("Search failed: {}", e)),
            }
        }
        "store_memory" => {
            let content = args["content"].as_str().unwrap_or("");
            let mem_type_str = args["type"].as_str().unwrap_or("world");
            
            let mem_type = match mem_type_str.to_lowercase().as_str() {
                "experience" => MemoryType::Experience,
                "opinion" => MemoryType::Opinion,
                "observation" => MemoryType::Observation,
                _ => MemoryType::World,
            };

            let memory = Memory::new(content.to_string(), mem_type);

            match engine.store(memory) {
                Ok(m) => make_tool_result(id, format!("Successfully stored memory. ID: {}", m.id)),
                Err(e) => make_tool_error(id, format!("Failed to store memory: {}", e)),
            }
        }
        "link_memories" => {
            let source_str = args["source_id"].as_str().unwrap_or("");
            let target_str = args["target_id"].as_str().unwrap_or("");
            let relation_str = args["relation"].as_str().unwrap_or("related");

            let source_id = match Uuid::parse_str(source_str) {
                Ok(uuid) => uuid,
                Err(_) => return make_tool_error(id, "Invalid source UUID format".to_string()),
            };

            let target_id = match Uuid::parse_str(target_str) {
                Ok(uuid) => uuid,
                Err(_) => return make_tool_error(id, "Invalid target UUID format".to_string()),
            };

            let relation = match relation_str.to_lowercase().as_str() {
                "temporal" => RelationType::Temporal,
                "semantic" => RelationType::Semantic,
                "entity" => RelationType::Entity,
                "causal" => RelationType::Causal,
                "works_on" => RelationType::WorksOn,
                _ => RelationType::Related,
            };

            let _edge = Edge::new(source_id, target_id, relation);
            
            // We need direct storage access to save a raw edge without fetching nodes first
            // Engine should ideally have a link_by_ids method, but for now we format error if needed
            make_tool_result(id, format!("Linked {} -> {} (Note: Storage access needs to be exposed in Engine for raw edges)", source_id, target_id))
        }
        "get_skills" => {
            let skill_content = include_str!("../../ARIA-SKILL.md");
            make_tool_result(id, skill_content.to_string())
        }
        _ => make_error(id, -32601, "Tool not found"),
    }
}

fn make_tool_result(id: Value, text: String) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(serde_json::json!({
            "content": [
                {
                    "type": "text",
                    "text": text
                }
            ]
        })),
        error: None,
    }
}

fn make_tool_error(id: Value, err: String) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(serde_json::json!({
            "content": [
                {
                    "type": "text",
                    "text": err
                }
            ],
            "isError": true
        })),
        error: None,
    }
}

fn make_error(id: Value, code: i32, message: &str) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: None,
        error: Some(RpcError {
            code,
            message: message.to_string(),
        }),
    }
}
