use crate::core::{Memory, MemoryType, RelationType};
use crate::core::engine::MemoryEngine;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use axum::{
    extract::State,
    routing::post,
    Json, Router,
};

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
    pub fn new(engine: Arc<MemoryEngine>) -> Self {
        Self {
            engine,
        }
    }

    pub async fn run_stdio(&self) -> Result<(), Box<dyn std::error::Error>> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin).lines();

        while let Some(line) = reader.next_line().await? {
            if let Ok(req) = serde_json::from_str::<RpcRequest>(&line) {
                let id = req.id.clone().unwrap_or(Value::Null);
                
                // Notifications (no ID) should not be responded to
                if req.id.is_none() && req.method.contains("/") {
                    // Handle notification logic if needed, but don't send RpcResponse
                    continue;
                }

                let response = handle_request(&req.method, req.params, self.engine.clone(), id).await;
                
                // If it was a notification that we didn't filter above, response might be 'Null' 
                // in result, but for safety in stdio, we only send if it's a proper response.
                let response_json = serde_json::to_string(&response)?;
                stdout.write_all(response_json.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
            }
        }

        Ok(())
    }

    pub async fn run_http(&self, port: u16) -> Result<(), Box<dyn std::error::Error>> {
        use tower::ServiceBuilder;
        
        let app = Router::new()
            .route("/", post(handle_http_post))
            .layer(
                ServiceBuilder::new()
                    .concurrency_limit(10) // Limit to 10 simultaneous requests
                    .into_inner()
            )
            .with_state(self.engine.clone());

        let addr = format!("0.0.0.0:{}", port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        tracing::info!("MCP HTTP server listening on {}", addr);
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
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "ariamem",
                "version": "0.1.0"
            }
        })),
        "ping" => make_tool_result(id, serde_json::json!({})),
        "notifications/initialized" => {
            make_tool_result(id, Value::Null)
        },
        "tools/list" | "list_tools" => {
            make_tool_result(id, serde_json::json!({
                "tools": [
                    {
                        "name": "store_memory",
                        "description": "Store a new memory in ARIA. Can optionally link to existing memories (Graph Context).",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "content": { "type": "string", "description": "The textual content to remember" },
                                "type": { 
                                    "type": "string", 
                                    "enum": ["world", "experience", "opinion", "observation"],
                                    "description": "Type of memory. 'world' for facts, 'experience' for events, etc."
                                },
                                "links": {
                                    "type": "array",
                                    "description": "Optional list of links to existing memories to create a graph context.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "to": { "type": "string", "description": "UUID of the target memory" },
                                            "rel": { 
                                                "type": "string", 
                                                "enum": ["temporal", "semantic", "entity", "causal", "related", "works_on"],
                                                "description": "Type of relationship"
                                            }
                                        },
                                        "required": ["to", "rel"]
                                    }
                                }
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
                        "name": "delete_memory",
                        "description": "Permanently remove a memory and its associated relations",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string", "description": "The UUID of the memory to delete" }
                            },
                            "required": ["id"]
                        }
                    },
                    {
                        "name": "get_stats",
                        "description": "Get memory engine statistics",
                        "inputSchema": { "type": "object", "properties": {} }
                    }
                ]
            }))
        },
        "tools/call" | "call_tool" => {
            let params = params.unwrap_or(Value::Null);
            let name = params["name"].as_str().unwrap_or("");
            let args = &params["arguments"];
            handle_tool_call(name, args, engine, id).await
        }
        _ => {
            eprintln!("Unknown MCP method: {}", method);
            make_tool_error(id, -32601, &format!("Method {} not found", method))
        }
    }
}

async fn handle_tool_call(
    name: &str,
    args: &Value,
    engine: Arc<MemoryEngine>,
    id: Value,
) -> RpcResponse {
    let result = match name {
        "store_memory" => {
            let content = args["content"].as_str().unwrap_or("");
            let type_str = args["type"].as_str().unwrap_or("world");
            let mem_type = type_str.parse::<MemoryType>().unwrap_or(MemoryType::World);

            let mut links = Vec::new();
            if let Some(links_array) = args["links"].as_array() {
                for link_obj in links_array {
                    if let (Some(to_str), Some(rel_str)) = (link_obj["to"].as_str(), link_obj["rel"].as_str()) {
                        let rel = rel_str.parse::<RelationType>().unwrap_or(RelationType::Related);
                        links.push((to_str.to_string(), rel));
                    }
                }
            }

            let memory = Memory::new(content.to_string(), mem_type);
            let num_links = links.len();

            match engine.store_contextual(memory, links).await {
                Ok(m) => Ok(format!("Memory stored successfully with ID: {}. ({} links created)", m.id, num_links)),
                Err(e) => Err(format!("Failed to store memory: {}", e)),
            }
        }
        "search_memories" => {
            let query = args["query"].as_str().unwrap_or("");
            let limit = args["limit"].as_u64().unwrap_or(5) as usize;

            match engine.search_by_text(query, limit).await {
                Ok(results) => Ok(engine.format_search_results(&results)),
                Err(e) => Err(format!("Search failed: {}", e)),
            }
        }
        "link_memories" => {
            let source_id = args["source_id"].as_str().unwrap_or("");
            let target_id = args["target_id"].as_str().unwrap_or("");
            let relation_str = args["relation"].as_str().unwrap_or("related");
            let relation = relation_str.parse::<RelationType>().unwrap_or(RelationType::Related);

            match engine.link_by_ids(source_id, target_id, relation) {
                Ok(edge) => Ok(format!("Successfully linked {} -> {} with relation {:?}. Edge ID: {}", source_id, target_id, relation, edge.id)),
                Err(e) => Err(format!("Failed to link memories: {}", e)),
            }
        }
        "delete_memory" => {
            let id_str = args["id"].as_str().unwrap_or("");
            match engine.exists(id_str) {
                Ok(true) => {
                    match engine.delete(id_str) {
                        Ok(_) => Ok(format!("Memory {} and its relations were permanently deleted.", id_str)),
                        Err(e) => Err(format!("Failed to delete memory {}: {}", id_str, e)),
                    }
                },
                Ok(false) => Err(format!("Memory {} not found", id_str)),
                Err(e) => Err(format!("Error checking memory existence: {}", e)),
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
                    Ok(stats)
                },
                Err(e) => Err(format!("Failed to get stats: {}", e)),
            }
        }
        _ => return make_tool_error(id, -32601, "Tool not found"),
    };

    match result {
        Ok(text) => make_tool_result(id, serde_json::json!({
            "content": [
                {
                    "type": "text",
                    "text": text
                }
            ]
        })),
        Err(msg) => make_tool_result(id, serde_json::json!({
            "content": [
                {
                    "type": "text",
                    "text": msg
                }
            ],
            "isError": true
        })),
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
