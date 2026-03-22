use crate::{
    Config, Embedder, Memory, MemoryEngine, MemoryType, Model2VecEmbedder, RelationType, SqliteStorage, Edge, Storage, core::MemoryQuery
};
use crate::api::mcp::McpServer;
use crate::api::rest::RestApi;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "ariamem")]
#[command(
    about = "AriaMem - Hybrid Memory Engine for AI Agents",
    version = "0.1.0"
)]
pub struct Cli {
    #[arg(short, long)]
    pub database: Option<PathBuf>,

    #[arg(short, long)]
    pub model: Option<String>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Initialize a new memory database
    Init,
    /// Run as an MCP server (stdio or http)
    Serve {
        #[arg(short, long)]
        port: Option<u16>,
        /// Also start the REST API on this port
        #[arg(short = 'r', long)]
        rest_port: Option<u16>,
    },
    /// Run as a REST API server
    ServeRest {
        #[arg(short, long)]
        port: Option<u16>,
    },
    /// Store a new memory
    Store {
        #[arg(short, long)]
        content: String,
        #[arg(short = 'm', long, default_value = "world")]
        memory_type: String,
    },
    /// Get a specific memory by ID
    Get {
        id: String,
    },
    /// List memories with optional filtering
    List {
        #[arg(short, long)]
        memory_type: Option<String>,
        #[arg(short, long, default_value = "50")]
        limit: usize,
    },
    /// Search memories using hybrid retrieval
    Search {
        #[arg(short, long)]
        query: String,
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    /// Delete a memory by ID
    Delete {
        id: String,
    },
    /// Show engine statistics
    Stats,
    /// Create a link between two memories
    Link {
        source: String,
        target: String,
        #[arg(short, long, default_value = "related")]
        relation: String,
    },
    /// List related memories for a given ID
    Related {
        id: String,
    },
}

pub async fn run() {
    let cli = Cli::parse();
    
    // Load configuration, create default if not found
    let config = Config::load().expect("Failed to load or create configuration");

    // Initialize logging
    setup_logging(&config.system.log_level);

    match &cli.command {
        Commands::Serve { port, rest_port } => {
            let engine_arc = create_engine(&config, cli.database.as_ref(), cli.model.as_ref());
            
            let mcp_engine = engine_arc.clone();
            let rest_engine = engine_arc.clone();
            
            let mcp_port = *port;
            // Default to 9090 if not specified, as requested for the background process
            let r_port = rest_port.unwrap_or(9090);

            let mcp_handle = tokio::spawn(async move {
                let server = McpServer::new(mcp_engine);
                if let Some(p) = mcp_port {
                    println!("Starting MCP HTTP server on port {}...", p);
                    if let Err(e) = server.run_http(p).await {
                        eprintln!("MCP HTTP Server Error: {}", e);
                    }
                } else {
                    println!("Starting MCP stdio server...");
                    if let Err(e) = server.run_stdio().await {
                        eprintln!("MCP Stdio Server Error: {}", e);
                    }
                }
            });

            let rest_handle = tokio::spawn(async move {
                let api = RestApi::new(r_port, rest_engine);
                println!("Starting REST API server on port {}...", r_port);
                if let Err(e) = api.start().await {
                    eprintln!("REST API Server Error: {}", e);
                }
            });

            // Wait for shutdown signal or server error
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    println!("\nShutdown signal received. Saving index...");
                    if let Err(e) = engine_arc.save_index() {
                        eprintln!("Error saving index during shutdown: {}", e);
                    }
                    println!("✓ Cleanup complete. Exiting.");
                }
                res = tokio::try_join!(mcp_handle, rest_handle) => {
                    if let Err(e) = res {
                        eprintln!("Server error: {}", e);
                    }
                }
            }
        }
        Commands::ServeRest { port } => {
            let engine_arc = create_engine(&config, cli.database.as_ref(), cli.model.as_ref());
            let server_port = port.unwrap_or(8081);
            let api = RestApi::new(server_port, engine_arc.clone());
            
            println!("Starting REST API server on port {}...", server_port);
            
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    println!("\nShutdown signal received. Saving index...");
                    let _ = engine_arc.save_index();
                }
                res = api.start() => {
                    if let Err(e) = res {
                        eprintln!("REST API Server Error: {}", e);
                    }
                }
            }
        }
        Commands::Init => {
            let db_path = cli.database.clone().unwrap_or_else(|| config.get_db_path());
            println!("Initializing AriaMem at: {:?}", db_path);
            let _ = create_storage(&config, cli.database.as_ref());
            println!("✓ Initialized!");
        }

        Commands::Store {
            content,
            memory_type,
        } => {
            // Try calling the background service first for instant response
            if let Some(response) = call_mcp_tool("store_memory", serde_json::json!({
                "content": content,
                "type": memory_type
            })) {
                println!("✓ (via service) {}", response);
                return;
            }

            // Fallback to local loading if service is down
            let engine = create_engine(&config, cli.database.as_ref(), cli.model.as_ref());
            let mem_type = memory_type.parse::<MemoryType>().unwrap_or(MemoryType::World);

            let memory = Memory::new(content.clone(), mem_type);

            match engine.store(memory).await {
                Ok(stored) => {
                    println!("✓ Stored! ID: {}", stored.id);
                },
                Err(e) => eprintln!("✗ Error: {}", e),
            }
        }

        Commands::Get { id } => {
            let storage = create_storage(&config, cli.database.as_ref());

            match storage.load_memory(id) {
                Ok(m) => {
                    println!("ID: {}", m.id);
                    println!("Type: {:?}", m.memory_type);
                    println!("Content: {}", m.content);
                    println!("Accesses: {}", m.access_count);
                }
                Err(_) => eprintln!("✗ Memory not found"),
            }
        }

        Commands::List { memory_type, limit } => {
            let storage = create_storage(&config, cli.database.as_ref());

            let query = MemoryQuery {
                memory_type: memory_type.as_ref().map(|t| match t.to_lowercase().as_str() {
                    "experience" => MemoryType::Experience,
                    "opinion" => MemoryType::Opinion,
                    "observation" => MemoryType::Observation,
                    _ => MemoryType::World,
                }),
                ..Default::default()
            };

            if let Ok(memories) = storage.list_memories(&query) {
                println!("{} memories:", memories.len());
                for (i, m) in memories.iter().take(*limit).enumerate() {
                    println!(
                        "  {}. [{}] ({}) {}",
                        i + 1,
                        format!("{:?}", m.memory_type),
                        m.id,
                        truncate(&m.content, 55)
                    );
                }
            }
        }

        Commands::Search { query, limit } => {
            // Try calling the background service first for instant response
            if let Some(response) = call_mcp_tool("search_memories", serde_json::json!({
                "query": query,
                "limit": limit
            })) {
                println!("Results (via service):\n{}", response);
                return;
            }

            // Fallback to local loading
            let engine = create_engine(&config, cli.database.as_ref(), cli.model.as_ref());

            match engine.search_by_text(query, *limit).await {
                Ok(results) => {
                    println!("{}", engine.format_search_results(&results));
                }
                Err(e) => eprintln!("✗ Error: {}", e),
            }
        }

        Commands::Delete { id } => {
            // Try calling the background service first
            if let Some(response) = call_mcp_tool("delete_memory", serde_json::json!({
                "id": id
            })) {
                println!("✓ (via service) {}", response);
                return;
            }

            // Fallback to local engine
            let engine = create_engine(&config, cli.database.as_ref(), cli.model.as_ref());

            match engine.delete(id) {
                Ok(_) => {
                    println!("✓ Deleted!");
                },
                Err(e) => eprintln!("✗ Error: {}", e),
            }
        }

        Commands::Stats => {
            // Try calling the background service first
            if let Some(response) = call_mcp_tool("get_stats", serde_json::json!({})) {
                println!("AriaMem Service Status:\n{}", response);
                return;
            }

            // Fallback to direct DB access
            let storage = create_storage(&config, cli.database.as_ref());
            let db_path = cli.database.clone().unwrap_or_else(|| config.get_db_path());
            println!("Stats (direct DB access) - {:?}", db_path);
            
            if let Ok(count) = storage.count_memories() {
                println!("Total: {}", count);
            }

            for (name, mt) in [
                ("World", MemoryType::World),
                ("Experience", MemoryType::Experience),
                ("Opinion", MemoryType::Opinion),
                ("Observation", MemoryType::Observation),
            ] {
                let query = MemoryQuery { memory_type: Some(mt), ..Default::default() };
                if let Ok(memories) = storage.list_memories(&query) {
                    println!("  {}: {}", name, memories.len());
                }
            }
        }

        Commands::Link {
            source,
            target,
            relation,
        } => {
            // Try calling the background service first
            if let Some(response) = call_mcp_tool("link_memories", serde_json::json!({
                "source_id": source,
                "target_id": target,
                "relation": relation
            })) {
                println!("✓ (via service) {}", response);
                return;
            }

            let engine = create_engine(&config, cli.database.as_ref(), cli.model.as_ref());
            let rel_type = match relation.to_lowercase().as_str() {
                "temporal" => RelationType::Temporal,
                "entity" => RelationType::Entity,
                "causal" => RelationType::Causal,
                "works_on" => RelationType::WorksOn,
                _ => RelationType::Related,
            };

            match engine.link_by_ids(&source, &target, rel_type) {
                Ok(_) => println!("✓ Linked! {} → {}", source, target),
                Err(e) => eprintln!("✗ Error: {}", e),
            }
        }

        Commands::Related { id } => {
            let storage = create_storage(&config, cli.database.as_ref());

            match storage.query_edges(id) {
                Ok(edges) => {
                    println!("{} related edges:", edges.len());
                    for edge in edges {
                        if let Ok(m) = storage.load_memory(&edge.target_id) {
                            println!("  • [{:?}] ({}) {}", edge.relation_type, m.id, truncate(&m.content, 60));
                        }
                    }
                }
                Err(e) => eprintln!("✗ Error: {}", e),
            }
        }
    }
}

fn create_engine(
    config: &Config,
    cli_db: Option<&PathBuf>,
    cli_model: Option<&String>,
) -> Arc<MemoryEngine> {
    let storage = create_storage(config, cli_db);

    let is_serve = std::env::args().any(|arg| arg == "serve");
    
    // Determine primary model to try
    let primary_model = if let Some(cli_m) = cli_model {
        cli_m.clone()
    } else {
        let model_path = config.get_model_path();

        if model_path.exists() && model_path.join("model.safetensors").exists() {
            model_path.to_string_lossy().to_string()
        } else {
            config.embedder.model2vec.model_name.clone()
        }
    };

    if !is_serve {
        eprintln!("Loading Model2Vec model: {}...", primary_model);
    }

    let embedder = Model2VecEmbedder::from_hub(&primary_model)
        .expect("Failed to load Model2Vec model. Make sure you have an internet connection.");

    let dim = embedder.dimension();
    
    if !is_serve {
        eprintln!("✓ Model loaded! Dimension: {}", dim);
    }

    let db_path = cli_db.cloned().unwrap_or_else(|| config.get_db_path());
    let mut index_path = db_path.clone();
    index_path.set_extension("hnsw");

    Arc::new(MemoryEngine::new_with_path(storage, embedder, dim, Some(index_path), config.clone()).expect("Failed to initialize Memory Engine"))
}

fn create_storage(config: &Config, cli_db: Option<&PathBuf>) -> SqliteStorage {
    let db_path = cli_db.cloned().unwrap_or_else(|| config.get_db_path());
    SqliteStorage::new(db_path.to_str().unwrap()).expect("Failed to open storage")
}

fn call_mcp_tool(tool_name: &str, arguments: serde_json::Value) -> Option<String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_millis(1500)) // Fast timeout
        .build()
        .ok()?;

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": "cli",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    });

    let response = client.post("http://localhost:8080/")
        .json(&request)
        .send()
        .ok()?;

    if response.status().is_success() {
        let json: serde_json::Value = response.json().ok()?;
        if let Some(result) = json.get("result") {
            if let Some(content) = result.get("content") {
                if let Some(text) = content[0].get("text") {
                    return Some(text.as_str().unwrap_or("").to_string());
                }
            }
            return if let Some(s) = result.as_str() {
                Some(s.to_string())
            } else {
                Some(result.to_string())
            };
        }
    }
    None
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max - 3])
    } else {
        s.to_string()
    }
}

fn setup_logging(level: &str) {
    use tracing_subscriber::{fmt, EnvFilter, prelude::*};
    
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();
}
