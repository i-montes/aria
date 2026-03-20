use ariamem::{
    Config, Embedder, Memory, MemoryEngine, MemoryType, Model2VecEmbedder, RelationType, SqliteStorage, Edge, Storage, core::MemoryQuery
};
use ariamem::api::mcp::{start_mcp_server, start_mcp_http_server};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "ariamem")]
#[command(
    about = "AriaMem - Hybrid Memory Engine for AI Agents",
    version = "0.1.0"
)]
struct Cli {
    #[arg(short, long)]
    database: Option<PathBuf>,

    #[arg(short, long)]
    model: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Init,
    Serve {
        #[arg(short, long)]
        port: Option<u16>,
    },
    Store {
        #[arg(short, long)]
        content: String,
        #[arg(short = 'm', long, default_value = "world")]
        memory_type: String,
    },
    Get {
        id: String,
    },
    List {
        #[arg(short, long)]
        memory_type: Option<String>,
        #[arg(short, long, default_value = "50")]
        limit: usize,
    },
    Search {
        #[arg(short, long)]
        query: String,
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    Delete {
        id: String,
    },
    Stats,
    Link {
        source: String,
        target: String,
        #[arg(short, long, default_value = "related")]
        relation: String,
    },
    Related {
        id: String,
    },
}

fn create_engine(
    config: &Config,
    cli_db: Option<&PathBuf>,
    cli_model: Option<&String>,
) -> Arc<MemoryEngine<SqliteStorage, Model2VecEmbedder>> {
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

    Arc::new(MemoryEngine::new(storage, embedder, dim))
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

    let response = client.post("http://localhost:8080/mcp")
        .json(&request)
        .send()
        .ok()?;

    if response.status().is_success() {
        let json: serde_json::Value = response.json().ok()?;
        if let Some(content) = json["result"]["content"].as_array() {
            if let Some(text) = content[0]["text"].as_str() {
                return Some(text.to_string());
            }
        }
    }
    None
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    
    // Load configuration, create default if not found
    let config = Config::load().expect("Failed to load or create configuration");

    match &cli.command {
        Commands::Serve { port } => {
            let engine = create_engine(&config, cli.database.as_ref(), cli.model.as_ref());
            if let Some(p) = port {
                if let Err(e) = start_mcp_http_server(engine, *p).await {
                    eprintln!("MCP HTTP Server Error: {}", e);
                }
            } else {
                if let Err(e) = start_mcp_server(engine).await {
                    eprintln!("MCP Stdio Server Error: {}", e);
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
            let mem_type = match memory_type.to_lowercase().as_str() {
                "experience" => MemoryType::Experience,
                "opinion" => MemoryType::Opinion,
                "observation" => MemoryType::Observation,
                _ => MemoryType::World,
            };

            let memory = Memory::new(content.clone(), mem_type);

            match engine.store(memory) {
                Ok(stored) => println!("✓ Stored! ID: {}", stored.id),
                Err(e) => eprintln!("✗ Error: {}", e),
            }
        }

        Commands::Get { id } => {
            let storage = create_storage(&config, cli.database.as_ref());
            let uuid = uuid::Uuid::parse_str(id).expect("Invalid UUID");

            match storage.load_memory(&uuid) {
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
                        "  {}. [{}] {}",
                        i + 1,
                        format!("{:?}", m.memory_type),
                        truncate(&m.content, 55)
                    );
                }
            }
        }

        Commands::Search { query, limit } => {
            // Try calling the background service first for instant response
            if let Some(response) = call_mcp_tool("search_memory", serde_json::json!({
                "query": query,
                "limit": limit
            })) {
                println!("Results (via service):\n{}", response);
                return;
            }

            // Fallback to local loading
            let engine = create_engine(&config, cli.database.as_ref(), cli.model.as_ref());

            match engine.search_by_text(query, *limit) {
                Ok(results) => {
                    println!("{} results:", results.len());
                    for (i, r) in results.iter().enumerate() {
                        println!(
                            "  {}. [score: {:.3}] {}",
                            i + 1,
                            r.score,
                            truncate(&r.memory.content, 55)
                        );
                    }
                }
                Err(e) => eprintln!("✗ Error: {}", e),
            }
        }

        Commands::Delete { id } => {
            let storage = create_storage(&config, cli.database.as_ref());
            let uuid = uuid::Uuid::parse_str(id).expect("Invalid UUID");

            match storage.delete_memory(&uuid) {
                Ok(_) => println!("✓ Deleted!"),
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
            let storage = create_storage(&config, cli.database.as_ref());
            let rel_type = match relation.to_lowercase().as_str() {
                "temporal" => RelationType::Temporal,
                "entity" => RelationType::Entity,
                "causal" => RelationType::Causal,
                "works_on" => RelationType::WorksOn,
                _ => RelationType::Related,
            };

            let source_id = uuid::Uuid::parse_str(source).unwrap_or(uuid::Uuid::new_v4());
            let target_id = uuid::Uuid::parse_str(target).unwrap_or(uuid::Uuid::new_v4());
            let edge = Edge::new(source_id, target_id, rel_type);

            match storage.save_edge(&edge) {
                Ok(_) => println!("✓ Linked! {} → {}", source, target),
                Err(e) => eprintln!("✗ Error: {}", e),
            }
        }

        Commands::Related { id } => {
            let storage = create_storage(&config, cli.database.as_ref());
            let uuid = uuid::Uuid::parse_str(id).expect("Invalid UUID");

            match storage.query_edges(&uuid) {
                Ok(edges) => {
                    println!("{} related edges:", edges.len());
                    for edge in edges {
                        if let Ok(m) = storage.load_memory(&edge.target_id) {
                            println!("  • [{:?}] {}", edge.relation_type, truncate(&m.content, 60));
                        }
                    }
                }
                Err(e) => eprintln!("✗ Error: {}", e),
            }
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max - 3])
    } else {
        s.to_string()
    }
}
