use ariamem::{
    Config, Embedder, Memory, MemoryEngine, MemoryType, Model2VecEmbedder, RelationType, SqliteStorage,
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
    let db_path = cli_db.cloned().unwrap_or_else(|| config.get_db_path());
    let storage = SqliteStorage::new(db_path.to_str().unwrap()).expect("Failed to open storage");

    let is_serve = std::env::args().any(|arg| arg == "serve");
    
    // 1. Determine model identifier (path or name)
    let model_path = config.get_model_path();
    let model_name = &config.embedder.model2vec.model_name;
    
    let model_to_load = if let Some(cli_m) = cli_model {
        cli_m.clone()
    } else if model_path.exists() && model_path.join("model.safetensors").exists() {
        model_path.to_string_lossy().to_string()
    } else {
        // Prepend author if it's one of our defaults
        if model_name == "bge-micro-v2" {
            format!("TaylorAI/{}", model_name)
        } else if model_name.contains("potion") || model_name.contains("bge") {
            format!("minishlab/{}", model_name)
        } else {
            model_name.clone()
        }
    };
    if !is_serve {
        eprintln!("Loading Model2Vec model: {}...", model_to_load);
    }
    
    let embedder = match Model2VecEmbedder::from_hub(&model_to_load) {
        Ok(e) => e,
        Err(e) => {
            // If the primary model fails (like the 401 on bge-micro), 
            // fallback to the ultra-reliable and lightweight potion model.
            if model_to_load.contains("bge-micro") {
                if !is_serve {
                    eprintln!("⚠ Primary model '{}' failed: {}. Falling back to 'minishlab/potion-base-32M'...", model_to_load, e);
                }
                Model2VecEmbedder::from_hub("minishlab/potion-base-32M")
                    .expect("Critical error: Even fallback model failed to load. Check your internet connection.")
            } else {
                panic!("Failed to load Model2Vec model '{}': {}. Make sure you have an internet connection.", model_to_load, e);
            }
        }
    };

    let dim = embedder.dimension();
    
    if !is_serve {
        eprintln!("✓ Model loaded! Dimension: {}", dim);
    }

    Arc::new(MemoryEngine::new(storage, embedder, dim))
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    
    // Load configuration, create default if not found
    let config = Config::load().expect("Failed to load or create configuration");

    match &cli.command {
        Commands::Serve { port } => {
            // Suppress standard Config::load stdout printing by using eprintln or just accepting we might need to redirect
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
            let engine = create_engine(&config, cli.database.as_ref(), cli.model.as_ref());
            println!("✓ Initialized! Memories: {}", engine.count().unwrap_or(0));
        }

        Commands::Store {
            content,
            memory_type,
        } => {
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
            let engine = create_engine(&config, cli.database.as_ref(), cli.model.as_ref());
            let uuid = uuid::Uuid::parse_str(id).expect("Invalid UUID");

            match engine.get(&uuid) {
                Ok(m) => {
                    println!("ID: {}", m.id);
                    println!("Type: {:?}", m.memory_type);
                    println!("Content: {}", m.content);
                    println!("Accesses: {}", m.access_count);
                }
                Err(e) => eprintln!("✗ Error: {}", e),
            }
        }

        Commands::List { memory_type, limit } => {
            let engine = create_engine(&config, cli.database.as_ref(), cli.model.as_ref());

            let results = match memory_type {
                Some(t) => {
                    let mt = match t.to_lowercase().as_str() {
                        "experience" => MemoryType::Experience,
                        "opinion" => MemoryType::Opinion,
                        "observation" => MemoryType::Observation,
                        _ => MemoryType::World,
                    };
                    engine.list_by_type(mt)
                }
                None => engine.all_memories(),
            };

            if let Ok(memories) = results {
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
            let engine = create_engine(&config, cli.database.as_ref(), cli.model.as_ref());
            let uuid = uuid::Uuid::parse_str(id).expect("Invalid UUID");

            match engine.delete(&uuid) {
                Ok(_) => println!("✓ Deleted!"),
                Err(e) => eprintln!("✗ Error: {}", e),
            }
        }

        Commands::Stats => {
            let engine = create_engine(&config, cli.database.as_ref(), cli.model.as_ref());
            let db_path = cli.database.clone().unwrap_or_else(|| config.get_db_path());
            println!("Stats - {:?}", db_path);
            println!("Total: {}", engine.count().unwrap_or(0));

            for (name, mt) in [
                ("World", MemoryType::World),
                ("Experience", MemoryType::Experience),
                ("Opinion", MemoryType::Opinion),
                ("Observation", MemoryType::Observation),
            ] {
                if let Ok(memories) = engine.list_by_type(mt) {
                    println!("  {}: {}", name, memories.len());
                }
            }
        }

        Commands::Link {
            source,
            target,
            relation,
        } => {
            let engine = create_engine(&config, cli.database.as_ref(), cli.model.as_ref());
            let rel_type = match relation.to_lowercase().as_str() {
                "temporal" => RelationType::Temporal,
                "entity" => RelationType::Entity,
                "causal" => RelationType::Causal,
                "works_on" => RelationType::WorksOn,
                _ => RelationType::Related,
            };

            match engine.store_with_edge(
                Memory::new(source.clone(), MemoryType::World),
                Memory::new(target.clone(), MemoryType::World),
                rel_type,
            ) {
                Ok((_, _edge, _)) => println!("✓ Linked! {} → {}", source, target),
                Err(e) => eprintln!("✗ Error: {}", e),
            }
        }

        Commands::Related { id } => {
            let engine = create_engine(&config, cli.database.as_ref(), cli.model.as_ref());
            let uuid = uuid::Uuid::parse_str(id).expect("Invalid UUID");

            match engine.get_related(&uuid) {
                Ok(related) => {
                    println!("{} related:", related.len());
                    for (_, m) in related {
                        println!("  • {}", truncate(&m.content, 60));
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
