use ariamem::{
    Embedder, Memory, MemoryEngine, MemoryType, Model2VecEmbedder, RelationType, SqliteStorage,
};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;

const DEFAULT_MODEL: &str = "/root/.local/share/ariamem/models/potion-base-32M";

#[derive(Parser)]
#[command(name = "ariamem")]
#[command(
    about = "AriaMem - Hybrid Memory Engine for AI Agents",
    version = "0.1.0"
)]
struct Cli {
    #[arg(short, long, default_value = "ariamem.db")]
    database: PathBuf,

    #[arg(short, long, default_value = DEFAULT_MODEL)]
    model: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Init,
    Store {
        #[arg(short, long)]
        content: String,
        #[arg(short, long, default_value = "world")]
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
    db_path: &PathBuf,
    model_path: &str,
) -> Arc<MemoryEngine<SqliteStorage, Model2VecEmbedder>> {
    let storage = SqliteStorage::new(db_path.to_str().unwrap()).expect("Failed to open storage");

    println!("Loading Model2Vec model: {}...", model_path);
    let embedder = Model2VecEmbedder::from_hub(model_path)
        .expect("Failed to load Model2Vec model. Make sure you have an internet connection.");

    let dim = embedder.dimension();
    println!("✓ Model loaded! Dimension: {}", dim);

    Arc::new(MemoryEngine::new(storage, embedder, dim))
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Init => {
            println!("Initializing AriaMem at: {:?}", cli.database);
            let engine = create_engine(&cli.database, &cli.model);
            println!("✓ Initialized! Memories: {}", engine.count().unwrap_or(0));
        }

        Commands::Store {
            content,
            memory_type,
        } => {
            let engine = create_engine(&cli.database, &cli.model);
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
            let engine = create_engine(&cli.database, &cli.model);
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
            let engine = create_engine(&cli.database, &cli.model);

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
            let engine = create_engine(&cli.database, &cli.model);

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
            let engine = create_engine(&cli.database, &cli.model);
            let uuid = uuid::Uuid::parse_str(id).expect("Invalid UUID");

            match engine.delete(&uuid) {
                Ok(_) => println!("✓ Deleted!"),
                Err(e) => eprintln!("✗ Error: {}", e),
            }
        }

        Commands::Stats => {
            let engine = create_engine(&cli.database, &cli.model);
            println!("Stats - {:?}", cli.database);
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
            let engine = create_engine(&cli.database, &cli.model);
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
            let engine = create_engine(&cli.database, &cli.model);
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
