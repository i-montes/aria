use ariamem::core::{Memory, MemoryEngine, MemoryType, RelationType};
use ariamem::storage::sqlite::SqliteStorage;
use ariamem::plugins::tfidf_embedder::TfIdfEmbedder;

fn main() {
    println!("--- Testing Graph YAML Output ---");
    
    // 1. Setup in-memory engine
    let storage = SqliteStorage::in_memory().unwrap();
    let embedder = TfIdfEmbedder::new(512);
    let engine = MemoryEngine::new(storage, embedder, 512);

    // 2. Store two related memories
    let m1 = Memory::new("Aria es una Inteligencia Artificial avanzada.".to_string(), MemoryType::World);
    let m2 = Memory::new("Aria puede ayudarte a escribir código en Rust.".to_string(), MemoryType::World);
    
    println!("Storing memories and linking them...");
    let (m1_saved, _edge, _m2_saved) = engine.store_with_edge(m1, m2, RelationType::Causal).unwrap();

    // 3. Search for "Aria"
    println!("Searching for 'Aria'...");
    let results = engine.search_by_text("Aria", 5).unwrap();
    
    // 4. Print YAML
    println!("--- YAML RESULT ---");
    println!("{}", engine.format_search_results(&results));
    
    // Check if links exist in the output
    let yaml = engine.format_search_results(&results);
    if yaml.contains("links:") && yaml.contains("target:") {
        println!("SUCCESS: Graph links found in YAML!");
    } else {
        println!("FAILURE: Graph links missing in YAML.");
    }
}
