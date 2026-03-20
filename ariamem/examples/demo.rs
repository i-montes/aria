use ariamem::{MemoryEngine, SqliteStorage, WordCountEmbedder, Memory, MemoryType, Embedder};
use std::sync::Arc;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         AriaMem Demo - Hybrid Memory Engine             ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("[1] Creando motor de memoria con WordCountEmbedder...\n");
    let storage = SqliteStorage::in_memory().expect("Failed to create storage");
    let mut embedder = WordCountEmbedder::new(500);
    
    let corpus = vec![
        "python programming rust go language software",
        "react typescript frontend javascript web development",
        "juan api rest database postgres backend",
        "coffee night work preference evening",
        "opinion python fast scripts automation",
        "proyecto frontend backend fullstack development",
        "agente coder especializado inteligencia artificial",
        "usuario preferencias configuracion sistema",
    ];
    embedder.fit(&corpus);
    let dimension = embedder.dimension();
    
    let engine = Arc::new(MemoryEngine::new(storage, embedder, dimension));

    println!("[2] Almacenando memorias de prueba...\n");
    
    let memory1 = Memory::new(
        "Usuario Juan trabaja en el proyecto API REST con Python".to_string(),
        MemoryType::World,
    );
    let stored1 = engine.store(memory1).expect("Failed to store");
    println!("  ✓ Guardado: {}", truncate(&stored1.content, 50));

    let memory2 = Memory::new(
        "El agente coder es especialista en Rust y Go".to_string(),
        MemoryType::Experience,
    );
    let stored2 = engine.store(memory2).expect("Failed to store");
    println!("  ✓ Guardado: {}", truncate(&stored2.content, 50));

    let memory3 = Memory::new(
        "El proyecto frontend usa React y TypeScript".to_string(),
        MemoryType::World,
    );
    let stored3 = engine.store(memory3).expect("Failed to store");
    println!("  ✓ Guardado: {}", truncate(&stored3.content, 50));

    let memory4 = Memory::new(
        "Juan prefiere trabajar de noche y tomar café".to_string(),
        MemoryType::Observation,
    );
    let stored4 = engine.store(memory4).expect("Failed to store");
    println!("  ✓ Guardado: {}", truncate(&stored4.content, 50));

    let memory5 = Memory::new(
        "Opinion: Python es mejor para scripts rápidos".to_string(),
        MemoryType::Opinion,
    )
    .with_confidence(0.75);
    let stored5 = engine.store(memory5).expect("Failed to store");
    println!("  ✓ Guardado: {}", truncate(&stored5.content, 50));

    println!("\n[3] Estadísticas de la base de datos:");
    let count = engine.count().expect("Failed to count");
    println!("  • Total de memorias: {}", count);

    println!("\n[4] Listando memorias por tipo:\n");
    
    for (mem_type, name) in [
        (MemoryType::World, "World"),
        (MemoryType::Experience, "Experience"),
        (MemoryType::Opinion, "Opinion"),
        (MemoryType::Observation, "Observation"),
    ] {
        let memories = engine.list_by_type(mem_type).expect("Failed to list");
        println!("  {} ({}):", name, memories.len());
        for m in &memories {
            println!("    • {}", truncate(&m.content, 60));
        }
        println!();
    }

    println!("[5] Simulando accesos para aumentar relevancia...\n");
    for _ in 0..4 {
        let _ = engine.get(&stored1.id);
    }
    println!("  ✓ Accedido {} veces a la memoria de Juan", engine.get(&stored1.id).unwrap().access_count);

    println!("\n[6] Buscando memorias relacionadas con 'Python'...\n");
    let results = engine.search_by_text("python programming scripts fast", 5).expect("Search failed");
    for (i, result) in results.iter().enumerate() {
        println!("  {}. [score: {:.3}] [relevance: {:.3}]", 
            i + 1, result.score, result.relevance_score);
        println!("     \"{}\"", truncate(&result.memory.content, 70));
    }

    println!("\n[7] Buscando memorias relacionadas con 'Rust programming'...\n");
    let results = engine.search_by_text("rust go language software", 5).expect("Search failed");
    for (i, result) in results.iter().enumerate() {
        println!("  {}. [score: {:.3}]", i + 1, result.score);
        println!("     \"{}\"", truncate(&result.memory.content, 70));
    }

    println!("\n[8] Buscando memorias relacionadas con 'Frontend'...\n");
    let results = engine.search_by_text("frontend react typescript web development", 5).expect("Search failed");
    for (i, result) in results.iter().enumerate() {
        println!("  {}. [score: {:.3}]", i + 1, result.score);
        println!("     \"{}\"", truncate(&result.memory.content, 70));
    }

    println!("[9] Buscando memorias relacionadas con 'Usuario Juan'...\n");
    let results = engine.search_by_text("juan api rest database backend", 5).expect("Search failed");
    for (i, result) in results.iter().enumerate() {
        println!("  {}. [score: {:.3}]", i + 1, result.score);
        println!("     \"{}\"", truncate(&result.memory.content, 70));
    }

    println!("\n[10] Creando relaciones entre memorias...\n");
    
    let _ = engine.store_with_edge(
        stored1.clone(),
        stored2.clone(),
        ariamem::RelationType::WorksOn,
    );
    println!("  ✓ Relacionado: Juan -> trabaja_con -> Coder");

    let _ = engine.store_with_edge(
        stored2.clone(),
        stored3.clone(),
        ariamem::RelationType::Related,
    );
    println!("  ✓ Relacionado: Coder -> relacionado_con -> Frontend");

    println!("\n[11] Obteniendo memorias relacionadas con 'Coder'...\n");
    let related = engine.get_related(&stored2.id).expect("Failed to get related");
    for (edge, memory) in related {
        println!("  → [{}] \"{}\"", 
            format!("{:?}", edge.relation_type),
            truncate(&memory.content, 60));
    }

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║              Demo completada exitosamente!                 ║");
    println!("╚══════════════════════════════════════════════════════════╝");
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max.saturating_sub(3)])
    } else {
        s.to_string()
    }
}
