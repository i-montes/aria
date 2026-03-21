use ariamem::{Memory, MemoryEngine, MemoryType, RelationType, SqliteStorage};
use ariamem::plugins::Embedder;
use uuid::Uuid;

// A mock embedder that knows specific semantic relations for testing
struct MockSemanticEmbedder;

impl Embedder for MockSemanticEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, ariamem::plugins::EmbedderError> {
        let mut vec = vec![0.0; 384];
        let text = text.to_lowercase();
        
        // Use a base orthogonal signal to avoid zero vectors and division by zero
        vec[383] = 1.0;

        // Scenario 2: Semantic mapping without word overlap
        if text.contains("azul") || text.contains("tonalidad") {
            vec[0] = 1.0;
            vec[383] = 0.0; // Keep it pure for the scenario
        }
        
        // Scenario 3: Initial node for graph discovery
        if text.contains("proyecto") || text.contains("status") {
            vec[100] = 1.0;
            vec[383] = 0.0;
        }

        Ok(vec)
    }

    fn dimension(&self) -> usize { 384 }
    fn name(&self) -> &str { "MockSemantic" }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ariamem::plugins::EmbedderError> {
        let mut results = Vec::new();
        for text in texts {
            results.push(self.embed(text)?);
        }
        Ok(results)
    }
}

#[test]
fn test_hybrid_coverage_complementarity() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join(format!("test_hybrid_{}.db", Uuid::new_v4()));
    let storage = SqliteStorage::new(db_path.to_str().unwrap()).unwrap();
    let embedder = MockSemanticEmbedder;
    let engine = MemoryEngine::new(storage, embedder, 384);

    // SETUP DATA
    
    // 1. For Vector Blind Spot (Technical acronym)
    let mut mem_hnsw = Memory::new(
        "El motor utiliza el algoritmo HNSW para indexación vectorial.".to_string(), 
        MemoryType::World
    );
    mem_hnsw.metadata.insert("base_weight".to_string(), "10.0".to_string());
    
    // 2. For Lexical Blind Spot (Semantic match)
    let mut mem_blue = Memory::new(
        "Su color favorito es el azul.".to_string(), 
        MemoryType::Experience
    );
    mem_blue.metadata.insert("base_weight".to_string(), "10.0".to_string());
    
    // 3. For Graph Magic (Chained discovery)
    let mem_project = Memory::new(
        "El Proyecto ARIA falló en la fase de despliegue.".to_string(), 
        MemoryType::Observation
    );
    let mem_error = Memory::new(
        "Se detectó un error crítico de desbordamiento de memoria.".to_string(), 
        MemoryType::Observation
    );

    engine.store(mem_hnsw.clone()).unwrap();
    engine.store(mem_blue.clone()).unwrap();
    let (_, _, _) = engine.store_with_edge(mem_project.clone(), mem_error.clone(), RelationType::Causal).unwrap();

    // --- EXECUTION & VALIDATION ---

    // Escenario 1: Victoria de FTS5 (Acrónimo técnico)
    let results = engine.search_by_text("HNSW", 5).unwrap();
    println!("Resultados HNSW: {:?}", results.iter().map(|r| &r.memory.content).collect::<Vec<_>>());
    assert!(!results.is_empty(), "FTS5 debería encontrar 'HNSW' por coincidencia exacta");
    assert!(results.iter().any(|r| r.memory.id == mem_hnsw.id), "Debe encontrar la memoria de HNSW");
    assert_eq!(results[0].memory.id, mem_hnsw.id, "El acrónimo exacto debería ser el primer resultado");
    println!("✅ Escenario 1 (FTS5 Win) Pasado: Encontrado por acrónimo exacto.");

    // Escenario 2: Victoria de HNSW (Sin traslape de palabras)
    let results = engine.search_by_text("preferencias de tonalidad", 5).unwrap();
    println!("Resultados Semánticos: {:?}", results.iter().map(|r| &r.memory.content).collect::<Vec<_>>());
    assert!(!results.is_empty(), "HNSW debería encontrar el recuerdo de 'azul' por el mock semántico");
    assert!(results.iter().any(|r| r.memory.id == mem_blue.id), "Debe encontrar la memoria de 'azul'");
    assert_eq!(results[0].memory.id, mem_blue.id, "El match semántico debería ser el primer resultado");
    println!("✅ Escenario 2 (HNSW Win) Pasado: Encontrado por significado semántico.");

    // Escenario 3: La magia del Grafo (Spreading Activation)
    let results = engine.search_by_text("status del proyecto", 5).unwrap();
    
    let found_project = results.iter().any(|r| r.memory.id == mem_project.id);
    let found_error = results.iter().any(|r| r.memory.id == mem_error.id);
    
    assert!(found_project, "Debe encontrar el nodo raíz del proyecto");
    assert!(found_error, "El Spreading Activation debe recuperar el error conectado causalmente");
    
    // Verificar que el error (descubierto por grafo) tenga score vectorial insignificante
    let error_res = results.iter().find(|r| r.memory.id == mem_error.id).unwrap();
    assert!(error_res.score < f32::EPSILON, "El nodo descubierto por grafo no tuvo coincidencia vectorial significativa");
    assert!(error_res.relevance_score > 0.0, "La relevancia final debe ser positiva por el boost del grafo");
    
    println!("✅ Escenario 3 (Graph Magic) Pasado: Nodo hijo recuperado por conexión causal.");
}
