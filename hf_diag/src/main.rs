use hf_hub::api::sync::ApiBuilder;

fn test_model(repo_id: &str) {
    println!("\n--- DIAGNÓSTICO PARA: {} ---", repo_id);

    let api = match ApiBuilder::new()
        .with_progress(true)
        .with_token(None) 
        .build() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("❌ Fallo al construir la API: {:?}", e);
            return;
        }
    };

    let repo = api.model(repo_id.to_string());
    
    let file = "config.json";
    println!("Probando descarga de '{}'...", file);
    
    match repo.get(file) {
        Ok(path) => {
            println!("✅ ¡ÉXITO! Archivo localizado en: {:?}", path);
        },
        Err(e) => {
            eprintln!("❌ ERROR DETALLADO:");
            eprintln!("{:#?}", e);
        }
    }
}

fn main() {
    println!("=== PROBANDO MODELO SUGERIDO (TaylorAI) ===");
    
    // Prueba: El modelo micro sugerido
    test_model("TaylorAI/bge-micro-v2");
    
    println!("\n=== FIN DEL DIAGNÓSTICO ===");
}
