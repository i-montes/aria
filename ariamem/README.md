# AriaMem - Hybrid Memory Engine for AI Agents

**ARIA**: *Adaptive Roles & Intelligent Assignment*

Motor de memoria híbrida para agentes de IA que combina almacenamiento vectorial y de grafos.

---

## 🚀 Inicio Rápido

### Setup Automático (Linux/macOS)

```bash
cd ..
./setup.sh
```

Esto:
1. Verifica Rust instalado
2. Compila el proyecto
3. Descarga el modelo `potion-base-32M`
4. Ejecuta todos los tests
5. Verifica funcionalidad

### Manual

```bash
# Compilar
cargo build --release

# El binario estará en: target/release/ariamem

# Iniciar
./target/release/ariamem -d memoria.db init

# Guardar una memoria
./target/release/ariamem -d memoria.db store -c "Python es genial"

# Buscar
./target/release/ariamem -d memoria.db search -q "programming"
```

---

## 🧪 Testing

### Ejecutar Todos los Tests

```bash
cargo test
```

**Estado actual: 23 tests pasando**

### Ejecutar Tests Específicos

```bash
# Solo tests de storage
cargo test storage

# Solo tests de integración
cargo test integration

# Tests con output detallado
cargo test -- --nocapture
```

### Ver Coverage (si está instalado)

```bash
cargo install cargo-tarpaulin
cargo tarpaulin --out Html
```

---

## 💻 Uso del CLI

### 1. Inicializar Base de Datos

```bash
# Crear base de datos en archivo por defecto (ariamem.db)
./target/release/ariamem init

# Especificar ruta
./target/release/ariamem --database /path/to/db.db init
```

### 2. Almacenar Memorias

```bash
# Memoria tipo World (default)
./target/release/ariamem store --content "Juan trabaja en proyecto API con Python"

# Memoria tipo Experience
./target/release/ariamem store --content "El coder es especialista en Rust" --memory-type experience

# Memoria tipo Opinion (con confidence)
./target/release/ariamem store --content "Python es mejor para scripts" --memory-type opinion
```

**Tipos de memoria disponibles:**
- `world` - Facts objetivos (default)
- `experience` - Experiencias del agente
- `opinion` - Juicios subjetivos
- `observation` - Observaciones neutrales

### 3. Buscar Memorias

```bash
# Buscar por texto similar
./target/release/ariamem search --query "python programming"

# Limitar resultados
./target/release/ariamem search --query "python" --limit 5
```

### 4. Listar y Ver Memorias

```bash
# Listar todas las memorias
./target/release/ariamem list

# Filtrar por tipo
./target/release/ariamem list --memory-type world

# Ver detalle de una memoria (necesitas el UUID)
./target/release/ariamem get <uuid>
```

### 5. Crear Relaciones

```bash
# Crear link entre dos memorias
./target/release/ariamem link <source_uuid> <target_uuid> --relation works_on
```

**Tipos de relación:**
- `related` - Relación general (default)
- `temporal` - Relación temporal
- `entity` - Relación entre entidades
- `causal` - Relación causal
- `works_on` - Trabaja en

### 6. Ver Memorias Relacionadas

```bash
./target/release/ariamem related <uuid>
```

### 7. Eliminar

```bash
./target/release/ariamem delete <uuid>
```

### 8. Estadísticas

```bash
./target/release/ariamem stats
```

---

## 📁 Estructura del Proyecto

```
ariamem/
├── src/
│   ├── main.rs           # CLI
│   ├── lib.rs            # Exports públicos
│   ├── core/
│   │   ├── types.rs      # Memory, Edge, enums
│   │   └── engine.rs    # MemoryEngine
│   ├── plugins/
│   │   ├── storage.rs     # Trait Storage
│   │   ├── embedder.rs   # Trait Embedder
│   │   ├── dummy_embedder.rs
│   │   ├── tfidf_embedder.rs  # WordCountEmbedder, TfIdfEmbedder
│   │   └── llm_provider.rs
│   ├── storage/
│   │   └── sqlite.rs     # SqliteStorage
│   ├── vector/
│   │   ├── index.rs      # Trait VectorIndex
│   │   └── naive.rs      # NaiveIndex
│   ├── relevance/
│   │   └── engine.rs    # calculate_relevance, calculate_coherence
│   └── retrieval/
│       └── engine.rs    # RetrievalEngine
├── tests/
│   ├── integration_tests.rs
│   └── storage_tests.rs
└── examples/
    └── demo.rs           # Demo visual
```

---

## 🔧 Desarrollo

### Compilar en Modo Debug

```bash
cargo build
```

### Ver Warnings

```bash
cargo check
```

### Fix Automático de Warnings

```bash
cargo fix --lib
```

### Limpiar y Recompilar

```bash
cargo clean
cargo build
```

### Formatear Código

```bash
cargo fmt
```

### Lint

```bash
cargo clippy
```

---

## 📊 Ejemplo Completo

```bash
# 1. Inicializar
$ ./target/release/ariamem init
✓ Initialized! Memories: 0

# 2. Agregar memorias
$ ./target/release/ariamem store --content "Juan trabaja en API con Python"
✓ Stored! ID: abc-123

$ ./target/release/ariamem store --content "Coder especialista en Rust"
✓ Stored! ID: def-456

# 3. Buscar
$ ./target/release/ariamem search --query "python programming"
2 results:
  1. [score: 0.316] Juan trabaja en API con Python
  2. [score: 0.204] Coder especialista en Rust

# 4. Ver estadísticas
$ ./target/release/ariamem stats
Stats - ariamem.db
Total: 2
  World: 2
```

---

## 📝 API Rust

### Crear Motor

```rust
use ariamem::{MemoryEngine, SqliteStorage, Model2VecEmbedder, Memory, MemoryType};

let storage = SqliteStorage::new("db.sqlite")?;
let embedder = Model2VecEmbedder::from_hub("minishlab/potion-base-32M")?;
let engine = MemoryEngine::new(storage, embedder, 512);
```

### Almacenar y Buscar

```rust
// Guardar
let memory = Memory::new("texto".to_string(), MemoryType::World);
let stored = engine.store(memory)?;

// Buscar
let results = engine.search_by_text("query", 10)?;
for r in results {
    println!("{} - score: {:.3}", r.memory.content, r.score);
}
```

---

## 🤖 Embedders Disponibles

### 1. Model2VecEmbedder (Default)
Ligero, rápido, embeddings semánticos de alta calidad. Usa el modelo `potion-base-32M`.

```rust
use ariamem::{MemoryEngine, SqliteStorage, Model2VecEmbedder};

let embedder = Model2VecEmbedder::from_hub("minishlab/potion-base-32M")?;
let engine = MemoryEngine::new(storage, embedder, 512);
```

### 2. WordCountEmbedder
Basado en TF-IDF, no requiere modelo ML.

```rust
use ariamem::{MemoryEngine, SqliteStorage, WordCountEmbedder};

let embedder = WordCountEmbedder::new(500);
let engine = MemoryEngine::new(storage, embedder, 500);
```

### 3. HttpEmbedder (para Ollama)
Usa un servidor Ollama para embeddings semánticos reales.

**Primero, inicia Ollama:**
```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Descargar modelo
ollama pull nomic-embed-text

# Iniciar servidor
ollama serve
```

**Luego en Rust:**
```rust
use ariamem::{MemoryEngine, SqliteStorage, HttpEmbedder, Memory, MemoryType};
use std::sync::Arc;
use tokio::runtime::Runtime;

fn main() {
    let rt = Runtime::new().unwrap();
    let embedder = rt.block_on(async {
        HttpEmbedder::from_ollama("http://localhost:11434", "nomic-embed-text")
            .await
            .expect("Failed to connect to Ollama")
    });
    
    let storage = SqliteStorage::new("db.sqlite").unwrap();
    let engine = Arc::new(MemoryEngine::new(storage, embedder, 768));
    
    // Ahora tienes embeddings semánticos de verdad
    let memory = Memory::new("Python es mejor para scripts".to_string(), MemoryType::Opinion);
    let stored = engine.store(memory).unwrap();
}
```

### Modelos Recomendados para Ollama

| Modelo | Dimensiones | Tamaño | Notas |
|--------|-------------|--------|-------|
| `nomic-embed-text` | 768 | ~274MB | Mejor calidad, open source |
| `mxbai-embed-large` | 1024 | ~670MB | Alta calidad |
| `all-minilm` | 384 | ~38MB | Rápido, menor calidad |

---

## 🏗️ Roadmap

| Fase | Estado | Descripción |
|------|--------|-------------|
| 1 | ✅ | Setup y estructura base |
| 2 | ✅ | SQLite Storage con CRUD |
| 3 | ✅ | Vector Index y búsqueda semántica |
| 4 | ✅ | Retrieval con grafos |
| 5 | ✅ | Plugin System (Embedders) |
| 6 | ✅ | CLI Tool |
| 7 | ✅ | Model2VecEmbedder (potion-base-32M) |
| 8 | ✅ | HttpEmbedder (Ollama) |
| 9 | 📋 | REST API |
| 10 | 📋 | MCP Server |

---

## 📄 Licencia

MIT
