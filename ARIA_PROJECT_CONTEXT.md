# ARIA - Adaptive Roles & Intelligent Assignment
## Contexto del Proyecto de Memoria Híbrida para Agentes IA

---

## 1. ACRÓNIMO

**A**daptive **R**oles & **I**ntelligent **A**ssignment

Un orquestador de agentes de IA que aprende y mejora con cada interacción.

---

## 2. VISIÓN GENERAL

ARIA es un proyecto que busca construir un motor de memoria híbrida (vectores + grafos) 
diseñado específicamente para agentes de Inteligencia Artificial.

### Problema que resuelve:
- LLMs pierden contexto a largo plazo
- Soluciones existentes (Elasticsearch, Qdrant, Chroma) solo hacen similitud semántica
- Soluciones completas requieren arquitecturas complejas (Neo4j + DB vectorial + Docker)

### Objetivos:
- Reducir tokens sin perder contexto ni "poder"
- Zero-config: bajar y usar
- Modular como piezas de lego
- Sin olvidar nunca (no hay "olvido" automático)

---

## 3. ARQUITECTURA GENERAL

```
┌─────────────────────────────────────────────────────────────┐
│                        ARIA CORE                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Almacenamiento Híbrido                   │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐  │  │
│  │  │ Nodes   │  │ Edges   │  │  Vector Index       │  │  │
│  │  │ (Grafo) │  │ (Links) │  │  (Embeddings)       │  │  │
│  │  └─────────┘  └─────────┘  └─────────────────────┘  │  │
│  └─────────────────────────────────────────────────────┘  │
│                            │                               │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Plugin System (Modular)                │  │
│  │  ┌──────────┐ ┌──────────┐ ┌────────────────────┐  │  │
│  │  │Embedder  │ │Storage   │ │LLM Provider        │  │  │
│  │  │(Model2Vec│ │(SQLite)  │ │(OpenAI/Local/Ollama│  │  │
│  │  │ONNX/HTTP)│ │          │ │                    │  │  │
│  │  └──────────┘ └──────────┘ └────────────────────┘  │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```
┌─────────────────────────────────────────────────────────────┐
│                        ARIA CORE                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Almacenamiento Híbrido                   │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐  │  │
│  │  │ Nodes   │  │ Edges   │  │  Vector Index       │  │  │
│  │  │ (Grafo) │  │ (Links) │  │  (Embeddings)       │  │  │
│  │  └─────────┘  └─────────┘  └─────────────────────┘  │  │
│  └─────────────────────────────────────────────────────┘  │
│                            │                               │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Plugin System (Modular)                  │  │
│  │  ┌──────────┐ ┌──────────┐ ┌────────────────────┐  │  │
│  │  │Embedder  │ │Storage   │ │LLM Provider        │  │  │
│  │  │(ONNX/API)│ │(SQLite)  │ │(OpenAI/Local/Ollama│  │  │
│  │  └──────────┘ └──────────┘ └────────────────────┘  │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. ESTRUCTURA DE CARPETAS DEL PROYECTO

```
aria/
├── ARIA_PROJECT_CONTEXT.md    # Este archivo
├── ariamem/                   # MVP: Motor de memoria
│   ├── src/
│   │   ├── core/              # Node, Edge, Graph, Coherence
│   │   ├── storage/           # SQLite (plugin)
│   │   ├── vector/            # Índice vectorial (plugin)
│   │   ├── relevance/         # Fórmulas de peso
│   │   ├── retrieval/         # Sistema de retrieval
│   │   ├── extraction/        # Fact extraction
│   │   ├── api/               # CLI/REST
│   │   └── plugins/           # Sistema de plugins
│   └── Cargo.toml
├── aria-core/                 # Fase 2: Orquestador
├── aria-agents/               # Fase 3: Agentes especializados
└── aria-skills/              # Fase 4: Skills/MCP
```

---

## 5. ESTRUCTURA DE DATOS

### 5.1 Memory (Nodo)
```rust
struct Memory {
    id: UUID,
    memory_type: MemoryType,  // World, Experience, Opinion, Observation
    content: String,
    embedding: Vec<f32>,
    temporal: TemporalMetadata {
        occurrence_start: Timestamp,
        occurrence_end: Timestamp,
        mention_time: Timestamp,
    },
    metadata: HashMap<String, String>,
    confidence: Option<f32>,  // Solo para Opinion
}
```

### 5.2 Edge (Relación)
```rust
struct Edge {
    id: UUID,
    source_id: UUID,
    target_id: UUID,
    relation_type: RelationType,  // temporal, semantic, entity, causal, related
    weight: f32,
    metadata: HashMap<String, String>,
}
```

### 5.3 Redes de Memoria (4 tipos - definidos para implementación)
```
𝒲 (World Network)       → Facts objetivos sobre el mundo
ℬ (Experience Network) → Experiencias propias del agente  
𝒪 (Opinion Network)    → Juicios subjetivos con confidence [0-1]
𝒮 (Observation Network)→ Summaries neutrales de entities
```

---

## 6. FÓRMULAS MATEMÁTICAS

### 6.1 Distancia Semántica (MemoriesDB)
```
d(Mi, Mj) = ||f_fuse(v_i^(H)) - f_fuse(v_j^(H))||₂
```

### 6.2 Coherencia Parcial (MemoriesDB)
```
C_pair(Mi, Mj) = e^(-d(Mi, Mj))
```
Rango: (0, 1] - Alta coherencia = alta similitud semántica

### 6.3 Coherencia Local (ventana temporal)
```
C_local,t = (1/|Et|) * Σ_{(i,j)∈Et} e^(-d(Mi, Mj))
```
Mide estabilidad semántica en una ventana de tiempo.

### 6.4 Relevancia de Nodo (basada en acceso)
```
relevance = base_weight × log(1 + access_count) × recency_decay

where:
  base_weight = 1.0 (default)
  access_count = número de veces consultado
  recency_decay = 1 / (1 + days_since_last_access)
```

### 6.5 Retrieval Score Combinado
```
S_i = α × sim(v_i^(H), q) + β × e^(-Δt_i/τ) + γ × Φ_i

where:
  sim = cosine similarity
  Δt_i = tiempo desde última actualización
  τ = constante de decay
  Φ_i = edge_density o relation_strength
  α, β, γ = pesos ajustables (default: 0.7, 0.2, 0.1)
```

---

## 7. SISTEMA DE RETRIEVAL (4-way - Basado en Hindsight)

| Estrategia | Implementación |
|-----------|---------------|
| **Semantic** | Vector similarity (cosine) |
| **Keyword** | BM25 o similar |
| **Graph** | Traversal de relaciones ponderadas |
| **Temporal** | Queries con rango de tiempo |

**MVP:** Semantic + Graph (Temporal simplificado)

---

## 8. SISTEMA DE PLUGINS

### 8.1 Trait: Embedder
```rust
trait Embedder {
    fn embed(&self, text: &str) -> Vec<f32>;
}
```
- Default: ONNX Runtime + modelo liviano
- Alternativa: OpenAI API, Ollama, etc.

### 8.2 Trait: Storage
```rust
trait Storage {
    fn save_memory(&self, memory: &Memory) -> Result<()>;
    fn load_memory(&self, id: &UUID) -> Result<Memory>;
    fn save_edge(&self, edge: &Edge) -> Result<()>;
    fn query_edges(&self, source_id: &UUID) -> Result<Vec<Edge>>;
}
```
- Default: SQLite
- Alternativa: LMDB, binario custom

### 8.3 Trait: LLMProvider
```rust
trait LLMProvider {
    fn extract_facts(&self, text: &str) -> Vec<Fact>;
}
```

---

## 9. DOS MODOS DE INSTALACIÓN

| Modo | Comportamiento |
|------|----------------|
| **Zero-Config** | Descarga automática del modelo ONNX en primer uso |
| **Manual-Config** | Usuario especifica path a modelo, API keys, etc. |

---

## 10. ROADMAP DE DESARROLLO

### Paso 0: Documentación ⭐
- [x] Crear este archivo

### Fase 1: Setup ✅ COMPLETADA
- [x] Estructura de carpetas `ariamem/`
- [x] Cargo.toml con dependencias mínimas
- [x] Definir traits básicos (Storage, Embedder, LLMProvider)
- [x] Tests de integración (11 tests)

### Fase 2: Storage Base ✅ COMPLETADA
- [x] Implementar SQLite storage
- [x] CRUD para Memories (save, load, update, delete)
- [x] CRUD para Edges (save, load, delete, query)
- [x] Tests de integración (12 tests)

### Fase 3: Vector Index ✅ COMPLETADA
- [x] NaiveVectorIndex con similitud coseno
- [x] Búsqueda por similitud
- [x] MemoryEngine integrado (storage + vector + retrieval)

### Fase 4: Retrieval ✅ COMPLETADA
- [x] Graph traversal
- [x] Implementar fórmula de relevancia
- [x] Queries combinados (vector + graph)

### Fase 5: Plugin System ✅ COMPLETADA
- [x] Trait Embedder definido
- [x] WordCountEmbedder funcional (TF-IDF)
- [x] Model2VecEmbedder funcional (potion-base-32M)
- [x] HttpEmbedder para Ollama
- [ ] Plugin ONNX nativo (futuro)

### Fase 6: API Layer ✅ COMPLETADA
- [x] CLI tool funcional
- [ ] REST API básica

### Fase 7: Integration Prep 📋 PENDIENTE
- [ ] MCP server preparation
- [ ] Tests de rendimiento
- [ ] Documentación

---

**ESTADO ACTUAL:** 23 tests pasando, Motor de búsqueda semántica funcional con Model2VecEmbedder (potion-base-32M, 512 dim)

---

## 11. DEPENDENCIAS MÍNIMAS (Rust)

```toml
rusqlite = "0.32"        # SQLite
serde = "1.0"            # Serialización
uuid = "1.0"             # IDs únicos
tokio = "1.0"            # Async runtime
tracing = "0.1"          # Logging
```

*Nota: Vector search se implementará de forma custom o con sqlite-vss según necesidad*

---

## 12. PRINCIPIOS DE DISEÑO

1. **Append-only**: No se borra, solo se marca inactivo
2. **Modularidad**: Cada componente es un plugin
3. **Zero-overhead**: No dependencias innecesarias
4. **Testeabilidad**: Tests en cada fase
5. **Documentación**: Código documentado + docs

---

## 13. REFERENCIAS

- MemoriesDB (arXiv:2511.06179) - Modelo de datos temporal-semántico-relacional
- Mem0 (arXiv:2504.19413) - Reducción de tokens + grafo
- Hindsight (arXiv:2512.12818) - 4 redes de memoria + retrieval 4-way
- FalkorDB - Graph DB liviano
- Letta (MemGPT) - Runtime de agentes con memoria

---

## 14. GLOSARIO

| Término | Definición |
|---------|------------|
| Memory | Unidad de información almacenada |
| Edge | Relación entre dos memories |
| Embedding | Vector numérico que representa texto |
| Retrieval | Recuperación de memories relevantes |
| Coherence | Similitud semántica entre memories |
| Relevance | Importancia de un nodo basándose en acceso |

---

## 15. HISTORIAL DE DECISIONES

| Fecha | Decisión | Justificación |
|-------|----------|---------------|
| (hoy) | 4 redes de memoria | Separación epistémica facts/opinions |
| (hoy) | No olvidar | Máquina ≠ humano, no hay razón para borrar |
| (hoy) | SQL como storage default | Madurez + simplicidad |
| (hoy) | ONNX como embedder default | Offline + eficiente |

---

*Documento creado como referencia para el desarrollo de AriaMem*
*Última actualización: Thu Mar 19 2026*
