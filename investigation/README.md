# Investigación: Motor de Memoria Híbrido para IA

Colección de papers y recursos relevantes para el proyecto Aria.

## Papers Principales

### 1. MemoriesDB (`memoriesdb.pdf`)
**"MemoriesDB: A Temporal-Semantic-Relational Database for Long-Term Agent Memory"**
- **Autores:** Joel "val" Ward (CircleClick Labs)
- **Fecha:** Octubre 2025
- ** arXiv:** 2511.06179
- **Relevancia:** MUY ALTA - Propone una arquitectura que combina:
  - Time-series datastore
  - Vector database  
  - Graph system
  - Todo en un schema append-only

### 2. Mem0 Research (`mem0_paper.pdf`)
**"Mem0: Improving Memory via Intelligent Selection for LLM Applications"**
- **Autores:** Mem0 Team
- **Fecha:** Abril 2025
- ** arXiv:** 2504.19413
- **Relevancia:** ALTA - Paper de investigación de Mem0 (47.8K GitHub stars)

### 3. Graph Memory Survey (`graph_memory_survey.pdf`)
**"Graph-based Agent Memory: Taxonomy, Techniques, and Applications"**
- **Autores:** Chang Yang et al. (Hong Kong PolyU + others)
- **Fecha:** 2026
- ** arXiv:** 2602.05665
- **Relevancia:** MUY ALTA - Survey comprehensivo de 14 páginas que cubre:
  - Taxonomía de memoria de agentes
  - Técnicas de extracción, almacenamiento, retrieval
  - Librerías open-source y benchmarks

### 4. Hindsight (`hindsight_paper.pdf`)
**"Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects"**
- **Autores:** Chris Latimer et al.
- **Fecha:** Diciembre 2025
- ** arXiv:** 2512.12818
- **Relevancia:** MEDIA - Enfoque en retención y reflexión de memoria

## Recursos Externos

### Mem0
- GitHub: https://github.com/mem0ai/mem0
- Docs: https://docs.mem0.ai/
- $24M Series A (Oct 2025)

### Letta (MemGPT)
- GitHub: https://github.com/letta-ai/letta
- 21.6K stars
- Paradigma: "LLM as Operating System"

### FalkorDB
- GitHub: https://github.com/FalkorDB/falkorDB
- 140ms p99 latency vs Neo4j's 46,900ms

## Hallazgos Clave del Research

1. **No existe solución Rust native** - Todas las soluciones requieren dependencias externas
2. **El modelo de MemoriesDB es el más cercano a nuestra visión**
3. **Zero-config es el diferenciador principal**
4. **El mercado está validado** - Mem0 levantó $24M
