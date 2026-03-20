# AriaMem: Contexto del Proyecto y Refactorización Integral

Este documento resume la transformación de **AriaMem** de un prototipo de memoria RAG a un motor de memoria híbrida de clase empresarial, modular y autónomo.

## 1. Arquitectura del Sistema
El proyecto se ha reestructurado como un **Rust Workspace** modular:
*   **`aria-cli` (Aria)**: Orquestador global y administrador de servicios. Proporciona el comando `aria` en el PATH.
*   **`ariamem`**: Motor de memoria híbrida (Vectores + Grafos). Funciona como un servicio independiente.

## 2. Optimizaciones de Rendimiento
*   **Búsqueda Vectorial HNSW**: Se reemplazó el escaneo lineal O(N) por un índice **HNSW (Hierarchical Navigable Small World)** en Rust puro. Esto permite búsquedas en milisegundos sobre miles de registros con persistencia local.
*   **Concurrencia Lock-Free**: Se sustituyeron los `RwLock<HashMap>` por `DashMap`, eliminando bloqueos globales en la memoria RAM.
*   **Persistencia SQLite Avanzada**: 
    *   Activación de modo **WAL (Write-Ahead Logging)** para permitir lecturas y escrituras simultáneas.
    *   Implementación de **Connection Pooling (`r2d2`)** con 15 conexiones concurrentes.

## 3. Inteligencia de Memoria (Híbrida)
*   **Spreading Activation**: Algoritmo de expansión de grafos que inyecta relevancia (Boost) a recuerdos conectados tras una búsqueda vectorial.
*   **Fórmula de Relevancia**: `50% Similitud Vectorial + 30% Recencia/Frecuencia + 20% Boost de Grafo`.
*   **Tipos de Memoria**: Soporte nativo para `World`, `Experience`, `Opinion` y `Observation`.

## 4. Zero-Config y Multiplataforma
*   **Detección de Hardware**: El sistema detecta automáticamente GPUs (NVIDIA/Metal) o CPUs y configura el modelo adecuado.
*   **Descarga Inteligente**: Uso de `hf-hub` para descargar modelos automáticamente de Hugging Face.
    *   *Modelo Pro:* `TaylorAI/bge-micro-v2` (GPU).
    *   *Modelo Base:* `minishlab/potion-base-32M` (CPU/Fallback).
*   **Rutas Nativas**: Resolución automática de rutas mediante la crate `directories` (`%APPDATA%` en Windows, `~/.local/share` en Unix).

## 5. AriaMem como Servicio (MCP)
*   **Protocolo MCP**: Implementación del *Model Context Protocol* para una integración transparente con agentes de IA.
*   **Transporte Dual**: Soporta tanto `stdio` como **HTTP (Axum)** en el puerto `8080`.
*   **Background Daemon**: El orquestador puede lanzar AriaMem en segundo plano (`aria start mem`) como un servicio oculto.
*   **Aria Skill**: Inclusión de `ARIA-SKILL.md` que instruye al agente sobre cómo usar la memoria de forma autónoma y ahorrar tokens.

## 6. Comandos Globales Disponibles
*   `aria status`: Estado de los servicios.
*   `aria start/stop mem`: Gestión del ciclo de vida del servicio de fondo.
*   `aria mem stats/search/store`: Interacción directa con el motor de memoria.

## 7. Estado de las Pruebas
La batería de tests en `tests/integration_tests.rs` y `tests/storage_tests.rs` ha sido totalmente refactorizada para validar:
*   Identificadores UUID.
*   Integridad del índice HNSW.
*   Algoritmo de expansión de grafos.

---
*Fecha de actualización: 20 de marzo de 2026*
