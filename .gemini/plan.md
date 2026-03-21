# Plan de Mejora: Integridad Híbrida y Autonomía del Agente

## 1. Background & Motivation
ARIA posee inconsistencias en el ciclo de vida de los datos y carece de las herramientas necesarias para que un agente autónomo conecte conceptos de forma independiente.

## 2. Scope & Impact
- ariamem/src/storage/sqlite.rs
- ariamem/src/vector/hnsw.rs
- ariamem/src/core/engine.rs
- ariamem/src/api/mcp.rs

## 3. Proposed Solution & Implementation Steps
Fase 1: Integridad de Grafo (Borrado en Cascada)
Fase 2: Integridad Vectorial (Tombstones)
Fase 3: Autonomia del Agente (API del Engine)
Fase 4: Autonomia del Agente (Completar MCP Tool)
