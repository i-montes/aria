use rusqlite::{Connection, params, Row};
use std::collections::HashMap;
use std::sync::Mutex;

use crate::core::{Memory, Edge, MemoryType, RelationType, TemporalMetadata};
use crate::plugins::Storage;

pub struct SqliteStorage {
    conn: Mutex<Connection>,
}

impl SqliteStorage {
    pub fn new(path: &str) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(path)?;
        let storage = Self { conn: Mutex::new(conn) };
        storage.init_schema()?;
        Ok(storage)
    }

    pub fn in_memory() -> Result<Self, rusqlite::Error> {
        let conn = Connection::open_in_memory()?;
        let storage = Self { conn: Mutex::new(conn) };
        storage.init_schema()?;
        Ok(storage)
    }

    fn init_schema(&self) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                occurrence_start TEXT NOT NULL,
                occurrence_end TEXT,
                mention_time TEXT NOT NULL,
                metadata TEXT NOT NULL,
                confidence REAL,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                metadata TEXT NOT NULL
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)",
            [],
        )?;

        Ok(())
    }

    fn memory_type_to_string(t: &MemoryType) -> &'static str {
        match t {
            MemoryType::World => "world",
            MemoryType::Experience => "experience",
            MemoryType::Opinion => "opinion",
            MemoryType::Observation => "observation",
        }
    }

    fn string_to_memory_type(s: &str) -> MemoryType {
        match s {
            "experience" => MemoryType::Experience,
            "opinion" => MemoryType::Opinion,
            "observation" => MemoryType::Observation,
            _ => MemoryType::World,
        }
    }

    fn relation_type_to_string(t: &RelationType) -> &'static str {
        match t {
            RelationType::Temporal => "temporal",
            RelationType::Semantic => "semantic",
            RelationType::Entity => "entity",
            RelationType::Causal => "causal",
            RelationType::Related => "related",
            RelationType::WorksOn => "works_on",
        }
    }

    fn string_to_relation_type(s: &str) -> RelationType {
        match s {
            "temporal" => RelationType::Temporal,
            "entity" => RelationType::Entity,
            "causal" => RelationType::Causal,
            "related" => RelationType::Related,
            "works_on" => RelationType::WorksOn,
            _ => RelationType::Semantic,
        }
    }

    fn row_to_memory(row: &Row) -> rusqlite::Result<Memory> {
        let id_str: String = row.get(0)?;
        let memory_type_str: String = row.get(1)?;
        let content: String = row.get(2)?;
        let embedding_json: String = row.get(3)?;
        let occurrence_start_str: String = row.get(4)?;
        let occurrence_end_str: Option<String> = row.get(5)?;
        let mention_time_str: String = row.get(6)?;
        let metadata_json: String = row.get(7)?;
        let confidence: Option<f32> = row.get(8)?;
        let access_count: u32 = row.get(9)?;
        let last_accessed_str: Option<String> = row.get(10)?;

        let embedding: Vec<f32> = serde_json::from_str(&embedding_json).unwrap_or_default();
        let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json).unwrap_or_default();

        let occurrence_start = chrono::DateTime::parse_from_rfc3339(&occurrence_start_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());
        let occurrence_end = occurrence_end_str.and_then(|s| {
            chrono::DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .ok()
        });
        let mention_time = chrono::DateTime::parse_from_rfc3339(&mention_time_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());
        let last_accessed = last_accessed_str.and_then(|s| {
            chrono::DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .ok()
        });

        Ok(Memory {
            id: uuid::Uuid::parse_str(&id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
            memory_type: Self::string_to_memory_type(&memory_type_str),
            content,
            embedding,
            temporal: TemporalMetadata {
                occurrence_start,
                occurrence_end,
                mention_time,
            },
            metadata,
            confidence,
            access_count,
            last_accessed,
        })
    }
}

impl Storage for SqliteStorage {
    fn save_memory(&self, memory: &Memory) -> Result<(), crate::plugins::StorageError> {
        let conn = self.conn.lock().unwrap();
        
        let metadata_json = serde_json::to_string(&memory.metadata)
            .map_err(|e| crate::plugins::StorageError::Serialization(e.to_string()))?;

        let embedding_blob = serde_json::to_string(&memory.embedding)
            .map_err(|e| crate::plugins::StorageError::Serialization(e.to_string()))?;

        conn.execute(
            "INSERT INTO memories (id, memory_type, content, embedding, occurrence_start, occurrence_end, mention_time, metadata, confidence, access_count, last_accessed)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                memory.id.to_string(),
                Self::memory_type_to_string(&memory.memory_type),
                memory.content,
                embedding_blob,
                memory.temporal.occurrence_start.to_rfc3339(),
                memory.temporal.occurrence_end.map(|t| t.to_rfc3339()),
                memory.temporal.mention_time.to_rfc3339(),
                metadata_json,
                memory.confidence,
                memory.access_count,
                memory.last_accessed.map(|t| t.to_rfc3339()),
            ],
        ).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        Ok(())
    }

    fn load_memory(&self, id: &uuid::Uuid) -> Result<Memory, crate::plugins::StorageError> {
        let conn = self.conn.lock().unwrap();
        
        let mut stmt = conn.prepare(
            "SELECT id, memory_type, content, embedding, occurrence_start, occurrence_end, mention_time, metadata, confidence, access_count, last_accessed
             FROM memories WHERE id = ?1"
        ).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        let memory = stmt.query_row(params![id.to_string()], |row| {
            Self::row_to_memory(row)
        }).map_err(|_| crate::plugins::StorageError::NotFound(*id))?;

        Ok(memory)
    }

    fn update_memory(&self, memory: &Memory) -> Result<(), crate::plugins::StorageError> {
        let conn = self.conn.lock().unwrap();
        
        let metadata_json = serde_json::to_string(&memory.metadata)
            .map_err(|e| crate::plugins::StorageError::Serialization(e.to_string()))?;

        let embedding_blob = serde_json::to_string(&memory.embedding)
            .map_err(|e| crate::plugins::StorageError::Serialization(e.to_string()))?;

        conn.execute(
            "UPDATE memories SET memory_type = ?2, content = ?3, embedding = ?4, occurrence_start = ?5, occurrence_end = ?6, mention_time = ?7, metadata = ?8, confidence = ?9, access_count = ?10, last_accessed = ?11
             WHERE id = ?1",
            params![
                memory.id.to_string(),
                Self::memory_type_to_string(&memory.memory_type),
                memory.content,
                embedding_blob,
                memory.temporal.occurrence_start.to_rfc3339(),
                memory.temporal.occurrence_end.map(|t| t.to_rfc3339()),
                memory.temporal.mention_time.to_rfc3339(),
                metadata_json,
                memory.confidence,
                memory.access_count,
                memory.last_accessed.map(|t| t.to_rfc3339()),
            ],
        ).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        Ok(())
    }

    fn delete_memory(&self, id: &uuid::Uuid) -> Result<(), crate::plugins::StorageError> {
        let conn = self.conn.lock().unwrap();
        
        conn.execute("DELETE FROM memories WHERE id = ?1", params![id.to_string()])
            .map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        Ok(())
    }

    fn list_memories(&self, query: &crate::core::MemoryQuery) -> Result<Vec<Memory>, crate::plugins::StorageError> {
        let conn = self.conn.lock().unwrap();
        
        let mut result = Vec::new();
        
        match query.memory_type {
            Some(mem_type) => {
                let sql = "SELECT id, memory_type, content, embedding, occurrence_start, occurrence_end, mention_time, metadata, confidence, access_count, last_accessed FROM memories WHERE memory_type = ?1";
                let mut stmt = conn.prepare(sql)
                    .map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
                
                let rows = stmt.query_map([Self::memory_type_to_string(&mem_type)], |row| {
                    Self::row_to_memory(row)
                }).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
                
                for row in rows {
                    if let Ok(m) = row {
                        result.push(m);
                    }
                }
            }
            None => {
                let sql = "SELECT id, memory_type, content, embedding, occurrence_start, occurrence_end, mention_time, metadata, confidence, access_count, last_accessed FROM memories";
                let mut stmt = conn.prepare(sql)
                    .map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
                
                let rows = stmt.query_map([], |row| {
                    Self::row_to_memory(row)
                }).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
                
                for row in rows {
                    if let Ok(m) = row {
                        result.push(m);
                    }
                }
            }
        }

        Ok(result)
    }

    fn count_memories(&self) -> Result<usize, crate::plugins::StorageError> {
        let conn = self.conn.lock().unwrap();
        
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories",
            [],
            |row| row.get(0),
        ).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        Ok(count as usize)
    }

    fn save_edge(&self, edge: &Edge) -> Result<(), crate::plugins::StorageError> {
        let conn = self.conn.lock().unwrap();
        
        let metadata_json = serde_json::to_string(&edge.metadata)
            .map_err(|e| crate::plugins::StorageError::Serialization(e.to_string()))?;

        conn.execute(
            "INSERT INTO edges (id, source_id, target_id, relation_type, weight, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                edge.id.to_string(),
                edge.source_id.to_string(),
                edge.target_id.to_string(),
                Self::relation_type_to_string(&edge.relation_type),
                edge.weight,
                metadata_json,
            ],
        ).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        Ok(())
    }

    fn load_edge(&self, id: &uuid::Uuid) -> Result<Edge, crate::plugins::StorageError> {
        let conn = self.conn.lock().unwrap();
        
        let mut stmt = conn.prepare(
            "SELECT id, source_id, target_id, relation_type, weight, metadata FROM edges WHERE id = ?1"
        ).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        let edge = stmt.query_row(params![id.to_string()], |row| {
            let id_str: String = row.get(0)?;
            let source_id_str: String = row.get(1)?;
            let target_id_str: String = row.get(2)?;
            let relation_type_str: String = row.get(3)?;
            let weight: f32 = row.get(4)?;
            let metadata_json: String = row.get(5)?;

            let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json).unwrap_or_default();

            Ok(Edge {
                id: uuid::Uuid::parse_str(&id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                source_id: uuid::Uuid::parse_str(&source_id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                target_id: uuid::Uuid::parse_str(&target_id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                relation_type: Self::string_to_relation_type(&relation_type_str),
                weight,
                metadata,
            })
        }).map_err(|_| crate::plugins::StorageError::EdgeNotFound(*id))?;

        Ok(edge)
    }

    fn delete_edge(&self, id: &uuid::Uuid) -> Result<(), crate::plugins::StorageError> {
        let conn = self.conn.lock().unwrap();
        
        conn.execute("DELETE FROM edges WHERE id = ?1", params![id.to_string()])
            .map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        Ok(())
    }

    fn query_edges(&self, source_id: &uuid::Uuid) -> Result<Vec<Edge>, crate::plugins::StorageError> {
        let conn = self.conn.lock().unwrap();
        
        let mut stmt = conn.prepare(
            "SELECT id, source_id, target_id, relation_type, weight, metadata FROM edges WHERE source_id = ?1"
        ).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        let rows = stmt.query_map(params![source_id.to_string()], |row| {
            let id_str: String = row.get(0)?;
            let source_id_str: String = row.get(1)?;
            let target_id_str: String = row.get(2)?;
            let relation_type_str: String = row.get(3)?;
            let weight: f32 = row.get(4)?;
            let metadata_json: String = row.get(5)?;

            let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json).unwrap_or_default();

            Ok(Edge {
                id: uuid::Uuid::parse_str(&id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                source_id: uuid::Uuid::parse_str(&source_id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                target_id: uuid::Uuid::parse_str(&target_id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                relation_type: Self::string_to_relation_type(&relation_type_str),
                weight,
                metadata,
            })
        }).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        let mut result = Vec::new();
        for row in rows {
            if let Ok(e) = row {
                result.push(e);
            }
        }

        Ok(result)
    }

    fn query_edges_by_target(&self, target_id: &uuid::Uuid) -> Result<Vec<Edge>, crate::plugins::StorageError> {
        let conn = self.conn.lock().unwrap();
        
        let mut stmt = conn.prepare(
            "SELECT id, source_id, target_id, relation_type, weight, metadata FROM edges WHERE target_id = ?1"
        ).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        let rows = stmt.query_map(params![target_id.to_string()], |row| {
            let id_str: String = row.get(0)?;
            let source_id_str: String = row.get(1)?;
            let target_id_str: String = row.get(2)?;
            let relation_type_str: String = row.get(3)?;
            let weight: f32 = row.get(4)?;
            let metadata_json: String = row.get(5)?;

            let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json).unwrap_or_default();

            Ok(Edge {
                id: uuid::Uuid::parse_str(&id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                source_id: uuid::Uuid::parse_str(&source_id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                target_id: uuid::Uuid::parse_str(&target_id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                relation_type: Self::string_to_relation_type(&relation_type_str),
                weight,
                metadata,
            })
        }).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        let mut result = Vec::new();
        for row in rows {
            if let Ok(e) = row {
                result.push(e);
            }
        }

        Ok(result)
    }
}
