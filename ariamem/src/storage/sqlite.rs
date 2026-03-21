use rusqlite::{params, Row, ToSql};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use std::collections::HashMap;

use crate::core::{Memory, Edge, MemoryType, RelationType, TemporalMetadata};
use crate::plugins::Storage;

pub struct SqliteStorage {
    pool: Pool<SqliteConnectionManager>,
    _master_conn: Option<std::sync::Mutex<rusqlite::Connection>>, // Keep in-memory DB alive
}

impl SqliteStorage {
    pub fn new(path: &str) -> Result<Self, rusqlite::Error> {
        let manager = SqliteConnectionManager::file(path);
        let pool = r2d2::Pool::builder()
            .max_size(15)
            .build(manager)
            .map_err(|_| rusqlite::Error::InvalidPath(std::path::PathBuf::from(path)))?;
            
        let storage = Self { pool, _master_conn: None };
        storage.init_schema()?;
        Ok(storage)
    }

    pub fn in_memory() -> Result<Self, rusqlite::Error> {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!("ariamem_{}.db", uuid::Uuid::new_v4()));
        let path_str = db_path.to_str().unwrap();
        
        let manager = SqliteConnectionManager::file(path_str);
        let pool = r2d2::Pool::builder()
            .max_size(5)
            .build(manager)
            .unwrap();
        
        let storage = Self { pool, _master_conn: None };
        storage.init_schema()?;
        Ok(storage)
    }

    fn init_schema(&self) -> Result<(), rusqlite::Error> {
        let conn = self.pool.get().map_err(|_| rusqlite::Error::QueryReturnedNoRows)?;
        self.init_schema_on_conn(&conn)
    }

    fn init_schema_on_conn(&self, conn: &rusqlite::Connection) -> Result<(), rusqlite::Error> {
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memories (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL,
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

        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_id ON memories(id)", [])?;

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

        // FTS5 Virtual Table
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                content_id UNINDEXED,
                content='memories',
                tokenize='unicode61'
            )",
            [],
        )?;

        // Triggers
        conn.execute_batch(
            "CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, content_id) VALUES (new.rowid, new.content, new.id);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, content_id) VALUES('delete', old.rowid, old.content, old.id);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, content_id) VALUES('delete', old.rowid, old.content, old.id);
                INSERT INTO memories_fts(rowid, content, content_id) VALUES (new.rowid, new.content, new.id);
            END;"
        )?;

        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)", [])?;
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)", [])?;
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)", [])?;

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
        let id_str: String = row.get(1)?; 
        let memory_type_str: String = row.get(2)?;
        let content: String = row.get(3)?;
        let embedding_json: String = row.get(4)?;
        let occurrence_start_str: String = row.get(5)?;
        let occurrence_end_str: Option<String> = row.get(6)?;
        let mention_time_str: String = row.get(7)?;
        let metadata_json: String = row.get(8)?;
        let confidence: Option<f32> = row.get(9)?;
        let access_count: u32 = row.get(10)?;
        let last_accessed_str: Option<String> = row.get(11)?;

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
        let mention_time = mention_time_str.parse::<chrono::DateTime<chrono::Utc>>().unwrap_or_else(|_| {
             chrono::DateTime::parse_from_rfc3339(&mention_time_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now())
        });
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
        let conn = self.pool.get().map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        
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

        let row_id = conn.last_insert_rowid();
        let _ = conn.execute(
            "INSERT INTO memories_fts(rowid, content, content_id) VALUES (?1, ?2, ?3)",
            params![row_id, memory.content, memory.id.to_string()],
        );

        Ok(())
    }

    fn load_memory(&self, id: &uuid::Uuid) -> Result<Memory, crate::plugins::StorageError> {
        let conn = self.pool.get().map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        
        let mut stmt = conn.prepare(
            "SELECT rowid, id, memory_type, content, embedding, occurrence_start, occurrence_end, mention_time, metadata, confidence, access_count, last_accessed
             FROM memories WHERE id = ?1"
        ).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        let memory = stmt.query_row(params![id.to_string()], |row| {
            Self::row_to_memory(row)
        }).map_err(|_| crate::plugins::StorageError::NotFound(*id))?;

        Ok(memory)
    }

    fn update_memory(&self, memory: &Memory) -> Result<(), crate::plugins::StorageError> {
        let conn = self.pool.get().map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        
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
        let conn = self.pool.get().map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        
        conn.execute("DELETE FROM edges WHERE source_id = ?1 OR target_id = ?1", params![id.to_string()])
            .map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        conn.execute("DELETE FROM memories WHERE id = ?1", params![id.to_string()])
            .map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        Ok(())
    }

    fn list_memories(&self, query: &crate::core::MemoryQuery) -> Result<Vec<Memory>, crate::plugins::StorageError> {
        let conn = self.pool.get().map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        
        let mut sql = "SELECT rowid, id, memory_type, content, embedding, occurrence_start, occurrence_end, mention_time, metadata, confidence, access_count, last_accessed FROM memories".to_string();
        
        let mut params_vec: Vec<Box<dyn ToSql>> = Vec::new();
        if let Some(mt) = query.memory_type {
            sql.push_str(" WHERE memory_type = ?1");
            params_vec.push(Box::new(Self::memory_type_to_string(&mt).to_string()));
        }

        let mut stmt = conn.prepare(&sql)
            .map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        
        let rows = stmt.query_map(rusqlite::params_from_iter(params_vec.iter().map(|p| p.as_ref())), |row| {
            Self::row_to_memory(row)
        }).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        
        let mut result = Vec::new();
        for row in rows {
            if let Ok(m) = row {
                result.push(m);
            }
        }
        Ok(result)
    }

    fn count_memories(&self) -> Result<usize, crate::plugins::StorageError> {
        let conn = self.pool.get().map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0)).unwrap_or(0);
        Ok(count as usize)
    }

    fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<crate::vector::SearchResult>, crate::plugins::StorageError> {
        let conn = self.pool.get().map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        
        let mut stmt = conn.prepare(
            "SELECT content_id, rank FROM memories_fts WHERE memories_fts MATCH ?1 ORDER BY rank ASC LIMIT ?2"
        ).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        let rows = stmt.query_map(params![query, limit], |row| {
            let id_str: String = row.get(0)?;
            let rank: f32 = row.get(1)?;
            Ok(crate::vector::SearchResult {
                id: uuid::Uuid::parse_str(&id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                score: rank,
            })
        }).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            if let Ok(res) = row {
                results.push(res);
            }
        }

        if results.is_empty() {
            let mut stmt = conn.prepare("SELECT id FROM memories WHERE content LIKE ?1 LIMIT ?2")
                .map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
            let like_query = format!("%{}%", query);
            let rows = stmt.query_map(params![like_query, limit], |row| {
                let id_str: String = row.get(0)?;
                Ok(crate::vector::SearchResult {
                    id: uuid::Uuid::parse_str(&id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                    score: 0.01, // Small positive score for LIKE matches
                })
            }).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
            for row in rows {
                if let Ok(res) = row {
                    results.push(res);
                }
            }
        }

        Ok(results)
    }

    fn save_edge(&self, edge: &Edge) -> Result<(), crate::plugins::StorageError> {
        let conn = self.pool.get().map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        let metadata_json = serde_json::to_string(&edge.metadata).unwrap();
        conn.execute(
            "INSERT INTO edges (id, source_id, target_id, relation_type, weight, metadata) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![edge.id.to_string(), edge.source_id.to_string(), edge.target_id.to_string(), Self::relation_type_to_string(&edge.relation_type), edge.weight, metadata_json],
        ).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        Ok(())
    }

    fn load_edge(&self, id: &uuid::Uuid) -> Result<Edge, crate::plugins::StorageError> {
        let conn = self.pool.get().map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        let mut stmt = conn.prepare("SELECT id, source_id, target_id, relation_type, weight, metadata FROM edges WHERE id = ?1").unwrap();
        let edge = stmt.query_row(params![id.to_string()], |row| {
            let id_str: String = row.get(0)?;
            let source_id_str: String = row.get(1)?;
            let target_id_str: String = row.get(2)?;
            let relation_type_str: String = row.get(3)?;
            let weight: f32 = row.get(4)?;
            let metadata_json: String = row.get(5)?;
            let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json).unwrap_or_default();
            Ok(Edge {
                id: uuid::Uuid::parse_str(&id_str).unwrap(),
                source_id: uuid::Uuid::parse_str(&source_id_str).unwrap(),
                target_id: uuid::Uuid::parse_str(&target_id_str).unwrap(),
                relation_type: Self::string_to_relation_type(&relation_type_str),
                weight,
                metadata,
            })
        }).map_err(|_| crate::plugins::StorageError::EdgeNotFound(*id))?;
        Ok(edge)
    }

    fn delete_edge(&self, id: &uuid::Uuid) -> Result<(), crate::plugins::StorageError> {
        let conn = self.pool.get().map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        conn.execute("DELETE FROM edges WHERE id = ?1", params![id.to_string()]).map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        Ok(())
    }

    fn query_edges(&self, source_id: &uuid::Uuid) -> Result<Vec<Edge>, crate::plugins::StorageError> {
        let conn = self.pool.get().map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        let mut stmt = conn.prepare("SELECT id, source_id, target_id, relation_type, weight, metadata FROM edges WHERE source_id = ?1").unwrap();
        let rows = stmt.query_map(params![source_id.to_string()], |row| {
            let id_str: String = row.get(0)?;
            let source_id_str: String = row.get(1)?;
            let target_id_str: String = row.get(2)?;
            let relation_type_str: String = row.get(3)?;
            let weight: f32 = row.get(4)?;
            let metadata_json: String = row.get(5)?;
            let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json).unwrap_or_default();
            Ok(Edge {
                id: uuid::Uuid::parse_str(&id_str).unwrap(),
                source_id: uuid::Uuid::parse_str(&source_id_str).unwrap(),
                target_id: uuid::Uuid::parse_str(&target_id_str).unwrap(),
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
        let conn = self.pool.get().map_err(|e| crate::plugins::StorageError::Database(e.to_string()))?;
        let mut stmt = conn.prepare("SELECT id, source_id, target_id, relation_type, weight, metadata FROM edges WHERE target_id = ?1").unwrap();
        let rows = stmt.query_map(params![target_id.to_string()], |row| {
            let id_str: String = row.get(0)?;
            let source_id_str: String = row.get(1)?;
            let target_id_str: String = row.get(2)?;
            let relation_type_str: String = row.get(3)?;
            let weight: f32 = row.get(4)?;
            let metadata_json: String = row.get(5)?;
            let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json).unwrap_or_default();
            Ok(Edge {
                id: uuid::Uuid::parse_str(&id_str).unwrap(),
                source_id: uuid::Uuid::parse_str(&source_id_str).unwrap(),
                target_id: uuid::Uuid::parse_str(&target_id_str).unwrap(),
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
