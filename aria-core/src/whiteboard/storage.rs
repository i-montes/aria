use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use anyhow::Result;
use chrono::{DateTime, Utc};
use crate::whiteboard::schema::{Task, TaskNote, TaskStatus, TaskNoteType};

#[derive(Clone)]
pub struct WhiteboardStorage {
    pool: Pool<SqliteConnectionManager>,
}

impl WhiteboardStorage {
    pub fn new(pool: Pool<SqliteConnectionManager>) -> Self {
        Self { pool }
    }

    pub fn init(&self) -> Result<()> {
        let conn = self.pool.get()?;
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS shared_whiteboard (
                id          TEXT PRIMARY KEY,
                project_id  TEXT NOT NULL,
                assigned_to TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'PENDING'
                            CHECK(status IN ('PENDING', 'IN_PROGRESS', 'REVIEW', 'COMPLETED', 'LOCKED')),
                description TEXT NOT NULL,
                context_snapshot TEXT,
                result_summary  TEXT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );",
            [],
        )?;

        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_wb_project    ON shared_whiteboard(project_id);
             CREATE INDEX IF NOT EXISTS idx_wb_assigned   ON shared_whiteboard(assigned_to);
             CREATE INDEX IF NOT EXISTS idx_wb_status     ON shared_whiteboard(status);"
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS task_notes (
                id          TEXT PRIMARY KEY,
                task_id     TEXT NOT NULL
                            REFERENCES shared_whiteboard(id) ON DELETE CASCADE,
                author      TEXT NOT NULL,
                note_type   TEXT NOT NULL DEFAULT 'INFO'
                            CHECK(note_type IN ('INFO', 'DECISION', 'PROBLEM', 'BLOCKER', 'RESULT', 'WARNING')),
                content     TEXT NOT NULL,
                metadata    TEXT,
                created_at  TEXT NOT NULL
            );",
            [],
        )?;

        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_notes_task    ON task_notes(task_id);
             CREATE INDEX IF NOT EXISTS idx_notes_author  ON task_notes(author);
             CREATE INDEX IF NOT EXISTS idx_notes_type    ON task_notes(note_type);"
        )?;

        Ok(())
    }

    pub fn create_task(&self, task: &Task) -> Result<()> {
        let conn = self.pool.get()?;
        conn.execute(
            "INSERT INTO shared_whiteboard 
            (id, project_id, assigned_to, status, description, context_snapshot, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            (
                &task.id,
                &task.project_id,
                &task.assigned_to,
                task.status.as_str(),
                &task.description,
                &task.context_snapshot,
                task.created_at.to_rfc3339(),
                task.updated_at.to_rfc3339(),
            ),
        )?;
        Ok(())
    }

    pub fn add_note(&self, note: &TaskNote) -> Result<()> {
        let conn = self.pool.get()?;
        conn.execute(
            "INSERT INTO task_notes 
            (id, task_id, author, note_type, content, metadata, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            (
                &note.id,
                &note.task_id,
                &note.author,
                note.note_type.as_str(),
                &note.content,
                &note.metadata,
                note.created_at.to_rfc3339(),
            ),
        )?;
        Ok(())
    }

    pub fn update_task_status(&self, id: &str, status: TaskStatus) -> Result<()> {
        let conn = self.pool.get()?;
        let now = Utc::now();
        conn.execute(
            "UPDATE shared_whiteboard SET status = ?1, updated_at = ?2 WHERE id = ?3",
            (status.as_str(), now.to_rfc3339(), id),
        )?;
        Ok(())
    }

    pub fn get_tasks_by_project(&self, project_id: &str) -> Result<Vec<Task>> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, assigned_to, status, description, context_snapshot, result_summary, created_at, updated_at 
             FROM shared_whiteboard WHERE project_id = ?1"
        )?;
        
        let task_iter = stmt.query_map([project_id], |row| {
            let status_str: String = row.get(3)?;
            let created_str: String = row.get(7)?;
            let updated_str: String = row.get(8)?;
            
            Ok(Task {
                id: row.get(0)?,
                project_id: row.get(1)?,
                assigned_to: row.get(2)?,
                status: TaskStatus::from(status_str),
                description: row.get(4)?,
                context_snapshot: row.get(5)?,
                result_summary: row.get(6)?,
                created_at: DateTime::parse_from_rfc3339(&created_str).unwrap_or_default().with_timezone(&Utc),
                updated_at: DateTime::parse_from_rfc3339(&updated_str).unwrap_or_default().with_timezone(&Utc),
            })
        })?;

        let mut tasks = Vec::new();
        for task in task_iter {
            tasks.push(task?);
        }
        Ok(tasks)
    }

    pub fn get_notes_by_task(&self, task_id: &str) -> Result<Vec<TaskNote>> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT id, task_id, author, note_type, content, metadata, created_at 
             FROM task_notes WHERE task_id = ?1"
        )?;
        
        let note_iter = stmt.query_map([task_id], |row| {
            let type_str: String = row.get(3)?;
            let created_str: String = row.get(6)?;
            
            Ok(TaskNote {
                id: row.get(0)?,
                task_id: row.get(1)?,
                author: row.get(2)?,
                note_type: TaskNoteType::from(type_str),
                content: row.get(4)?,
                metadata: row.get(5)?,
                created_at: DateTime::parse_from_rfc3339(&created_str).unwrap_or_default().with_timezone(&Utc),
            })
        })?;

        let mut notes = Vec::new();
        for note in note_iter {
            notes.push(note?);
        }
        Ok(notes)
    }
}
