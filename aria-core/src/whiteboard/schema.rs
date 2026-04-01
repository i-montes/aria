use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Review,
    Completed,
    Locked,
}

impl TaskStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskStatus::Pending => "PENDING",
            TaskStatus::InProgress => "IN_PROGRESS",
            TaskStatus::Review => "REVIEW",
            TaskStatus::Completed => "COMPLETED",
            TaskStatus::Locked => "LOCKED",
        }
    }
}

impl From<String> for TaskStatus {
    fn from(s: String) -> Self {
        match s.as_str() {
            "PENDING" => TaskStatus::Pending,
            "IN_PROGRESS" => TaskStatus::InProgress,
            "REVIEW" => TaskStatus::Review,
            "COMPLETED" => TaskStatus::Completed,
            "LOCKED" => TaskStatus::Locked,
            _ => TaskStatus::Pending,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TaskNoteType {
    Info,
    Decision,
    Problem,
    Blocker,
    Result,
    Warning,
}

impl TaskNoteType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskNoteType::Info => "INFO",
            TaskNoteType::Decision => "DECISION",
            TaskNoteType::Problem => "PROBLEM",
            TaskNoteType::Blocker => "BLOCKER",
            TaskNoteType::Result => "RESULT",
            TaskNoteType::Warning => "WARNING",
        }
    }
}

impl From<String> for TaskNoteType {
    fn from(s: String) -> Self {
        match s.as_str() {
            "INFO" => TaskNoteType::Info,
            "DECISION" => TaskNoteType::Decision,
            "PROBLEM" => TaskNoteType::Problem,
            "BLOCKER" => TaskNoteType::Blocker,
            "RESULT" => TaskNoteType::Result,
            "WARNING" => TaskNoteType::Warning,
            _ => TaskNoteType::Info,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub project_id: String,
    pub assigned_to: String,
    pub status: TaskStatus,
    pub description: String,
    pub context_snapshot: Option<String>, // YAML string
    pub result_summary: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskNote {
    pub id: String,
    pub task_id: String,
    pub author: String,
    pub note_type: TaskNoteType,
    pub content: String,
    pub metadata: Option<String>, // JSON string
    pub created_at: DateTime<Utc>,
}

// --- Context Snapshot Structures ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSnapshot {
    pub project: ProjectContext,
    pub task: TaskContext,
    pub agent: AgentContext,
    pub dependencies: Vec<DependencyContext>,
    pub sibling_tasks: Vec<SiblingTaskContext>,
    pub memory_context: Vec<MemoryContext>,
    pub constraints: ConstraintsContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectContext {
    pub id: String,
    pub goal: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    pub id: String,
    pub description: String,
    pub priority: Priority,
    pub deadline_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentContext {
    pub name: String,
    pub role: String,
    pub params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyContext {
    pub task_id: String,
    pub status: TaskStatus,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiblingTaskContext {
    pub task_id: String,
    pub assigned_to: String,
    pub status: TaskStatus,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContext {
    pub id: String,
    pub content: String,
    pub r#type: String,
    pub relevance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintsContext {
    pub max_tool_calls: u32,
    pub allowed_paths: Vec<String>,
    pub timeout_minutes: u32,
}
