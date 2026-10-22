//! The dock's scheduled tasks (docs/superpowers/specs/2026-10-21-schedules-design.md): one shared list, kept in
//! `~/.aria/schedules.json`, changed through the sessions' jobs sockets and read by the scheduler.

use aria_core::paths::AriaPaths;
use aria_core::schedules::{self, Clock, Grants, RunResult, SystemClock, Task, When, DAYS, MAX_TASKS};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

type OnDelete = Arc<Mutex<Option<Box<dyn Fn(&str) + Send>>>>;

#[derive(Clone)]
pub struct ScheduleStore {
    pub(crate) paths: AriaPaths,
    tasks: Arc<Mutex<Vec<Task>>>,
    on_delete: OnDelete,
    on_update: OnDelete,
    on_awaiting: OnDelete,
}

fn strings(value: &Value) -> Option<Vec<String>> {
    value.as_array().map(|items| items.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
}

fn text(value: &Value, field: &str) -> Result<Option<String>, String> {
    match &value[field] {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s.clone())),
        _ => Err(format!("{field} must be text")),
    }
}

impl ScheduleStore {
    pub fn open(paths: AriaPaths) -> ScheduleStore {
        let tasks = schedules::load(&paths);
        ScheduleStore {
            paths,
            tasks: Arc::new(Mutex::new(tasks)),
            on_delete: Arc::default(),
            on_update: Arc::default(),
            on_awaiting: Arc::default(),
        }
    }

    /// What runs when a task is deleted (the dock closes its tab).
    pub fn on_delete(&self, f: impl Fn(&str) + Send + 'static) {
        *self.on_delete.lock().unwrap() = Some(Box::new(f));
    }

    /// What runs when a task changes (the dock gives its tab's Pi the new grants).
    pub fn on_update(&self, f: impl Fn(&str) + Send + 'static) {
        *self.on_update.lock().unwrap() = Some(Box::new(f));
    }

    /// What runs when a task waits for the person's approval (the page opens `/schedules`).
    pub fn on_awaiting(&self, f: impl Fn(&str) + Send + 'static) {
        *self.on_awaiting.lock().unwrap() = Some(Box::new(f));
    }

    fn tell(hook: &OnDelete, id: &str) {
        if let Some(f) = hook.lock().unwrap().as_ref() {
            f(id);
        }
    }

    fn servers(&self) -> Vec<String> {
        aria_core::mcp::servers(&self.paths)
    }

    /// Checks a task as it would be kept: the core rules, and every writable folder there and narrow enough. Folders are
    /// kept by their real path, so a link cannot be pointed elsewhere later.
    fn check(&self, task: &mut Task) -> Result<(), String> {
        schedules::validate(task, &self.servers())?;
        let mut real = Vec::new();
        for dir in &task.grants.writable {
            let path = std::fs::canonicalize(dir).map_err(|_| format!("{dir} is not a folder"))?;
            if !path.is_dir() {
                return Err(format!("{dir} is not a folder"));
            }
            if let Some(why) = too_wide(&path, &self.paths.root) {
                return Err(format!("{dir}: {why}"));
            }
            real.push(path.display().to_string());
        }
        task.grants.writable = real;
        Ok(())
    }

    fn save(&self, tasks: &[Task]) -> Result<(), String> {
        schedules::save(&self.paths, tasks).map_err(|e| format!("could not save the tasks: {e}"))
    }

    /// Applies the fields `input` names to `task` (any of title, prompt, at, days, date, connectors, writable, enabled).
    fn apply(task: &mut Task, input: &Value) -> Result<(), String> {
        if let Some(title) = text(input, "title")? {
            task.title = title.trim().to_string();
        }
        if let Some(prompt) = text(input, "prompt")? {
            task.prompt = prompt.trim().to_string();
        }
        if let Some(at) = text(input, "at")? {
            task.when.at = at;
        }
        if !input["days"].is_null() {
            task.when.days = strings(&input["days"]).ok_or("days must be a list")?;
        }
        if input.get("date").is_some() {
            task.when.date = text(input, "date")?;
        }
        if !input["connectors"].is_null() {
            let map = input["connectors"].as_object().ok_or("connectors must map a connector to read or allow")?;
            task.grants.connectors =
                map.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string())).collect();
        }
        if !input["writable"].is_null() {
            task.grants.writable = strings(&input["writable"]).ok_or("writable must be a list of folders")?;
        }
        if let Some(enabled) = input["enabled"].as_bool() {
            task.enabled = enabled;
        }
        Ok(())
    }

    pub fn create(&self, input: &Value, now: i64) -> Result<Task, String> {
        let mut task = Task {
            id: schedules::new_id(),
            title: String::new(),
            prompt: String::new(),
            when: When { at: String::new(), days: DAYS.iter().map(|d| d.to_string()).collect(), date: None },
            grants: Grants::default(),
            enabled: true,
            awaiting: false,
            created: now,
            last_run: None,
            last_result: None,
        };
        Self::apply(&mut task, input)?;
        self.check(&mut task)?;
        // A task that can write waits for the person: the model asked for it, the person approves it in /schedules.
        task.awaiting = schedules::writes(&task);
        task.enabled = !task.awaiting;
        let mut tasks = self.tasks.lock().unwrap();
        if tasks.len() >= MAX_TASKS {
            return Err(format!("there are already {MAX_TASKS} scheduled tasks"));
        }
        let mut next = tasks.clone();
        next.push(task.clone());
        self.save(&next)?;
        *tasks = next;
        drop(tasks);
        if task.awaiting {
            Self::tell(&self.on_awaiting, &task.id);
        }
        Ok(task)
    }

    pub fn update(&self, input: &Value) -> Result<Task, String> {
        let id = input["id"].as_str().ok_or("id is missing")?;
        let mut tasks = self.tasks.lock().unwrap();
        let at = tasks.iter().position(|t| t.id == id).ok_or_else(|| format!("there is no task {id}"))?;
        let before = tasks[at].clone();
        let mut task = before.clone();
        Self::apply(&mut task, input)?;
        self.check(&mut task)?;
        if before.awaiting && task.enabled {
            return Err("only the person can approve it, with Approve in /schedules".into());
        }
        // A change that lets it write more, or a new prompt for a task that writes, waits for the person again.
        let allowed = |t: &Task| -> Vec<String> {
            t.grants.connectors.iter().filter(|(_, l)| *l == "allow").map(|(n, _)| n.clone()).collect()
        };
        let widened = allowed(&task).iter().any(|n| !allowed(&before).contains(n))
            || task.grants.writable.iter().any(|w| !before.grants.writable.contains(w));
        let reworded = task.prompt != before.prompt;
        let waits = schedules::writes(&task) && (widened || reworded);
        if waits {
            task.awaiting = true;
            task.enabled = false;
        }
        let mut next = tasks.clone();
        next[at] = task.clone();
        self.save(&next)?;
        *tasks = next;
        drop(tasks);
        Self::tell(&self.on_update, &task.id);
        if waits {
            Self::tell(&self.on_awaiting, &task.id);
        }
        Ok(task)
    }

    /// The person's own switch (`/schedules`): resuming a task that waits approves it.
    pub fn set_enabled_by_person(&self, id: &str, enabled: bool) -> Result<Task, String> {
        let mut tasks = self.tasks.lock().unwrap();
        let at = tasks.iter().position(|t| t.id == id).ok_or_else(|| format!("there is no task {id}"))?;
        let mut next = tasks.clone();
        next[at].enabled = enabled;
        if enabled {
            next[at].awaiting = false;
        }
        let task = next[at].clone();
        self.save(&next)?;
        *tasks = next;
        drop(tasks);
        Self::tell(&self.on_update, &task.id);
        Ok(task)
    }

    pub fn delete(&self, id: &str) -> Result<(), String> {
        {
            let mut tasks = self.tasks.lock().unwrap();
            let next: Vec<Task> = tasks.iter().filter(|t| t.id != id).cloned().collect();
            if next.len() == tasks.len() {
                return Err(format!("there is no task {id}"));
            }
            self.save(&next)?;
            *tasks = next;
        }
        Self::tell(&self.on_delete, id);
        Ok(())
    }

    /// Every task with `next`, the time it will run next (null when paused or done).
    pub fn list(&self, now: i64) -> Vec<Value> {
        self.tasks
            .lock()
            .unwrap()
            .iter()
            .map(|task| {
                let mut value = serde_json::to_value(task).unwrap_or(Value::Null);
                value["next"] = json!(schedules::next_run(task, now, &SystemClock));
                value
            })
            .collect()
    }

    pub fn ids(&self) -> Vec<String> {
        self.tasks.lock().unwrap().iter().map(|t| t.id.clone()).collect()
    }

    pub fn get(&self, id: &str) -> Option<Task> {
        self.tasks.lock().unwrap().iter().find(|t| t.id == id).cloned()
    }

    /// The tasks due now, each with whether it is late. Their `lastRun` is saved before they are returned, so a crash never
    /// runs one twice; a one-off task is paused once taken.
    pub fn take_due(&self, now: i64, clock: &dyn Clock) -> Vec<(Task, bool)> {
        let mut tasks = self.tasks.lock().unwrap();
        let due: Vec<(Task, bool)> = tasks
            .iter()
            .filter(|task| schedules::due(task, now, clock))
            .map(|task| {
                let late = schedules::occurrence(task, now, clock).is_some_and(|at| schedules::late(at, now));
                (task.clone(), late)
            })
            .collect();
        if due.is_empty() {
            return due;
        }
        let mut next = tasks.clone();
        for task in next.iter_mut().filter(|t| due.iter().any(|(d, _)| d.id == t.id)) {
            task.last_run = Some(now);
            if task.when.date.is_some() {
                task.enabled = false;
            }
        }
        if let Err(error) = self.save(&next) {
            crate::note(&format!("scheduled tasks not run, as their run could not be recorded: {error}"));
            return Vec::new();
        }
        *tasks = next;
        due
    }

    pub fn record(&self, id: &str, result: RunResult) {
        let mut tasks = self.tasks.lock().unwrap();
        let mut next = tasks.clone();
        let Some(task) = next.iter_mut().find(|t| t.id == id) else { return };
        task.last_result = Some(result);
        if self.save(&next).is_ok() {
            *tasks = next;
        }
    }
}

/// Why a folder is too much to let a task write in: the root, the home or one of its parents, Aria's own home, a folder of
/// secrets or one that starts programs at login. None when it is narrow enough.
fn too_wide(path: &std::path::Path, aria_root: &std::path::Path) -> Option<&'static str> {
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from).and_then(|h| std::fs::canonicalize(h).ok());
    if path == std::path::Path::new("/") || home.as_ref().is_some_and(|h| h.starts_with(path)) {
        return Some("the whole disk or the home folder");
    }
    if std::fs::canonicalize(aria_root).is_ok_and(|root| path.starts_with(&root) || root.starts_with(path)) {
        return Some("Aria's own folder");
    }
    const GUARDED: [&str; 15] = [
        ".ssh",
        ".gnupg",
        ".aws",
        ".config/gh",
        ".docker",
        ".codex",
        ".kube",
        ".config/gcloud",
        ".config/opencode",
        ".local/share/opencode",
        ".pi",
        "Library/LaunchAgents",
        "Library/Keychains",
        ".config/autostart",
        ".config/systemd",
    ];
    let home = home?;
    GUARDED
        .iter()
        .any(|g| path.starts_with(home.join(g)))
        .then_some("a folder of secrets or of programs that start by themselves")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn store() -> (tempfile::TempDir, ScheduleStore) {
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths::new(dir.path().to_path_buf());
        std::fs::write(paths.mcp(), r#"{"mcpServers":{"notion":{"url":"https://n"},"engram":{"command":"e"}}}"#)
            .unwrap();
        (dir, ScheduleStore::open(paths))
    }

    #[test]
    fn create_fills_the_defaults_and_saves() {
        let (_dir, store) = store();
        let task = store.create(&json!({"title": "Notion tasks", "prompt": "My tasks", "at": "07:30"}), 1_000).unwrap();
        assert!(task.id.starts_with("s-"));
        assert_eq!(task.when.days.len(), 7, "every day unless said");
        assert!(task.enabled && task.created == 1_000);
        assert_eq!(aria_core::schedules::load(&store.paths), vec![task]);
    }

    #[test]
    fn create_refuses_an_unknown_connector_and_ask() {
        let (_dir, store) = store();
        let unknown = json!({"title": "t", "prompt": "p", "at": "07:30", "connectors": {"gmail": "read"}});
        assert!(store.create(&unknown, 0).unwrap_err().contains("gmail"));
        let asking = json!({"title": "t", "prompt": "p", "at": "07:30", "connectors": {"notion": "ask"}});
        assert!(store.create(&asking, 0).is_err());
        let ok = json!({"title": "t", "prompt": "p", "at": "07:30", "connectors": {"notion": "allow"}});
        assert_eq!(store.create(&ok, 0).unwrap().grants.connectors["notion"], "allow");
    }

    #[test]
    fn a_writable_folder_must_exist() {
        let (_dir, store) = store();
        let missing = json!({"title": "t", "prompt": "p", "at": "07:30", "writable": ["/no/such/folder"]});
        assert!(store.create(&missing, 0).is_err());
        let elsewhere = tempfile::tempdir().unwrap();
        let here = json!({"title": "t", "prompt": "p", "at": "07:30", "writable": [elsewhere.path()]});
        assert!(store.create(&here, 0).is_ok());
    }

    #[test]
    fn update_changes_only_what_it_names_and_validates() {
        let (_dir, store) = store();
        let task = store.create(&json!({"title": "t", "prompt": "p", "at": "07:30"}), 0).unwrap();
        let paused = store.update(&json!({"id": task.id, "enabled": false, "at": "08:15"})).unwrap();
        assert!(!paused.enabled);
        assert_eq!((paused.when.at.as_str(), paused.prompt.as_str()), ("08:15", "p"));
        assert!(store.update(&json!({"id": task.id, "at": "8"})).is_err());
        assert_eq!(store.list(0)[0]["when"]["at"], "08:15", "a refused change leaves the task");
        assert!(store.update(&json!({"id": "s-nothere00000"})).is_err());
    }

    #[test]
    fn delete_removes_and_tells() {
        let (_dir, store) = store();
        let deleted = Arc::new(Mutex::new(Vec::new()));
        let seen = deleted.clone();
        store.on_delete(move |id| seen.lock().unwrap().push(id.to_string()));
        let task = store.create(&json!({"title": "t", "prompt": "p", "at": "07:30"}), 0).unwrap();
        store.delete(&task.id).unwrap();
        assert!(store.list(0).is_empty());
        assert_eq!(*deleted.lock().unwrap(), vec![task.id.clone()]);
        assert!(store.delete(&task.id).is_err());
    }

    #[test]
    fn more_than_fifty_is_refused() {
        let (_dir, store) = store();
        for n in 0..50 {
            store.create(&json!({"title": format!("t{n}"), "prompt": "p", "at": "07:30"}), 0).unwrap();
        }
        assert!(store.create(&json!({"title": "one more", "prompt": "p", "at": "07:30"}), 0).is_err());
    }

    #[test]
    fn list_shows_the_next_run() {
        let (_dir, store) = store();
        store.create(&json!({"title": "t", "prompt": "p", "at": "07:30"}), 0).unwrap();
        let listed = store.list(0);
        assert!(listed[0]["next"].as_i64().is_some(), "{listed:?}");
    }

    #[test]
    fn a_read_only_task_starts_at_once() {
        let (_dir, store) = store();
        let task = store
            .create(&json!({"title": "t", "prompt": "p", "at": "07:30", "connectors": {"notion": "read"}}), 0)
            .unwrap();
        assert!(task.enabled && !task.awaiting);
    }

    #[test]
    fn a_writing_task_from_the_model_waits_for_the_person() {
        let (_dir, store) = store();
        let dir = tempfile::tempdir().unwrap(); // a folder outside Aria's home
        let told = Arc::new(Mutex::new(Vec::new()));
        let seen = told.clone();
        store.on_awaiting(move |id| seen.lock().unwrap().push(id.to_string()));
        let allow = store
            .create(&json!({"title": "t", "prompt": "p", "at": "07:30", "connectors": {"notion": "allow"}}), 0)
            .unwrap();
        assert!(!allow.enabled && allow.awaiting);
        let writes =
            store.create(&json!({"title": "t", "prompt": "p", "at": "07:30", "writable": [dir.path().join("out")]}), 0);
        assert!(writes.is_err(), "the folder must exist");
        std::fs::create_dir(dir.path().join("out")).unwrap();
        let writes = store
            .create(&json!({"title": "t", "prompt": "p", "at": "07:30", "writable": [dir.path().join("out")]}), 0)
            .unwrap();
        assert!(writes.awaiting);
        assert_eq!(*told.lock().unwrap(), vec![allow.id.clone(), writes.id.clone()]);
    }

    #[test]
    fn widening_a_task_or_changing_a_writing_one_makes_it_wait_again() {
        let (_dir, store) = store();
        let task = store.create(&json!({"title": "t", "prompt": "p", "at": "07:30"}), 0).unwrap();
        let moved = store.update(&json!({"id": task.id, "at": "08:00"})).unwrap();
        assert!(moved.enabled && !moved.awaiting, "a read-only change applies");
        let widened = store.update(&json!({"id": task.id, "connectors": {"notion": "allow"}})).unwrap();
        assert!(widened.awaiting && !widened.enabled);
        store.set_enabled_by_person(&task.id, true).unwrap();
        let reworded = store.update(&json!({"id": task.id, "prompt": "write everything"})).unwrap();
        assert!(reworded.awaiting, "a writing task's new prompt needs the person again");
    }

    #[test]
    fn only_the_person_approves() {
        let (_dir, store) = store();
        let task = store
            .create(&json!({"title": "t", "prompt": "p", "at": "07:30", "connectors": {"notion": "allow"}}), 0)
            .unwrap();
        assert!(store.update(&json!({"id": task.id, "enabled": true})).unwrap_err().contains("approve"));
        let approved = store.set_enabled_by_person(&task.id, true).unwrap();
        assert!(approved.enabled && !approved.awaiting);
        let paused = store.update(&json!({"id": task.id, "enabled": false})).unwrap();
        assert!(!paused.enabled && !paused.awaiting, "the model may pause");
        assert!(store.update(&json!({"id": task.id, "enabled": true})).is_ok(), "and resume what the person approved");
    }

    #[test]
    fn a_writable_folder_that_is_too_wide_or_secret_is_refused() {
        let (dir, store) = store();
        let home = std::path::PathBuf::from(std::env::var_os("HOME").unwrap());
        let refused = |folder: &std::path::Path| {
            store.create(&json!({"title": "t", "prompt": "p", "at": "07:30", "writable": [folder]}), 0).is_err()
        };
        assert!(refused(std::path::Path::new("/")));
        assert!(refused(&home));
        assert!(refused(home.parent().unwrap()), "an ancestor of the home");
        assert!(refused(dir.path()), "Aria's own home");
        if home.join(".ssh").is_dir() {
            assert!(refused(&home.join(".ssh")));
        }
        let link = dir.path().join("../link-to-root");
        let _ = std::os::unix::fs::symlink("/", &link);
        if link.exists() {
            assert!(refused(&link), "a link is judged by where it points");
            let _ = std::fs::remove_file(&link);
        }
    }

    #[test]
    fn take_due_writes_last_run_first() {
        let (_dir, store) = store();
        let now = SystemClock.epoch(2026, 10, 21, 7, 31);
        let task = store.create(&json!({"title": "t", "prompt": "p", "at": "07:30"}), now - 86_400_000).unwrap();
        let due = store.take_due(now, &SystemClock);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].0.id, task.id);
        assert!(!due[0].1, "on time");
        assert_eq!(aria_core::schedules::load(&store.paths)[0].last_run, Some(now), "written before it runs");
        assert!(store.take_due(now + 1_000, &SystemClock).is_empty(), "never twice");
    }

    #[test]
    fn a_one_off_disables_itself_once_taken() {
        let (_dir, store) = store();
        let now = SystemClock.epoch(2026, 10, 21, 7, 31);
        store.create(&json!({"title": "t", "prompt": "p", "at": "07:30", "date": "2026-10-21"}), 0).unwrap();
        assert_eq!(store.take_due(now, &SystemClock).len(), 1);
        assert!(!store.list(now)[0]["enabled"].as_bool().unwrap());
    }

    #[test]
    fn record_keeps_the_result() {
        let (_dir, store) = store();
        let task = store.create(&json!({"title": "t", "prompt": "p", "at": "07:30"}), 0).unwrap();
        store.record(&task.id, RunResult { at: 5, ok: true, summary: "3 tasks".into() });
        assert_eq!(aria_core::schedules::load(&store.paths)[0].last_result.as_ref().unwrap().summary, "3 tasks");
    }
}
