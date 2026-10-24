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

fn home() -> std::path::PathBuf {
    std::env::var_os("HOME").map(std::path::PathBuf::from).unwrap_or_default()
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
    ///
    /// Every folder must follow `schedules::writable_problem` (inside the projects folder or `/tmp`); the ones in `kept`
    /// (the folders the task already had) are not judged by it again, so a task stored before the rule can still be
    /// paused or reworded: a run drops them (`runnable_writable`).
    fn check(&self, task: &mut Task, kept: &[String]) -> Result<(), String> {
        schedules::validate(task, &self.servers())?;
        let (projects, home) = (self.projects(), home());
        let mut real = Vec::new();
        for dir in &task.grants.writable {
            if !kept.contains(dir) {
                if let Some(why) = schedules::writable_problem(dir, &projects, &home) {
                    return Err(why);
                }
            }
            let path = std::fs::canonicalize(dir).map_err(|_| format!("{dir} is not a folder"))?;
            if !path.is_dir() {
                return Err(format!("{dir} is not a folder"));
            }
            if let Some(why) = too_wide(&path, &self.paths.root) {
                return Err(format!("{dir}: {why}"));
            }
            // Where a link points is judged too: a link in the projects folder cannot reach a dot folder. A kept folder
            // is judged by its real path when that is not the path it was stored under (it was swapped for a link):
            // the store never takes a grant that now leads out of bounds. One stored before the rule, still where it was,
            // stays (a run drops it).
            let real_path = path.display().to_string();
            if !kept.contains(dir) || real_path != *dir {
                if let Some(why) = schedules::writable_problem(&real_path, &projects, &home) {
                    return Err(why);
                }
            }
            real.push(real_path);
        }
        task.grants.writable = real;
        Ok(())
    }

    /// The settings' projects folder (`sandbox.projectsDir`, `~/Projects` without settings).
    fn projects(&self) -> std::path::PathBuf {
        aria_core::agent::workdir(aria_core::settings::load(&self.paths).ok().flatten().as_ref())
    }

    /// The writable folders a run may have: the task's own that follow the rule now, judged where they really are at
    /// this moment (a folder the session replaced by a link after the approval is judged by the link's target). A grant
    /// that breaks the rule, or cannot be resolved, is dropped with one line in the log, never widened; one whose real
    /// path follows the rule is given as that real path, which is what the sandbox binds. A task stored before the rule
    /// (or under another projects folder) keeps its grants on disk.
    pub fn runnable_writable(&self, task: &Task) -> Vec<String> {
        let (projects, home) = (self.projects(), home());
        let drop = |dir: &str, why: &str| {
            crate::note(&format!(
                "scheduled task {}: its writable folder {dir} {why}, so this run does not get it",
                task.id
            ));
        };
        task.grants
            .writable
            .iter()
            .filter_map(|dir| {
                if schedules::writable_problem(dir, &projects, &home).is_some() {
                    drop(dir, "is outside the projects folder and /tmp");
                    return None;
                }
                let Ok(real) = std::fs::canonicalize(dir) else {
                    drop(dir, "cannot be resolved");
                    return None;
                };
                let real = real.display().to_string();
                if real != *dir && schedules::writable_problem(&real, &projects, &home).is_some() {
                    drop(dir, &format!("now leads to {real}, outside the projects folder and /tmp"));
                    return None;
                }
                Some(real)
            })
            .collect()
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
        self.check(&mut task, &[])?;
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
        self.check(&mut task, &before.grants.writable)?;
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

    /// A plain pause or resume (`/schedules` in the dock or a terminal). A task that waits for approval stays
    /// waiting and disabled when asked to resume: only `approve` clears the wait.
    pub fn set_enabled(&self, id: &str, enabled: bool) -> Result<Task, String> {
        self.change(id, |task| {
            if enabled && task.awaiting {
                return Err(format!("task {id} waits for the person's approval in the dock"));
            }
            task.enabled = enabled;
            Ok(())
        })
    }

    /// The dock page's Approve button: the only way the wait ends (it enables the task). Called for nothing a client
    /// can send.
    pub fn approve(&self, id: &str) -> Result<Task, String> {
        self.change(id, |task| {
            task.enabled = true;
            task.awaiting = false;
            Ok(())
        })
    }

    fn change(&self, id: &str, edit: impl FnOnce(&mut Task) -> Result<(), String>) -> Result<Task, String> {
        let mut tasks = self.tasks.lock().unwrap();
        let at = tasks.iter().position(|t| t.id == id).ok_or_else(|| format!("there is no task {id}"))?;
        let mut next = tasks.clone();
        edit(&mut next[at])?;
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

    /// The tasks due now, each with the time it was due. Their `lastRun` is saved before they are returned, so a crash
    /// never runs one twice; a one-off task is paused once taken.
    pub fn take_due(&self, now: i64, clock: &dyn Clock) -> Vec<(Task, i64)> {
        let mut tasks = self.tasks.lock().unwrap();
        let due: Vec<(Task, i64)> = tasks
            .iter()
            .filter(|task| schedules::due(task, now, clock) && !task.awaiting)
            .filter_map(|task| schedules::occurrence(task, now, clock).map(|at| (task.clone(), at)))
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
    // Each platform's keyring and login items: macOS's Keychains and LaunchAgents; Linux's Secret Service (gnome-keyring),
    // XDG autostart, systemd user units and Hyprland's exec-once.
    const GUARDED: [&str; 17] = [
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
        ".local/share/keyrings",
        ".config/autostart",
        ".config/systemd",
        ".config/hypr",
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
        store.approve(&task.id).unwrap();
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
        let approved = store.approve(&task.id).unwrap();
        assert!(approved.enabled && !approved.awaiting);
        let paused = store.update(&json!({"id": task.id, "enabled": false})).unwrap();
        assert!(!paused.enabled && !paused.awaiting, "the model may pause");
        assert!(store.update(&json!({"id": task.id, "enabled": true})).is_ok(), "and resume what the person approved");
    }

    #[test]
    fn a_plain_switch_never_approves_a_waiting_task() {
        let (_dir, store) = store();
        let waiting = store
            .create(&json!({"title": "t", "prompt": "p", "at": "07:30", "connectors": {"notion": "allow"}}), 0)
            .unwrap();
        assert!(waiting.awaiting && !waiting.enabled);
        let refused = store.set_enabled(&waiting.id, true).unwrap_err();
        assert!(refused.contains("approv"), "{refused}");
        let still = store.get(&waiting.id).unwrap();
        assert!(still.awaiting && !still.enabled, "it stays waiting and disabled");
        assert!(store.set_enabled(&waiting.id, false).is_ok(), "pausing a waiting task is harmless");
        assert!(store.get(&waiting.id).unwrap().awaiting, "and does not approve it");
        let approved = store.approve(&waiting.id).unwrap();
        assert!(approved.enabled && !approved.awaiting, "only approve clears the wait");
    }

    #[test]
    fn a_plain_switch_pauses_and_resumes_a_task_that_is_not_waiting() {
        let (_dir, store) = store();
        let task = store.create(&json!({"title": "t", "prompt": "p", "at": "07:30"}), 0).unwrap();
        assert!(!store.set_enabled(&task.id, false).unwrap().enabled);
        assert!(store.set_enabled(&task.id, true).unwrap().enabled);
        assert!(store.set_enabled("s-none", true).is_err());
        assert!(store.approve("s-none").is_err());
    }

    /// The store of `store()` with `projects` as the settings' projects folder.
    fn with_projects(store: &ScheduleStore, projects: &std::path::Path) {
        let mut settings = aria_core::settings::Settings::defaults("en");
        settings.sandbox.projects_dir = projects.display().to_string();
        aria_core::settings::save(&store.paths, &settings).unwrap();
    }

    /// A folder that is not under `/tmp`, so "inside the projects folder" is not "inside /tmp".
    fn outside_tmp() -> tempfile::TempDir {
        tempfile::tempdir_in("/var/tmp").unwrap()
    }

    fn create_writing(store: &ScheduleStore, folder: &std::path::Path) -> Result<Task, String> {
        store.create(&json!({"title": "t", "prompt": "p", "at": "07:30", "writable": [folder]}), 0)
    }

    #[test]
    fn a_task_may_write_only_inside_the_projects_folder_or_tmp() {
        let (_dir, store) = store();
        let work = outside_tmp();
        let projects = work.path().join("work");
        std::fs::create_dir_all(projects.join("acme/out")).unwrap();
        with_projects(&store, &projects);
        let task = create_writing(&store, &projects.join("acme/out")).unwrap();
        assert!(task.awaiting, "{task:?}");
        let tmp = tempfile::Builder::new().tempdir_in("/tmp").unwrap();
        assert!(create_writing(&store, tmp.path()).is_ok(), "/tmp is allowed");
        assert!(!projects.starts_with("/tmp"), "the projects folder is not /tmp, so the two are told apart");
        let why = create_writing(&store, std::path::Path::new("/usr")).unwrap_err(); // exists, outside both
        assert!(why.contains("projects folder") && why.contains("/tmp"), "{why}");
    }

    #[test]
    fn home_dot_folders_and_traversal_are_refused_whether_or_not_they_exist() {
        let (_dir, store) = store();
        let home = std::path::PathBuf::from(std::env::var_os("HOME").unwrap());
        with_projects(&store, &home); // even with the home folder as the projects folder
        for folder in [".config", ".local/bin", ".config/systemd/user", ".ssh", ".aria", ".not-there"] {
            let why = create_writing(&store, &home.join(folder)).unwrap_err();
            assert!(why.contains("projects folder"), "{folder}: {why}");
        }
        assert!(create_writing(&store, &home).is_err(), "the home folder itself");
        assert!(create_writing(&store, std::path::Path::new("/etc")).is_err());
        let work = outside_tmp();
        let projects = work.path().join("work");
        std::fs::create_dir_all(projects.join("a")).unwrap();
        with_projects(&store, &projects);
        let why = create_writing(&store, &projects.join("a/../../work")).unwrap_err();
        assert!(why.contains("projects folder"), "a `..` is refused even where it lands inside: {why}");
        assert!(store.list(0).is_empty(), "nothing was stored");
    }

    #[test]
    fn the_rule_holds_when_a_task_is_changed_and_a_stored_violation_is_not_widened() {
        let (_dir, store) = store();
        let work = outside_tmp();
        let projects = work.path().join("work");
        std::fs::create_dir_all(projects.join("ok")).unwrap();
        with_projects(&store, &projects);
        let task = create_writing(&store, &projects.join("ok")).unwrap();
        let why = store.update(&json!({"id": task.id, "writable": ["/usr"]})).unwrap_err();
        assert!(why.contains("projects folder"), "{why}");
        // A task stored before the rule: it stays, can be paused, and a new folder is judged.
        let mut old = task.clone();
        old.id = "s-oldoldoldold".into();
        old.grants.writable = vec!["/usr".into()];
        schedules::save(&store.paths, &[old.clone()]).unwrap();
        let store = ScheduleStore::open(store.paths.clone());
        assert_eq!(store.get(&old.id).unwrap().grants.writable, old.grants.writable, "stays stored");
        assert!(store.update(&json!({"id": old.id, "enabled": false})).is_ok(), "and can still be paused");
        assert!(store.update(&json!({"id": old.id, "writable": ["/usr", "/etc"]})).is_err(), "a new folder is judged");
    }

    #[test]
    fn a_run_drops_the_writable_grants_that_break_the_rule() {
        let (_dir, store) = store();
        let work = outside_tmp();
        let projects = work.path().join("work");
        std::fs::create_dir_all(projects.join("ok")).unwrap();
        with_projects(&store, &projects);
        let mut task = create_writing(&store, &projects.join("ok")).unwrap();
        let kept = task.grants.writable.clone();
        assert_eq!(store.runnable_writable(&task), kept, "a grant that follows the rule is kept");
        task.grants.writable.push("/home/ana/.config".into());
        task.grants.writable.push("/etc".into());
        assert_eq!(store.runnable_writable(&task), kept, "the others are dropped, never widened");
    }

    #[test]
    fn a_link_in_the_projects_folder_is_judged_by_where_it_points_when_the_task_is_made() {
        let (_dir, store) = store();
        let work = outside_tmp();
        let projects = work.path().join("work");
        std::fs::create_dir_all(projects.join("real")).unwrap();
        with_projects(&store, &projects);
        std::os::unix::fs::symlink("/usr", projects.join("to-usr")).unwrap();
        let why = create_writing(&store, &projects.join("to-usr")).unwrap_err();
        assert!(why.contains("projects folder"), "{why}");
        let home = std::path::PathBuf::from(std::env::var_os("HOME").unwrap());
        if home.join(".config").is_dir() {
            std::os::unix::fs::symlink(home.join(".config"), projects.join("to-config")).unwrap();
            assert!(create_writing(&store, &projects.join("to-config")).is_err(), "a link to ~/.config");
        }
        std::os::unix::fs::symlink(projects.join("real"), projects.join("inside")).unwrap();
        let task = create_writing(&store, &projects.join("inside")).unwrap();
        assert_eq!(task.grants.writable, vec![projects.join("real").canonicalize().unwrap().display().to_string()]);
        assert!(store.list(0).len() == 1, "only the good one was stored");
    }

    #[test]
    fn an_unchanged_grant_swapped_for_a_link_out_is_refused_when_the_task_is_changed_again() {
        let (_dir, store) = store();
        let work = outside_tmp();
        let projects = work.path().join("work");
        std::fs::create_dir_all(projects.join("out")).unwrap();
        with_projects(&store, &projects);
        let task = create_writing(&store, &projects.join("out")).unwrap();
        let out = task.grants.writable[0].clone();
        // The same folder submitted again, untouched: fine.
        assert!(store.update(&json!({"id": task.id, "writable": [&out]})).is_ok());
        // After the approval the folder becomes a link out; the task is changed again with the grant as it was.
        std::fs::remove_dir(&out).unwrap();
        std::os::unix::fs::symlink("/usr", &out).unwrap();
        let why = store.update(&json!({"id": task.id, "writable": [&out]})).unwrap_err();
        assert!(why.contains("projects folder"), "{why}");
        let why = store.update(&json!({"id": task.id, "title": "other"})).unwrap_err();
        assert!(why.contains("projects folder"), "a change that leaves the grant alone is refused too: {why}");
        assert_eq!(store.get(&task.id).unwrap().grants.writable, vec![out], "the store still holds the old entry");
    }

    #[test]
    fn a_run_judges_a_grant_by_its_real_path_and_drops_one_that_became_a_link_out() {
        let (_dir, store) = store();
        let work = outside_tmp();
        let projects = work.path().join("work");
        for dir in ["out", "other", "gone"] {
            std::fs::create_dir_all(projects.join(dir)).unwrap();
        }
        with_projects(&store, &projects);
        let task = store
            .create(&json!({"title": "t", "prompt": "p", "at": "07:30", "writable": [projects.join("out"), projects.join("gone")]}), 0)
            .unwrap();
        let (out, gone) = (task.grants.writable[0].clone(), task.grants.writable[1].clone());
        assert_eq!(store.runnable_writable(&task), vec![out.clone(), gone.clone()], "untouched grants are kept");
        // After the approval the folders are replaced: one by a link out, one removed.
        std::fs::remove_dir(&out).unwrap();
        std::os::unix::fs::symlink("/usr", &out).unwrap();
        std::fs::remove_dir(&gone).unwrap();
        assert_eq!(store.runnable_writable(&task), Vec::<String>::new(), "a link out and a missing folder are dropped");
        // A link to another folder inside the projects folder: the grant is the real folder.
        std::fs::remove_file(&out).unwrap();
        std::os::unix::fs::symlink(projects.join("other"), &out).unwrap();
        let real = projects.join("other").canonicalize().unwrap().display().to_string();
        assert_eq!(store.runnable_writable(&task), vec![real]);
    }

    #[test]
    fn a_task_names_at_most_sixteen_writable_folders() {
        let (_dir, store) = store();
        let work = outside_tmp();
        let projects = work.path().join("work");
        let dirs: Vec<_> = (0..17).map(|n| projects.join(format!("d{n}"))).collect();
        for dir in &dirs {
            std::fs::create_dir_all(dir).unwrap();
        }
        with_projects(&store, &projects);
        let make =
            |n: usize| store.create(&json!({"title": "t", "prompt": "p", "at": "07:30", "writable": &dirs[..n]}), 0);
        assert!(make(16).is_ok());
        let why = make(17).unwrap_err();
        assert!(why.contains("16"), "{why}");
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
    fn both_platforms_keyrings_and_login_items_are_guarded() {
        let home = std::fs::canonicalize(std::env::var_os("HOME").unwrap()).unwrap();
        let root = std::path::Path::new("/nonexistent-aria-root");
        for folder in ["Library/Keychains", "Library/LaunchAgents"] {
            assert!(too_wide(&home.join(folder), root).is_some(), "macOS: {folder}");
        }
        for folder in [".local/share/keyrings", ".config/autostart", ".config/systemd/user", ".config/hypr"] {
            assert!(too_wide(&home.join(folder), root).is_some(), "Linux: {folder}");
        }
        assert!(too_wide(&home.join("Documents/notes"), root).is_none(), "an ordinary folder is narrow enough");
    }

    #[test]
    fn take_due_writes_last_run_first() {
        let (_dir, store) = store();
        let now = SystemClock.epoch(2026, 10, 21, 7, 31);
        let task = store.create(&json!({"title": "t", "prompt": "p", "at": "07:30"}), now - 86_400_000).unwrap();
        let due = store.take_due(now, &SystemClock);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].0.id, task.id);
        assert!(!aria_core::schedules::late(due[0].1, now), "on time");
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
