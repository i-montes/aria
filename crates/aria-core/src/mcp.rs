//! The two MCP services the setup configures (Notion and Tavily), in Pi's own file formats so the
//! bundled Pi of phase 2 reads them as they are: `mcp.json` lists the servers (Tavily's key goes in
//! its `Authorization` header) and `mcp-auth.json` holds the OAuth state of the servers that sign in
//! (Notion). Both files are mode 600 and written atomically; one that does not parse is never replaced.

use crate::json_file::{read_json, write_json_atomic, JsonFileError};
use crate::paths::AriaPaths;
use serde_json::{json, Map, Value};
use std::os::unix::fs::DirBuilderExt;
use std::path::Path;

pub const NOTION_URL: &str = "https://mcp.notion.com/mcp";
pub const TAVILY_URL: &str = "https://mcp.tavily.com/mcp/";

#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error(transparent)]
    File(#[from] JsonFileError),
    #[error("could not write the MCP file: {0}")]
    Write(#[from] std::io::Error),
}

/// How Pi keys a server's OAuth record: `mcp__<name>|<url>`.
pub fn oauth_key(name: &str, url: &str) -> String {
    format!("mcp__{name}|{url}")
}

fn read_object(path: &Path) -> Result<Map<String, Value>, McpError> {
    match read_json(path)? {
        None => Ok(Map::new()),
        Some(Value::Object(map)) => Ok(map),
        Some(_) => Err(JsonFileError::Invalid { path: path.to_path_buf(), message: "expected a JSON object".into() }.into()),
    }
}

/// Changes `mcpServers` and writes the file back (mode 600), keeping every other key.
fn update_servers(paths: &AriaPaths, change: impl FnOnce(&mut Map<String, Value>)) -> Result<(), McpError> {
    let mut file = read_object(&paths.mcp())?;
    let mut servers = match file.remove("mcpServers") {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    };
    change(&mut servers);
    file.insert("mcpServers".into(), Value::Object(servers));
    write_json_atomic(&paths.mcp(), &Value::Object(file), 0o600)?;
    Ok(())
}

/// Stores Tavily's key. The server's entry is replaced whole, so an old key or header cannot linger.
pub fn set_tavily_key(paths: &AriaPaths, key: &str) -> Result<(), McpError> {
    update_servers(paths, |servers| {
        servers.insert(
            "tavily".into(),
            json!({
                "url": TAVILY_URL,
                "headers": { "Authorization": format!("Bearer {key}") },
                "description": "Search the web and extract, crawl or map web pages with Tavily",
            }),
        );
    })
}

/// Lists Notion as an MCP server (it signs in with OAuth, so there is no secret here). An entry the
/// user already has is left alone.
pub fn add_notion_server(paths: &AriaPaths) -> Result<(), McpError> {
    update_servers(paths, |servers| {
        servers.entry("notion").or_insert_with(|| {
            json!({
                "url": NOTION_URL,
                "description": "Search, read, create and update pages and databases in the user's Notion workspace",
            })
        });
    })
}

/// The seven Engram tools the owner's persona calls directly (ARIA-SETUP item 16).
const ENGRAM_DIRECT: [&str; 7] = ["mem_context", "mem_search", "mem_get_observation", "mem_save", "mem_update", "mem_judge", "mem_session_summary"];

/// Aria's own Engram entry: the binary the installer puts in `~/.aria/bin` and a database of its own in
/// `~/.aria/engram` (the owner's decision: Aria's memories do not mix with his terminal Pi's).
fn aria_engram(paths: &AriaPaths, project: &str) -> Value {
    let exposure: Map<String, Value> = ENGRAM_DIRECT.iter().map(|t| (t.to_string(), json!("direct"))).collect();
    json!({
        "command": paths.bin().join("engram").display().to_string(),
        "args": ["mcp", "--project", project, "--tools=agent"],
        "env": { "ENGRAM_DATA_DIR": paths.root.join("engram").display().to_string() },
        "toolExposure": exposure,
    })
}

/// An `engram` entry is Aria's when it runs Aria's binary on Aria's database; any other is the owner's.
fn is_arias_engram(paths: &AriaPaths, entry: &Value) -> bool {
    entry["command"].as_str() == Some(&paths.bin().join("engram").display().to_string())
        && entry["env"]["ENGRAM_DATA_DIR"].as_str() == Some(&paths.root.join("engram").display().to_string())
}

/// Memory on: Aria's Engram entry, with the settings' project. Memory off: Aria's entry goes. An `engram` entry Aria
/// did not write is never changed, and the file is only written when something changes.
pub fn sync_memory(paths: &AriaPaths, memory: &crate::settings::Memory) -> Result<(), McpError> {
    let file = read_object(&paths.mcp())?;
    let current = file.get("mcpServers").and_then(|s| s.get("engram")).cloned();
    let project = if memory.project.trim().is_empty() { "aria" } else { memory.project.trim() };
    let wanted = aria_engram(paths, project);
    match (&current, memory.enabled) {
        (Some(entry), _) if !is_arias_engram(paths, entry) => Ok(()),
        (Some(entry), true) if *entry == wanted => Ok(()),
        (None, false) => Ok(()),
        (_, true) => {
            std::fs::DirBuilder::new().recursive(true).mode(0o700).create(paths.root.join("engram"))?;
            update_servers(paths, |servers| {
                servers.insert("engram".into(), wanted);
            })
        }
        (Some(_), false) => update_servers(paths, |servers| {
            servers.remove("engram");
        }),
    }
}

/// Saves a server's OAuth record under Pi's key, keeping the other servers' records.
pub fn store_oauth_record(paths: &AriaPaths, name: &str, url: &str, record: Value) -> Result<(), McpError> {
    let mut file = read_object(&paths.mcp_auth())?;
    file.insert(oauth_key(name, url), record);
    write_json_atomic(&paths.mcp_auth(), &Value::Object(file), 0o600)?;
    Ok(())
}

/// What is really set up: `mcp:tavily` when its entry carries a key, `mcp:notion` when its OAuth
/// record holds an access token. Unreadable files count as nothing.
pub fn connected(paths: &AriaPaths) -> Vec<String> {
    let mut found = Vec::new();
    let non_empty = |v: Option<&Value>| v.and_then(Value::as_str).is_some_and(|s| !s.trim().is_empty());
    if let Ok(file) = read_object(&paths.mcp()) {
        if non_empty(file.get("mcpServers").and_then(|s| s.pointer("/tavily/headers/Authorization"))) {
            found.push("mcp:tavily".to_string());
        }
    }
    if let Ok(file) = read_object(&paths.mcp_auth()) {
        if non_empty(file.get(&oauth_key("notion", NOTION_URL)).and_then(|r| r.pointer("/tokens/access_token"))) {
            found.push("mcp:notion".to_string());
        }
    }
    found.sort();
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths::AriaPaths;
    use serde_json::json;
    use std::os::unix::fs::PermissionsExt;

    fn paths() -> (tempfile::TempDir, AriaPaths) {
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths { root: dir.path().to_path_buf() };
        (dir, paths)
    }

    fn read(path: &std::path::Path) -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn tavily_is_written_in_pis_mcp_format_with_mode_600() {
        let (_dir, paths) = paths();
        set_tavily_key(&paths, "tvly-secret").unwrap();
        let file = paths.mcp();
        assert_eq!(std::fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600);
        let tavily = &read(&file)["mcpServers"]["tavily"];
        assert_eq!(tavily["url"], "https://mcp.tavily.com/mcp/");
        assert_eq!(tavily["headers"]["Authorization"], "Bearer tvly-secret");
        assert!(tavily["description"].as_str().unwrap().contains("web"));
    }

    #[test]
    fn other_servers_and_unknown_fields_survive_and_a_new_key_replaces_the_old() {
        let (_dir, paths) = paths();
        std::fs::write(
            paths.mcp(),
            r#"{"mcpServers":{"mine":{"command":"x"},"tavily":{"url":"old","headers":{"Authorization":"Bearer old","X-Extra":"1"}}},"note":"keep"}"#,
        )
        .unwrap();
        set_tavily_key(&paths, "new").unwrap();
        let file = read(&paths.mcp());
        assert_eq!(file["mcpServers"]["mine"], json!({"command": "x"}));
        assert_eq!(file["note"], "keep");
        assert_eq!(file["mcpServers"]["tavily"]["headers"], json!({"Authorization": "Bearer new"}));
        assert_eq!(file["mcpServers"]["tavily"]["url"], "https://mcp.tavily.com/mcp/");
    }

    #[test]
    fn a_broken_mcp_file_is_reported_and_kept() {
        let (_dir, paths) = paths();
        std::fs::write(paths.mcp(), "{ nope").unwrap();
        assert!(set_tavily_key(&paths, "k").is_err());
        assert_eq!(std::fs::read_to_string(paths.mcp()).unwrap(), "{ nope");
    }

    #[test]
    fn notion_is_listed_without_any_secret() {
        let (_dir, paths) = paths();
        add_notion_server(&paths).unwrap();
        let notion = &read(&paths.mcp())["mcpServers"]["notion"];
        assert_eq!(notion["url"], "https://mcp.notion.com/mcp");
        assert!(notion.get("headers").is_none());
    }

    #[test]
    fn connected_reports_only_what_is_really_there() {
        let (_dir, paths) = paths();
        assert!(connected(&paths).is_empty());
        set_tavily_key(&paths, "k").unwrap();
        assert_eq!(connected(&paths), vec!["mcp:tavily".to_string()]);
        // a Notion entry in mcp.json without tokens is not a connection
        add_notion_server(&paths).unwrap();
        assert_eq!(connected(&paths), vec!["mcp:tavily".to_string()]);
        std::fs::write(
            paths.mcp_auth(),
            r#"{"mcp__notion|https://mcp.notion.com/mcp":{"tokens":{"access_token":"a","refresh_token":"r"}}}"#,
        )
        .unwrap();
        assert_eq!(connected(&paths), vec!["mcp:notion".to_string(), "mcp:tavily".to_string()]);
    }

    #[test]
    fn an_oauth_record_is_stored_under_pis_key_and_other_servers_stay() {
        let (_dir, paths) = paths();
        std::fs::write(paths.mcp_auth(), r#"{"mcp__figma|https://mcp.figma.com/mcp":{"tokens":{"access_token":"f"}}}"#).unwrap();
        store_oauth_record(&paths, "notion", NOTION_URL, json!({"serverUrl": NOTION_URL, "tokens": {"access_token": "a"}})).unwrap();
        let file = read(&paths.mcp_auth());
        assert_eq!(std::fs::metadata(paths.mcp_auth()).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(file["mcp__notion|https://mcp.notion.com/mcp"]["tokens"]["access_token"], "a");
        assert_eq!(file["mcp__figma|https://mcp.figma.com/mcp"]["tokens"]["access_token"], "f");
    }

    fn memory(enabled: bool, project: &str) -> crate::settings::Memory {
        crate::settings::Memory { enabled, project: project.into() }
    }

    fn servers(paths: &AriaPaths) -> Value {
        serde_json::from_str::<Value>(&std::fs::read_to_string(paths.mcp()).unwrap()).unwrap()["mcpServers"].clone()
    }

    #[test]
    fn memory_on_writes_arias_engram_entry_with_the_project_and_its_own_database() {
        let (_dir, paths) = paths();
        sync_memory(&paths, &memory(true, "casa")).unwrap();
        let engram = &servers(&paths)["engram"];
        assert_eq!(engram["command"], json!(paths.bin().join("engram").display().to_string()));
        assert_eq!(engram["args"], json!(["mcp", "--project", "casa", "--tools=agent"]));
        assert_eq!(engram["env"]["ENGRAM_DATA_DIR"], json!(paths.root.join("engram").display().to_string()));
        assert_eq!(engram["toolExposure"]["mem_save"], "direct");
        assert_eq!(engram["toolExposure"].as_object().unwrap().len(), 7);
    }

    #[test]
    fn memory_off_removes_only_arias_entry() {
        let (_dir, paths) = paths();
        sync_memory(&paths, &memory(true, "aria")).unwrap();
        set_tavily_key(&paths, "k").unwrap();
        sync_memory(&paths, &memory(false, "aria")).unwrap();
        let list = servers(&paths);
        assert!(list.get("engram").is_none());
        assert!(list.get("tavily").is_some());
    }

    #[test]
    fn a_hand_written_engram_entry_is_never_touched() {
        let (_dir, paths) = paths();
        let own = json!({"mcpServers": {"engram": {"command": "/usr/local/bin/engram", "args": ["mcp"]}, "rea": {"command": "rea"}}});
        std::fs::write(paths.mcp(), serde_json::to_string_pretty(&own).unwrap()).unwrap();
        sync_memory(&paths, &memory(true, "casa")).unwrap();
        assert_eq!(servers(&paths), own["mcpServers"]);
        sync_memory(&paths, &memory(false, "casa")).unwrap();
        assert_eq!(servers(&paths), own["mcpServers"]);
    }

    #[test]
    fn nothing_is_written_when_nothing_changes() {
        let (_dir, paths) = paths();
        sync_memory(&paths, &memory(false, "aria")).unwrap();
        assert!(!paths.mcp().exists(), "memory off and no file: no file is created");
    }
}
