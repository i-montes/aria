//! The MCP servers the person already set up in other apps (Claude Code, Claude Desktop, Cursor, VS Code), and how the
//! ones they pick are copied into Aria's `mcp.json` in Pi's format. Other apps' files are only ever read. Secrets in
//! `env` or `headers` are copied as they are (`mcp.json` is mode 0600); the page only gets their names.

use crate::json_file::{read_json, write_json_atomic, JsonFileError};
use crate::paths::AriaPaths;
use serde::Serialize;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

/// A server found in another app's configuration. `config` is the entry already in Pi's format; it can hold secrets, so
/// it is never serialized (the page gets `secrets`, the names of the variables and headers).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Found {
    /// `source:name`.
    pub id: String,
    pub name: String,
    /// `claude-code` | `claude-desktop` | `cursor` | `vscode`.
    pub source: String,
    /// `stdio` | `http`.
    pub kind: String,
    /// The command's file name (stdio) or the URL's host (http).
    pub target: String,
    pub secrets: Vec<String>,
    #[serde(skip)]
    config: Value,
}

impl Found {
    /// The entry as it is written to Aria's `mcp.json`.
    pub fn config(&self) -> &Value {
        &self.config
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error(transparent)]
    File(#[from] JsonFileError),
    #[error("could not write the MCP file: {0}")]
    Write(#[from] std::io::Error),
    #[error("none of the servers asked for was found")]
    NothingToImport,
}

/// Where each app keeps its servers, and under which key, in the order they are listed.
fn sources(home: &Path) -> Vec<(&'static str, PathBuf, &'static str)> {
    #[cfg(target_os = "macos")]
    let (desktop, vscode) = {
        let support = home.join("Library/Application Support");
        (support.join("Claude/claude_desktop_config.json"), support.join("Code/User/mcp.json"))
    };
    #[cfg(not(target_os = "macos"))]
    let (desktop, vscode) =
        (home.join(".config/Claude/claude_desktop_config.json"), home.join(".config/Code/User/mcp.json"));
    vec![
        ("claude-code", home.join(".claude.json"), "mcpServers"),
        ("claude-desktop", desktop, "mcpServers"),
        ("cursor", home.join(".cursor/mcp.json"), "mcpServers"),
        ("vscode", vscode, "servers"),
    ]
}

/// Every server the other apps configure, except `engram` and the names already in Aria's `mcp.json` (`existing`).
/// A missing or broken file, and an entry of a shape Aria does not know, are skipped.
pub fn find(home: &Path, existing: &[String]) -> Vec<Found> {
    let mut found = Vec::new();
    for (source, path, key) in sources(home) {
        let Ok(Some(file)) = read_json(&path) else { continue };
        let Some(servers) = file.get(key).and_then(Value::as_object) else { continue };
        for (name, entry) in servers {
            if name == "engram" || existing.iter().any(|e| e == name) {
                continue;
            }
            if let Some(server) = parse(source, name, entry) {
                found.push(server);
            }
        }
    }
    found
}

/// A string-to-string object, or `None` when any value is not a string. Absent is an empty map.
fn string_map(value: Option<&Value>) -> Option<Option<Map<String, Value>>> {
    match value {
        None | Some(Value::Null) => Some(None),
        Some(Value::Object(map)) if map.values().all(Value::is_string) => {
            Some(if map.is_empty() { None } else { Some(map.clone()) })
        }
        Some(_) => None,
    }
}

fn parse(source: &str, name: &str, entry: &Value) -> Option<Found> {
    let entry = entry.as_object()?;
    let kind = entry.get("type").map(|t| t.as_str()).unwrap_or(Some(""))?;
    let url = entry.get("url").or_else(|| entry.get("serverUrl"));
    let stdio = match kind {
        "stdio" => true,
        "http" | "sse" | "streamable-http" => false,
        "" => entry.contains_key("command"),
        _ => return None,
    };
    let mut config = Map::new();
    let (target, secret_map) = if stdio {
        let command = entry.get("command")?.as_str().filter(|c| !c.trim().is_empty())?;
        config.insert("command".into(), command.into());
        match entry.get("args") {
            None | Some(Value::Null) => {}
            Some(Value::Array(args)) if args.iter().all(Value::is_string) => {
                if !args.is_empty() {
                    config.insert("args".into(), Value::Array(args.clone()));
                }
            }
            Some(_) => return None,
        }
        let env = string_map(entry.get("env"))?;
        if let Some(env) = &env {
            config.insert("env".into(), Value::Object(env.clone()));
        }
        let target = Path::new(command).file_name()?.to_string_lossy().into_owned();
        (target, env)
    } else {
        let url = url?.as_str()?;
        let host = url::Url::parse(url).ok()?.host_str()?.to_string();
        config.insert("url".into(), url.into());
        let headers = string_map(entry.get("headers"))?;
        if let Some(headers) = &headers {
            config.insert("headers".into(), Value::Object(headers.clone()));
        }
        (host, headers)
    };
    let config = Value::Object(config);
    let mut secrets: Vec<String> = secret_map.map(|m| m.keys().cloned().collect()).unwrap_or_default();
    for id in input_references(&config) {
        if !secrets.contains(&id) {
            secrets.push(id);
        }
    }
    Some(Found {
        id: format!("{source}:{name}"),
        name: name.to_string(),
        source: source.to_string(),
        kind: if stdio { "stdio" } else { "http" }.to_string(),
        target,
        secrets,
        config,
    })
}

/// VS Code's `${input:<id>}` references anywhere in the entry, as `input:<id>`, in order of appearance.
fn input_references(value: &Value) -> Vec<String> {
    let mut found = Vec::new();
    let mut strings = Vec::new();
    collect_strings(value, &mut strings);
    for text in strings {
        let mut rest = text;
        while let Some(start) = rest.find("${input:") {
            let after = &rest[start + "${input:".len()..];
            let Some(end) = after.find('}') else { break };
            let id = format!("input:{}", &after[..end]);
            if !found.contains(&id) {
                found.push(id);
            }
            rest = &after[end + 1..];
        }
    }
    found
}

fn collect_strings<'a>(value: &'a Value, out: &mut Vec<&'a str>) {
    match value {
        Value::String(s) => out.push(s),
        Value::Array(items) => items.iter().for_each(|v| collect_strings(v, out)),
        Value::Object(map) => map.values().for_each(|v| collect_strings(v, out)),
        _ => {}
    }
}

/// Copies the servers whose ids are in `ids` into Aria's `mcp.json`, keeping every server and key already there and
/// the file's mode 0600. A name already taken gets `-2`, `-3`… Returns the names written, in order.
pub fn import(paths: &AriaPaths, found: &[Found], ids: &[String]) -> Result<Vec<String>, ImportError> {
    let chosen: Vec<&Found> = found.iter().filter(|f| ids.contains(&f.id)).collect();
    if chosen.is_empty() {
        return Err(ImportError::NothingToImport);
    }
    let mut file = match read_json(&paths.mcp())? {
        None => Map::new(),
        Some(Value::Object(map)) => map,
        Some(_) => {
            return Err(JsonFileError::Invalid { path: paths.mcp(), message: "expected a JSON object".into() }.into())
        }
    };
    let mut servers = match file.remove("mcpServers") {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    };
    let mut written = Vec::new();
    for server in chosen {
        let mut name = server.name.clone();
        let mut n = 2;
        while servers.contains_key(&name) {
            name = format!("{}-{n}", server.name);
            n += 1;
        }
        servers.insert(name.clone(), server.config.clone());
        written.push(name);
    }
    file.insert("mcpServers".into(), Value::Object(servers));
    write_json_atomic(&paths.mcp(), &Value::Object(file), 0o600)?;
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::os::unix::fs::PermissionsExt;

    const CLAUDE_CODE: &str = include_str!("../tests/fixtures/mcp/claude.json");
    const CLAUDE_DESKTOP: &str = include_str!("../tests/fixtures/mcp/claude_desktop_config.json");
    const CURSOR: &str = include_str!("../tests/fixtures/mcp/cursor-mcp.json");
    const VSCODE: &str = include_str!("../tests/fixtures/mcp/vscode-mcp.json");

    fn put(home: &Path, relative: PathBuf, text: &str) {
        let path = home.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    /// A home with every app's file, at the paths of the platform the tests run on.
    fn home() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        let texts = [CLAUDE_CODE, CLAUDE_DESKTOP, CURSOR, VSCODE];
        for ((_, path, _), text) in sources(home).into_iter().zip(texts) {
            put(home, path.strip_prefix(home).unwrap().to_path_buf(), text);
        }
        dir
    }

    fn by_id<'a>(found: &'a [Found], id: &str) -> &'a Found {
        found.iter().find(|f| f.id == id).unwrap_or_else(|| panic!("{id} not found"))
    }

    fn ids(found: &[Found]) -> Vec<&str> {
        found.iter().map(|f| f.id.as_str()).collect()
    }

    fn aria() -> (tempfile::TempDir, AriaPaths) {
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths { root: dir.path().to_path_buf() };
        (dir, paths)
    }

    #[test]
    fn claude_code_user_servers_are_found() {
        let home = home();
        let found = find(home.path(), &[]);
        let files = by_id(&found, "claude-code:files");
        assert_eq!(files.name, "files");
        assert_eq!(files.source, "claude-code");
        assert_eq!(files.kind, "stdio");
        assert_eq!(files.target, "files-mcp");
        assert_eq!(files.secrets, vec!["FILES_TOKEN"]);
        assert_eq!(
            files.config(),
            &json!({"command": "/home/ana/.local/bin/files-mcp", "args": ["--root", "/home/ana/notes"], "env": {"FILES_TOKEN": "files-secret-value"}})
        );
        // Project-scoped servers belong to their project.
        assert!(!found.iter().any(|f| f.name == "project-db"));
        // The other apps are read too.
        assert_eq!(by_id(&found, "claude-desktop:calendar").target, "npx");
        assert_eq!(
            by_id(&found, "cursor:docs").config(),
            &json!({"url": "https://docs.example.com/mcp", "headers": {"X-Api-Key": "docs-secret-value"}})
        );
    }

    #[test]
    fn vscode_uses_servers() {
        let home = home();
        let found = find(home.path(), &[]);
        let tracker = by_id(&found, "vscode:tracker");
        assert_eq!(tracker.kind, "stdio");
        assert_eq!(tracker.target, "node");
        assert_eq!(tracker.secrets, vec!["TRACKER_TOKEN", "input:tracker-token"]);
        // Imported as it is, the input reference included.
        assert_eq!(tracker.config()["env"]["TRACKER_TOKEN"], "${input:tracker-token}");
        let search = by_id(&found, "vscode:search");
        assert_eq!(search.kind, "http");
        assert_eq!(search.secrets, vec!["Authorization", "input:search-key"]);
        assert!(!found.iter().any(|f| f.id == "vscode:ignored"), "VS Code's key is `servers`");
    }

    #[test]
    fn http_and_sse_are_http() {
        let home = home();
        let found = find(home.path(), &[]);
        let weather = by_id(&found, "claude-code:weather");
        assert_eq!((weather.kind.as_str(), weather.target.as_str()), ("http", "weather.example.com"));
        assert_eq!(
            weather.config(),
            &json!({"url": "https://weather.example.com/mcp", "headers": {"Authorization": "Bearer weather-secret-value"}})
        );
        let events = by_id(&found, "claude-code:events");
        assert_eq!((events.kind.as_str(), events.target.as_str()), ("http", "events.example.com"));
        assert_eq!(events.config(), &json!({"url": "https://events.example.com:8443/sse"}));
        assert!(events.secrets.is_empty());
        assert_eq!(by_id(&found, "cursor:docs").target, "docs.example.com");
    }

    #[test]
    fn odd_entries_are_skipped_and_the_rest_found() {
        let home = home();
        let found = find(home.path(), &[]);
        let claude: Vec<&str> = ids(&found).into_iter().filter(|id| id.starts_with("claude-code:")).collect();
        assert_eq!(claude, vec!["claude-code:events", "claude-code:files", "claude-code:weather"]);
        // A broken file is skipped; the other apps still count.
        put(home.path(), PathBuf::from(".cursor/mcp.json"), "{ not json");
        let found = find(home.path(), &[]);
        assert!(!found.iter().any(|f| f.source == "cursor"));
        assert!(found.iter().any(|f| f.source == "vscode"));
        // No files at all: nothing.
        assert!(find(tempfile::tempdir().unwrap().path(), &[]).is_empty());
    }

    #[test]
    fn secrets_are_names_only() {
        let home = home();
        let found = find(home.path(), &[]);
        let text = serde_json::to_string(&found).unwrap();
        assert!(!text.contains("secret-value"), "{text}");
        assert!(!text.contains("config"), "{text}");
        assert_eq!(
            serde_json::to_value(by_id(&found, "claude-code:weather")).unwrap(),
            json!({"id": "claude-code:weather", "name": "weather", "source": "claude-code", "kind": "http", "target": "weather.example.com", "secrets": ["Authorization"]})
        );
    }

    #[test]
    fn existing_and_engram_are_hidden() {
        let home = home();
        let found = find(home.path(), &["files".to_string(), "tracker".to_string()]);
        assert!(!found.iter().any(|f| f.name == "engram"));
        assert!(!found.iter().any(|f| f.name == "files" || f.name == "tracker"));
        assert!(found.iter().any(|f| f.name == "weather"));
    }

    #[test]
    fn names_are_deduplicated() {
        let home = home();
        let (_dir, paths) = aria();
        let found = find(home.path(), &[]);
        let wanted: Vec<String> =
            ["claude-code:files", "claude-desktop:files", "cursor:files"].map(String::from).to_vec();
        assert_eq!(import(&paths, &found, &wanted).unwrap(), vec!["files", "files-2", "files-3"]);
        let file = read_json(&paths.mcp()).unwrap().unwrap();
        assert_eq!(file["mcpServers"]["files-3"], json!({"command": "uvx", "args": ["files-mcp"]}));
        // Taken again by a later import.
        assert_eq!(import(&paths, &found, &["cursor:files".to_string()]).unwrap(), vec!["files-4"]);
    }

    #[test]
    fn import_keeps_the_others_and_the_mode() {
        let home = home();
        let (_dir, paths) = aria();
        let before =
            json!({"mcpServers": {"tavily": {"url": "https://tavily.example.com"}}, "settings": {"keep": true}});
        write_json_atomic(&paths.mcp(), &before, 0o600).unwrap();
        let found = find(home.path(), &crate::mcp::servers(&paths));
        let names = import(&paths, &found, &["claude-code:weather".into(), "vscode:tracker".into()]).unwrap();
        assert_eq!(names, vec!["weather", "tracker"]);
        let file = read_json(&paths.mcp()).unwrap().unwrap();
        assert_eq!(file["settings"], json!({"keep": true}));
        assert_eq!(file["mcpServers"]["tavily"], before["mcpServers"]["tavily"]);
        assert_eq!(file["mcpServers"]["weather"]["headers"]["Authorization"], "Bearer weather-secret-value");
        assert_eq!(std::fs::metadata(paths.mcp()).unwrap().permissions().mode() & 0o777, 0o600);
        // The other apps' files are untouched.
        assert_eq!(std::fs::read_to_string(home.path().join(".claude.json")).unwrap(), CLAUDE_CODE);
    }

    #[test]
    fn nothing_to_import_is_an_error() {
        let home = home();
        let (_dir, paths) = aria();
        let found = find(home.path(), &[]);
        assert!(matches!(import(&paths, &found, &[]), Err(ImportError::NothingToImport)));
        assert!(matches!(import(&paths, &found, &["cursor:gone".into()]), Err(ImportError::NothingToImport)));
        assert!(!paths.mcp().exists());
        // A broken mcp.json is never replaced.
        std::fs::write(paths.mcp(), "{ broken").unwrap();
        assert!(matches!(import(&paths, &found, &["cursor:docs".into()]), Err(ImportError::File(_))));
        assert_eq!(std::fs::read_to_string(paths.mcp()).unwrap(), "{ broken");
    }
}
