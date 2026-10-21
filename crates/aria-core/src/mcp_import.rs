//! The MCP servers the person already set up in other apps (Claude Code, Claude Desktop, Cursor, VS Code), and how the
//! ones they pick are copied into Aria's `mcp.json` in Pi's format. Other apps' files are only ever read. Secrets in
//! `env` or `headers` are copied as they are (`mcp.json` is mode 0600); the page only gets their names.
//!
//! Pi resolves the values of `env` and `headers` (Pi 1.3.0, `resolve-config-value.js`): `${NAME}` and `$NAME` are
//! environment variables, `$$` is a `$`, `$!` is a `!`, and a value that starts with `!` runs as a shell command. So a
//! value is translated on the way in: `${env:NAME}` (VS Code, Cursor) becomes `${NAME}`, a lone `$` becomes `$$` (the
//! other apps only expand `${...}`), and a leading `!` becomes `$!`, so nothing the other app took as text ever runs.
//! `${input:<id>}` (VS Code prompts) is not a name Pi knows, so it stays as it is and is listed as the secret `input:<id>`.
//! Pi refuses the legacy SSE transport: SSE servers are listed (`importable: false`) but never written.

use crate::json_file::{read_json, write_json_atomic, JsonFileError};
use crate::paths::AriaPaths;
use serde::Serialize;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

/// A server found in another app's configuration. `config` is the entry already in Pi's format; it can hold secrets, so
/// it is never serialized nor printed (the page gets `secrets`, the names of the variables and headers).
#[derive(Clone, PartialEq, Serialize)]
pub struct Found {
    /// `source:name`.
    pub id: String,
    /// The name in the other app, for display.
    pub name: String,
    /// The name it is written under: `name` with what Pi refuses replaced (`pi_name`). A name already taken in Aria's
    /// `mcp.json` still gets `-2`, `-3`… when it is imported.
    #[serde(rename = "importAs")]
    pub import_as: String,
    /// `claude-code` | `claude-desktop` | `cursor` | `vscode`.
    pub source: String,
    /// `stdio` | `http` | `sse`.
    pub kind: String,
    /// The command's file name (stdio) or the URL's host (http, sse).
    pub target: String,
    pub secrets: Vec<String>,
    /// False for SSE, which Pi does not support: listed so the page can say so, never imported.
    pub importable: bool,
    /// Turned off in the other app; imported turned off too (Pi's `enabled: false`).
    pub disabled: bool,
    #[serde(skip)]
    config: Value,
}

impl Found {
    /// The entry as it is written to Aria's `mcp.json`.
    pub fn config(&self) -> &Value {
        &self.config
    }
}

impl std::fmt::Debug for Found {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Found")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("import_as", &self.import_as)
            .field("source", &self.source)
            .field("kind", &self.kind)
            .field("target", &self.target)
            .field("secrets", &self.secrets)
            .field("importable", &self.importable)
            .field("disabled", &self.disabled)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error(transparent)]
    File(#[from] JsonFileError),
    #[error("could not write the MCP file: {0}")]
    Write(#[from] std::io::Error),
    #[error("none of the servers asked for can be imported")]
    NothingToImport,
}

/// Where each app keeps its servers, and under which key, in the order they are listed. `config` is the Linux config
/// directory (`$XDG_CONFIG_HOME` or `~/.config`); macOS keeps them in `~/Library/Application Support`.
fn sources(home: &Path, config: &Path) -> Vec<(&'static str, PathBuf, &'static str)> {
    #[cfg(target_os = "macos")]
    let (desktop, vscode) = {
        let _ = config;
        let support = home.join("Library/Application Support");
        (support.join("Claude/claude_desktop_config.json"), support.join("Code/User/mcp.json"))
    };
    #[cfg(not(target_os = "macos"))]
    let (desktop, vscode) = (config.join("Claude/claude_desktop_config.json"), config.join("Code/User/mcp.json"));
    vec![
        ("claude-code", home.join(".claude.json"), "mcpServers"),
        ("claude-desktop", desktop, "mcpServers"),
        ("cursor", home.join(".cursor/mcp.json"), "mcpServers"),
        ("vscode", vscode, "servers"),
    ]
}

/// `$XDG_CONFIG_HOME` when it is set and absolute, else `~/.config`.
pub fn config_dir(home: &Path, xdg: Option<&std::ffi::OsStr>) -> PathBuf {
    match xdg.map(PathBuf::from) {
        Some(dir) if dir.is_absolute() => dir,
        _ => home.join(".config"),
    }
}

/// Every server the other apps configure, except `engram` and the names already in Aria's `mcp.json` (`existing`).
/// A missing or broken file, and an entry of a shape Aria does not know, are skipped. A `home` that is empty or not
/// absolute (no `$HOME`) finds nothing: relative paths are never read.
pub fn find(home: &Path, existing: &[String]) -> Vec<Found> {
    let xdg = std::env::var_os("XDG_CONFIG_HOME");
    find_in(home, &config_dir(home, xdg.as_deref()), existing)
}

/// `find` with the Linux config directory given (see `config_dir`).
pub fn find_in(home: &Path, config: &Path, existing: &[String]) -> Vec<Found> {
    if !home.is_absolute() {
        return Vec::new();
    }
    let mut found = Vec::new();
    for (source, path, key) in sources(home, config) {
        let Ok(Some(file)) = read_json(&path) else { continue };
        let Some(servers) = file.get(key).and_then(Value::as_object) else { continue };
        for (name, entry) in servers {
            let import_as = pi_name(name);
            if name == "engram" || import_as == "engram" || existing.contains(&import_as) {
                continue;
            }
            if let Some(server) = parse(source, name, entry) {
                found.push(server);
            }
        }
    }
    found
}

/// A string-to-string object with its values translated for Pi (`pi_value`), or `None` when any value is not a
/// string. Absent or empty is `Some(None)`.
fn string_map(value: Option<&Value>) -> Option<Option<Map<String, Value>>> {
    match value {
        None | Some(Value::Null) => Some(None),
        Some(Value::Object(map)) => {
            let mut out = Map::new();
            for (key, value) in map {
                out.insert(key.clone(), pi_value(value.as_str()?).into());
            }
            Some(if out.is_empty() { None } else { Some(out) })
        }
        Some(_) => None,
    }
}

/// A server name Pi accepts (letters, digits, `_` and `-`): every run of other characters becomes one `-`, a `-` at
/// either end goes, and nothing left is `server`.
fn pi_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            out.push(c);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let out = out.trim_matches('-');
    if out.is_empty() {
        "server".to_string()
    } else {
        out.to_string()
    }
}

/// An `env` or `headers` value as Pi must read it to mean what the other app meant (see the module's doc).
fn pi_value(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    let mut rest = value;
    while let Some(at) = rest.find('$') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        if let Some(after) = rest.strip_prefix("${env:") {
            if let Some(end) = after.find('}') {
                out.push_str("${");
                out.push_str(&after[..end]);
                out.push('}');
                rest = &after[end + 1..];
                continue;
            }
        }
        if rest.starts_with("${") {
            out.push('$');
        } else {
            out.push_str("$$");
        }
        rest = &rest[1..];
    }
    out.push_str(rest);
    if out.starts_with('!') {
        out.insert(0, '$');
    }
    out
}

fn parse(source: &str, name: &str, entry: &Value) -> Option<Found> {
    let entry = entry.as_object()?;
    let kind = entry.get("type").map(|t| t.as_str()).unwrap_or(Some(""))?;
    let url = entry.get("url").or_else(|| entry.get("serverUrl"));
    let kind = match kind {
        "stdio" => "stdio",
        "http" | "streamable-http" => "http",
        "sse" => "sse",
        "" if entry.contains_key("command") => "stdio",
        "" => "http",
        _ => return None,
    };
    let mut config = Map::new();
    let (target, secret_map) = if kind == "stdio" {
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
        match entry.get("cwd") {
            None | Some(Value::Null) => {}
            Some(Value::String(cwd)) => {
                config.insert("cwd".into(), cwd.as_str().into());
            }
            Some(_) => return None,
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
    let disabled =
        entry.get("disabled") == Some(&Value::Bool(true)) || entry.get("enabled") == Some(&Value::Bool(false));
    if disabled {
        config.insert("enabled".into(), false.into());
    }
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
        import_as: pi_name(name),
        source: source.to_string(),
        kind: kind.to_string(),
        target,
        secrets,
        importable: kind != "sse",
        disabled,
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
/// the file's mode 0600. A name already taken gets `-2`, `-3`… An SSE server is never written. Returns the names
/// written, in order.
pub fn import(paths: &AriaPaths, found: &[Found], ids: &[String]) -> Result<Vec<String>, ImportError> {
    let chosen: Vec<&Found> = found.iter().filter(|f| f.importable && ids.contains(&f.id)).collect();
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
        let mut name = server.import_as.clone();
        let mut n = 2;
        while servers.contains_key(&name) {
            name = format!("{}-{n}", server.import_as);
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
        for ((_, path, _), text) in sources(home, &home.join(".config")).into_iter().zip(texts) {
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
        let found = find_in(home.path(), &home.path().join(".config"), &[]);
        let files = by_id(&found, "claude-code:files");
        assert_eq!(files.name, "files");
        assert_eq!(files.source, "claude-code");
        assert_eq!(files.kind, "stdio");
        assert_eq!(files.target, "files-mcp");
        assert_eq!(files.secrets, vec!["FILES_TOKEN"]);
        assert_eq!(
            files.config(),
            &json!({"command": "/home/ana/.local/bin/files-mcp", "args": ["--root", "/home/ana/notes"], "env": {"FILES_TOKEN": "files-secret-value"}, "cwd": "/home/ana/notes"})
        );
        // Project-scoped servers belong to their project.
        assert!(!found.iter().any(|f| f.name == "project-db"));
        // The other apps are read too.
        assert_eq!(by_id(&found, "claude-desktop:calendar").target, "npx");
        let docs = by_id(&found, "cursor:docs");
        assert!(docs.disabled, "turned off in Cursor");
        assert_eq!(
            docs.config(),
            &json!({"url": "https://docs.example.com/mcp", "headers": {"X-Api-Key": "docs-secret-value"}, "enabled": false})
        );
        assert!(!files.disabled);
    }

    #[test]
    fn vscode_uses_servers() {
        let home = home();
        let found = find_in(home.path(), &home.path().join(".config"), &[]);
        let tracker = by_id(&found, "vscode:tracker");
        assert_eq!(tracker.kind, "stdio");
        assert_eq!(tracker.target, "node");
        assert_eq!(tracker.secrets, vec!["TRACKER_HOME", "TRACKER_TOKEN", "input:tracker-token"]);
        // An input reference is imported as it is; `${env:X}` becomes Pi's `${X}`.
        assert_eq!(tracker.config()["env"]["TRACKER_TOKEN"], "${input:tracker-token}");
        assert_eq!(tracker.config()["env"]["TRACKER_HOME"], "${HOME}/tracker");
        // `enabled: false` in VS Code.
        assert!(tracker.disabled);
        assert_eq!(tracker.config()["enabled"], false);
        let search = by_id(&found, "vscode:search");
        assert_eq!(search.kind, "http");
        assert_eq!(search.secrets, vec!["Authorization", "input:search-key"]);
        assert!(!found.iter().any(|f| f.id == "vscode:ignored"), "VS Code's key is `servers`");
    }

    #[test]
    fn http_is_http_and_sse_is_listed_but_not_importable() {
        let home = home();
        let found = find_in(home.path(), &home.path().join(".config"), &[]);
        let weather = by_id(&found, "claude-code:weather");
        assert_eq!((weather.kind.as_str(), weather.target.as_str()), ("http", "weather.example.com"));
        assert_eq!(
            weather.config(),
            &json!({"url": "https://weather.example.com/mcp", "headers": {"Authorization": "Bearer weather-secret-value"}})
        );
        let events = by_id(&found, "claude-code:events");
        assert_eq!((events.kind.as_str(), events.target.as_str()), ("sse", "events.example.com"));
        assert!(!events.importable, "Pi refuses the legacy SSE transport");
        assert!(events.secrets.is_empty());
        assert!(weather.importable);
        assert_eq!(by_id(&found, "cursor:docs").target, "docs.example.com");
    }

    #[test]
    fn odd_entries_are_skipped_and_the_rest_found() {
        let home = home();
        let found = find_in(home.path(), &home.path().join(".config"), &[]);
        let claude: Vec<&str> = ids(&found).into_iter().filter(|id| id.starts_with("claude-code:")).collect();
        assert_eq!(claude, vec!["claude-code:events", "claude-code:files", "claude-code:weather"]);
        // A broken file is skipped; the other apps still count.
        put(home.path(), PathBuf::from(".cursor/mcp.json"), "{ not json");
        let found = find_in(home.path(), &home.path().join(".config"), &[]);
        assert!(!found.iter().any(|f| f.source == "cursor"));
        assert!(found.iter().any(|f| f.source == "vscode"));
        // No files at all: nothing.
        assert!(find_in(tempfile::tempdir().unwrap().path(), Path::new("/nowhere"), &[]).is_empty());
    }

    #[test]
    fn secrets_are_names_only() {
        let home = home();
        let found = find_in(home.path(), &home.path().join(".config"), &[]);
        let text = serde_json::to_string(&found).unwrap();
        assert!(!text.contains("secret-value"), "{text}");
        assert!(!text.contains("config"), "{text}");
        assert_eq!(
            serde_json::to_value(by_id(&found, "claude-code:weather")).unwrap(),
            json!({"id": "claude-code:weather", "name": "weather", "source": "claude-code", "kind": "http", "target": "weather.example.com", "secrets": ["Authorization"], "importAs": "weather", "importable": true, "disabled": false})
        );
        let debug = format!("{found:?}");
        assert!(!debug.contains("secret-value") && !debug.contains("config"), "{debug}");
    }

    #[test]
    fn existing_and_engram_are_hidden() {
        let home = home();
        let found = find_in(home.path(), &home.path().join(".config"), &["files".to_string(), "tracker".to_string()]);
        assert!(!found.iter().any(|f| f.name == "engram"));
        assert!(!found.iter().any(|f| f.name == "files" || f.name == "tracker"));
        assert!(found.iter().any(|f| f.name == "weather"));
    }

    #[test]
    fn names_are_deduplicated() {
        let home = home();
        let (_dir, paths) = aria();
        let found = find_in(home.path(), &home.path().join(".config"), &[]);
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
        let found = find_in(home.path(), &home.path().join(".config"), &crate::mcp::servers(&paths));
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
        let found = find_in(home.path(), &home.path().join(".config"), &[]);
        assert!(matches!(import(&paths, &found, &[]), Err(ImportError::NothingToImport)));
        assert!(matches!(import(&paths, &found, &["cursor:gone".into()]), Err(ImportError::NothingToImport)));
        assert!(matches!(import(&paths, &found, &["claude-code:events".into()]), Err(ImportError::NothingToImport)));
        assert!(!paths.mcp().exists());
        // A broken mcp.json is never replaced.
        std::fs::write(paths.mcp(), "{ broken").unwrap();
        assert!(matches!(import(&paths, &found, &["cursor:docs".into()]), Err(ImportError::File(_))));
        assert_eq!(std::fs::read_to_string(paths.mcp()).unwrap(), "{ broken");
    }

    #[test]
    fn an_sse_server_is_never_written() {
        let home = home();
        let (_dir, paths) = aria();
        let found = find_in(home.path(), &home.path().join(".config"), &[]);
        let names = import(&paths, &found, &["claude-code:events".into(), "claude-code:weather".into()]).unwrap();
        assert_eq!(names, vec!["weather"]);
        assert_eq!(crate::mcp::servers(&paths), vec!["weather"]);
    }

    #[test]
    fn values_mean_to_pi_what_they_meant_to_the_other_app() {
        // A leading `!` would run as a shell command in Pi: `$!` is a literal `!`.
        assert_eq!(pi_value("!hello"), "$!hello");
        // Pi expands a bare `$NAME`; the other apps do not.
        assert_eq!(pi_value("costs $5 or $HOME"), "costs $$5 or $$HOME");
        assert_eq!(pi_value("a $$ b"), "a $$$$ b");
        // `${NAME}` means the same in both; `${env:NAME}` is VS Code's and Cursor's spelling of it.
        assert_eq!(pi_value("${TOKEN}"), "${TOKEN}");
        assert_eq!(pi_value("Bearer ${env:TOKEN}"), "Bearer ${TOKEN}");
        assert_eq!(pi_value("${input:key}"), "${input:key}");
        assert_eq!(pi_value("${env:broken"), "${env:broken");
        assert_eq!(pi_value("plain"), "plain");
        let home = home();
        let found = find_in(home.path(), &home.path().join(".config"), &[]);
        let calendar = by_id(&found, "claude-desktop:calendar");
        assert_eq!(calendar.config()["env"]["GREETING"], "$!hello");
        assert_eq!(calendar.config()["env"]["PRICE"], "costs $$5");
    }

    #[test]
    fn the_config_directory_and_a_missing_home() {
        let home = Path::new("/home/ana");
        assert_eq!(config_dir(home, None), PathBuf::from("/home/ana/.config"));
        assert_eq!(config_dir(home, Some("/data/conf".as_ref())), PathBuf::from("/data/conf"));
        assert_eq!(config_dir(home, Some("relative".as_ref())), PathBuf::from("/home/ana/.config"));
        assert_eq!(config_dir(home, Some("".as_ref())), PathBuf::from("/home/ana/.config"));
        // No `$HOME`: nothing is read, not even relative to the working directory.
        assert!(find(Path::new(""), &[]).is_empty());
        assert!(find_in(Path::new("relative"), Path::new("relative/.config"), &[]).is_empty());
    }

    /// Linux only: Claude Desktop and VS Code are read from the given config directory, not `~/.config`.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn claude_desktop_and_vscode_follow_the_config_directory() {
        let dir = tempfile::tempdir().unwrap();
        let (home, config) = (dir.path().join("home"), dir.path().join("xdg"));
        put(&config, PathBuf::from("Code/User/mcp.json"), VSCODE);
        put(&config, PathBuf::from("Claude/claude_desktop_config.json"), CLAUDE_DESKTOP);
        let found = find_in(&home, &config, &[]);
        assert!(found.iter().any(|f| f.id == "vscode:tracker"));
        assert!(found.iter().any(|f| f.id == "claude-desktop:calendar"));
        assert!(find_in(&home, &home.join(".config"), &[]).is_empty());
    }

    #[test]
    fn names_pi_refuses_are_written_under_a_valid_one() {
        assert_eq!(pi_name("my docs.v2"), "my-docs-v2");
        assert_eq!(pi_name("a  . b"), "a-b");
        assert_eq!(pi_name(".hidden."), "hidden");
        assert_eq!(pi_name("ok_name-1"), "ok_name-1");
        assert_eq!(pi_name("día"), "d-a");
        assert_eq!(pi_name("..."), "server");
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        put(
            home,
            PathBuf::from(".cursor/mcp.json"),
            r#"{"mcpServers": {"my docs.v2": {"command": "docs"}, "my-docs-v2": {"command": "other"}}}"#,
        );
        let (_aria, paths) = aria();
        let found = find_in(home, &home.join(".config"), &[]);
        let spaced = by_id(&found, "cursor:my docs.v2");
        assert_eq!((spaced.name.as_str(), spaced.import_as.as_str()), ("my docs.v2", "my-docs-v2"));
        let names = import(&paths, &found, &["cursor:my docs.v2".into(), "cursor:my-docs-v2".into()]).unwrap();
        // In the order found: the first takes the name, the one already called that gets `-2`.
        assert_eq!(names, vec!["my-docs-v2", "my-docs-v2-2"]);
        // Already in Aria's mcp.json under its written name: no longer offered.
        assert!(find_in(home, &home.join(".config"), &crate::mcp::servers(&paths)).is_empty());
    }
}
