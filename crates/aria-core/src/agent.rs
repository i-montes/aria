//! The agent directory Pi runs with (`~/.aria/agent/`, via `PI_CODING_AGENT_DIR`). Aria regenerates
//! it from `settings.json`; it holds Pi's own `settings.json`, links to the credential files Aria
//! writes in `~/.aria/`, and the persona. Credentials are linked, never copied or read here.

use crate::json_file::{merge_preserving, read_json, write_json_atomic};
use crate::paths::AriaPaths;
use crate::settings::Settings;
use serde_json::{json, Value};
use std::io;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Path, PathBuf};

/// The provider for Claude Pro/Max sign-in (`claude-bridge`), pinned like Pi itself. It runs Claude Code through the Agent
/// SDK, uses Pi's Anthropic credential (OAuth = the subscription, an API key = per token) and is about 300 MB.
/// The packages the owner's extensions rely on (`pi-packages` in the repo), pinned; the installer installs them.
pub fn required_packages() -> Vec<&'static str> {
    include_str!("../../../pi-packages").lines().map(str::trim).filter(|line| !line.is_empty()).collect()
}

/// From superpowers the owner loads one skill and no extensions (as in his own Pi's settings); the settings'
/// `subagents.autoSkills` replace it.
const SUPERPOWERS_SKILLS: [&str; 1] = ["systematic-debugging"];

/// `pi install` lists a package as a plain string; the superpowers one becomes a filtered entry with no extensions
/// and only `skills` (names under the package's `skills/`), the ponytail one keeps its extension and gets two skills.
/// An entry filtered earlier gets the current skills.
fn with_package_filters(packages: Value, skills: &[String]) -> Value {
    let Value::Array(list) = packages else { return packages };
    let skills: Vec<String> = skills.iter().map(|name| format!("skills/{name}")).collect();
    Value::Array(
        list.into_iter()
            .map(|entry| {
                let source = entry.as_str().or_else(|| entry["source"].as_str()).map(str::to_string);
                match source {
                    Some(source) if source.starts_with("git:github.com/obra/superpowers") => {
                        json!({"source": source, "extensions": [], "skills": skills})
                    }
                    // Ponytail: its Pi extension and two of its skills (the audit, debt, gain and help ones are not Aria's).
                    Some(source) if source.starts_with("git:github.com/DietrichGebert/ponytail") => {
                        json!({"source": source, "skills": ["skills/ponytail", "skills/ponytail-review"]})
                    }
                    _ => entry,
                }
            })
            .collect(),
    )
}

fn listed_packages(paths: &AriaPaths) -> Vec<String> {
    read_json(&paths.agent().join("settings.json"))
        .ok()
        .flatten()
        .and_then(|value| value["packages"].as_array().cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|p| p.as_str().or_else(|| p["source"].as_str()).map(str::to_string))
        .collect()
}

/// Are the required packages listed in Pi's settings and present on disk?
pub fn packages_installed(paths: &AriaPaths) -> bool {
    let listed = listed_packages(paths);
    let agent = paths.agent();
    required_packages().iter().all(|wanted| {
        let present = if let Some(npm) = wanted.strip_prefix("npm:") {
            let name = npm.rsplit_once('@').map(|(name, _)| name).unwrap_or(npm);
            agent.join("npm/node_modules").join(name).is_dir()
        } else if let Some(git) = wanted.strip_prefix("git:") {
            let repo = git.split_once('@').map(|(repo, _)| repo).unwrap_or(git);
            agent.join("git").join(repo).is_dir()
        } else {
            false
        };
        present && listed.iter().any(|l| l == wanted)
    })
}

pub const CLAUDE_BRIDGE_PACKAGE: &str = "npm:pi-claude-agent-sdk@0.9.2";

/// The one-line persona earlier versions wrote; it is replaced by the owner's.
const PLACEHOLDER_SYSTEM: &str = "You are Aria, a personal desktop assistant. Keep answers short and direct.\n";

/// The owner's persona and experts (`agent-files/` in the repo), written into the agent directory.
static AGENT_FILES: include_dir::Dir = include_dir::include_dir!("$CARGO_MANIFEST_DIR/../../agent-files");

/// The owner's Pi extensions (`extensions/` in the repo), written into `agent/extensions/`.
static EXTENSIONS: include_dir::Dir = include_dir::include_dir!("$CARGO_MANIFEST_DIR/../../extensions");

/// Which files Aria wrote, with the sha256 of what it wrote: a file whose content still has that hash was not edited by
/// the owner and may follow a new version; any other content is the owner's and is kept.
const MARKER: &str = ".aria-files";

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn read_marker(agent: &Path) -> serde_json::Map<String, Value> {
    read_json(&agent.join(MARKER)).ok().flatten().and_then(|v| v.as_object().cloned()).unwrap_or_default()
}

/// Notes that Aria wrote `bytes` at `rel` (relative to the agent directory).
pub fn record_written(agent: &Path, rel: &str, bytes: &[u8]) -> io::Result<()> {
    let mut marker = read_marker(agent);
    marker.insert(rel.to_string(), json!(sha256_hex(bytes)));
    write_json_atomic(&agent.join(MARKER), &Value::Object(marker), 0o600)
}

/// Writes every file of `dir` under `agent/<prefix>`, except one the owner edited or deleted (see `MARKER`); returns the
/// edited files it kept, so the owner can be told. A file is written to a temp file and renamed (never through a symlink,
/// never truncated under a reader). Files whose name ends with `README.md` or `.test.ts` are not installed.
/// What Aria writes for one embedded file: its (rendered) contents, or nothing, in which case a copy Aria wrote earlier
/// is removed (a disabled expert) and the owner's own file is left alone.
pub enum Render {
    Write(Vec<u8>),
    Remove,
}

pub fn install_embedded(
    agent: &Path,
    dir: &include_dir::Dir,
    prefix: &str,
    extra_replaceable: &[&str],
) -> io::Result<Vec<String>> {
    install_rendered(agent, dir, prefix, extra_replaceable, &|_, bytes| Render::Write(bytes.to_vec()))
}

/// `install_embedded`, with each file's contents decided by `render(relative path, embedded bytes)`.
pub fn install_rendered(
    agent: &Path,
    dir: &include_dir::Dir,
    prefix: &str,
    extra_replaceable: &[&str],
    render: &dyn Fn(&str, &[u8]) -> Render,
) -> io::Result<Vec<String>> {
    let mut marker = read_marker(agent);
    let mut files = Vec::new();
    collect(dir, &mut files);
    let mut kept = Vec::new();
    let mut changed = false;
    for file in files {
        let rel_in_dir = file.path().to_string_lossy().into_owned();
        if rel_in_dir.ends_with("README.md") || rel_in_dir.ends_with(".test.ts") {
            continue;
        }
        let rel = if prefix.is_empty() { rel_in_dir } else { format!("{prefix}/{rel_in_dir}") };
        let target = agent.join(&rel);
        let rendered = render(&rel, file.contents());
        let recorded = marker.get(&rel).and_then(Value::as_str).map(str::to_string);
        let is_link = std::fs::symlink_metadata(&target).is_ok_and(|m| m.file_type().is_symlink());
        let current = if is_link { None } else { std::fs::read(&target).ok() };
        let wanted: &[u8] = match &rendered {
            Render::Write(bytes) => bytes,
            Render::Remove => {
                // Aria's own copy goes, and its record with it (so turning it on again writes it); the owner's stays.
                match &current {
                    Some(current) if recorded.as_deref() == Some(sha256_hex(current).as_str()) => {
                        std::fs::remove_file(&target)?;
                        marker.remove(&rel);
                        changed = true;
                    }
                    Some(_) => kept.push(rel),
                    None => {
                        changed |= marker.remove(&rel).is_some();
                    }
                }
                continue;
            }
        };
        let replaceable = match (&current, &recorded) {
            (Some(current), _) if current == wanted => {
                if recorded.as_deref() != Some(sha256_hex(wanted).as_str()) {
                    marker.insert(rel.clone(), json!(sha256_hex(wanted)));
                    changed = true;
                }
                continue;
            }
            // Aria wrote it once and it is gone now: the owner deleted it on purpose.
            (None, Some(_)) if !is_link => continue,
            (None, _) => true,
            (Some(current), recorded) => {
                let hash = sha256_hex(current);
                recorded.as_deref() == Some(hash.as_str()) || extra_replaceable.iter().any(|h| *h == hash)
            }
        };
        if !replaceable {
            kept.push(rel);
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = target.with_extension(format!("aria-{}.tmp", std::process::id()));
        std::fs::write(&tmp, wanted)?;
        std::fs::rename(&tmp, &target)?; // replaces a symlink itself, not what it points at
        marker.insert(rel, json!(sha256_hex(wanted)));
        changed = true;
    }
    if changed {
        write_json_atomic(&agent.join(MARKER), &Value::Object(marker), 0o600)?;
    }
    Ok(kept)
}

fn collect<'a>(dir: &'a include_dir::Dir<'a>, out: &mut Vec<&'a include_dir::File<'a>>) {
    for file in dir.files() {
        out.push(file);
    }
    for sub in dir.dirs() {
        collect(sub, out);
    }
}

/// Where Pi works: `sandbox.projectsDir` with `~` expanded, `~/Projects` when there are no settings.
pub fn workdir(settings: Option<&Settings>) -> PathBuf {
    let configured =
        settings.map(|s| s.sandbox.projects_dir.as_str()).filter(|dir| !dir.is_empty()).unwrap_or("~/Projects");
    let home = std::env::var("HOME").unwrap_or_default();
    match configured.strip_prefix('~') {
        Some(rest) => PathBuf::from(format!("{home}{rest}")),
        None => PathBuf::from(configured),
    }
}

/// `opencode-go/deepseek-v4.1-flash` -> `("opencode-go", "deepseek-v4.1-flash")`.
pub fn split_model(model: &str) -> Option<(&str, &str)> {
    model.split_once('/').filter(|(provider, id)| !provider.is_empty() && !id.is_empty())
}

fn link(target: &Path, at: &Path) -> io::Result<()> {
    match std::fs::symlink_metadata(at) {
        Ok(meta) if !meta.file_type().is_symlink() => return Ok(()), // a real file is the user's; leave it
        Ok(_) => std::fs::remove_file(at)?,
        Err(_) => {}
    }
    std::os::unix::fs::symlink(target, at)
}

/// Every session's file, whatever the working folder (Pi is started with `--session-dir` pointing here).
pub fn sessions_dir(paths: &AriaPaths) -> PathBuf {
    paths.agent().join("sessions").join("aria")
}

/// Pi used to keep sessions in one folder per working folder: move them into `sessions_dir`, never over a newer one.
fn gather_sessions(paths: &AriaPaths) -> io::Result<()> {
    let root = paths.agent().join("sessions");
    let target = sessions_dir(paths);
    std::fs::create_dir_all(&target)?;
    let Ok(folders) = std::fs::read_dir(&root) else { return Ok(()) };
    for folder in folders.filter_map(Result::ok) {
        let path = folder.path();
        if path == target || !path.is_dir() {
            continue;
        }
        for file in std::fs::read_dir(&path)?.filter_map(Result::ok) {
            let to = target.join(file.file_name());
            if file.path().extension().is_some_and(|e| e == "jsonl") && !to.exists() {
                std::fs::rename(file.path(), to)?;
            }
        }
        let _ = std::fs::remove_dir(&path); // only when it is empty now
    }
    Ok(())
}

/// Starts of several sessions can prepare at once; the files are written one at a time.
static PREPARE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Creates or refreshes the agent directory for the given settings (`None` before the setup). A Pi `settings.json` that
/// does not parse is an `InvalidData` error and is left as it is (it lists the installed packages).
pub fn prepare(paths: &AriaPaths, settings: Option<&Settings>) -> io::Result<()> {
    prepare_reporting(paths, settings).map(|_| ())
}

/// `prepare`, and the files Aria kept because the owner edited them (updates to them are not applied).
pub fn prepare_reporting(paths: &AriaPaths, settings: Option<&Settings>) -> io::Result<Vec<String>> {
    let mut kept = Vec::new();
    let _one_at_a_time = PREPARE.lock().unwrap_or_else(|e| e.into_inner());
    let agent = paths.agent();
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&agent)?;
    std::fs::set_permissions(&agent, std::fs::Permissions::from_mode(0o700))?;

    // Memory: Aria's Engram server in mcp.json (before mcp.json is linked below, so a first one is linked too).
    if let Some(settings) = settings {
        // An mcp.json that does not parse is left as it is (mcp.rs never replaces one) and the chat still starts.
        let _ = crate::mcp::sync_memory(paths, &settings.memory);
        // A memory project Engram does not know yet makes every lookup fail; it gets a first memory. A failure here
        // leaves the chat to start anyway.
        let _ = crate::mcp::seed_memory(paths, &settings.memory);
    }

    for (source, name) in [(paths.auth(), "auth.json"), (paths.mcp(), "mcp.json"), (paths.mcp_auth(), "mcp-auth.json")]
    {
        let at = agent.join(name);
        if source.exists() {
            link(&source, &at)?;
        } else if std::fs::symlink_metadata(&at).is_ok_and(|meta| meta.file_type().is_symlink()) {
            std::fs::remove_file(&at)?; // the source is gone; do not leave a dangling link
        }
    }

    let settings_path = agent.join("settings.json");
    let mut ours = json!({"quietStartup": true});
    if let Some((provider, model)) = settings.and_then(|s| s.models.daily.as_deref()).and_then(split_model) {
        ours["defaultProvider"] = json!(provider);
        ours["defaultModel"] = json!(model);
    }
    let mut original = match read_json(&settings_path) {
        Ok(found) => found.unwrap_or_else(|| Value::Object(Default::default())),
        Err(error) => return Err(io::Error::new(io::ErrorKind::InvalidData, error.to_string())),
    };
    if let Some(packages) = original.get_mut("packages") {
        let skills: Vec<String> = match settings {
            Some(settings) => settings.subagents.auto_skills.clone(),
            None => SUPERPOWERS_SKILLS.iter().map(|s| s.to_string()).collect(),
        };
        *packages = with_package_filters(packages.take(), &skills);
    }
    // A folder of Pi packages the setup names (integrations.packages, source "local") is listed once.
    if let Some(local) = settings.and_then(|s| local_packages(&s.integrations.packages)) {
        let list = original.as_object_mut().map(|o| o.entry("packages").or_insert_with(|| json!([])));
        if let Some(Value::Array(list)) = list {
            if !list.iter().any(|p| p.as_str() == Some(local.as_str())) {
                list.push(json!(local));
            }
        }
    }
    // The compaction share becomes per-model reserves Pi reads when it starts. The autocompact extension writes them too
    // (at a session start and for `/autocompact`) and notes the share it applied; Aria writes them only when the
    // settings ask for another share, so a change from the settings screen reaches the next Pi start.
    let compaction =
        settings.map(|s| s.compaction.percent).filter(|percent| applied_compaction(&agent) != Some(*percent));
    let compaction = compaction.and_then(|percent| context_windows(&agent).map(|models| (percent, models)));
    if let Some((percent, models)) = &compaction {
        original = with_compaction(original, models, *percent);
    }
    write_json_atomic(&settings_path, &merge_preserving(original, ours), 0o600)?;
    if let Some((percent, _)) = compaction {
        std::fs::write(agent.join(COMPACTION_APPLIED), format!("{percent}\n"))?;
    }

    // The extensions' settings (extensions/lib/config.ts); before the setup they keep the values in their code.
    let config_path = agent.join("aria.json");
    match settings {
        Some(settings) => write_json_atomic(&config_path, &crate::agent_config::extension_config(settings), 0o600)?,
        None => {
            let _ = std::fs::remove_file(&config_path);
        }
    }

    gather_sessions(paths)?;

    // The owner's persona and experts; one the owner edited is kept.
    // Rendered from the settings (persona.rs): the owner's text, with what a setting varies filled in.
    let owners = Settings::defaults("es");
    let chosen = settings.unwrap_or(&owners);
    kept.extend(install_rendered(
        &agent,
        &AGENT_FILES,
        "",
        &[&sha256_hex(PLACEHOLDER_SYSTEM.as_bytes())],
        &|rel, bytes| {
            let text = String::from_utf8_lossy(bytes);
            if rel == "SYSTEM.md" {
                return Render::Write(crate::persona::render_system(&text, chosen).into_bytes());
            }
            let name = rel.strip_prefix("agents/").and_then(|n| n.strip_suffix(".md"));
            match name.and_then(|n| chosen.subagents.experts.iter().find(|e| e.name == n)) {
                Some(expert) if !expert.enabled => Render::Remove,
                Some(expert) => Render::Write(crate::persona::render_expert(&text, expert).into_bytes()),
                None => Render::Write(bytes.to_vec()),
            }
        },
    )?);
    // His extensions, adapted to run in Aria (see extensions/README.md); one he edited is kept.
    kept.extend(install_embedded(&agent, &EXTENSIONS, "extensions", &[])?);
    Ok(kept)
}

/// The local package folder of the setup (`~` expanded), when it names one.
fn local_packages(packages: &crate::settings::Packages) -> Option<String> {
    let path = packages.path.trim();
    // "path" is what earlier versions wrote for a local folder.
    if !matches!(packages.source.as_str(), "local" | "path") || path.is_empty() {
        return None;
    }
    let home = std::env::var("HOME").unwrap_or_default();
    Some(path.strip_prefix("~/").map(|rest| format!("{home}/{rest}")).unwrap_or_else(|| path.to_string()))
}

/// The share of the context window the compaction reserves were last written for (extensions/autocompact writes it too).
const COMPACTION_APPLIED: &str = ".compaction-percent";
/// Pi's own smallest reserve (extensions/autocompact/autocompact.ts, MIN_RESERVE).
const MIN_RESERVE: u64 = 16384;

fn applied_compaction(agent: &Path) -> Option<u8> {
    std::fs::read_to_string(agent.join(COMPACTION_APPLIED)).ok()?.trim().parse().ok()
}

/// `provider/id` and context window of every model Pi knows (its `models-store.json`), or None without one.
fn context_windows(agent: &Path) -> Option<Vec<(String, u64)>> {
    let store = read_json(&agent.join("models-store.json")).ok().flatten()?;
    let mut models = Vec::new();
    for (provider, entry) in store.as_object()? {
        for model in entry["models"].as_array().into_iter().flatten() {
            if let (Some(id), Some(window)) = (model["id"].as_str(), model["contextWindow"].as_u64().filter(|w| *w > 0))
            {
                models.push((format!("{}/{id}", model["provider"].as_str().unwrap_or(provider)), window));
            }
        }
    }
    Some(models)
}

/// Pi's settings with the compaction of `percent`, as extensions/autocompact/autocompact.ts `applyPercent` writes it:
/// 0 turns it off; otherwise each model reserves what is left of its window past the share (never under Pi's minimum).
fn with_compaction(mut settings: Value, models: &[(String, u64)], percent: u8) -> Value {
    let Some(object) = settings.as_object_mut() else { return settings };
    let compaction = object.entry("compaction").or_insert_with(|| json!({}));
    if !compaction.is_object() {
        *compaction = json!({});
    }
    let compaction = compaction.as_object_mut().expect("an object");
    if percent == 0 {
        compaction.insert("enabled".into(), json!(false));
        return settings;
    }
    compaction.remove("enabled");
    let overrides = compaction.entry("modelOverrides").or_insert_with(|| json!({}));
    if !overrides.is_object() {
        *overrides = json!({});
    }
    let overrides = overrides.as_object_mut().expect("an object");
    for (key, window) in models {
        let reserve = MIN_RESERVE.max(window - window * u64::from(percent) / 100);
        let entry = overrides.entry(key.clone()).or_insert_with(|| json!({}));
        if !entry.is_object() {
            *entry = json!({});
        }
        entry["reserveTokens"] = json!(reserve);
    }
    settings
}

/// A fingerprint of what Pi reads only when it starts: the persona, the experts, Pi's `settings.json`, `mcp.json` and
/// the `pi` shim. When a settings change leaves it as it was, the running Pis need no restart (the extensions read
/// `aria.json` on every use).
pub fn start_fingerprint(paths: &AriaPaths) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let agent = paths.agent();
    let mut files = vec![agent.join("SYSTEM.md"), agent.join("settings.json"), paths.mcp(), paths.bin().join("pi")];
    if let Ok(entries) = std::fs::read_dir(agent.join("agents")) {
        let mut experts: Vec<PathBuf> = entries.filter_map(Result::ok).map(|e| e.path()).collect();
        experts.sort();
        files.extend(experts);
    }
    for file in files {
        file.hash(&mut hasher);
        std::fs::read(&file).ok().hash(&mut hasher);
    }
    hasher.finish()
}

/// Is the Claude provider listed in Pi's settings and downloaded?
pub fn claude_bridge_installed(paths: &AriaPaths) -> bool {
    let listed = read_json(&paths.agent().join("settings.json"))
        .ok()
        .flatten()
        .and_then(|value| value["packages"].as_array().cloned())
        .is_some_and(|packages| packages.iter().any(|p| p.as_str().is_some_and(|p| p.contains("pi-claude-agent-sdk"))));
    listed && paths.agent().join("npm/node_modules/pi-claude-agent-sdk").is_dir()
}

/// What to run to install the Claude provider: `pi install` with Aria's agent directory, so Pi downloads the package and
/// lists it in its own settings.
#[derive(Debug, Clone, PartialEq)]
pub struct InstallCommand {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    /// Names removed from the inherited environment (`pi::unset_for_pi`).
    pub unset: Vec<String>,
    pub cwd: PathBuf,
}

pub fn claude_bridge_install_command(
    paths: &AriaPaths,
    install: &crate::pi::PiInstall,
    path_var: &str,
) -> InstallCommand {
    InstallCommand {
        program: install.binary.clone(),
        args: vec!["install".into(), CLAUDE_BRIDGE_PACKAGE.into()],
        env: vec![
            ("PI_CODING_AGENT_DIR".into(), paths.agent().display().to_string()),
            ("PATH".into(), crate::pi::path_with_bin(paths, path_var)),
            ("PI_SKIP_VERSION_CHECK".into(), "1".into()),
        ],
        unset: crate::pi::unset_for_pi(std::env::vars()),
        cwd: paths.agent(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths() -> (tempfile::TempDir, AriaPaths) {
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths::new(dir.path().to_path_buf());
        (dir, paths)
    }

    fn settings_with_daily(model: &str) -> Settings {
        let mut settings = Settings::default();
        settings.models.daily = Some(model.into());
        settings
    }

    #[test]
    fn links_credentials_that_exist() {
        let (_dir, paths) = paths();
        std::fs::write(paths.auth(), "{}").unwrap();
        prepare(&paths, None).unwrap();
        let link = paths.agent().join("auth.json");
        assert_eq!(std::fs::read_link(&link).unwrap(), paths.auth());
    }

    #[test]
    fn does_not_link_missing_files() {
        let (_dir, paths) = paths();
        prepare(&paths, None).unwrap();
        assert!(std::fs::symlink_metadata(paths.agent().join("mcp.json")).is_err());
    }

    #[test]
    fn removes_a_link_whose_source_disappeared() {
        let (_dir, paths) = paths();
        std::fs::write(paths.mcp(), "{}").unwrap();
        prepare(&paths, None).unwrap();
        std::fs::remove_file(paths.mcp()).unwrap();
        prepare(&paths, None).unwrap();
        assert!(std::fs::symlink_metadata(paths.agent().join("mcp.json")).is_err());
    }

    #[test]
    fn replaces_a_stale_link_but_keeps_a_real_file() {
        let (_dir, paths) = paths();
        std::fs::write(paths.auth(), "{}").unwrap();
        std::fs::create_dir_all(paths.agent()).unwrap();
        std::os::unix::fs::symlink("/nonexistent", paths.agent().join("auth.json")).unwrap();
        std::fs::write(paths.agent().join("mcp.json"), "mine").unwrap();
        std::fs::write(paths.mcp(), "{}").unwrap();
        prepare(&paths, None).unwrap();
        assert_eq!(std::fs::read_link(paths.agent().join("auth.json")).unwrap(), paths.auth());
        assert_eq!(std::fs::read_to_string(paths.agent().join("mcp.json")).unwrap(), "mine");
    }

    #[test]
    fn sets_default_provider_and_model_from_the_daily_model() {
        let (_dir, paths) = paths();
        prepare(&paths, Some(&settings_with_daily("opencode-go/deepseek-v4.1-flash"))).unwrap();
        let value = read_json(&paths.agent().join("settings.json")).unwrap().unwrap();
        assert_eq!(value["defaultProvider"], "opencode-go");
        assert_eq!(value["defaultModel"], "deepseek-v4.1-flash");
        assert_eq!(value["quietStartup"], true);
    }

    #[test]
    fn keeps_unknown_keys_in_pi_settings() {
        let (_dir, paths) = paths();
        std::fs::create_dir_all(paths.agent()).unwrap();
        std::fs::write(paths.agent().join("settings.json"), r#"{"packages":["npm:x"],"defaultModel":"old"}"#).unwrap();
        prepare(&paths, Some(&settings_with_daily("anthropic/claude-sonnet-5-5"))).unwrap();
        let value = read_json(&paths.agent().join("settings.json")).unwrap().unwrap();
        assert_eq!(value["packages"], json!(["npm:x"]));
        assert_eq!(value["defaultModel"], "claude-sonnet-5-5");
    }

    #[test]
    fn writes_the_owners_persona_and_experts() {
        let (_dir, paths) = paths();
        prepare(&paths, None).unwrap();
        let system = std::fs::read_to_string(paths.agent().join("SYSTEM.md")).unwrap();
        assert_eq!(system, include_str!("../tests/golden/SYSTEM.owner.md"));
        assert!(system.contains("No tables"), "the answer-style rule is there");
        for expert in ["frontend", "backend", "security", "reviewer", "researcher"] {
            assert!(paths.agent().join("agents").join(format!("{expert}.md")).is_file(), "{expert}");
        }
    }

    #[test]
    fn replaces_the_placeholder_persona_of_earlier_versions() {
        let (_dir, paths) = paths();
        std::fs::create_dir_all(paths.agent()).unwrap();
        std::fs::write(paths.agent().join("SYSTEM.md"), PLACEHOLDER_SYSTEM).unwrap();
        prepare(&paths, None).unwrap();
        assert_eq!(
            std::fs::read_to_string(paths.agent().join("SYSTEM.md")).unwrap(),
            include_str!("../tests/golden/SYSTEM.owner.md")
        );
    }

    #[test]
    fn keeps_a_persona_the_owner_edited() {
        let (_dir, paths) = paths();
        prepare(&paths, None).unwrap();
        std::fs::write(paths.agent().join("SYSTEM.md"), "my own rules").unwrap();
        std::fs::write(paths.agent().join("agents/security.md"), "my security expert").unwrap();
        prepare(&paths, None).unwrap();
        assert_eq!(std::fs::read_to_string(paths.agent().join("SYSTEM.md")).unwrap(), "my own rules");
        assert_eq!(std::fs::read_to_string(paths.agent().join("agents/security.md")).unwrap(), "my security expert");
    }

    #[test]
    fn the_install_skill_and_extension_are_written_into_pis_folders() {
        let (_dir, paths) = paths();
        prepare(&paths, None).unwrap();
        let skill = paths.agent().join("skills/install-from-link/SKILL.md");
        assert_eq!(
            std::fs::read_to_string(skill).unwrap(),
            include_str!("../../../agent-files/skills/install-from-link/SKILL.md")
        );
        assert!(paths.agent().join("extensions/install/index.ts").exists());
        assert!(!paths.agent().join("extensions/install/install.test.ts").exists());
    }

    #[test]
    fn the_browser_skill_is_written_into_pis_skills() {
        let (_dir, paths) = paths();
        prepare(&paths, None).unwrap();
        let skill = paths.agent().join("skills/verify-in-browser/SKILL.md");
        assert_eq!(
            std::fs::read_to_string(skill).unwrap(),
            include_str!("../../../agent-files/skills/verify-in-browser/SKILL.md")
        );
    }

    #[test]
    fn the_slice_skill_is_written_into_pis_skills_and_hidden_from_the_model() {
        let (_dir, paths) = paths();
        prepare(&paths, None).unwrap();
        let skill = std::fs::read_to_string(paths.agent().join("skills/slice/SKILL.md")).unwrap();
        assert_eq!(skill, include_str!("../../../agent-files/skills/slice/SKILL.md"));
        let front_matter = skill.split("---").nth(1).unwrap();
        assert!(front_matter.lines().any(|line| line == "disable-model-invocation: true"));
    }

    #[test]
    fn a_file_nobody_edited_follows_a_new_version_of_aria() {
        let (_dir, paths) = paths();
        std::fs::create_dir_all(paths.agent().join("agents")).unwrap();
        // what an older Aria wrote, and recorded as its own
        std::fs::write(paths.agent().join("agents/reviewer.md"), "old reviewer").unwrap();
        record_written(&paths.agent(), "agents/reviewer.md", b"old reviewer").unwrap();
        prepare(&paths, None).unwrap();
        assert_eq!(
            std::fs::read_to_string(paths.agent().join("agents/reviewer.md")).unwrap(),
            include_str!("../../../agent-files/agents/reviewer.md")
        );
    }

    #[test]
    fn unpacks_the_owners_extensions_without_their_tests() {
        let (_dir, paths) = paths();
        prepare(&paths, None).unwrap();
        let extensions = paths.agent().join("extensions");
        for name in
            ["jev-router", "sandbox", "quota-fallback", "autocompact", "ask-user", "subagent-mode", "subagent-watch"]
        {
            assert!(extensions.join(name).join("index.ts").is_file(), "{name}");
        }
        assert!(extensions.join("lib/agent-dir.ts").is_file(), "the shared helper");
        assert!(extensions.join("subagent-mode/SUBAGENT.md").is_file(), "the files an extension reads");
        assert!(!extensions.join("sandbox/policy.test.ts").exists(), "no tests");
        assert!(extensions.join("rotate/index.ts").is_file(), "Aria's own rotation command");
        assert!(!extensions.join("rotate/seed.test.ts").exists(), "no tests");
        assert!(!extensions.join("README.md").exists(), "no readme");
    }

    #[test]
    fn keeps_an_extension_the_owner_edited() {
        let (_dir, paths) = paths();
        prepare(&paths, None).unwrap();
        let file = paths.agent().join("extensions/autocompact/index.ts");
        std::fs::write(&file, "// my change").unwrap();
        prepare(&paths, None).unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "// my change");
    }

    #[test]
    fn the_required_packages_come_from_the_repo_file_and_are_pinned() {
        assert_eq!(
            required_packages(),
            vec![
                "npm:pi-parallel-agents@0.2.0",
                "git:github.com/obra/superpowers@bb92a77741419a4ab5f06e711a283343f1ada0c3",
                "git:github.com/DietrichGebert/ponytail@09c434ab73ca43761c5d8b0c5d341926c2fba710"
            ]
        );
        assert!(required_packages().iter().all(|p| p.contains('@')));
    }

    #[test]
    fn a_git_package_is_pinned_to_a_full_commit() {
        // `git fetch origin <short sha>` fails on an existing clone: every update would break.
        for package in required_packages().iter().filter(|p| p.starts_with("git:")) {
            let sha = package.rsplit_once('@').unwrap().1;
            assert!(sha.len() == 40 && sha.chars().all(|c| c.is_ascii_hexdigit()), "{package}");
        }
    }

    #[test]
    fn prepare_gives_the_superpowers_package_the_owners_filter() {
        let (_dir, paths) = paths();
        std::fs::create_dir_all(paths.agent()).unwrap();
        std::fs::write(paths.agent().join("settings.json"), r#"{"packages":["npm:pi-parallel-agents@0.2.0","git:github.com/obra/superpowers@bb92a77741419a4ab5f06e711a283343f1ada0c3","npm:pi-claude-agent-sdk@0.9.2"]}"#).unwrap();
        prepare(&paths, None).unwrap();
        let packages = read_json(&paths.agent().join("settings.json")).unwrap().unwrap()["packages"].clone();
        assert_eq!(packages[0], json!("npm:pi-parallel-agents@0.2.0"));
        assert_eq!(
            packages[1],
            json!({"source": "git:github.com/obra/superpowers@bb92a77741419a4ab5f06e711a283343f1ada0c3", "extensions": [], "skills": ["skills/systematic-debugging"]})
        );
        assert_eq!(packages[2], json!("npm:pi-claude-agent-sdk@0.9.2"));
        // and again: the object stays as it is
        prepare(&paths, None).unwrap();
        assert_eq!(read_json(&paths.agent().join("settings.json")).unwrap().unwrap()["packages"], packages);
    }

    #[test]
    fn prepare_gives_the_ponytail_package_its_two_skills() {
        let (_dir, paths) = paths();
        std::fs::create_dir_all(paths.agent()).unwrap();
        std::fs::write(paths.agent().join("settings.json"), r#"{"packages":["npm:pi-parallel-agents@0.2.0","git:github.com/DietrichGebert/ponytail@09c434ab73ca43761c5d8b0c5d341926c2fba710"]}"#).unwrap();
        prepare(&paths, None).unwrap();
        let packages = read_json(&paths.agent().join("settings.json")).unwrap().unwrap()["packages"].clone();
        assert_eq!(packages[0], json!("npm:pi-parallel-agents@0.2.0"));
        // no `extensions` key: Pi loads the package's extension when it is absent
        assert_eq!(
            packages[1],
            json!({"source": "git:github.com/DietrichGebert/ponytail@09c434ab73ca43761c5d8b0c5d341926c2fba710", "skills": ["skills/ponytail", "skills/ponytail-review"]})
        );
        prepare(&paths, None).unwrap();
        assert_eq!(read_json(&paths.agent().join("settings.json")).unwrap().unwrap()["packages"], packages);
    }

    #[test]
    fn the_packages_are_installed_when_settings_list_them_and_their_files_exist() {
        let (_dir, paths) = paths();
        prepare(&paths, None).unwrap();
        assert!(!packages_installed(&paths));
        std::fs::write(paths.agent().join("settings.json"), r#"{"packages":["npm:pi-parallel-agents@0.2.0","git:github.com/obra/superpowers@bb92a77741419a4ab5f06e711a283343f1ada0c3","git:github.com/DietrichGebert/ponytail@09c434ab73ca43761c5d8b0c5d341926c2fba710"]}"#).unwrap();
        std::fs::create_dir_all(paths.agent().join("npm/node_modules/pi-parallel-agents")).unwrap();
        assert!(!packages_installed(&paths), "superpowers is not cloned yet");
        std::fs::create_dir_all(paths.agent().join("git/github.com/obra/superpowers/skills")).unwrap();
        assert!(!packages_installed(&paths), "ponytail is not cloned yet");
        std::fs::create_dir_all(paths.agent().join("git/github.com/DietrichGebert/ponytail/skills")).unwrap();
        assert!(packages_installed(&paths));
    }

    #[test]
    fn the_superpowers_filter_lists_the_automatic_skills_from_the_settings() {
        let packages = json!(["git:github.com/obra/superpowers@abc"]);
        let skills = vec!["brainstorming".to_string(), "systematic-debugging".to_string()];
        assert_eq!(
            with_package_filters(packages, &skills),
            json!([{"source": "git:github.com/obra/superpowers@abc", "extensions": [], "skills": ["skills/brainstorming", "skills/systematic-debugging"]}])
        );
    }

    #[test]
    fn prepare_writes_aria_json_and_removes_it_without_settings() {
        let (_dir, paths) = paths();
        let settings: Settings =
            serde_json::from_str(include_str!("../tests/golden/settings-provider-opencode-go.json")).unwrap();
        prepare(&paths, Some(&settings)).unwrap();
        let file = paths.agent().join("aria.json");
        let written: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        assert_eq!(written["quota"]["switchAt"], 98);
        assert_eq!(std::fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600);
        prepare(&paths, None).unwrap();
        assert!(!file.exists(), "before the setup the extensions keep their own values");
    }

    fn owners_settings() -> Settings {
        let mut settings: Settings =
            serde_json::from_str(include_str!("../tests/golden/settings-provider-opencode-go.json")).unwrap();
        settings.assistant.language = "es".into();
        settings
    }

    /// Pi's model store with two models (windows of 200k and 1M tokens).
    fn store_models(paths: &AriaPaths) {
        std::fs::create_dir_all(paths.agent()).unwrap();
        let store = json!({
            "opencode-go": {"models": [{"id": "glm-5.3", "provider": "opencode-go", "contextWindow": 200000}]},
            "anthropic": {"models": [{"id": "claude-fable-5-1", "contextWindow": 1000000}, {"id": "no-window"}]},
        });
        std::fs::write(paths.agent().join("models-store.json"), store.to_string()).unwrap();
    }

    fn pi_settings(paths: &AriaPaths) -> Value {
        serde_json::from_str(&std::fs::read_to_string(paths.agent().join("settings.json")).unwrap()).unwrap()
    }

    #[test]
    fn a_new_compaction_share_is_written_as_pis_reserves_before_it_starts() {
        let (_dir, paths) = paths();
        store_models(&paths);
        let mut settings = owners_settings();
        settings.compaction.percent = 70;
        prepare(&paths, Some(&settings)).unwrap();
        let overrides = &pi_settings(&paths)["compaction"]["modelOverrides"];
        assert_eq!(overrides["opencode-go/glm-5.3"]["reserveTokens"], 60000);
        assert_eq!(overrides["anthropic/claude-fable-5-1"]["reserveTokens"], 300000);
        assert!(overrides.get("anthropic/no-window").is_none());
        assert_eq!(std::fs::read_to_string(paths.agent().join(COMPACTION_APPLIED)).unwrap().trim(), "70");

        settings.compaction.percent = 0;
        prepare(&paths, Some(&settings)).unwrap();
        assert_eq!(pi_settings(&paths)["compaction"]["enabled"], false);
    }

    #[test]
    fn a_share_already_applied_leaves_pis_reserves_alone() {
        // `/autocompact` (or a session start) wrote the reserves and noted the share: Aria does not write them again,
        // so the running Pis need no restart.
        let (_dir, paths) = paths();
        store_models(&paths);
        std::fs::write(
            paths.agent().join("settings.json"),
            r#"{"compaction":{"modelOverrides":{"x/y":{"reserveTokens":5}}}}"#,
        )
        .unwrap();
        std::fs::write(paths.agent().join(COMPACTION_APPLIED), "40\n").unwrap();
        prepare(&paths, Some(&owners_settings())).unwrap();
        assert_eq!(pi_settings(&paths)["compaction"]["modelOverrides"], json!({"x/y": {"reserveTokens": 5}}));
    }

    #[test]
    fn memory_on_gives_arias_pi_its_engram_server() {
        let (_dir, paths) = paths();
        prepare(&paths, Some(&owners_settings())).unwrap();
        let mcp: Value =
            serde_json::from_str(&std::fs::read_to_string(paths.agent().join("mcp.json")).unwrap()).unwrap();
        assert_eq!(mcp["mcpServers"]["engram"]["args"][2], "aria");
    }

    #[test]
    fn a_local_package_folder_from_the_settings_is_listed_once() {
        let (_dir, paths) = paths();
        let mut settings = owners_settings();
        settings.integrations.packages.source = "local".into();
        settings.integrations.packages.path = "/srv/pi-packages/meeting".into();
        prepare(&paths, Some(&settings)).unwrap();
        prepare(&paths, Some(&settings)).unwrap();
        let pi: Value =
            serde_json::from_str(&std::fs::read_to_string(paths.agent().join("settings.json")).unwrap()).unwrap();
        let listed =
            pi["packages"].as_array().unwrap().iter().filter(|p| p == &&json!("/srv/pi-packages/meeting")).count();
        assert_eq!(listed, 1);
    }

    #[test]
    fn the_persona_is_rendered_from_the_settings() {
        let (_dir, paths) = paths();
        let mut settings = owners_settings();
        prepare(&paths, Some(&settings)).unwrap();
        let system = std::fs::read_to_string(paths.agent().join("SYSTEM.md")).unwrap();
        assert_eq!(
            system,
            include_str!("../tests/golden/SYSTEM.owner.md"),
            "the owner's settings give his own SYSTEM.md"
        );
        settings.assistant.name = "Nova".into();
        prepare(&paths, Some(&settings)).unwrap();
        assert!(
            std::fs::read_to_string(paths.agent().join("SYSTEM.md")).unwrap().starts_with("# Nova"),
            "a change is written again"
        );
    }

    #[test]
    fn turning_an_expert_off_and_on_brings_its_file_back() {
        let (_dir, paths) = paths();
        let mut settings = owners_settings();
        prepare(&paths, Some(&settings)).unwrap();
        let file = paths.agent().join("agents/backend.md");
        assert!(file.exists());
        settings.subagents.experts.iter_mut().filter(|e| e.name == "backend").for_each(|e| e.enabled = false);
        prepare(&paths, Some(&settings)).unwrap();
        assert!(!file.exists(), "a disabled expert is not offered");
        settings.subagents.experts.iter_mut().filter(|e| e.name == "backend").for_each(|e| e.enabled = true);
        prepare(&paths, Some(&settings)).unwrap();
        assert!(file.exists(), "enabled again, it comes back");
    }

    #[test]
    fn a_read_only_setting_changes_the_experts_tools() {
        let (_dir, paths) = paths();
        let mut settings = owners_settings();
        settings.subagents.experts.iter_mut().filter(|e| e.name == "frontend").for_each(|e| e.read_only = true);
        prepare(&paths, Some(&settings)).unwrap();
        assert!(std::fs::read_to_string(paths.agent().join("agents/frontend.md"))
            .unwrap()
            .contains("tools: read, bash, codemode"));
    }

    #[test]
    fn an_expert_the_owner_edited_is_still_kept_when_it_is_turned_off() {
        let (_dir, paths) = paths();
        let mut settings = owners_settings();
        prepare(&paths, Some(&settings)).unwrap();
        let file = paths.agent().join("agents/backend.md");
        std::fs::write(&file, "my own backend").unwrap();
        settings.subagents.experts.iter_mut().filter(|e| e.name == "backend").for_each(|e| e.enabled = false);
        prepare(&paths, Some(&settings)).unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "my own backend");
    }

    #[test]
    fn a_file_the_owner_deleted_stays_deleted() {
        let (_dir, paths) = paths();
        prepare(&paths, None).unwrap();
        std::fs::remove_file(paths.agent().join("agents/backend.md")).unwrap();
        prepare(&paths, None).unwrap();
        assert!(!paths.agent().join("agents/backend.md").exists());
    }

    #[test]
    fn prepare_says_which_edited_files_it_kept() {
        let (_dir, paths) = paths();
        prepare(&paths, None).unwrap();
        std::fs::write(paths.agent().join("extensions/sandbox/policy.ts"), "// weakened").unwrap();
        let kept = prepare_reporting(&paths, None).unwrap();
        assert_eq!(kept, vec!["extensions/sandbox/policy.ts".to_string()]);
        assert!(prepare_reporting(&paths, None).unwrap().len() == 1, "said every time, while it lasts");
    }

    #[test]
    fn a_symlink_in_the_way_is_replaced_not_followed() {
        let (dir, paths) = paths();
        prepare(&paths, None).unwrap();
        let outside = dir.path().join("outside.md");
        let target = paths.agent().join("agents/reviewer.md");
        std::fs::remove_file(&target).unwrap();
        // the owner did not delete it: Aria's marker entry is gone too, as on a fresh directory
        std::fs::remove_file(paths.agent().join(".aria-files")).unwrap();
        std::os::unix::fs::symlink(&outside, &target).unwrap();
        prepare(&paths, None).unwrap();
        assert!(!outside.exists(), "nothing written through the link");
        assert!(std::fs::symlink_metadata(&target).unwrap().file_type().is_file());
    }

    #[test]
    fn the_marker_lists_what_aria_wrote() {
        let (_dir, paths) = paths();
        prepare(&paths, None).unwrap();
        let marker = read_json(&paths.agent().join(".aria-files")).unwrap().unwrap();
        assert!(marker["SYSTEM.md"].as_str().is_some_and(|h| h.len() == 64));
        assert!(marker["agents/frontend.md"].is_string());
    }

    #[test]
    fn the_agent_dir_is_private() {
        let (_dir, paths) = paths();
        prepare(&paths, None).unwrap();
        assert_eq!(std::fs::metadata(paths.agent()).unwrap().permissions().mode() & 0o777, 0o700);
    }

    #[test]
    fn the_claude_provider_is_installed_when_settings_list_it_and_its_files_exist() {
        let (_dir, paths) = paths();
        assert!(!claude_bridge_installed(&paths));
        prepare(&paths, None).unwrap();
        std::fs::write(paths.agent().join("settings.json"), format!(r#"{{"packages":["{CLAUDE_BRIDGE_PACKAGE}"]}}"#))
            .unwrap();
        assert!(!claude_bridge_installed(&paths), "listed but not downloaded");
        std::fs::create_dir_all(paths.agent().join("npm/node_modules/pi-claude-agent-sdk")).unwrap();
        assert!(claude_bridge_installed(&paths));
        std::fs::write(paths.agent().join("settings.json"), r#"{"packages":[]}"#).unwrap();
        assert!(!claude_bridge_installed(&paths), "downloaded but not listed");
    }

    #[test]
    fn the_install_command_runs_the_pinned_pi_in_aria_s_agent_dir_with_the_shims_first() {
        let (_dir, paths) = paths();
        let install = crate::pi::PiInstall {
            dir: paths.pi_install("1.3.0"),
            binary: paths.pi_install("1.3.0").join("node_modules/.bin/pi"),
        };
        let command = claude_bridge_install_command(&paths, &install, "/usr/bin");
        assert_eq!(command.program, install.binary);
        assert_eq!(command.args, vec!["install".to_string(), CLAUDE_BRIDGE_PACKAGE.to_string()]);
        let env = |key: &str| command.env.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone());
        assert_eq!(env("PI_CODING_AGENT_DIR"), Some(paths.agent().display().to_string()));
        assert_eq!(env("PATH"), Some(format!("{}:/usr/bin", paths.bin().display())));
        assert_eq!(env("PI_SKIP_VERSION_CHECK"), Some("1".to_string()));
    }

    #[test]
    fn the_install_command_drops_anthropic_variables() {
        std::env::set_var("ANTHROPIC_ARIA_INSTALL_TEST", "x");
        let (_dir, paths) = paths();
        let install = crate::pi::PiInstall {
            dir: paths.pi_install("1.3.0"),
            binary: paths.pi_install("1.3.0").join("node_modules/.bin/pi"),
        };
        let command = claude_bridge_install_command(&paths, &install, "/usr/bin");
        assert!(command.unset.contains(&"ANTHROPIC_ARIA_INSTALL_TEST".to_string()), "{:?}", command.unset);
    }

    #[test]
    fn preparing_keeps_the_packages_pi_installed() {
        let (_dir, paths) = paths();
        prepare(&paths, None).unwrap();
        std::fs::write(paths.agent().join("settings.json"), format!(r#"{{"packages":["{CLAUDE_BRIDGE_PACKAGE}"]}}"#))
            .unwrap();
        prepare(&paths, Some(&settings_with_daily("opencode-go/glm-5.3"))).unwrap();
        let value = read_json(&paths.agent().join("settings.json")).unwrap().unwrap();
        assert_eq!(value["packages"], json!([CLAUDE_BRIDGE_PACKAGE]));
    }

    #[test]
    fn sessions_live_in_one_folder_and_older_ones_move_there() {
        let (_dir, paths) = paths();
        let old = paths.agent().join("sessions").join("--home-u-Projects--");
        std::fs::create_dir_all(&old).unwrap();
        std::fs::write(old.join("2026-10-18T00-00-00-000Z_tab-1.jsonl"), "{}").unwrap();
        prepare(&paths, None).unwrap();
        assert_eq!(sessions_dir(&paths), paths.agent().join("sessions").join("aria"));
        assert!(
            sessions_dir(&paths).join("2026-10-18T00-00-00-000Z_tab-1.jsonl").exists(),
            "moved, so the tab keeps its history"
        );
        assert!(!old.join("2026-10-18T00-00-00-000Z_tab-1.jsonl").exists());
    }

    #[test]
    fn a_session_already_in_the_folder_is_not_overwritten_by_an_old_copy() {
        let (_dir, paths) = paths();
        std::fs::create_dir_all(sessions_dir(&paths)).unwrap();
        std::fs::write(sessions_dir(&paths).join("a_tab.jsonl"), "new").unwrap();
        let old = paths.agent().join("sessions").join("--x--");
        std::fs::create_dir_all(&old).unwrap();
        std::fs::write(old.join("a_tab.jsonl"), "old").unwrap();
        prepare(&paths, None).unwrap();
        assert_eq!(std::fs::read_to_string(sessions_dir(&paths).join("a_tab.jsonl")).unwrap(), "new");
    }

    #[test]
    fn a_broken_pi_settings_file_is_reported_not_replaced() {
        let (_dir, paths) = paths();
        std::fs::create_dir_all(paths.agent()).unwrap();
        std::fs::write(paths.agent().join("settings.json"), "{ broken").unwrap();
        let error = prepare(&paths, None).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(std::fs::read_to_string(paths.agent().join("settings.json")).unwrap(), "{ broken");
    }

    #[test]
    fn many_starts_at_once_all_prepare_fine() {
        let (_dir, paths) = paths();
        std::fs::write(paths.auth(), "{}").unwrap();
        let settings = settings_with_daily("opencode-go/glm-5.3");
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let (paths, settings) = (paths.clone(), settings.clone());
                std::thread::spawn(move || prepare(&paths, Some(&settings)).map_err(|e| e.to_string()))
            })
            .collect();
        for handle in handles {
            handle.join().unwrap().unwrap();
        }
        assert_eq!(read_json(&paths.agent().join("settings.json")).unwrap().unwrap()["defaultModel"], "glm-5.3");
    }

    #[test]
    fn workdir_expands_tilde_and_defaults() {
        let home = std::env::var("HOME").unwrap();
        assert_eq!(workdir(None), PathBuf::from(format!("{home}/Projects")));
        let mut settings = Settings::default();
        settings.sandbox.projects_dir = "~/Work".into();
        assert_eq!(workdir(Some(&settings)), PathBuf::from(format!("{home}/Work")));
        settings.sandbox.projects_dir = "/srv/code".into();
        assert_eq!(workdir(Some(&settings)), PathBuf::from("/srv/code"));
    }

    #[test]
    fn model_ids_split_at_the_first_slash() {
        assert_eq!(split_model("opencode-go/glm-5.3"), Some(("opencode-go", "glm-5.3")));
        assert_eq!(split_model("claude-bridge/claude-opus-5-5"), Some(("claude-bridge", "claude-opus-5-5")));
        assert_eq!(split_model("nope"), None);
        assert_eq!(split_model("/x"), None);
    }
}
