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

const DEFAULT_SYSTEM: &str = "You are Aria, a personal desktop assistant. Keep answers short and direct.\n";

/// Where Pi works: `sandbox.projectsDir` with `~` expanded, `~/Projects` when there are no settings.
pub fn workdir(settings: Option<&Settings>) -> PathBuf {
    let configured = settings.map(|s| s.sandbox.projects_dir.as_str()).filter(|dir| !dir.is_empty()).unwrap_or("~/Projects");
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

/// Creates or refreshes the agent directory for the given settings (`None` before the setup).
pub fn prepare(paths: &AriaPaths, settings: Option<&Settings>) -> io::Result<()> {
    let agent = paths.agent();
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&agent)?;
    std::fs::set_permissions(&agent, std::fs::Permissions::from_mode(0o700))?;

    for (source, name) in [(paths.auth(), "auth.json"), (paths.mcp(), "mcp.json"), (paths.mcp_auth(), "mcp-auth.json")] {
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
    let original = read_json(&settings_path).ok().flatten().unwrap_or_else(|| Value::Object(Default::default()));
    write_json_atomic(&settings_path, &merge_preserving(original, ours), 0o600)?;

    let system = agent.join("SYSTEM.md");
    if !system.exists() {
        std::fs::write(&system, DEFAULT_SYSTEM)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths() -> (tempfile::TempDir, AriaPaths) {
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths { root: dir.path().to_path_buf() };
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
    fn leaves_an_existing_system_md_alone() {
        let (_dir, paths) = paths();
        std::fs::create_dir_all(paths.agent()).unwrap();
        std::fs::write(paths.agent().join("SYSTEM.md"), "custom").unwrap();
        prepare(&paths, None).unwrap();
        assert_eq!(std::fs::read_to_string(paths.agent().join("SYSTEM.md")).unwrap(), "custom");
        let (_dir2, paths2) = self::paths();
        prepare(&paths2, None).unwrap();
        assert!(std::fs::read_to_string(paths2.agent().join("SYSTEM.md")).unwrap().contains("Aria"));
    }

    #[test]
    fn the_agent_dir_is_private() {
        let (_dir, paths) = paths();
        prepare(&paths, None).unwrap();
        assert_eq!(std::fs::metadata(paths.agent()).unwrap().permissions().mode() & 0o777, 0o700);
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
