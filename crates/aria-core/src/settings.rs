//! `~/.aria/settings.json`: every setting of the project in one document (see docs/settings.md).
//! Writing keeps keys Aria does not know, and a file that cannot be parsed is never overwritten.

use crate::json_file::{merge_preserving, read_json, write_json_atomic, JsonFileError};
use crate::paths::AriaPaths;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub const SETTINGS_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error(transparent)]
    File(#[from] JsonFileError),
    #[error("settings do not have the expected shape: {0}")]
    Invalid(String),
    #[error("could not write settings: {0}")]
    Write(#[from] std::io::Error),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Settings {
    pub version: u32,
    pub setup: Setup,
    pub provider: ProviderSettings,
    pub router: Router,
    pub assistant: Assistant,
    pub models: Models,
    pub effort: Effort,
    pub compaction: Compaction,
    pub memory: Memory,
    pub integrations: Integrations,
    pub sandbox: SandboxSettings,
    pub git: Git,
    pub subagents: Subagents,
    /// The dock's shortcut, `MOD + MOD + KEY` (see `shortcut`); `""` is the platform's default.
    pub shortcut: String,
    pub credentials: Credentials,
}

/// Where keys live: in the system keyring (Keychain, Secret Service), or in plain files in `~/.aria` when `keyring` is
/// off (docs/superpowers/specs/2026-10-21-keyring-design.md).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Credentials {
    pub keyring: bool,
}

impl Default for Credentials {
    fn default() -> Self {
        Credentials { keyring: true }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Setup {
    pub completed: bool,
    /// "quick" | "full" | "skipped" | null
    pub mode: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ProviderSettings {
    /// "opencode-go" | "anthropic-api" | "anthropic-oauth" | null
    pub id: Option<String>,
    pub quota: Option<Quota>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Quota {
    pub switch_at: u8,
    pub plan_switch_at: u8,
    pub free_model: String,
    pub ask_first: bool,
    pub countdown_minutes: u32,
    /// "switch" | "stay"
    pub on_no_answer: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Router {
    pub enabled: bool,
    /// "opencode" | "typesafe"
    pub backend: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Assistant {
    pub user_name: String,
    pub name: String,
    /// A code (`es`, `en`, `pt`) or free text.
    pub language: String,
    /// "very-short" | "brief" | "detailed" | "technical" | "teaching"
    pub style: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Models {
    pub daily: Option<String>,
    pub planning: Option<String>,
    pub planning_effort: String,
    pub allow_pay_per_use: bool,
    /// A run that gets nothing from Pi for this many minutes is stopped (0 turns the watchdog off).
    pub turn_timeout_minutes: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Effort {
    /// "auto" | "fixed"
    pub mode: String,
    /// off | minimal | low | medium | high | xhigh | max
    pub fixed_level: Option<String>,
    pub max_on_free: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Compaction {
    pub percent: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Memory {
    pub enabled: bool,
    pub project: String,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Integrations {
    pub services: Vec<Service>,
    pub packages: Packages,
    /// Access level per connector (MCP server name): "read" | "ask" | "allow". A connector not listed is "ask".
    pub access: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Service {
    /// notion | tavily | figma | openpencil | rea | other
    pub id: String,
    pub name: String,
    /// "oauth" | "key"
    pub auth: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Packages {
    /// "gallery" | "npm" | "git" | "local"
    pub source: String,
    pub path: String,
    pub third_party_trusted: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct SandboxSettings {
    pub projects_dir: String,
    pub extra_writable: Vec<String>,
    pub ask_to_leave: bool,
    pub jev_review: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Git {
    pub commit_without_asking: bool,
    pub push_or_merge_without_asking: bool,
    pub ask_before_others_changes: bool,
    pub unasked_as_proposals: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Subagents {
    pub experts: Vec<Expert>,
    pub save_sessions: bool,
    pub turn_limit: Option<u32>,
    pub plan_execute: bool,
    pub auto_skills: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Expert {
    pub name: String,
    pub enabled: bool,
    pub read_only: bool,
}

impl Default for Router {
    fn default() -> Self {
        Router { enabled: true, backend: "opencode".into() }
    }
}
impl Default for Assistant {
    fn default() -> Self {
        Assistant { user_name: String::new(), name: "Aria".into(), language: "en".into(), style: "very-short".into() }
    }
}
impl Default for Models {
    fn default() -> Self {
        Models {
            daily: None,
            planning: None,
            planning_effort: "high".into(),
            allow_pay_per_use: false,
            turn_timeout_minutes: 10,
        }
    }
}
impl Default for Effort {
    fn default() -> Self {
        Effort { mode: "auto".into(), fixed_level: None, max_on_free: true }
    }
}
impl Default for Compaction {
    fn default() -> Self {
        Compaction { percent: 40 }
    }
}
impl Default for Memory {
    fn default() -> Self {
        Memory { enabled: true, project: "aria".into() }
    }
}
impl Default for Packages {
    fn default() -> Self {
        Packages { source: "gallery".into(), path: String::new(), third_party_trusted: false }
    }
}
impl Default for SandboxSettings {
    fn default() -> Self {
        SandboxSettings {
            projects_dir: "~/Projects".into(),
            extra_writable: Vec::new(),
            ask_to_leave: true,
            jev_review: true,
        }
    }
}
impl Default for Git {
    fn default() -> Self {
        Git {
            commit_without_asking: false,
            push_or_merge_without_asking: false,
            ask_before_others_changes: true,
            unasked_as_proposals: true,
        }
    }
}
impl Default for Subagents {
    fn default() -> Self {
        let expert = |name: &str, read_only: bool| Expert { name: name.into(), enabled: true, read_only };
        Subagents {
            experts: vec![
                expert("frontend", false),
                expert("backend", false),
                expert("security", true),
                expert("reviewer", true),
                expert("researcher", true),
            ],
            save_sessions: false,
            turn_limit: None,
            plan_execute: true,
            auto_skills: vec!["systematic-debugging".into(), "verification-before-completion".into()],
        }
    }
}
impl Default for Settings {
    fn default() -> Self {
        Settings {
            version: SETTINGS_VERSION,
            setup: Setup::default(),
            provider: ProviderSettings::default(),
            router: Router::default(),
            assistant: Assistant::default(),
            models: Models::default(),
            effort: Effort::default(),
            compaction: Compaction::default(),
            memory: Memory::default(),
            integrations: Integrations::default(),
            sandbox: SandboxSettings::default(),
            git: Git::default(),
            subagents: Subagents::default(),
            shortcut: String::new(),
            credentials: Credentials::default(),
        }
    }
}

impl Settings {
    /// The owner's current setup (the parity rule), with the given default conversation language.
    pub fn defaults(language: &str) -> Settings {
        let mut settings = Settings::default();
        settings.assistant.language = language.into();
        settings
    }
}

/// `es`, `pt` or `en` from a locale such as `es_CO.UTF-8`; anything else is `en`.
pub fn language_from_locale(lang: &str) -> String {
    match lang.get(..2) {
        Some("es") => "es".into(),
        Some("pt") => "pt".into(),
        _ => "en".into(),
    }
}

/// `Ok(None)` when no settings file exists yet. A file that does not parse, or has values of the
/// wrong type, is an error: the caller shows it and does not replace the file.
pub fn load(paths: &AriaPaths) -> Result<Option<Settings>, SettingsError> {
    let Some(raw) = read_json(&paths.settings())? else { return Ok(None) };
    serde_json::from_value(raw).map(Some).map_err(|e| SettingsError::Invalid(e.to_string()))
}

/// Writes `settings` over the existing file, keeping every key Aria does not know.
pub fn save(paths: &AriaPaths, settings: &Settings) -> Result<(), SettingsError> {
    let original = read_json(&paths.settings())?.unwrap_or_else(|| json!({}));
    let ours = serde_json::to_value(settings).map_err(|e| SettingsError::Invalid(e.to_string()))?;
    let merged: Value = merge_preserving(original, ours);
    write_json_atomic(&paths.settings(), &merged, 0o644)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths::AriaPaths;
    use serde_json::json;

    fn owner_defaults(language: &str) -> serde_json::Value {
        json!({
            "version": 1,
            "setup": { "completed": false, "mode": null },
            "provider": { "id": null, "quota": null },
            "router": { "enabled": true, "backend": "opencode" },
            "assistant": { "userName": "", "name": "Aria", "language": language, "style": "very-short" },
            "models": { "daily": null, "planning": null, "planningEffort": "high", "allowPayPerUse": false, "turnTimeoutMinutes": 10 },
            "effort": { "mode": "auto", "fixedLevel": null, "maxOnFree": true },
            "compaction": { "percent": 40 },
            "memory": { "enabled": true, "project": "aria" },
            "integrations": { "services": [], "packages": { "source": "gallery", "path": "", "thirdPartyTrusted": false }, "access": {} },
            "sandbox": { "projectsDir": "~/Projects", "extraWritable": [], "askToLeave": true, "jevReview": true },
            "git": { "commitWithoutAsking": false, "pushOrMergeWithoutAsking": false, "askBeforeOthersChanges": true, "unaskedAsProposals": true },
            "subagents": {
                "experts": [
                    { "name": "frontend", "enabled": true, "readOnly": false },
                    { "name": "backend", "enabled": true, "readOnly": false },
                    { "name": "security", "enabled": true, "readOnly": true },
                    { "name": "reviewer", "enabled": true, "readOnly": true },
                    { "name": "researcher", "enabled": true, "readOnly": true }
                ],
                "saveSessions": false, "turnLimit": null, "planExecute": true,
                "autoSkills": ["systematic-debugging", "verification-before-completion"]
            },
            "shortcut": "",
            "credentials": { "keyring": true }
        })
    }

    fn temp_paths() -> (tempfile::TempDir, AriaPaths) {
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths::new(dir.path().to_path_buf());
        (dir, paths)
    }

    #[test]
    fn defaults_match_owner_setup() {
        assert_eq!(serde_json::to_value(Settings::defaults("en")).unwrap(), owner_defaults("en"));
    }

    #[test]
    fn defaults_match_the_golden_file_the_ui_is_tested_against() {
        let golden: serde_json::Value =
            serde_json::from_str(include_str!("../tests/golden/settings-defaults-en.json")).unwrap();
        assert_eq!(serde_json::to_value(Settings::defaults("en")).unwrap(), golden);
    }

    #[test]
    fn a_file_without_credentials_keeps_the_keyring_on() {
        let settings: Settings = serde_json::from_str(r#"{"models":{"daily":null}}"#).unwrap();
        assert!(settings.credentials.keyring);
        assert!(Settings::defaults("en").credentials.keyring);
    }

    #[test]
    fn credentials_keyring_round_trips_false() {
        let settings: Settings = serde_json::from_str(r#"{"credentials":{"keyring":false}}"#).unwrap();
        assert!(!settings.credentials.keyring);
        assert_eq!(serde_json::to_value(&settings).unwrap()["credentials"], serde_json::json!({"keyring": false}));
    }

    #[test]
    fn turn_timeout_defaults_to_ten() {
        assert_eq!(Settings::defaults("en").models.turn_timeout_minutes, 10);
        let settings: Settings = serde_json::from_str(r#"{"models":{"daily":null}}"#).unwrap();
        assert_eq!(settings.models.turn_timeout_minutes, 10, "an older file gets the default");
        let settings: Settings = serde_json::from_str(r#"{"models":{"turnTimeoutMinutes":0}}"#).unwrap();
        assert_eq!(settings.models.turn_timeout_minutes, 0);
    }

    #[test]
    fn shortcut_defaults_to_empty() {
        assert_eq!(Settings::defaults("en").shortcut, "");
        let settings: Settings = serde_json::from_str(r#"{"version":1}"#).unwrap();
        assert_eq!(settings.shortcut, "", "an older file gets the platform's default");
        let settings: Settings = serde_json::from_str(r#"{"shortcut":"SUPER + SHIFT + A"}"#).unwrap();
        assert_eq!(settings.shortcut, "SUPER + SHIFT + A");
    }

    #[test]
    fn access_defaults_to_empty() {
        assert!(Settings::defaults("en").integrations.access.is_empty());
        let settings: Settings = serde_json::from_str(r#"{"integrations":{"services":[]}}"#).unwrap();
        assert!(settings.integrations.access.is_empty());
    }

    #[test]
    fn access_round_trips() {
        let text = r#"{"integrations":{"access":{"notion":"read","tavily":"allow"}}}"#;
        let settings: Settings = serde_json::from_str(text).unwrap();
        assert_eq!(settings.integrations.access.get("notion").map(String::as_str), Some("read"));
        let value = serde_json::to_value(&settings).unwrap();
        assert_eq!(value["integrations"]["access"], json!({"notion": "read", "tavily": "allow"}));
        assert_eq!(serde_json::from_value::<Settings>(value).unwrap(), settings);
    }

    #[test]
    fn language_from_locale_picks_supported_codes() {
        assert_eq!(language_from_locale("es_CO.UTF-8"), "es");
        assert_eq!(language_from_locale("pt_BR.UTF-8"), "pt");
        assert_eq!(language_from_locale("de_DE.UTF-8"), "en");
        assert_eq!(language_from_locale(""), "en");
    }

    #[test]
    fn save_keeps_unknown_keys() {
        let (_dir, paths) = temp_paths();
        std::fs::write(paths.settings(), r#"{"version":1,"future":{"x":1},"assistant":{"mood":"calm"}}"#).unwrap();
        save(&paths, &Settings::defaults("es")).unwrap();
        let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(paths.settings()).unwrap()).unwrap();
        assert_eq!(raw["future"]["x"], 1);
        assert_eq!(raw["assistant"]["mood"], "calm");
        assert_eq!(raw["assistant"]["language"], "es");
    }

    #[test]
    fn save_refuses_broken_file() {
        let (_dir, paths) = temp_paths();
        std::fs::write(paths.settings(), "{ nope").unwrap();
        assert!(matches!(save(&paths, &Settings::defaults("en")), Err(SettingsError::File(_))));
        assert_eq!(std::fs::read_to_string(paths.settings()).unwrap(), "{ nope");
    }

    #[test]
    fn load_round_trips_and_fills_missing_groups() {
        let (_dir, paths) = temp_paths();
        assert!(load(&paths).unwrap().is_none());
        std::fs::write(paths.settings(), r#"{"version":1,"assistant":{"userName":"Ivan"}}"#).unwrap();
        let loaded = load(&paths).unwrap().unwrap();
        assert_eq!(loaded.assistant.user_name, "Ivan");
        assert_eq!(loaded.compaction.percent, 40);
    }

    #[test]
    fn load_reports_wrong_shapes() {
        let (_dir, paths) = temp_paths();
        std::fs::write(paths.settings(), r#"{"compaction":{"percent":"lots"}}"#).unwrap();
        assert!(matches!(load(&paths), Err(SettingsError::Invalid(_))));
    }
}
