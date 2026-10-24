//! The one file Aria's Pi extensions read their settings from (`agent/aria.json`), made from Aria's `settings.json`.
//! Without it (the owner's own Pi) every extension keeps the values it has in its code; see `extensions/lib/config.ts`.
//! It never holds a secret: keys live in `auth.json` and `mcp.json`.

use crate::settings::Settings;
use serde_json::{json, Value};
use std::path::Path;

/// The config for `settings`, with `~` in folders expanded against the user's home.
pub fn extension_config(settings: &Settings) -> Value {
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from).unwrap_or_default();
    extension_config_with_home(settings, &home)
}

pub fn extension_config_with_home(settings: &Settings, home: &Path) -> Value {
    let model = |id: &str| crate::agent::split_model(id).map(|(provider, id)| json!({"provider": provider, "id": id}));
    let mut config = json!({
        "version": 1,
        "models": { "planThinking": settings.models.planning_effort, "allowPayPerUse": settings.models.allow_pay_per_use },
        // Jev comes from OpenCode Go or from TypeSafe; the extensions pick the classifier model from the backend.
        "router": {
            "enabled": settings.router.enabled && matches!(settings.router.backend.as_str(), "opencode" | "typesafe"),
            "backend": settings.router.backend,
        },
        "effort": { "mode": settings.effort.mode, "fixedLevel": settings.effort.fixed_level, "maxOnFree": settings.effort.max_on_free },
        "compaction": { "percent": settings.compaction.percent },
        "style": settings.assistant.style,
        // The conversation language by name, as the persona says it ("Spanish"); `ask_user` writes its questions in it.
        "assistant": { "language": crate::persona::language_name(&settings.assistant.language) },
        "sandbox": {
            "extraWritable": settings.sandbox.extra_writable.iter().filter_map(|dir| absolute(dir, home)).collect::<Vec<_>>(),
            "askToLeave": settings.sandbox.ask_to_leave,
            "jevReview": settings.sandbox.jev_review,
        },
        "subagents": {
            "saveSessions": settings.subagents.save_sessions,
            "turnLimit": settings.subagents.turn_limit,
            "planExecute": settings.subagents.plan_execute,
        },
        // The access level per connector; one not listed (or with an unknown level) is "ask" in the extension.
        "connectors": settings.integrations.access,
    });
    if let Some(plan) = settings.models.planning.as_deref().and_then(model) {
        config["models"]["plan"] = plan;
    }
    if let Some(quota) = &settings.provider.quota {
        config["quota"] = json!({
            "switchAt": quota.switch_at,
            "planSwitchAt": quota.plan_switch_at,
            "freeModel": model(&quota.free_model),
            "askFirst": quota.ask_first,
            "countdownMs": u64::from(quota.countdown_minutes) * 60_000,
            "onNoAnswer": quota.on_no_answer,
        });
    }
    config
}

/// `~/x` against `home`, an absolute path as it is; anything else (relative) is dropped.
fn absolute(dir: &str, home: &Path) -> Option<String> {
    let dir = dir.trim();
    if let Some(rest) = dir.strip_prefix("~/") {
        return Some(home.join(rest).display().to_string());
    }
    Path::new(dir).is_absolute().then(|| dir.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden() -> Settings {
        serde_json::from_str(include_str!("../tests/golden/settings-provider-opencode-go.json")).unwrap()
    }

    fn todays() -> Value {
        json!({
            "version": 1,
            "models": { "plan": { "provider": "opencode-go", "id": "mimo-v2.6-pro" }, "planThinking": "high", "allowPayPerUse": false },
            "quota": { "switchAt": 98, "planSwitchAt": 90, "freeModel": { "provider": "opencode-go", "id": "cat-jam-free" },
                       "askFirst": true, "countdownMs": 300000, "onNoAnswer": "switch" },
            "router": { "enabled": true, "backend": "opencode" },
            "effort": { "mode": "auto", "fixedLevel": null, "maxOnFree": true },
            "compaction": { "percent": 40 },
            "style": "very-short",
            "assistant": { "language": "English" },
            "sandbox": { "extraWritable": [], "askToLeave": true, "jevReview": true },
            "subagents": { "saveSessions": false, "turnLimit": null, "planExecute": true },
            "connectors": {}
        })
    }

    #[test]
    fn the_golden_settings_give_todays_values() {
        assert_eq!(extension_config_with_home(&golden(), Path::new("/home/u")), todays());
    }

    #[test]
    fn the_config_names_the_conversation_language_as_the_persona_does() {
        for (code, name) in
            [("es", "Spanish"), ("en", "English"), ("pt", "Portuguese"), ("French", "French"), ("", "Spanish")]
        {
            let mut settings = golden();
            settings.assistant.language = code.into();
            let config = extension_config_with_home(&settings, Path::new("/home/u"));
            assert_eq!(config["assistant"]["language"], name, "{code:?}");
        }
    }

    #[test]
    fn a_provider_without_a_free_tier_has_no_quota_section() {
        let mut settings = golden();
        settings.provider.quota = None;
        assert!(extension_config_with_home(&settings, Path::new("/home/u")).get("quota").is_none());
    }

    #[test]
    fn the_router_carries_its_backend() {
        for backend in ["typesafe", "opencode"] {
            let mut settings = golden();
            settings.router.backend = backend.into();
            let config = extension_config_with_home(&settings, Path::new("/home/u"));
            assert_eq!(config["router"]["enabled"], true);
            assert_eq!(config["router"]["backend"], backend);
        }
        let mut settings = golden();
        settings.router.backend = "typesafe".into();
        settings.router.enabled = false;
        assert_eq!(extension_config_with_home(&settings, Path::new("/home/u"))["router"]["enabled"], false);
    }

    #[test]
    fn a_relative_extra_folder_is_dropped_and_tilde_is_expanded() {
        let mut settings = golden();
        settings.sandbox.extra_writable = vec!["~/notes".into(), "relative/dir".into(), "/srv/data".into()];
        assert_eq!(
            extension_config_with_home(&settings, Path::new("/home/u"))["sandbox"]["extraWritable"],
            json!(["/home/u/notes", "/srv/data"])
        );
    }

    #[test]
    fn aria_json_carries_the_connectors() {
        let mut settings = golden();
        settings.integrations.access =
            [("notion".to_string(), "read".to_string()), ("rea".to_string(), "allow".to_string())].into();
        assert_eq!(
            extension_config_with_home(&settings, Path::new("/home/u"))["connectors"],
            json!({"notion": "read", "rea": "allow"})
        );
    }

    #[test]
    fn the_config_never_holds_a_secret() {
        let mut settings = golden();
        settings.integrations.services =
            vec![crate::settings::Service { id: "tavily".into(), name: "Tavily".into(), auth: "key".into() }];
        let text = extension_config_with_home(&settings, Path::new("/home/u")).to_string();
        assert!(!text.to_lowercase().contains("tavily") && !text.contains("key\":\""), "{text}");
    }
}
