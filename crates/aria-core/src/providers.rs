//! Provider modules. A module declares what a provider supports (models, free models, quota,
//! whether Jev comes with it); the rest of Aria only reads these declarations. To add a provider,
//! add a `ProviderModule` to `providers()` (see docs/adding-a-provider.md).

use crate::settings::{Quota, Settings};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthKind {
    ApiKey,
    OAuth,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: &'static str,
    pub label: &'static str,
}

#[derive(Debug, PartialEq, Eq)]
pub struct QuotaDefaults {
    pub switch_at: u8,
    pub plan_switch_at: u8,
    pub free_model: &'static str,
    pub ask_first: bool,
    pub countdown_minutes: u32,
    pub on_no_answer: &'static str,
}

#[derive(Debug)]
pub struct ProviderModule {
    pub id: &'static str,
    pub label: &'static str,
    pub auth: AuthKind,
    /// The keys of `auth.json` this provider's credential is stored under.
    pub auth_entries: &'static [&'static str],
    pub models: &'static [ModelInfo],
    pub free_models: &'static [ModelInfo],
    /// `Some` only for providers whose quota can be read (and that have a free tier to fall back to).
    pub quota: Option<QuotaDefaults>,
    /// The provider's own account reaches Jev, so no separate Jev key is needed.
    pub includes_jev: bool,
    pub daily_default: &'static str,
    pub planning_default: &'static str,
}

const fn model(id: &'static str, label: &'static str) -> ModelInfo {
    ModelInfo { id, label }
}

static PROVIDERS: [ProviderModule; 3] = [
    ProviderModule {
        id: "opencode-go",
        label: "OpenCode Go",
        auth: AuthKind::ApiKey,
        auth_entries: &["opencode-go", "opencode"],
        models: &[
            model("opencode-go/mimo-v2.6-flash", "MiMo V2.6 Flash"),
            model("opencode-go/mimo-v2.6-pro", "MiMo V2.6 Pro"),
            model("opencode-go/deepseek-v4.1-flash", "DeepSeek V4.1 Flash"),
            model("opencode-go/glm-5.3", "GLM 5.3"),
            model("opencode-go/kimi-k3", "Kimi K3"),
            model("opencode-go/qwen3.8-max", "Qwen 3.8 Max"),
        ],
        free_models: &[
            model("opencode-go/cat-jam-free", "Cat Jam (free)"),
            model("opencode-go/longcat-2.5-preview-free", "Longcat 2.5 preview (free)"),
        ],
        quota: Some(QuotaDefaults {
            switch_at: 98,
            plan_switch_at: 90,
            free_model: "opencode-go/cat-jam-free",
            ask_first: true,
            countdown_minutes: 5,
            on_no_answer: "switch",
        }),
        includes_jev: true,
        daily_default: "opencode-go/mimo-v2.6-flash",
        planning_default: "opencode-go/mimo-v2.6-pro",
    },
    ProviderModule {
        id: "anthropic-api",
        label: "Anthropic · API key",
        auth: AuthKind::ApiKey,
        auth_entries: &["anthropic"],
        models: &[
            model("anthropic/claude-sonnet-5-5", "Claude Sonnet 5.5"),
            model("anthropic/claude-opus-5-5", "Claude Opus 5.5"),
            model("anthropic/claude-haiku-5-5", "Claude Haiku 5.5"),
            model("anthropic/claude-fable-5-1", "Claude Fable 5.1"),
        ],
        free_models: &[],
        quota: None,
        includes_jev: false,
        daily_default: "anthropic/claude-sonnet-5-5",
        planning_default: "anthropic/claude-opus-5-5",
    },
    ProviderModule {
        id: "anthropic-oauth",
        label: "Anthropic · Sign in with Claude",
        auth: AuthKind::OAuth,
        auth_entries: &["anthropic"],
        models: &[
            model("claude-bridge/claude-sonnet-5-5", "Claude Sonnet 5.5"),
            model("claude-bridge/claude-opus-5-5", "Claude Opus 5.5"),
            model("claude-bridge/claude-haiku-5-5", "Claude Haiku 5.5"),
            model("claude-bridge/claude-fable-5-1", "Claude Fable 5.1"),
        ],
        free_models: &[],
        quota: None,
        includes_jev: false,
        daily_default: "claude-bridge/claude-sonnet-5-5",
        planning_default: "claude-bridge/claude-opus-5-5",
    },
];

pub fn providers() -> &'static [ProviderModule] {
    &PROVIDERS
}

pub fn provider(id: &str) -> Option<&'static ProviderModule> {
    PROVIDERS.iter().find(|p| p.id == id)
}

/// Points `settings` at a provider: its default models, quota (or none), and the router, effort and
/// Jev-review defaults that make sense with it. An unknown id changes nothing.
pub fn apply_provider(settings: &mut Settings, id: &str) {
    let Some(module) = provider(id) else { return };
    settings.provider.id = Some(module.id.into());
    settings.provider.quota = module.quota.as_ref().map(|q| Quota {
        switch_at: q.switch_at,
        plan_switch_at: q.plan_switch_at,
        free_model: q.free_model.into(),
        ask_first: q.ask_first,
        countdown_minutes: q.countdown_minutes,
        on_no_answer: q.on_no_answer.into(),
    });
    settings.models.daily = Some(module.daily_default.into());
    settings.models.planning = Some(module.planning_default.into());
    if module.includes_jev {
        settings.router.enabled = true;
        settings.router.backend = "opencode".into();
        settings.effort.mode = "auto".into();
        settings.effort.fixed_level = None;
        settings.sandbox.jev_review = true;
    } else {
        settings.router.enabled = false;
        settings.router.backend = "typesafe".into();
        settings.effort.mode = "fixed".into();
        settings.effort.fixed_level = Some("medium".into());
        settings.sandbox.jev_review = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::Settings;

    #[test]
    fn only_three_providers() {
        let ids: Vec<_> = providers().iter().map(|p| p.id).collect();
        assert_eq!(ids, ["opencode-go", "anthropic-api", "anthropic-oauth"]);
    }

    #[test]
    fn apply_opencode_go() {
        let mut s = Settings::defaults("en");
        apply_provider(&mut s, "opencode-go");
        let quota = s.provider.quota.as_ref().unwrap();
        assert_eq!((quota.switch_at, quota.plan_switch_at), (98, 90));
        assert_eq!(quota.free_model, "opencode-go/cat-jam-free");
        assert!(quota.ask_first);
        assert_eq!((quota.countdown_minutes, quota.on_no_answer.as_str()), (5, "switch"));
        assert_eq!(s.models.daily.as_deref(), Some("opencode-go/mimo-v2.6-flash"));
        assert_eq!(s.models.planning.as_deref(), Some("opencode-go/mimo-v2.6-pro"));
        assert_eq!((s.router.enabled, s.router.backend.as_str()), (true, "opencode"));
        assert_eq!(s.effort.mode, "auto");
        assert!(s.sandbox.jev_review);
    }

    #[test]
    fn apply_anthropic_oauth() {
        let mut s = Settings::defaults("en");
        apply_provider(&mut s, "opencode-go");
        apply_provider(&mut s, "anthropic-oauth");
        assert!(s.provider.quota.is_none());
        assert_eq!(s.provider.id.as_deref(), Some("anthropic-oauth"));
        assert_eq!(s.models.planning.as_deref(), Some("claude-bridge/claude-opus-5-5"));
        assert!(!s.router.enabled);
        assert_eq!(s.router.backend, "typesafe");
        assert_eq!((s.effort.mode.as_str(), s.effort.fixed_level.as_deref()), ("fixed", Some("medium")));
        assert!(!s.sandbox.jev_review);
    }

    // Golden files: the web UI (ui/onboarding/steps.js) is tested against the same JSON, so the two
    // implementations of the provider defaults cannot drift apart. Regenerate after an intended
    // change with `cargo test -p aria-core regenerate_golden -- --ignored`.
    fn after_provider(id: &str) -> serde_json::Value {
        let mut settings = Settings::defaults("en");
        apply_provider(&mut settings, id);
        serde_json::to_value(settings).unwrap()
    }

    fn golden(text: &str) -> serde_json::Value {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn settings_after_each_provider_match_the_golden_files() {
        assert_eq!(
            after_provider("opencode-go"),
            golden(include_str!("../tests/golden/settings-provider-opencode-go.json"))
        );
        assert_eq!(
            after_provider("anthropic-api"),
            golden(include_str!("../tests/golden/settings-provider-anthropic-api.json"))
        );
        assert_eq!(
            after_provider("anthropic-oauth"),
            golden(include_str!("../tests/golden/settings-provider-anthropic-oauth.json"))
        );
    }

    #[test]
    #[ignore = "rewrites the golden files"]
    fn regenerate_golden() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, value: serde_json::Value| {
            std::fs::write(dir.join(name), serde_json::to_string_pretty(&value).unwrap() + "\n").unwrap();
        };
        write("settings-defaults-en.json", serde_json::to_value(Settings::defaults("en")).unwrap());
        for id in ["opencode-go", "anthropic-api", "anthropic-oauth"] {
            write(&format!("settings-provider-{id}.json"), after_provider(id));
        }
    }

    #[test]
    fn opencode_go_writes_both_auth_entries() {
        assert_eq!(provider("opencode-go").unwrap().auth_entries, ["opencode-go", "opencode"]);
        assert_eq!(provider("anthropic-api").unwrap().auth_entries, ["anthropic"]);
        assert!(provider("openai").is_none());
    }
}
