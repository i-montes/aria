//! `~/.aria/auth.json`, in Pi's exact format so the bundled Pi can read it directly:
//! `{ "<provider>": { "type": "api_key", "key": … } }` or
//! `{ "<provider>": { "type": "oauth", "access": …, "refresh": …, "expires": <ms epoch> } }`.

use crate::json_file::{read_json, write_json_atomic, JsonFileError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Credential {
    #[serde(rename = "api_key")]
    ApiKey { key: String },
    #[serde(rename = "oauth")]
    OAuth { access: String, refresh: String, expires: i64 },
}

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error(transparent)]
    File(#[from] JsonFileError),
    #[error("could not write auth file: {0}")]
    Write(#[from] std::io::Error),
    #[error("credential for {provider} is not in a known format")]
    BadCredential { provider: String },
    #[error(transparent)]
    Vault(#[from] crate::vault::VaultError),
}

/// `auth.json`. With a vault (the dock holds the keyring) an API key goes to the keyring and the file keeps the command
/// Pi runs to read it; `get` gives the real key back.
pub struct AuthStore {
    path: PathBuf,
    root: PathBuf,
    vault: Option<crate::vault::Vault>,
}

impl AuthStore {
    /// The plain file at `path`.
    pub fn new(path: PathBuf) -> AuthStore {
        let root = path.parent().map(Path::to_path_buf).unwrap_or_default();
        AuthStore { path, root, vault: None }
    }

    /// The home's `auth.json`, through its vault when the dock opened one.
    pub fn for_paths(paths: &crate::paths::AriaPaths) -> AuthStore {
        AuthStore { path: paths.auth(), root: paths.root.clone(), vault: paths.vault.clone() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn get(&self, provider: &str) -> Result<Option<Credential>, JsonFileError> {
        let Some(Value::Object(map)) = read_json(&self.path)? else { return Ok(None) };
        let credential = map.get(provider).and_then(|v| serde_json::from_value(v.clone()).ok());
        // A key kept in the keyring is read from it; one that cannot be read stays the command (still "there").
        Ok(credential.map(|credential| match credential {
            Credential::ApiKey { key } => {
                Credential::ApiKey { key: crate::vault::resolve(self.vault.as_ref(), &key).unwrap_or(key) }
            }
            other => other,
        }))
    }

    /// `provider:type` for every credential that is really there (`anthropic:oauth`,
    /// `opencode-go:api_key`), sorted. Never the secret itself. An unreadable file lists nothing.
    pub fn summary(&self) -> Vec<String> {
        let Ok(Some(Value::Object(map))) = read_json(&self.path) else { return Vec::new() };
        let mut found: Vec<String> = map
            .iter()
            .filter_map(|(provider, value)| {
                let kind = match serde_json::from_value::<Credential>(value.clone()).ok()? {
                    Credential::ApiKey { .. } => "api_key",
                    Credential::OAuth { .. } => "oauth",
                };
                Some(format!("{provider}:{kind}"))
            })
            .collect();
        found.sort();
        found
    }

    pub fn has(&self, provider: &str) -> bool {
        matches!(self.get(provider), Ok(Some(_)))
    }

    /// Adds or replaces one provider; every OTHER provider and its unknown fields stay as they were.
    /// The provider's own entry is replaced whole (a merge would keep the old key inside a new
    /// sign-in, or the old tokens inside a new key). A file that does not parse is left untouched.
    pub fn set(&self, provider: &str, credential: &Credential) -> Result<(), AuthError> {
        let mut providers = match read_json(&self.path)? {
            None => serde_json::Map::new(),
            Some(Value::Object(map)) => map,
            Some(_) => {
                return Err(JsonFileError::Invalid {
                    path: self.path.clone(),
                    message: "expected a JSON object".into(),
                }
                .into());
            }
        };
        let stored = match (&self.vault, credential) {
            (Some(vault), Credential::ApiKey { key }) => {
                let account = format!("key:{provider}");
                vault.store(&account, key)?;
                Credential::ApiKey { key: crate::vault::command_for(&vault.exe, &self.root, &account) }
            }
            _ => credential.clone(),
        };
        let value =
            serde_json::to_value(&stored).map_err(|_| AuthError::BadCredential { provider: provider.into() })?;
        providers.insert(provider.to_string(), value);
        write_json_atomic(&self.path, &Value::Object(providers), 0o600)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::os::unix::fs::PermissionsExt;

    fn store() -> (tempfile::TempDir, AuthStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = AuthStore::new(dir.path().join("auth.json"));
        (dir, store)
    }

    #[test]
    fn with_a_vault_an_api_key_is_a_command_on_disk_and_real_in_get() {
        use crate::keyring::Backend;
        let dir = tempfile::tempdir().unwrap();
        let (paths, keyring) = crate::testing::vault_paths(dir.path());
        let store = AuthStore::for_paths(&paths);
        store.set("opencode-go", &Credential::ApiKey { key: "k1".into() }).unwrap();
        let on_disk: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(paths.auth()).unwrap()).unwrap();
        let command = crate::vault::command_for(std::path::Path::new("/bin/aria"), dir.path(), "key:opencode-go");
        assert_eq!(on_disk["opencode-go"], json!({"type": "api_key", "key": command}));
        assert_eq!(keyring.get("key:opencode-go").unwrap().as_deref(), Some("k1"));
        assert_eq!(store.get("opencode-go").unwrap(), Some(Credential::ApiKey { key: "k1".into() }));
        assert_eq!(store.summary(), vec!["opencode-go:api_key".to_string()]);
    }

    #[test]
    fn with_a_vault_oauth_goes_to_the_live_file() {
        let dir = tempfile::tempdir().unwrap();
        let (paths, _keyring) = crate::testing::vault_paths(dir.path());
        let store = AuthStore::for_paths(&paths);
        store.set("anthropic", &Credential::OAuth { access: "a".into(), refresh: "r".into(), expires: 5 }).unwrap();
        assert!(paths.auth().starts_with(dir.path().join("live")));
        assert!(!dir.path().join("auth.json").exists());
        assert_eq!(
            store.get("anthropic").unwrap(),
            Some(Credential::OAuth { access: "a".into(), refresh: "r".into(), expires: 5 })
        );
    }

    #[test]
    fn set_writes_pi_format_with_0600() {
        let (_dir, store) = store();
        store.set("opencode-go", &Credential::ApiKey { key: "k1".into() }).unwrap();
        let text = std::fs::read_to_string(store.path()).unwrap();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value, json!({"opencode-go": {"type": "api_key", "key": "k1"}}));
        assert_eq!(std::fs::metadata(store.path()).unwrap().permissions().mode() & 0o777, 0o600);
    }

    #[test]
    fn set_keeps_other_providers_and_unknown_fields() {
        let (_dir, store) = store();
        std::fs::write(
            store.path(),
            r#"{"opencode":{"type":"api_key","key":"z"},"x":{"type":"api_key","key":"y","extra":1}}"#,
        )
        .unwrap();
        store.set("anthropic", &Credential::OAuth { access: "a".into(), refresh: "r".into(), expires: 5 }).unwrap();
        let value: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(store.path()).unwrap()).unwrap();
        assert_eq!(value["opencode"], json!({"type": "api_key", "key": "z"}));
        assert_eq!(value["x"], json!({"type": "api_key", "key": "y", "extra": 1}));
        assert_eq!(value["anthropic"]["type"], "oauth");
    }

    #[test]
    fn switching_a_provider_between_key_and_sign_in_leaves_no_stale_secret() {
        let (_dir, store) = store();
        store.set("anthropic", &Credential::ApiKey { key: "old-key".into() }).unwrap();
        store.set("anthropic", &Credential::OAuth { access: "a".into(), refresh: "r".into(), expires: 7 }).unwrap();
        let after_oauth: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(store.path()).unwrap()).unwrap();
        assert_eq!(after_oauth["anthropic"], json!({"type": "oauth", "access": "a", "refresh": "r", "expires": 7}));
        store.set("anthropic", &Credential::ApiKey { key: "new-key".into() }).unwrap();
        let after_key: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(store.path()).unwrap()).unwrap();
        assert_eq!(after_key["anthropic"], json!({"type": "api_key", "key": "new-key"}));
    }

    #[test]
    fn summary_names_each_credential_by_provider_and_type_and_never_the_secret() {
        let (_dir, store) = store();
        assert!(store.summary().is_empty());
        store.set("opencode-go", &Credential::ApiKey { key: "k".into() }).unwrap();
        store.set("anthropic", &Credential::OAuth { access: "a".into(), refresh: "r".into(), expires: 1 }).unwrap();
        let summary = store.summary();
        assert_eq!(summary, vec!["anthropic:oauth".to_string(), "opencode-go:api_key".to_string()]);
        assert!(summary.iter().all(|s| !s.contains('k') || s.contains("api_key") || s.contains("opencode")));
        std::fs::write(store.path(), "{ nope").unwrap();
        assert!(store.summary().is_empty());
    }

    #[test]
    fn oauth_round_trip() {
        let (_dir, store) = store();
        let cred = Credential::OAuth { access: "a".into(), refresh: "r".into(), expires: 1_792_000_000_000 };
        store.set("anthropic", &cred).unwrap();
        assert_eq!(store.get("anthropic").unwrap(), Some(cred));
        assert!(store.has("anthropic"));
        assert!(!store.has("opencode-go"));
    }

    #[test]
    fn broken_auth_is_error_not_overwritten() {
        let (_dir, store) = store();
        std::fs::write(store.path(), "{ nope").unwrap();
        assert!(store.set("opencode-go", &Credential::ApiKey { key: "k".into() }).is_err());
        assert_eq!(std::fs::read_to_string(store.path()).unwrap(), "{ nope");
    }
}
