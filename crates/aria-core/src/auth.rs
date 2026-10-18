//! `~/.aria/auth.json`, in Pi's exact format so the bundled Pi can read it directly:
//! `{ "<provider>": { "type": "api_key", "key": … } }` or
//! `{ "<provider>": { "type": "oauth", "access": …, "refresh": …, "expires": <ms epoch> } }`.

use crate::json_file::{merge_preserving, read_json, write_json_atomic, JsonFileError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
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
}

pub struct AuthStore {
    path: PathBuf,
}

impl AuthStore {
    pub fn new(path: PathBuf) -> AuthStore {
        AuthStore { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn get(&self, provider: &str) -> Result<Option<Credential>, JsonFileError> {
        let Some(Value::Object(map)) = read_json(&self.path)? else { return Ok(None) };
        Ok(map.get(provider).and_then(|v| serde_json::from_value(v.clone()).ok()))
    }

    pub fn has(&self, provider: &str) -> bool {
        matches!(self.get(provider), Ok(Some(_)))
    }

    /// Adds or replaces one provider; every other provider and unknown field stays as it was.
    /// A file that exists but does not parse is left untouched and reported.
    pub fn set(&self, provider: &str, credential: &Credential) -> Result<(), AuthError> {
        let original = read_json(&self.path)?.unwrap_or_else(|| json!({}));
        let ours = json!({ provider: serde_json::to_value(credential).map_err(|_| AuthError::BadCredential { provider: provider.into() })? });
        write_json_atomic(&self.path, &merge_preserving(original, ours), 0o600)?;
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
        store
            .set("anthropic", &Credential::OAuth { access: "a".into(), refresh: "r".into(), expires: 5 })
            .unwrap();
        let value: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(store.path()).unwrap()).unwrap();
        assert_eq!(value["opencode"], json!({"type": "api_key", "key": "z"}));
        assert_eq!(value["x"], json!({"type": "api_key", "key": "y", "extra": 1}));
        assert_eq!(value["anthropic"]["type"], "oauth");
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
