//! The person's own secrets (passwords, API keys, tokens) that Aria keeps for them without the model ever seeing one
//! (docs/superpowers/plans/2026-10-22-secrets.md). Each value is a keyring account `user:<name>`; `secrets.json` in the
//! home holds the names, what each one is for and when it was saved, never a value. A command uses one as the
//! environment variable `ARIA_SECRET_<NAME>`.

use crate::paths::AriaPaths;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The longest name: enough for `github-token-work-laptop`, short enough to read in a list.
pub const MAX_NAME: usize = 48;

/// One saved secret, as the index and the page see it: no value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub name: String,
    #[serde(default)]
    pub purpose: String,
    /// When it was saved, in ms since the epoch.
    #[serde(default)]
    pub saved: i64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Index {
    #[serde(default)]
    secrets: Vec<Entry>,
}

/// A name the person and the model can both say: lower case letters, digits and dashes, starting with a letter or a
/// digit. Upper case and spaces are folded (`GitHub token` → `github-token`); anything else is refused.
pub fn valid_name(raw: &str) -> Result<String, String> {
    let folded: String =
        raw.trim().chars().map(|c| if c == ' ' || c == '_' { '-' } else { c.to_ascii_lowercase() }).collect();
    let ok = !folded.is_empty()
        && folded.len() <= MAX_NAME
        && folded.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !folded.starts_with('-')
        && !folded.ends_with('-');
    if ok {
        Ok(folded)
    } else {
        Err(format!("a secret's name is letters, digits and dashes (at most {MAX_NAME}): '{raw}'"))
    }
}

/// The keyring account of a secret.
pub fn account(name: &str) -> String {
    format!("user:{name}")
}

/// The environment variable a command reads it from: `github-token` → `ARIA_SECRET_GITHUB_TOKEN`.
pub fn env_name(name: &str) -> String {
    format!("ARIA_SECRET_{}", name.to_ascii_uppercase().replace('-', "_"))
}

/// The index of names, in the home.
pub fn index_path(paths: &AriaPaths) -> PathBuf {
    paths.root.join("secrets.json")
}

fn read_index(paths: &AriaPaths) -> Index {
    std::fs::read_to_string(index_path(paths))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn write_index(paths: &AriaPaths, index: &Index) -> Result<(), String> {
    let text = serde_json::to_string_pretty(index).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&paths.root).map_err(|e| e.to_string())?;
    crate::vault::write_private(&index_path(paths), text.as_bytes())
        .map_err(|e| format!("the index could not be saved: {e}"))
}

/// The saved secrets, by name.
pub fn list(paths: &AriaPaths) -> Vec<Entry> {
    let mut entries = read_index(paths).secrets;
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

/// Keeps `value` under `name`: in the keyring (read back to check it), then its name in the index. Saving a name again
/// replaces the value and the purpose. Without the keyring (`aria keyring off`) nothing is saved: `keyring-off`.
pub fn store(paths: &AriaPaths, raw_name: &str, value: &str, purpose: &str, now: i64) -> Result<Entry, String> {
    let name = valid_name(raw_name)?;
    let value = value.trim();
    if value.is_empty() {
        return Err("the secret is empty".into());
    }
    let vault = paths.vault.as_ref().ok_or("keyring-off")?;
    vault.store(&account(&name), value).map_err(|e| format!("keyring-failed: {e}"))?;
    let mut index = read_index(paths);
    index.secrets.retain(|e| e.name != name);
    let entry = Entry { name, purpose: purpose.trim().to_string(), saved: now };
    index.secrets.push(entry.clone());
    write_index(paths, &index)?;
    Ok(entry)
}

/// Forgets a secret: its value from the keyring and its name from the index. A name that is not saved is not an error.
pub fn delete(paths: &AriaPaths, raw_name: &str) -> Result<(), String> {
    let name = valid_name(raw_name)?;
    if let Some(vault) = paths.vault.as_ref() {
        vault.keyring.delete(&account(&name)).map_err(|e| format!("keyring-failed: {e}"))?;
    }
    let mut index = read_index(paths);
    let before = index.secrets.len();
    index.secrets.retain(|e| e.name != name);
    if index.secrets.len() != before {
        write_index(paths, &index)?;
    }
    Ok(())
}

/// The prefixes of keys that look like one on their own (the same list as `ui/chat/secrets.js` and
/// `extensions/lib/secrets.ts`).
const PREFIXES: [&str; 9] = ["sk-", "ghp_", "gho_", "github_pat_", "glpat-", "xoxb-", "xoxp-", "AKIA", "AIza"];

/// The value in `text` to keep: the value of `key` in a dotenv (`KEY=value`, `export KEY="value"`), JSON
/// (`"key": "value"`) or YAML (`key: value`) line; without a key, the one key-shaped string in the text. Never
/// returns more than one value: several candidates, or none, is an error naming what to pass.
pub fn extract(text: &str, key: Option<&str>) -> Result<String, String> {
    if let Some(key) = key.map(str::trim).filter(|k| !k.is_empty()) {
        for line in text.lines() {
            if let Some(value) = value_of(line, key) {
                return Ok(value);
            }
        }
        return Err(format!("no `{key}` in the file"));
    }
    let mut found: Vec<String> = Vec::new();
    for word in text.split(|c: char| c.is_whitespace() || "\"'=:,;()[]{}<>`".contains(c)) {
        if PREFIXES.iter().any(|p| word.starts_with(p)) && word.len() >= 16 && !found.iter().any(|f| f == word) {
            found.push(word.to_string());
        }
    }
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => Err("no key-shaped value in the file: say which variable holds it (`key`)".into()),
        n => Err(format!("{n} key-shaped values in the file: say which variable holds the one to keep (`key`)")),
    }
}

/// The value of `key` on one line, in the three forms `extract` reads.
fn value_of(line: &str, key: &str) -> Option<String> {
    let line = line.trim();
    let line = line.strip_prefix("export ").unwrap_or(line).trim_start();
    let rest = if let Some(rest) = line.strip_prefix(key) {
        rest
    } else {
        let quoted = format!("\"{key}\"");
        line.strip_prefix(&quoted)?
    };
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('=').or_else(|| rest.strip_prefix(':'))?;
    let value = rest.trim().trim_end_matches(',').trim();
    let value = if value.len() >= 2
        && ((value.starts_with('"') && value.ends_with('"')) || (value.starts_with('\'') && value.ends_with('\'')))
    {
        &value[1..value.len() - 1]
    } else {
        // An unquoted dotenv or YAML value ends at a comment.
        value.split(" #").next().unwrap_or(value).trim()
    };
    (!value.is_empty()).then(|| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyring::Backend;
    use crate::testing::vault_paths;

    #[test]
    fn names_are_folded_and_checked() {
        assert_eq!(valid_name("GitHub token").unwrap(), "github-token");
        assert_eq!(valid_name("openai_key").unwrap(), "openai-key");
        assert!(valid_name("").is_err());
        assert!(valid_name("-x").is_err());
        assert!(valid_name("a/b").is_err());
        assert!(valid_name(&"a".repeat(MAX_NAME + 1)).is_err());
        assert_eq!(env_name("github-token"), "ARIA_SECRET_GITHUB_TOKEN");
        assert_eq!(account("github-token"), "user:github-token");
    }

    #[test]
    fn a_secret_goes_to_the_keyring_and_only_its_name_to_the_index() {
        let dir = tempfile::tempdir().unwrap();
        let (paths, keyring) = vault_paths(dir.path());
        let entry = store(&paths, "GitHub token", "ghp_0123456789abcdefghij", "the CI", 7).unwrap();
        assert_eq!(entry, Entry { name: "github-token".into(), purpose: "the CI".into(), saved: 7 });
        assert_eq!(keyring.get("user:github-token").unwrap().as_deref(), Some("ghp_0123456789abcdefghij"));
        let index = std::fs::read_to_string(index_path(&paths)).unwrap();
        assert!(!index.contains("ghp_"), "the index never holds a value: {index}");
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(index_path(&paths)).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(list(&paths), vec![entry]);
    }

    #[test]
    fn saving_again_replaces_and_delete_forgets() {
        let dir = tempfile::tempdir().unwrap();
        let (paths, keyring) = vault_paths(dir.path());
        store(&paths, "db", "one", "", 1).unwrap();
        store(&paths, "db", "two", "the database", 2).unwrap();
        assert_eq!(keyring.get("user:db").unwrap().as_deref(), Some("two"));
        assert_eq!(list(&paths).len(), 1);
        delete(&paths, "db").unwrap();
        assert_eq!(keyring.get("user:db").unwrap(), None);
        assert!(list(&paths).is_empty());
        delete(&paths, "db").unwrap();
    }

    #[test]
    fn without_the_keyring_nothing_is_saved() {
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths::new(dir.path().to_path_buf());
        assert_eq!(store(&paths, "x", "value", "", 1).unwrap_err(), "keyring-off");
        assert!(!index_path(&paths).exists());
        assert!(store(&paths, "x", "  ", "", 1).is_err(), "an empty value is refused before the keyring");
    }

    #[test]
    fn the_value_of_a_variable_in_dotenv_json_or_yaml() {
        let env = "# keys\nexport OPENAI_API_KEY=\"sk-abc123\"\nDB_PASSWORD=hunter2 # local\n";
        assert_eq!(extract(env, Some("OPENAI_API_KEY")).unwrap(), "sk-abc123");
        assert_eq!(extract(env, Some("DB_PASSWORD")).unwrap(), "hunter2");
        assert_eq!(extract("{\n  \"token\": \"t-1\",\n}", Some("token")).unwrap(), "t-1");
        assert_eq!(extract("api:\n  key: 'k 2'\n", Some("key")).unwrap(), "k 2");
        assert!(extract(env, Some("MISSING")).is_err());
    }

    #[test]
    fn without_a_variable_only_one_key_shaped_value_is_taken() {
        assert_eq!(extract("token=ghp_0123456789abcdefghij\nname=x", None).unwrap(), "ghp_0123456789abcdefghij");
        let two = "A=sk-proj-0123456789abcdef\nB=ghp_0123456789abcdefghij";
        assert!(extract(two, None).unwrap_err().contains("2 key-shaped"));
        assert!(extract("PASSWORD=hunter2", None).unwrap_err().contains("which variable"));
    }
}
