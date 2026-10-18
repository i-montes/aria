//! JSON files Aria owns: read without guessing, write atomically, never drop keys we don't know.

use serde_json::Value;
use std::io::{self, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum JsonFileError {
    #[error("{path}: {message}")]
    Invalid { path: PathBuf, message: String },
    #[error("{path}: {source}")]
    Io { path: PathBuf, source: io::Error },
}

/// `Ok(None)` when the file does not exist; `Err(Invalid)` when it exists but does not parse.
pub fn read_json(path: &Path) -> Result<Option<Value>, JsonFileError> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(JsonFileError::Io { path: path.to_path_buf(), source }),
    };
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|e| JsonFileError::Invalid { path: path.to_path_buf(), message: e.to_string() })
}

/// Pretty JSON plus a trailing newline: temp file in the same directory, fsync, rename, chmod.
pub fn write_json_atomic(path: &Path, value: &Value, mode: u32) -> io::Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(&tmp)?;
        file.set_permissions(std::fs::Permissions::from_mode(mode))?;
        file.write_all(serde_json::to_string_pretty(value)?.as_bytes())?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Deep merge of objects: keys only in `original` stay, keys in `ours` win, other values are replaced.
pub fn merge_preserving(original: Value, ours: Value) -> Value {
    match (original, ours) {
        (Value::Object(mut base), Value::Object(new)) => {
            for (key, value) in new {
                let merged = match base.remove(&key) {
                    Some(old) => merge_preserving(old, value),
                    None => value,
                };
                base.insert(key, merged);
            }
            Value::Object(base)
        }
        (_, ours) => ours,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn read_missing_is_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_json(&dir.path().join("nope.json")).unwrap().is_none());
    }

    #[test]
    fn read_invalid_is_error() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("bad.json");
        std::fs::write(&file, "{ nope").unwrap();
        assert!(matches!(read_json(&file), Err(JsonFileError::Invalid { .. })));
    }

    #[test]
    fn write_is_atomic_and_mode() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("auth.json");
        write_json_atomic(&file, &json!({"a": 1}), 0o600).unwrap();
        assert_eq!(read_json(&file).unwrap().unwrap(), json!({"a": 1}));
        assert_eq!(std::fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600);
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn merge_keeps_unknown_keys() {
        let merged = merge_preserving(
            json!({"a": 1, "x": {"keep": true, "b": 1}}),
            json!({"x": {"b": 2}}),
        );
        assert_eq!(merged, json!({"a": 1, "x": {"keep": true, "b": 2}}));
    }
}
