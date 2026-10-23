//! The voice model's file: Whisper large-v3-turbo quantized to q5_0, from whisper.cpp's own repository on Hugging Face.
//! It is not part of the install; the first dictation asks the owner, then it is downloaded to a `.part` file, checked
//! (size and SHA-256) and only then put in its place, so a cut or a corrupt download never looks like a model.

use std::path::{Path, PathBuf};

pub const MODEL_FILE: &str = "ggml-large-v3-turbo-q5_0.bin";
pub const SIZE: u64 = 574_041_195;
pub const SHA256: &str = "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2";
pub const URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin";

#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    /// The download did not complete (no network, a server error, a write that failed).
    Failed(String),
    /// It completed, but it is not the model (the size or the hash differs).
    Broken,
}

/// Where the model lives: `<aria home>/models/`.
pub fn path(paths: &aria_core::paths::AriaPaths) -> PathBuf {
    paths.root.join("models").join(MODEL_FILE)
}

/// The model is there, whole (its size; the hash was checked when it was downloaded).
pub fn present(path: &Path, size: u64) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.len() == size)
}

/// Downloads `url` to `dest` (through `dest.part`), checks its size and hash, then moves it in place. `progress` hears
/// the share done (0..1), at most a hundred times, and 1.0 once it is in place.
pub fn download(
    url: &str,
    dest: &Path,
    size: u64,
    sha256: &str,
    mut progress: impl FnMut(f64),
) -> Result<(), ModelError> {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Write};
    let failed = |e: &dyn std::fmt::Display| ModelError::Failed(e.to_string());
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).map_err(|e| failed(&e))?;
    }
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let response = ureq::get(url).call().map_err(|e| failed(&e))?;
    let mut body = response.into_reader();
    let mut file = std::fs::File::create(&part).map_err(|e| failed(&e))?;
    let mut hasher = Sha256::new();
    let mut done: u64 = 0;
    let mut told = 0u64;
    let mut buf = vec![0u8; 1 << 16];
    let result = loop {
        let n = match body.read(&mut buf) {
            Ok(0) => break Ok(()),
            Ok(n) => n,
            Err(e) => break Err(failed(&e)),
        };
        if let Err(e) = file.write_all(&buf[..n]) {
            break Err(failed(&e));
        }
        hasher.update(&buf[..n]);
        done += n as u64;
        let percent = (done * 100).checked_div(size).unwrap_or(0).min(99);
        if percent > told {
            told = percent;
            progress(percent as f64 / 100.0);
        }
    };
    drop(file);
    let hash: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let checked = result.and_then(|()| if done == size && hash == sha256 { Ok(()) } else { Err(ModelError::Broken) });
    match checked {
        Ok(()) => {
            std::fs::rename(&part, dest).map_err(|e| failed(&e))?;
            progress(1.0);
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::io::{Read, Write};

    /// Serves `body` once over HTTP on a local port; returns the URL.
    fn serve(body: Vec<u8>) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 2048];
                let _ = stream.read(&mut buf);
                let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body);
            }
        });
        format!("http://127.0.0.1:{port}/model.bin")
    }

    fn hex(data: &[u8]) -> String {
        Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn a_complete_file_is_present_and_a_short_one_is_not() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join(MODEL_FILE);
        std::fs::write(&file, vec![0u8; 10]).unwrap();
        assert!(present(&file, 10));
        assert!(!present(&file, 11));
        assert!(!present(&dir.path().join("missing"), 10));
    }

    #[test]
    fn download_verifies_then_renames() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("models").join(MODEL_FILE);
        let body: Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
        let url = serve(body.clone());
        let mut last = 0.0;
        download(&url, &dest, body.len() as u64, &hex(&body), |p| last = p).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(!PathBuf::from(format!("{}.part", dest.display())).exists());
        assert_eq!(last, 1.0);
    }

    #[test]
    fn a_wrong_hash_is_broken_and_leaves_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join(MODEL_FILE);
        let body = vec![7u8; 512];
        let url = serve(body);
        assert_eq!(download(&url, &dest, 512, &"0".repeat(64), |_| {}), Err(ModelError::Broken));
        assert!(!dest.exists());
        assert!(!PathBuf::from(format!("{}.part", dest.display())).exists());
    }

    #[test]
    fn a_failed_connection_is_failed() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join(MODEL_FILE);
        let got = download("http://127.0.0.1:9/none", &dest, 1, SHA256, |_| {});
        assert!(matches!(got, Err(ModelError::Failed(_))), "{got:?}");
    }
}
